"""고정 후보의 질문만/질문·사례 영어 번역 비교. 번역 캐시 외 운영 자료는 열지 않습니다."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import time

from experiments.gliner_decide import CAPACITY, FIELDS, source_cells, digest, metrics, atomic_report, freeze
from experiments.gliner_followup import (PROMPT, RELATION_LABELS, case_text,
                                       pair_rerank, summarize)

ENGLISH_FIELDS = dict(zip(FIELDS, ('Title', 'Problem symptom', 'Symptom', 'Product family',
    'Unit name', 'Alarm code', 'Equipment', 'Originating module', 'Originating unit (part)')))
CJK = r'[가-힣\u3400-\u9fff]'
UNIT_INSTRUCTION = ('\nTranslate every Korean and Chinese phrase into English, including phrases containing English acronyms. '
                    'Keep questions as questions; do not answer them. Preserve every __CS_NUMBER_ placeholder '
                    'exactly once in its original order. Never add units or digits after placeholders. '
                    'Do not leave any Korean or Chinese characters in the output values.')
REPAIR_INSTRUCTION = ('\nCorrection: translate clause by clause in the source order. Use short English clauses '
                      'instead of moving measurements earlier in the sentence. All placeholders must appear '
                      'in exactly the same sequence as the input. Previous validation error (data): ')


def validate_unit(original, english):
    from ragkit.english import validate_translation
    validate_translation({'text': english}, {'text': original})
    if re.search(CJK, english):
        raise ValueError('Korean or Chinese text remains untranslated.')
    if re.findall(r'__CS_NUMBER_\d+__', original) != re.findall(r'__CS_NUMBER_\d+__', english):
        raise ValueError('Numeric placeholders changed.')


def inputs(source, previous):
    dataset = json.loads((source / 'dataset.json').read_text())['dataset']
    candidates = json.loads((source / 'candidates.json').read_text())
    old = json.loads((previous / 'results.json').read_text())
    assert old['complete'] and candidates['complete']
    assert freeze(dataset)[1] == candidates['freeze_hash']
    assert digest(source / 'results.json') == old['protocol']['source_results_hash']
    assert digest(Path(__file__).with_name('gliner_followup.py')) == old['protocol']['script_hash']
    assert digest(source / 'candidates.json') == old['protocol']['candidate_hash']
    challenge = json.loads((previous / 'challenge.json').read_text())
    original_challenge = Path(__file__).with_name('gliner_conditions.json')
    assert digest(original_challenge) == old['protocol']['challenge_hash']
    assert challenge == json.loads(original_challenge.read_text())
    groups = challenge['groups']
    rows = []
    for case, entry, prior in zip(dataset['cases'], candidates['results'], old['development'], strict=True):
        assert case['id'] == entry['id'] == prior['id'] and case['question'] == entry['question']
        rows.append(dict(case=case, hits=entry['hits'], prior=prior, section='development', language='ko',
                         exact=entry['details']['vector_search']['mode'] == 'exact'))
    for group in groups:
        for lang, question in group['queries'].items():
            case = dict(id=f"{group['id']}-{lang}", question=question,
                        expected_ids=[] if group['expected'] is None else [str(group['expected'])])
            prior = next(r for r in old['challenge'] if r['id'] == case['id'])
            hits = [dict(id=str(i), evidence=[{'cells': {'현상': text}}]) for i, text in enumerate(group['cases'])]
            rows.append(dict(case=case, hits=hits, prior=prior, section='challenge', language=lang, exact=False))
    return rows, dataset['template_titles'], old


def translated_pair(question, hits, predict, translations, *, query_only=False, exact=False):
    # 원문 후보는 그대로 두고 모델에 전달하는 문자열만 바꿉니다. 채점·인용은 원문 기준입니다.
    def english_predict(texts):
        query = translations[question]
        pairs = {f'Query: {question}\nCase: {case_text(hit)}':
                 f"Query: {query}\nCase: {case_text(hit) if query_only else translations[case_text(hit)]}"
                 for hit in hits}
        return predict([pairs[text] for text in texts])
    return pair_rerank(question, hits, english_predict, exact=exact)


def translate(rows, output, base_url, model):
    from openai import OpenAI
    from ragkit.english import request_json, validate_translation, TRANSLATE_PROMPT
    from ragkit.llm import protect_numbers, restore_numbers
    # 정답·점수·조치 결과 없이 동일한 원문 문자열은 한 번만 번역합니다.
    texts = list(dict.fromkeys(text for row in rows if not row['exact'] and row['hits']
                             for text in [row['case']['question'], *map(case_text, row['hits'])]))
    originals = list(dict.fromkeys(text for row in rows if not row['exact'] and row['hits']
        for text in [row['case']['question'], *[value for hit in row['hits']
            for key, value in source_cells(hit).items() if key in FIELDS and value.strip()]]))
    protected = {text: protect_numbers(text) for text in originals}
    units = list(dict.fromkeys(value[0] for value in protected.values()))
    config = dict(model=model, reasoning_effort='low', temperature=0, max_tokens=6000)
    signature = dict(base_url=base_url, config=config, prompt=TRANSLATE_PROMPT + UNIT_INSTRUCTION, texts=texts,
                     units=units, fields=ENGLISH_FIELDS,
                     helper_sha256=digest(Path('ragkit/english.py')), numeric_helper_sha256=digest(Path('ragkit/llm.py')))
    path = output / 'translations.json'
    cache = json.loads(path.read_text()) if path.exists() else dict(signature=signature, entries={}, units={}, errors={}, batches=[])
    if cache['signature'] != signature:
        raise ValueError('번역 조건이 바뀌었습니다. 새 출력 폴더를 사용하세요.')
    for original, english in cache['units'].items():
        validate_unit(original, english)
    for unit in units:
        if not re.search(CJK, unit):
            cache['units'][unit] = unit  # 이미 영어인 질문·코드는 재작성하지 않습니다.
    pending = [text for text in units if text not in cache['units']]
    print(f"translation units={len(units)} pending={len(pending)}", flush=True)
    client = OpenAI(base_url=base_url, api_key='EMPTY', timeout=240, max_retries=0)
    for start in range(0, len(pending), 6):
        batch = pending[start:start+6]
        cells = {f'text_{i}': text for i, text in enumerate(batch)}
        started = time.perf_counter()
        # 이미 숫자를 보호했으므로 다시 보호하는 translate_cells 대신 같은 JSON 요청 함수를 사용합니다.
        pending_cells = cells
        previous_errors = {text: cache['errors'][text] for text in batch if text in cache['errors']}
        error = json.dumps(previous_errors, ensure_ascii=False) if previous_errors else None
        attempts = []
        for attempt in range(3):
            if not pending_cells:
                break
            try:
                prompt = signature['prompt'] + (REPAIR_INSTRUCTION + json.dumps(error) if error else '')
                # low에서 반복 실패한 문장만 high로 보완합니다. 실제 설정은 결과에 남깁니다.
                request_config = config | {'reasoning_effort': 'high' if previous_errors or attempt == 2 else 'low'}
                attempts.append(request_config)
                values = request_json(client, request_config, prompt, pending_cells)
                if not isinstance(values, dict) or set(values) != set(pending_cells):
                    raise ValueError('Translation keys differ from requested keys.')
            except Exception as failure:
                error = type(failure).__name__ + ': ' + str(failure)[:500]
                cache['errors'].update({text: error for text in pending_cells.values()})
                continue
            failed = {}
            for key, original in pending_cells.items():
                try:
                    validate_unit(original, values[key])
                    # 표시자의 주변에 새 숫자·단위를 붙였는지도 복원 시점에 검증합니다.
                    for text, (masked, tokens, prefix) in protected.items():
                        if masked == original:
                            restored = json.loads(restore_numbers(json.dumps({'text': values[key]}),
                                ({}, {'text': (tokens, prefix)}, None), {'id': 'english', 'text': text}))
                            validate_translation(restored, {'text': text})
                    cache['units'][original] = values[key]
                    cache['errors'].pop(original, None)
                except ValueError as failure:
                    error = str(failure)
                    cache['errors'][original] = error
                    failed[key] = original
            pending_cells = failed
        cache['batches'].append(dict(texts=len(batch), seconds=time.perf_counter()-started,
                                    error=error if pending_cells else None, repair_instruction=REPAIR_INSTRUCTION,
                                    attempts=attempts))
        cache['complete'] = False
        atomic_report(path, cache)
        print(f"translation {len(cache['units'])}/{len(units)} errors={len(cache['errors'])}", flush=True)
    if len(cache['units']) != len(units):
        raise ValueError('번역 미완료: translations.json을 유지하고 같은 명령으로 재개할 수 있습니다.')
    restored = {}
    for original, (masked, tokens, prefix) in protected.items():
        value = restore_numbers(json.dumps({'text': cache['units'][masked]}),
            ({}, {'text': (tokens, prefix)}, None), {'id': 'english', 'text': original})
        restored[original] = json.loads(value)['text']
        validate_translation({'text': restored[original]}, {'text': original})
    for row in rows:
        if row['exact'] or not row['hits']:
            continue
        query = row['case']['question']
        cache['entries'][query] = restored[query]
        for hit in row['hits']:
            cells = source_cells(hit)
            original = case_text(hit)
            english = '\n'.join(f'{ENGLISH_FIELDS[key]}: {restored[cells[key]]}'
                                for key in FIELDS if cells.get(key, '').strip())
            validate_translation({'text': english}, {'text': original})
            cache['entries'][original] = english
    cache['complete'] = len(cache['entries']) == len(texts)
    atomic_report(path, cache)


def evaluate(rows, titles, old, source, previous, output):
    target = output / 'results.json'
    if target.exists():
        raise ValueError('기존 평가 결과를 덮어쓰지 않습니다.')
    cache = json.loads((output / 'translations.json').read_text())
    assert cache['complete'] and not cache['errors']
    expected = {text for row in rows if not row['exact'] and row['hits']
                for text in [row['case']['question'], *map(case_text, row['hits'])]}
    assert set(cache['entries']) == set(cache['signature']['texts']) == expected
    for name, checksum in old['protocol']['model_files'].items():
        assert digest(source / 'model' / name) == checksum
    assert all(importlib.metadata.version(name) == version for name, version in old['protocol']['packages'].items())
    import torch
    from gliner2 import AutoExtractor
    torch.set_num_threads(4)
    protocol = dict(date='2026-09-26', previous_sha256=digest(previous / 'results.json'),
                    dataset_sha256=digest(source / 'dataset.json'), candidate_sha256=digest(source / 'candidates.json'),
                    challenge_sha256=digest(previous / 'challenge.json'), translation_sha256=digest(output / 'translations.json'),
                    script_sha256=digest(__file__), previous_protocol=old['protocol'],
                    prompt=PROMPT, labels=RELATION_LABELS, capacity=CAPACITY, batch_size=4,
                    pair_boost=.1, no_match_threshold=.5, translation_repair=REPAIR_INSTRUCTION,
                    limits=['Frozen synthetic candidates; no fresh retrieval or database access',
                            'Existing English questions kept verbatim; numeric spans protected then restored',
                            'No tuning on results; original source retained for scoring'])
    atomic_report(output / 'protocol.json', protocol)
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(str(source / 'model'), local_files_only=True)
    model.eval()
    schema = model.create_schema().classification('relevance', RELATION_LABELS, prompt=PROMPT)
    report = dict(complete=False, protocol=protocol, load_seconds=time.perf_counter()-started,
                  development=[], challenge=[])
    counts = []
    def predict(texts):
        for text in texts:
            batch = model.processor.collate_fn_inference([(text, schema.build())], max_len=None,
                        architecture='boundary', error_policy='raise')
            count = batch.input_ids.shape[-1]
            counts.append(count)
            if count > CAPACITY:
                raise ValueError(f'Input capacity exceeded: {count}>{CAPACITY}')
        return model.batch_extract(texts, schema, batch_size=4, include_confidence=True, max_len=None)
    started = time.perf_counter()
    predict(['Query: Pump noise\nCase: Abnormal pump noise during operation'])
    report['warmup_seconds'] = time.perf_counter()-started
    for item in rows:
        case, hits, prior = item['case'], item['hits'], item['prior']
        methods = {'baseline': prior['methods']['baseline'], 'original_rank': prior['methods']['pair_rank']}
        row = dict(id=case['id'], language=item['language'], methods=methods, arms={})
        for arm in ('query', 'both'):
            ranked, info = translated_pair(case['question'], hits, predict, cache['entries'],
                                           query_only=arm == 'query', exact=item['exact'])
            row['arms'][arm] = info
            boosted = hits
            if info['status'] == 'ok':
                scores = [(1+.1*p)/(60+i) for i, p in enumerate(info['scores'], 1)]
                boosted = [hit for _, hit in sorted(zip(scores, hits), key=lambda pair: -pair[0])]
            for suffix, selected in (('rank', ranked), ('boost', boosted)):
                assert sorted(h['id'] for h in hits) == sorted(h['id'] for h in selected)
                methods[f'{arm}_{suffix}'] = dict(ids=[h['id'] for h in selected], metrics=metrics(case, selected, titles))
        row['pair'] = row['arms']['both']
        report[item['section']].append(row)
        atomic_report(target, report)
        print(f"evaluate {len(report['development'])+len(report['challenge'])}/{len(rows)} {case['id']} "
              f"query={row['arms']['query']['status']} both={row['pair']['status']}", flush=True)
    report['summary'] = {key: summarize(report[key]) for key in ('development', 'challenge')}
    report['by_language'] = {lang: summarize([r for r in report['challenge'] if r['language'] == lang])
                             for lang in ('ko', 'en', 'zh')}
    report['max_input_tokens'] = max(counts)
    for path, expected in ((previous / 'results.json', protocol['previous_sha256']),
                           (source / 'dataset.json', protocol['dataset_sha256']),
                           (source / 'candidates.json', protocol['candidate_sha256']),
                           (previous / 'challenge.json', protocol['challenge_sha256']),
                           (output / 'translations.json', protocol['translation_sha256'])):
        assert digest(path) == expected
    report['complete'] = True
    atomic_report(target, report)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('translate', 'evaluate'))
    parser.add_argument('--source', type=Path, default=Path('outputs/gliner-decide-20260926'))
    parser.add_argument('--previous', type=Path, default=Path('outputs/gliner-followup-20260926'))
    parser.add_argument('--output', type=Path, default=Path('outputs/gliner-translation-20260926'))
    parser.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    parser.add_argument('--model', default='openai/gpt-oss-20b')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, titles, old = inputs(args.source, args.previous)
    if args.phase == 'translate':
        translate(rows, args.output, args.base_url, args.model)
    else:
        evaluate(rows, titles, old, args.source, args.previous, args.output)
