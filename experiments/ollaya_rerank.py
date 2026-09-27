"""Ollaya 판단 모델을 같은 영어 후보로 비교합니다. 운영 DB와 설정은 열지 않습니다."""
import argparse
import json
import math
from pathlib import Path
import platform
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.gliner_decide import atomic_report, digest, metrics
from experiments.gliner_followup import RELATION_LABELS, summarize
from experiments.gliner_translation import inputs, translated_pair
from experiments.lev_rerank import PROBES, QUESTION, relevance_answer


def api(base_url, path, body=None):
    request = Request(base_url.rstrip('/') + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(f'Ollaya HTTP {error.code}: {error.read(500).decode(errors="replace")}') from error


def checked_answer(response, model):
    if (response['model'] != model or response['state_truncated'] is not False
            or response['done_reason'] != 'decide'
            or type(response['usage']['output_tokens']) is not int
            or response['usage']['output_tokens'] != 0):
        raise ValueError('Unexpected model, truncated state or generated output')
    if set(response['answers']) != {'relevance'}:
        raise ValueError('Missing or extra answers')
    answer = response['answers']['relevance']
    probabilities = answer['probabilities']
    if (answer['type'] != 'choice' or not isinstance(probabilities, dict)
            or set(probabilities) != set(RELATION_LABELS)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1., abs_tol=.0005)
            or answer['choice'] not in probabilities
            or probabilities[answer['choice']] != max(probabilities.values())):
        raise ValueError('Invalid Ollaya probability distribution')
    # API는 소수점 4자리로 반올림합니다. 허용 오차를 확인한 분포만 합계 1로 정규화합니다.
    total = sum(probabilities.values())
    return relevance_answer({name: p / total for name, p in probabilities.items()})


def snapshot(base_url, model):
    tag = next(m for m in api(base_url, '/api/tags')['models'] if m['name'] == model)
    show = api(base_url, '/api/show', {'model': model})
    return dict(version=api(base_url, '/api/version')['version'], model=model,
                digest=tag['digest'], details=tag['details'], model_info=show['model_info'])


def run(args):
    rows, titles, _ = inputs(args.source, args.previous)
    cache_path = args.translations / 'translations.json'
    prior_path = args.baseline / 'results.json'
    cache, prior = (json.loads(p.read_text()) for p in (cache_path, prior_path))
    assert cache['complete'] and not cache['errors'] and prior['complete']
    for path in (cache_path, args.source / 'dataset.json', args.source / 'candidates.json',
                 args.previous / 'challenge.json'):
        assert digest(path) == prior['protocol']['hashes'][str(path)]
    paths = [cache_path, prior_path, args.source / 'dataset.json', args.source / 'candidates.json',
             args.previous / 'challenge.json', Path(__file__),
             *[Path(__file__).with_name(p) for p in ('lev_rerank.py', 'gliner_translation.py',
                                                   'gliner_followup.py', 'gliner_decide.py')]]
    server = snapshot(args.base_url, args.model)
    load_start = time.perf_counter()
    api(args.base_url, '/api/decide', dict(model=args.model, keep_alive='10m'))
    load_seconds = time.perf_counter() - load_start
    loaded = next(m for m in api(args.base_url, '/api/ps')['models'] if m['name'] == args.model)
    assert loaded['digest'] == server['digest']
    protocol = dict(date='2026-09-27', server=server, device=loaded['device'],
        base_url=args.base_url, hashes={str(p): digest(p) for p in paths}, question=QUESTION,
        pair_boost=.1, no_match_threshold=.5, probability_rounding_tolerance=.0005,
        identical_input_reuse='within each question only', python=platform.python_version(),
        limits=['Fixed synthetic questions and saved English translations including known errors',
                'Different model and runtime from LEV/CLM; not a same-model speedup',
                'No training, prompt tuning, operating DB, fresh retrieval or production promotion'])
    target, contract = args.output / 'results.json', args.output / 'protocol.json'
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError('서버·모델·입력·코드가 달라졌습니다. 새 출력 폴더를 사용하세요.')
    report = json.loads(target.read_text()) if target.exists() else dict(complete=False,
        protocol=protocol, development=[], challenge=[], sessions=[])
    assert report['protocol'] == protocol
    if report['complete']:
        print('Already complete; results preserved.')
        return
    atomic_report(contract, protocol)
    session = dict(load_request_seconds=load_seconds, probe=[])
    for text in PROBES:
        started = time.perf_counter()
        response = api(args.base_url, '/api/decide', dict(model=args.model, state=text, questions=QUESTION))
        answer = checked_answer(response, args.model)
        session['probe'].append(dict(answer=answer, seconds=time.perf_counter()-started, response=response))
    report['sessions'].append(session)
    completed = [r['id'] for section in ('development', 'challenge') for r in report[section]]
    assert len(completed) == len(set(completed)) and set(completed) <= {r['case']['id'] for r in rows}
    for item in rows:
        case, hits = item['case'], item['hits']
        if case['id'] in completed:
            continue
        memo, responses = {}, []
        def predict(texts):
            answers = []
            # 원문 후보는 유지하고, 같은 문항 안의 동일 영어 입력만 계산을 재사용합니다.
            for text in texts:
                if text not in memo:
                    response = api(args.base_url, '/api/decide',
                                   dict(model=args.model, state=text, questions=QUESTION))
                    responses.append(response)
                    memo[text] = checked_answer(response, args.model)
                answers.append(memo[text])
            return answers
        ranked, info = translated_pair(case['question'], hits, predict, cache['entries'], exact=item['exact'])
        info.update(model_calls=len(responses), responses=responses,
                    reused_inputs=len(hits)-len(responses) if info['status'] == 'ok' else 0)
        previous = next(r for r in prior[item['section']] if r['id'] == case['id'])
        methods = dict(previous['methods'])
        boosted = hits
        if info['status'] == 'ok':
            values = [(1+.1*p)/(60+i) for i, p in enumerate(info['scores'], 1)]
            boosted = [h for _, h in sorted(zip(values, hits), key=lambda pair: -pair[0])]
        for method, selected in (('ollaya_rank', ranked), ('ollaya_boost', boosted)):
            assert sorted(h['id'] for h in hits) == sorted(h['id'] for h in selected)
            methods[method] = dict(ids=[h['id'] for h in selected], metrics=metrics(case, selected, titles))
        report[item['section']].append(dict(id=case['id'], language=item['language'], pair=info, methods=methods))
        atomic_report(target, report)
        print(f'evaluate {len(report["development"])+len(report["challenge"])}/{len(rows)} '
              f'{case["id"]} {info["status"]} {info["seconds"]:.3f}s', flush=True)
    assert snapshot(args.base_url, args.model) == server
    final_model = next(m for m in api(args.base_url, '/api/ps')['models'] if m['name'] == args.model)
    assert final_model['device'] == loaded['device'] and final_model['digest'] == loaded['digest']
    session['loaded_model_after'] = final_model
    for path, expected in protocol['hashes'].items():
        assert digest(path) == expected
    report['summary'] = {s: summarize(report[s]) for s in ('development', 'challenge')}
    report['by_language'] = {lang: summarize([r for r in report['challenge'] if r['language'] == lang])
                             for lang in ('ko', 'en', 'zh')}
    report['complete'] = True
    atomic_report(target, report)
    print('Complete: fixed questions, Ollaya ranking and bounded boost.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='laya:en')
    for name, default in (('source', 'outputs/gliner-decide-20260926'),
            ('previous', 'outputs/gliner-followup-20260926'),
            ('translations', 'outputs/gliner-translation-20260926'),
            ('baseline', 'outputs/clm-20260927/fp16-cpu-stage'),
            ('output', 'outputs/ollaya-20260927/laya-en')):
        parser.add_argument('--'+name, type=Path, default=Path(default))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
