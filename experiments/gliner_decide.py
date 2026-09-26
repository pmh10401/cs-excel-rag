"""고정 검색 후보에서만 실행하는 GLiNER 실험. DB·운영 검색·설정을 열거나 수정하지 않습니다.

python -m experiments.gliner_decide --source outputs/laya-coreml-20260922 \
  --output outputs/gliner-decide-20260926 --model outputs/gliner-decide-20260926/model
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time

from ragkit.retrieval_evaluation import atomic_report, freeze, score, source_cells

MODEL = 'fastino/GLiNER2.5-multi-Decide'
REVISION = '6bc1d43d201b0691e733626389af8c57eea3ea68'
LABELS = ('noise', 'vibration', 'leak', 'pressure_abnormal', 'flow_abnormal',
          'temperature_abnormal', 'motion_failure', 'contamination', 'electrical_fault',
          'communication_fault', 'other')
FIELDS = ('제목', '문제현상', '현상', '제품군', 'UNIT명', '알람코드', '설비', '기인 모듈', '기인 Unit(부품)')
THRESHOLD, BOOST, CAPACITY = .5, .1, 512


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tags(result):
    """모델의 잘못된 응답을 정상 분류나 0점으로 숨기지 않습니다."""
    if not isinstance(result, dict) or set(result) != {'symptom'} or not isinstance(result['symptom'], list):
        raise ValueError('Invalid symptom response')
    found = set()
    for item in result['symptom']:
        label, value = item['label'], item['confidence']
        if label not in LABELS or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid label or confidence')
        if value >= THRESHOLD and label != 'other':
            found.add(label)
    return found


def rerank(question, hits, predict, *, exact=False):
    if exact or not hits:
        return list(hits), {'status': 'skipped', 'seconds': 0}
    started = time.perf_counter()
    try:
        texts = [question]
        for hit in hits:
            cells = source_cells(hit)
            texts.append('\n'.join(f'{key}: {cells[key]}' for key in FIELDS if cells.get(key, '').strip()))
        if any(not text.strip() for text in texts):
            raise ValueError('Empty symptom fields')
        predictions = predict(texts)  # 정답·점수·접수번호·작업자·조치 결과는 전달하지 않습니다.
        if len(predictions) != len(texts):
            raise ValueError('Incomplete prediction batch')
        classified = [tags(item) for item in predictions]
        query = classified[0]
        similarities = [len(query & doc) / len(query | doc) if query | doc else 0 for doc in classified[1:]]
        # ponytail: 후보 20개 안의 최대 10% 가점만 평가합니다. 후보 밖 정답은 복구하지 못합니다.
        values = [(1 + BOOST * overlap) / (60 + rank) for rank, overlap in enumerate(similarities, 1)]
        ranked = [hit for _, hit in sorted(zip(values, hits), key=lambda pair: -pair[0])]
        info = {'status': 'ok', 'tags': [sorted(value) for value in classified], 'overlap': similarities}
    except Exception as error:
        ranked = list(hits)
        info = {'status': 'fallback', 'error': type(error).__name__ + ': ' + str(error)[:300]}
    info['seconds'] = time.perf_counter() - started
    return ranked, info


def metrics(case, hits, titles):
    result = score(case, hits, titles)
    expected = set(case.get('expected_ids', case.get('expected_templates', [])))
    lookup = {title: label for label, title in titles.items()}
    if 'expected_templates' in case:
        labels = [{lookup.get(source_cells(hit).get('제목', '').strip())} - {None} for hit in hits]
    else:
        labels = [{hit['id'], hit.get('record_id', hit['id'].removesuffix('/문서')),
                   *hit.get('evidence_ids', [])} for hit in hits]
    relevant = [bool(value & expected) for value in labels]
    result.update(precision_at_5=sum(relevant[:5]) / len(hits[:5]) if hits else 0.,
                  hit_at_5=any(relevant[:5]) if expected else None,
                  candidate_hit=any(relevant) if expected else None,
                  unexpected_hits=bool(hits) if not expected else None)
    return result


def run(source, output, model_path):
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'results.json'
    if target.exists():
        raise ValueError('기존 평가 결과를 덮어쓰지 않습니다. 새 출력 폴더를 사용하세요.')
    frozen = json.loads((source / 'dataset.json').read_text())
    candidates = json.loads((source / 'candidates.json').read_text())
    dataset, signature = freeze(frozen['dataset'])
    assert candidates['complete'] and signature == frozen['freeze_hash'] == candidates['freeze_hash']
    assert candidates['db_sha256_before'] == candidates['db_sha256_after']
    assert [(c['id'], c['question']) for c in dataset['cases']] == [(r['id'], r['question']) for r in candidates['results']]
    assert all(len({h['id'] for h in row['hits']}) == len(row['hits']) <= 20 for row in candidates['results'])
    files = {str(p.relative_to(model_path)): digest(p) for p in model_path.rglob('*')
             if p.is_file() and '.cache' not in p.parts}
    report = {'complete': False, 'date': '2026-09-26', 'model': MODEL, 'revision': REVISION,
              'freeze_hash': signature, 'candidate_sha256': digest(source / 'candidates.json'),
              'script_sha256': digest(__file__), 'model_files': files,
              'source_records': candidates['source_records'], 'python': platform.python_version(),
              'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name)
                  for name in ('gliner2', 'torch', 'transformers', 'tokenizers', 'protobuf', 'sentencepiece')},
              'protocol': {'labels': LABELS, 'fields': FIELDS, 'threshold': THRESHOLD, 'boost': BOOST,
                  'capacity': CAPACITY, 'device': 'cpu', 'threads': 4, 'batch_size': 4,
                  'primary': 'precision_at_5', 'no_regression_guard': 'hit_at_5',
                  'baseline': 'Frozen 2026-09-22 Korean hybrid candidates; not fresh retrieval latency',
                  'source_provenance': dataset['provenance'],
                  'limitations': ['Known synthetic regression questions, not independent field evaluation',
                      'Same candidate pool; no translation or answer generation',
                      'No learned tags used as citation evidence; no DB/UI/runtime dependency changes',
                      'No parameter tuning against evaluation labels; CPU latency measured locally']},
              'results': []}
    atomic_report(output / 'protocol.json', {k: v for k, v in report.items() if k != 'results'})
    # 입력을 복사해 고정합니다. 운영 DB를 열지 않고 이전 읽기 전용 검색 결과만 사용합니다.
    atomic_report(output / 'dataset.json', frozen)
    atomic_report(output / 'candidates.json', candidates)
    import torch
    from gliner2 import AutoExtractor
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    report['load_seconds'] = time.perf_counter() - started
    schema = model.create_schema().classification('symptom', list(LABELS), multi_label=True, cls_threshold=THRESHOLD)
    token_counts = []
    def predict(texts):
        # 실제 스키마와 텍스트를 함께 인코딩해 길이를 확인합니다. 잘라서 평가하지 않습니다.
        for text in texts:
            batch = model.processor.collate_fn_inference([(text, schema.build())],
                        max_len=None, architecture='boundary', error_policy='raise')
            count = batch.input_ids.shape[-1]
            token_counts.append(count)
            if count > CAPACITY:
                raise ValueError(f'Input capacity exceeded: {count}>{CAPACITY}')
        return model.batch_extract(texts, schema, batch_size=4, include_confidence=True, max_len=None)
    started = time.perf_counter()
    predict(['펌프 운전 중 큰 소음과 진동이 발생함.'])
    report['warmup_seconds'] = time.perf_counter() - started
    atomic_report(target, report)
    for case, entry in zip(dataset['cases'], candidates['results']):
        exact = entry['details']['vector_search']['mode'] == 'exact'
        ranked, info = rerank(entry['question'], entry['hits'], predict, exact=exact)
        before = metrics(case, entry['hits'], dataset['template_titles'])
        after = metrics(case, ranked, dataset['template_titles'])
        assert sorted(h['id'] for h in entry['hits']) == sorted(h['id'] for h in ranked)
        report['results'].append({'id': case['id'], 'category': case['category'],
            'baseline_ids': [h['id'] for h in entry['hits']], 'gliner_ids': [h['id'] for h in ranked],
            'baseline': before, 'gliner': after, 'classification': info})
        atomic_report(target, report)
        print(f"{len(report['results'])}/{len(dataset['cases'])} {case['id']} {info['status']} "
              f"P@5={before['precision_at_5']:.2f}->{after['precision_at_5']:.2f} +{info['seconds']:.3f}s", flush=True)
    entries, summary = report['results'], {}
    for metric in ('top1', 'hit_at_3', 'precision_at_3', 'precision_at_5', 'hit_at_5', 'candidate_hit'):
        measured = [e for e in entries if e['baseline'][metric] is not None]
        summary[metric] = {method: statistics.mean(float(e[method][metric]) for e in measured)
                           for method in ('baseline', 'gliner')}
        summary[metric]['n'] = len(measured)
    for name, metric, before, after in (('top1_improved', 'top1', False, True),
            ('top1_regressed', 'top1', True, False), ('hit_at_5_lost', 'hit_at_5', True, False)):
        summary[name] = [e['id'] for e in entries if e['baseline'][metric] is before and e['gliner'][metric] is after]
    latencies = sorted(e['classification']['seconds'] for e in entries if e['classification']['status'] != 'skipped')
    summary.update(questions=len(entries), applied=len(latencies),
        fallbacks=sum(e['classification']['status'] == 'fallback' for e in entries),
        unexpected_hits=sum(e['gliner']['unexpected_hits'] is True for e in entries),
        changed=sum(e['baseline_ids'] != e['gliner_ids'] for e in entries),
        extra_mean_seconds=statistics.mean(latencies), extra_median_seconds=statistics.median(latencies),
        extra_p95_seconds=latencies[math.ceil(.95 * len(latencies))-1], max_input_tokens=max(token_counts))
    assert digest(source / 'candidates.json') == report['candidate_sha256']
    report.update(complete=True, summary=summary)
    atomic_report(target, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.model)
