"""고정 영어 번역으로 LEV를 비교합니다. 운영 DB·검색·설정은 사용하지 않습니다."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

from experiments.gliner_decide import CAPACITY, atomic_report, digest, metrics
from experiments.gliner_followup import PROMPT, RELATION_LABELS, summarize
from experiments.gliner_translation import inputs, translated_pair

QUESTION = {'relevance': {'type': 'choice', 'instructions': PROMPT, 'criteria': RELATION_LABELS}}
# 평가 정답과 무관한 장치 비교 입력입니다. 이 점수로 프롬프트를 조정하지 않습니다.
PROBES = [
    'Query: Pump noise\nCase: Abnormal pump noise during operation.',
    'Query: Pump noise\nCase: Pump operates silently without abnormal noise.',
    'Query: Flow is exactly 7.2 L/min\nCase: Flow is exactly 7.2 L/min.',
    'Query: Flow is exactly 7.2 L/min\nCase: Flow is exactly 72 L/min.',
    'Query: Heater is slow to warm up\nCase: Heater requires longer than normal to reach the set temperature.',
    'Query: A suspected leak\nCase: A leak has been confirmed.',
]


def relevance_answer(probabilities):
    # LEV의 confidence는 분포의 집중도입니다. 관련성 점수로 쓰면 의미가 달라집니다.
    if (not isinstance(probabilities, dict) or set(probabilities) != set(RELATION_LABELS)
            or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-6)):
        raise ValueError('Invalid LEV probability distribution')
    label = max(probabilities, key=probabilities.get)
    return {'relevance': {'label': label, 'confidence': probabilities[label]}}


def move_engine(engine, device):
    import torch
    engine.model.to(device)
    if engine.mode_b_head is not None:
        engine.mode_b_head.to(device)
    engine._device = torch.device(device)  # 공식 로더는 CUDA/CPU만 자동 선택합니다.
    assert {p.device.type for p in engine.model.parameters()} == {device}
    torch.mps.synchronize()
    torch.mps.empty_cache()


def infer(engine, text):
    from lev.model import _QUESTIONS, serving_routes
    # 잘린 문장으로 평가하지 않도록 실제 스키마·두 선택지 순서를 포함해 길이를 확인합니다.
    questions = _QUESTIONS.validate_python(QUESTION)
    routes = serving_routes(questions, engine.tokenizer, engine.config)
    prefix, variants = engine._render(text, questions, routes)
    count = len(engine.tokenizer.encode(prefix, add_special_tokens=True)) + max(len(v.suffix_ids) for v in variants)
    if count > CAPACITY:
        raise ValueError(f'Input capacity exceeded: {count}>{CAPACITY}')
    result = engine.system_one(text, QUESTION)
    assert result.usage.output_tokens == 0
    return relevance_answer(result.answers['relevance'].probabilities), count


def probe(engine, output):
    import torch
    report = {'complete': False, 'inputs': PROBES, 'devices': {}, 'tolerance': .02}
    for device in ('mps', 'cpu'):
        move_engine(engine, device)
        started = time.perf_counter()
        infer(engine, PROBES[0])
        row = {'warmup_seconds': time.perf_counter()-started, 'results': []}
        for text in PROBES:
            started = time.perf_counter()
            value, count = infer(engine, text)
            row['results'].append(dict(answer=value, tokens=count, seconds=time.perf_counter()-started))
            print(f'probe {device} {len(row["results"])}/{len(PROBES)} {row["results"][-1]}', flush=True)
        if device == 'mps':
            row['allocated_bytes'] = torch.mps.current_allocated_memory()
            row['driver_bytes'] = torch.mps.driver_allocated_memory()
        report['devices'][device] = row
        atomic_report(output / 'device-probe.json', report)
    from experiments.gliner_followup import relevance
    pairs = list(zip(report['devices']['cpu']['results'], report['devices']['mps']['results'], strict=True))
    report['max_score_difference'] = max(abs(relevance(a['answer'])-relevance(b['answer'])) for a, b in pairs)
    report['same_labels'] = all(a['answer']['relevance']['label'] == b['answer']['relevance']['label'] for a, b in pairs)
    report['complete'] = True
    atomic_report(output / 'device-probe.json', report)
    if not report['same_labels'] or report['max_score_difference'] > report['tolerance']:
        raise ValueError('CPU/MPS probe differs beyond the fixed tolerance; inspect before evaluation.')


def run(args):
    import torch
    import lev
    if not torch.backends.mps.is_available():
        raise RuntimeError('이 실험에는 MPS GPU가 필요합니다.')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('CPU fallback을 끄고 GPU 실행 여부를 명확히 확인하세요.')
    torch.set_num_threads(4)
    rows, titles, _ = inputs(args.source, args.previous)
    cache_path = args.translations / 'translations.json'
    old_path = args.translations / 'results.json'
    cache, old = (json.loads(path.read_text()) for path in (cache_path, old_path))
    assert cache['complete'] and not cache['errors'] and old['complete']
    assert digest(cache_path) == old['protocol']['translation_sha256']
    assert digest(args.source / 'candidates.json') == old['protocol']['candidate_sha256']
    verification = json.loads((args.model_root / 'model-verification.json').read_text())
    for folder in ('base', 'adapter'):
        for name, checksum in verification[folder]['files'].items():
            assert digest(args.model_root / folder / name) == checksum
    assert all(importlib.metadata.version(name) == version for name, version in verification['packages'].items())
    paths = [cache_path, old_path, args.source / 'dataset.json', args.source / 'candidates.json',
             args.previous / 'challenge.json', Path(__file__),
             Path(__file__).with_name('gliner_translation.py'), Path(__file__).with_name('gliner_followup.py'),
             Path(__file__).with_name('gliner_decide.py'), args.model_root / 'runtime-adapter/lev_release.json']
    protocol = dict(date='2026-09-26', hashes={str(p): digest(p) for p in paths}, models=verification,
                    device='mps', dtype='bfloat16', cpu_threads=4, question=QUESTION, capacity=CAPACITY,
                    order_average=True, pair_boost=.1, no_match_threshold=.5,
                    identical_input_reuse='within each question only',
                    python=platform.python_version(), platform=platform.platform(),
                    limits=['Frozen synthetic development and correlated diagnostic questions',
                            'Saved English translations unchanged, including known semantic errors',
                            'No database access, fresh retrieval, answer generation or production promotion'])
    target = args.output / 'results.json'
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('기존 실행 조건이 다릅니다. 새 출력 폴더를 사용하세요.')
    report = json.loads(target.read_text()) if target.exists() else dict(complete=False, protocol=protocol,
                development=[], challenge=[], sessions=[])
    if report['protocol'] != protocol:
        raise ValueError('실행 조건이 달라졌습니다. 새 출력 폴더를 사용하세요.')
    if report['complete']:
        print('Already complete; results preserved.')
        return
    atomic_report(protocol_path, protocol)
    started = time.perf_counter()
    engine = lev.load(args.model_root / 'runtime-adapter')
    session = dict(load_seconds=time.perf_counter()-started)
    assert engine.config.order_average and not engine.config.compile and engine.config.prefix_mode == 'single'
    if not (args.output / 'device-probe.json').exists():
        probe(engine, args.output)
    check = json.loads((args.output / 'device-probe.json').read_text())
    assert check['complete'] and check['same_labels'] and check['max_score_difference'] <= check['tolerance']
    move_engine(engine, 'mps')
    started = time.perf_counter()
    infer(engine, PROBES[0])
    session['warmup_seconds'] = time.perf_counter()-started
    session['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
    session['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
    report['sessions'].append(session)
    completed = {r['id'] for section in ('development', 'challenge') for r in report[section]}
    assert completed <= {r['case']['id'] for r in rows}
    for item in rows:
        case, hits = item['case'], item['hits']
        if case['id'] in completed:
            continue
        counts = []
        memo = {}
        def predict(texts):
            # ponytail: 기본 LEV API로 후보를 순차 평가합니다. 배치 최적화는 동등성 확인 후 별도 비교합니다.
            result = []
            for text in texts:
                if text not in memo:
                    memo[text] = infer(engine, text)
                answer, count = memo[text]
                counts.append(count)
                result.append(answer)
            return result
        ranked, info = translated_pair(case['question'], hits, predict, cache['entries'], exact=item['exact'])
        info['model_calls'] = len(memo)
        info['reused_inputs'] = len(hits)-len(memo) if info['status'] == 'ok' else 0
        prior = next(r for r in old[item['section']] if r['id'] == case['id'])
        methods = {name: prior['methods'][key] for name, key in
                   (('baseline', 'baseline'), ('gliner_rank', 'both_rank'), ('gliner_boost', 'both_boost'))}
        boosted = hits
        if info['status'] == 'ok':
            values = [(1+.1*p)/(60+i) for i, p in enumerate(info['scores'], 1)]
            boosted = [hit for _, hit in sorted(zip(values, hits), key=lambda pair: -pair[0])]
        for name, selected in (('lev_rank', ranked), ('lev_boost', boosted)):
            assert sorted(h['id'] for h in selected) == sorted(h['id'] for h in hits)
            methods[name] = dict(ids=[h['id'] for h in selected], metrics=metrics(case, selected, titles))
        report[item['section']].append(dict(id=case['id'], language=item['language'], pair=info,
                                           token_counts=counts, methods=methods))
        atomic_report(target, report)
        print(f'evaluate {len(report["development"])+len(report["challenge"])}/{len(rows)} '
              f'{case["id"]} {info["status"]} {info["seconds"]:.3f}s', flush=True)
    report['summary'] = {section: summarize(report[section]) for section in ('development', 'challenge')}
    report['by_language'] = {lang: summarize([r for r in report['challenge'] if r['language'] == lang])
                             for lang in ('ko', 'en', 'zh')}
    for path, expected in protocol['hashes'].items():
        assert digest(path) == expected
    report['complete'] = True
    atomic_report(target, report)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/gliner-decide-20260926'))
    parser.add_argument('--previous', type=Path, default=Path('outputs/gliner-followup-20260926'))
    parser.add_argument('--translations', type=Path, default=Path('outputs/gliner-translation-20260926'))
    parser.add_argument('--model-root', type=Path, default=Path('outputs/lev-20260926'))
    parser.add_argument('--output', type=Path, default=Path('outputs/lev-20260926'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
