"""고정 합성 36건: 원문 규칙 → 실제 LLM 독립 번역 → 운영 영어 NLI를 대조합니다.

운영 DB·설정·기존 결과는 변경하지 않습니다. 라벨은 입력에 넣지 않습니다.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from openai import OpenAI
from experiments import claim_guard as probe
from ragkit import claim_verification as guard, nli
from ragkit.llm import validate_claims
from ragkit.retrieval_evaluation import atomic_report

ROOT = Path(__file__).resolve().parents[1]


def protocol(config, dataset):
    paths = dict(probe.code_paths()) | {
        'experiments/claim_guard_operating.py': Path(__file__).resolve(),
        'ragkit/claim_verification.py': ROOT / 'ragkit/claim_verification.py',
        'ragkit/english.py': ROOT / 'ragkit/english.py',
        'ragkit/semantic.py': ROOT / 'ragkit/semantic.py',
    }
    return {'format': 'claim-guard-operating-v1', 'dataset_sha256': probe.digest(dataset),
        'code_sha256': {key: probe.digest(path) for key, path in paths.items()},
        'model': nli.MODEL_ID, 'revision': nli.MODEL_REV, 'threshold': nli.THRESHOLD,
        'device': 'cpu', 'dtype': 'float32', 'manual_score_tolerance': 1e-5,
        'translation': {key: config.get(key) for key in ('model', 'base_url', 'reasoning_effort',
            'timeout_seconds', 'max_tokens', 'temperature', 'top_p', 'context_window', 'max_context_chars', 'claim_guard_timeout_seconds')},
        'scope': '고정 합성 개발 탐침. 업무 정확도나 Windows GPT-OSS 120B 인증이 아닙니다.'}


def evaluate(case, client, config):
    started = time.monotonic()
    baseline = probe.rule_check(case)
    premise, hypothesis = probe.cited_pair(case, language='en')
    manual = {'status': 'empty_evidence', 'passed': False}
    if premise.strip():
        try:
            manual = nli.score(premise, hypothesis) | {'status': 'ok'}
        except nli.NLIError as error:
            manual = {'status': 'error', 'passed': False, 'error': str(error)}
    if baseline['status'] == 'ok':
        payload = {'claims': [{'text': case['claim']['ko'], 'evidence': deepcopy(case['citations'])}]}
        validate_claims(payload, probe.build_hits(case))
        automatic = guard.verify_claims(client, config | {'claim_guard_mode': 'enforce'}, payload['claims'])
        combined = {'status': 'error' if automatic['status'] == 'error' else 'ok',
                    'passed': automatic['status'] == 'passed'}
    else:
        automatic = {'status': 'skipped', 'reason': '원문 규칙에서 먼저 거절 또는 오류'}
        combined = {'status': baseline['status'], 'passed': baseline['passed']}
    return {'id': case['id'], 'category': case['category'], 'expected_semantic': case['expected_semantic'],
        'baseline': baseline, 'manual': manual, 'automatic': automatic, 'combined': combined,
        'seconds': round(time.monotonic() - started, 4)}


def summary(rows, total):
    result = {'processed': len(rows), 'missing': total - len(rows), 'errors': [], 'baseline': {}, 'automatic': {}}
    for key, field in (('baseline', 'baseline'), ('automatic', 'combined')):
        decisions = [row for row in rows if row[field]['status'] not in ('error',)]
        result[key] = {'false_accepts': [r['id'] for r in decisions if r['expected_semantic'] != 'supported' and r[field]['passed']],
                      'false_rejects': [r['id'] for r in decisions if r['expected_semantic'] == 'supported' and not r[field]['passed']]}
    result['errors'] = [r['id'] for r in rows if r['combined']['status'] == 'error' or r['manual']['status'] == 'error']
    timings = [r['automatic']['seconds'] for r in rows if r['automatic'].get('seconds') is not None]
    result['mean_verification_seconds'] = round(sum(timings) / len(timings), 4) if timings else None
    result['max_verification_seconds'] = max(timings, default=None)
    result['verification_count'] = len(timings)
    result['complete'] = result['missing'] == 0 and not result['errors']
    return result


def run(config, dataset, output):
    data = probe.load_cases(dataset)
    conditions = protocol(config, dataset)
    signed = hashlib.sha256(json.dumps(conditions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != conditions:
        raise ValueError('입력·코드·모델 설정이 달라졌습니다. 기존 결과를 덮어쓰지 말고 새 출력 폴더를 사용하세요.')
    atomic_report(target, conditions)
    rows = []
    atomic_report(output / 'status.json', {'status': 'running', 'complete': False,
        'processed': 0, 'total': len(data['cases']), 'protocol_sha256': signed})
    with OpenAI(base_url=config['base_url'], api_key=config.get('api_key', 'EMPTY'),
                timeout=config.get('timeout_seconds', 120), max_retries=0) as client:
        for case in data['cases']:
            checkpoint = output / 'checkpoints' / (case['id'] + '.json')
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if (saved['protocol_sha256'] != signed or any(saved['row'][key] != case[key]
                        for key in ('id', 'category', 'expected_semantic'))):
                    raise ValueError('체크포인트의 자료 연결이 달라졌습니다.')
                row = saved['row']
            else:
                atomic_report(output / 'status.json', {'status': 'running', 'complete': False,
                    'processed': len(rows), 'total': len(data['cases']), 'current': case['id'], 'protocol_sha256': signed})
                print('start ' + case['id'], flush=True)
                row = evaluate(case, client, config)
                atomic_report(checkpoint, {'protocol_sha256': signed, 'row': row})
            rows.append(row)
            if protocol(config, dataset) != conditions:
                raise ValueError('평가 중 입력·코드가 변경됐습니다. 완료로 인증하지 않습니다.')
            report = {'protocol_sha256': signed, 'results': rows} | summary(rows, len(data['cases']))
            atomic_report(output / 'results.json', report)
            atomic_report(output / 'status.json', {k: v for k, v in report.items() if k != 'results'})
            print(f"{len(rows)}/{len(data['cases'])} {case['id']} baseline={row['baseline']['status']} "
                  f"automatic={row['automatic']['status']} seconds={row['seconds']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'mac_config.json')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'experiments/claim_guard_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/claim-guard-operating-20261001/automatic-en')
    parser.add_argument('--self-check', action='store_true', help='모델 호출 없이 오류와 완료 집계를 검사')
    args = parser.parse_args()
    if args.self_check:
        row = {'id': 'synthetic', 'expected_semantic': 'supported',
            'baseline': {'status': 'ok', 'passed': True}, 'manual': {'status': 'ok'},
            'automatic': {'seconds': 1}, 'combined': {'status': 'error', 'passed': False}}
        result = summary([row], 1)
        assert result['automatic']['false_rejects'] == [] and result['errors'] == ['synthetic'] and not result['complete']
        row['combined'] = {'status': 'ok', 'passed': True}
        assert summary([row], 1)['complete'] and not summary([row], 2)['complete']
        print('PASS: 검사 불가는 사실 거절과 구분하며 누락·오류가 있으면 완료하지 않습니다.')
        return 0
    config = json.loads(args.config.read_text())
    report = run(config, args.dataset, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
