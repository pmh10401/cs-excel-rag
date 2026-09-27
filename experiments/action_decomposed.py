"""조치 판단을 여섯 사실 질문으로 나눕니다. 기존 실험 코드는 수정하지 않습니다."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

from experiments.action_selection import state, validate_dataset
from experiments.gliner_decide import atomic_report, digest
from experiments.ollaya_rerank import api, snapshot


# 모델 응답을 보기 전에 고정한 질문과 기준입니다. 기존 24문항은 개발 자료로만 사용합니다.
QUESTIONS = {name: {'type': 'noul', 'instructions': question} for name, question in (
    ('context', 'Does `symptom` establish every condition required by `question`?'),
    ('performed', 'Does `history` explicitly say that `target_action` was performed?'),
    ('recovered', 'Does `history` report verified restoration of normal operation after `target_action`?'),
    ('relapse', 'Does `history` report recurrence after `target_action`?'),
    ('limited', 'Does `history` explicitly limit the result of `target_action` to partial improvement or inconclusive observation?'),
    ('joint', 'Does `history` say the effect of `target_action` cannot be separated from other actions?'),
)}
POSITIVE = ('context', 'performed', 'recovered')
NEGATIVE = ('relapse', 'limited', 'joint')


def checked(response, model):
    if (response['model'] != model or response['state_truncated'] is not False
            or response['done_reason'] != 'decide' or type(response['usage']['output_tokens']) is not int
            or response['usage']['output_tokens'] != 0 or set(response['answers']) != set(QUESTIONS)):
        raise ValueError('Wrong model, missing answers, truncated state or generated output')
    probabilities = {}
    for name, answer in response['answers'].items():
        p = answer['noul']
        if answer['type'] != 'noul' or type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Invalid probability: {name}')
        probabilities[name] = p
    return probabilities


def eligible(probabilities):
    # 0.8/0.2는 공식 가이드의 시작값이며 이 업무에서 보정된 기준이 아닙니다. 결과에 맞춰 바꾸지 않습니다.
    return all(probabilities[k] >= .8 for k in POSITIVE) and all(probabilities[k] <= .2 for k in NEGATIVE)


def summarize(data, rows):
    assert [c['id'] for c in data['cases']] == [r['id'] for r in rows]
    counts = Counter(questions=len(rows), positive_questions=0, negative_questions=0, correct=0,
        correct_selection=0, correct_abstention=0, wrong_recommendation=0, missed_positive=0, execution_failures=0)
    for case, row in zip(data['cases'], rows, strict=True):
        truth, choice = case['expected_ids'], row['selected']
        assert choice is None or choice in {c['id'] for c in case['candidates']}
        failed = any(p['status'] != 'ok' for p in row['predictions'])
        correct = not failed and (choice in truth if truth else choice is None)
        counts['positive_questions' if truth else 'negative_questions'] += 1
        counts['correct'] += correct
        counts['correct_selection' if truth else 'correct_abstention'] += correct
        counts['wrong_recommendation'] += choice is not None and choice not in truth
        counts['missed_positive'] += bool(truth) and choice is None
        counts['execution_failures'] += failed
    seconds = [r['seconds'] for r in rows]
    return dict(counts= dict(counts), mean_seconds=statistics.mean(seconds),
                p95_seconds=sorted(seconds)[math.ceil(.95*len(seconds))-1])


def run(args):
    data = json.loads(args.dataset.read_text())
    validate_dataset(data)
    paths = [args.dataset, Path(__file__), Path('experiments/action_selection.py'),
             Path('experiments/ollaya_rerank.py'), Path('experiments/gliner_decide.py')]
    server = snapshot(args.base_url, args.model)
    api(args.base_url, '/api/decide', dict(model=args.model, keep_alive='10m'))
    loaded = next(m for m in api(args.base_url, '/api/ps')['models'] if m['name'] == args.model)
    assert loaded['digest'] == server['digest']
    protocol = dict(server=server, device=loaded['device'], precision=loaded['details']['quantization_level'],
        questions=QUESTIONS, hashes={str(p): digest(p) for p in paths},
        thresholds={'positive_min': .8, 'negative_max': .2}, positive=list(POSITIVE), negative=list(NEGATIVE),
        selection='first candidate passing every gate; any execution error abstains and counts as failure',
        scope='Synthetic English decision input with original Korean evidence; no training or production integration')
    target, contract = args.output / 'results.json', args.output / 'protocol.json'
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError('입력·모델·코드가 바뀌었습니다. 새 출력 폴더를 사용하세요.')
    report = json.loads(target.read_text()) if target.exists() else dict(complete=False, protocol=protocol,
        started_at=datetime.now().astimezone().isoformat(), rows=[])
    assert report['protocol'] == protocol
    completed = [r['id'] for r in report['rows']]
    assert completed == [c['id'] for c in data['cases'][:len(completed)]]
    if report['complete']:
        assert report['summary'] == summarize(data, report['rows'])
        print('Already complete; preserved.')
        return
    atomic_report(contract, protocol)
    for case in data['cases'][len(completed):]:
        predictions, started = [], time.perf_counter()
        for candidate in case['candidates']:
            item = dict(id=candidate['id'], status='failed')
            try:
                response = api(args.base_url, '/api/decide', dict(model=args.model,
                    state=state(case, candidate), questions=QUESTIONS, keep_alive='10m'))
                item['response'] = response
                probabilities = checked(response, args.model)
                item.update(probabilities=probabilities, eligible=eligible(probabilities), status='ok')
            except Exception as error:
                item['error'] = type(error).__name__ + ': ' + str(error)[:300]
            predictions.append(item)
        seconds = time.perf_counter() - started
        chosen = None if any(p['status'] != 'ok' for p in predictions) else next(
            (c for c, p in zip(case['candidates'], predictions, strict=True) if p['eligible']), None)
        evidence = None if chosen is None else dict(id=chosen['id'], action=chosen['action']['ko'],
            cells={'현상': chosen['symptom']['ko'], '조치': chosen['record']['ko']})
        report['rows'].append(dict(id=case['id'], predictions=predictions,
            selected=None if chosen is None else chosen['id'], evidence=evidence, seconds=seconds))
        atomic_report(target, report)
        print(f'{len(report["rows"])}/{len(data["cases"])} {case["id"]}', flush=True)
    assert snapshot(args.base_url, args.model) == server
    final = next(m for m in api(args.base_url, '/api/ps')['models'] if m['name'] == args.model)
    assert (final['device'], final['details']['quantization_level'], final['digest']) == (
        loaded['device'], protocol['precision'], loaded['digest'])
    for path, expected in protocol['hashes'].items():
        assert digest(path) == expected
    report.update(complete=True, summary=summarize(data, report['rows']), loaded_model_after=final,
                  completed_at=datetime.now().astimezone().isoformat())
    atomic_report(target, report)
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='laya:typed-decisions')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
