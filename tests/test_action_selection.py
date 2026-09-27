"""모델 없이 정답 누출·응답 검사·추천 보류·원문 보존을 확인합니다."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.action_selection import QUESTIONS, checked, score, select, source_rule, state, validate_dataset


def response(labels):
    return dict(model='laya:en', state_truncated=False, done_reason='decide', usage={'output_tokens': 0},
        answers={key: dict(type='choice', choice=value, probabilities={name: float(name == value)
            for name in QUESTIONS[key]['criteria']}) for key, value in labels.items()})


def main():
    data = json.loads(Path('experiments/action_cases.json').read_text())
    before = deepcopy(data)
    validate_dataset(data)
    case, candidate = data['cases'][0], data['cases'][0]['candidates'][0]
    changed = deepcopy(case)
    changed['expected_ids'] = ['DO_NOT_SEND']
    changed['candidates'][0]['expected'] = {'context': 'DO_NOT_SEND', 'outcome': 'DO_NOT_SEND'}
    assert state(case, candidate) == state(changed, changed['candidates'][0])
    assert set(json.loads(state(case, candidate))) == {'question', 'symptom', 'target_action', 'history'}
    assert {c['id']: source_rule(c) for c in case['candidates']} == {
        'ACT-01-1': 'ineffective', 'ACT-01-2': 'confirmed'}
    good = response({'context': 'match', 'outcome': 'confirmed'})
    assert checked(good, 'laya:en') == {'context': 'match', 'outcome': 'confirmed'}
    bad = [good | {'state_truncated': True}, good | {'model': 'other'}, good | {'answers': {}},
           good | {'usage': {'output_tokens': True}}, good | {'done_reason': 'load'}]
    for value in (float('nan'), True, -.1, .7):
        item = deepcopy(good)
        item['answers']['outcome']['probabilities']['confirmed'] = value
        bad.append(item)
    item = deepcopy(good)
    item['answers']['outcome']['choice'] = 'unknown'
    bad.append(item)
    for item in bad:
        try:
            checked(item, 'laya:en')
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError('Invalid response accepted')
    rows = []
    for case in data['cases']:
        predictions = [dict(id=c['id'], status='ok', labels=checked(response(c['expected']), 'laya:en'))
                       for c in case['candidates']]
        choice = select(case, predictions)
        assert choice in case['expected_ids'] if case['expected_ids'] else choice is None
        rows.append(dict(id=case['id'], selected=dict(first_candidate=choice, source_rule=choice, laya=choice),
                         predictions=predictions, seconds=0.))
    assert score(data, rows)['methods']['laya']['correct'] == len(data['cases'])
    # 실패를 정상적인 추천 보류로 세면, 정답 없는 질문의 점수가 부풀려집니다.
    index = next(i for i, case in enumerate(data['cases']) if not case['expected_ids'])
    rows[index]['predictions'][0] = {'status': 'failed', 'error': 'timeout'}
    assert select(data['cases'][index], rows[index]['predictions']) is None
    result = score(data, rows)['methods']['laya']
    assert result['correct'] == len(data['cases'])-1 and result['execution_failures'] == 1
    assert data == before
    print('PASS: no label leakage, source rules, invalid responses, selection, abstention and failure scoring')


if __name__ == '__main__':
    main()
