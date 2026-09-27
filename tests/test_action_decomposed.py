"""확률 경계·잘림·정답 누출과 실행 실패를 정상 보류로 세지 않는지 검사합니다."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.action_decomposed import NEGATIVE, POSITIVE, QUESTIONS, checked, eligible, state, summarize


def main():
    probabilities = {**dict.fromkeys(POSITIVE, .8), **dict.fromkeys(NEGATIVE, .2)}
    response = dict(model='laya:typed-decisions', state_truncated=False, done_reason='decide',
        usage={'output_tokens': 0}, answers={k: dict(type='noul', noul=v) for k, v in probabilities.items()})
    assert eligible(checked(response, 'laya:typed-decisions'))
    for k in POSITIVE:
        assert not eligible(probabilities | {k: .7999})
    for k in NEGATIVE:
        assert not eligible(probabilities | {k: .2001})
    bad = [response | {'state_truncated': True}, response | {'model': 'laya:en'}, response | {'answers': {}},
           response | {'usage': {'output_tokens': True}}, response | {'done_reason': 'load'}]
    for value in (float('nan'), float('inf'), True, -.01, 1.01):
        item = deepcopy(response)
        item['answers']['recovered']['noul'] = value
        bad.append(item)
    item = deepcopy(response)
    item['answers']['context']['type'] = 'choice'
    bad.append(item)
    for item in bad:
        try:
            checked(item, 'laya:typed-decisions')
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError('Invalid response accepted')
    data = json.loads(Path('experiments/action_cases.json').read_text())
    original = deepcopy(data)
    case = data['cases'][0]
    changed = deepcopy(case)
    changed['expected_ids'] = ['DO_NOT_SEND']
    changed['candidates'][0]['expected'] = {'context': 'DO_NOT_SEND', 'outcome': 'DO_NOT_SEND'}
    assert state(case, case['candidates'][0]) == state(changed, changed['candidates'][0])
    assert set(json.loads(state(case, case['candidates'][0]))) == {'question', 'symptom', 'target_action', 'history'}
    rows = [dict(id=c['id'], selected=c['expected_ids'][0] if c['expected_ids'] else None,
                 seconds=0., predictions=[dict(status='ok') for _ in c['candidates']]) for c in data['cases']]
    assert summarize(data, rows)['counts']['correct'] == 24
    i = next(i for i, c in enumerate(data['cases']) if not c['expected_ids'])
    rows[i]['predictions'][0]['status'] = 'failed'
    counts = summarize(data, rows)['counts']
    assert counts['correct'] == 23 and counts['execution_failures'] == 1 and counts['correct_abstention'] == 7
    assert data == original and set(QUESTIONS) == set(POSITIVE + NEGATIVE)
    print('PASS: probability gates, invalid responses, no label leakage and failure accounting')


if __name__ == '__main__':
    main()
