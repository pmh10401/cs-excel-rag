"""동일 원문·언어 교대·영문 보정·완료 재사용을 네트워크 없이 확인합니다."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from experiments import prompt_language as experiment
from experiments.answer_baseline import prepare
from ragkit import llm
from ragkit.records import public_cells, record_id


def main():
    original_prompt = llm.CLAIM_PROMPT
    config = {'model': 'mock', 'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY', 'reasoning_effort': 'high'}
    request = {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': '원문 자료: 30분'},
                           {'role': 'assistant', 'content': '이전 한국어 응답'},
                           {'role': 'user', 'content': '검사 오류: 숫자 90 / 조치내용(1)' + experiment.REPAIR_KO}]}
    saved = deepcopy(request)
    translated = experiment.translate_request(request, config)
    assert request == saved and translated['messages'][:3] == saved['messages'][:3]
    assert translated['messages'][-1]['content'] == 'Validation error: 숫자 90 / 조치내용(1)' + experiment.REPAIR_EN
    request['messages'][-1]['content'] = experiment.LENGTH_KO
    assert experiment.translate_request(request, config)['messages'][-1]['content'] == experiment.LENGTH_EN
    request['messages'][-1]['content'] = 'new instructions'
    try:
        experiment.translate_request(request, config)
    except ValueError:
        pass
    else:
        raise AssertionError('Unreviewed repair text accepted')
    english = experiment.PROMPT.read_text().strip()
    assert len(english.splitlines()) == len(original_prompt.splitlines())
    rows, _, dataset = prepare(json.loads(Path('experiments/action_cases.json').read_text()))
    by_id = {record_id(row): row for row in rows}
    baseline = {'complete': True, 'results': []}
    for case in dataset['cases']:
        hits = [{'id': identity + '/문서', 'record_id': identity, 'expected_label': 'DO_NOT_SEND',
                 'evidence': [{'id': identity + '/접수', 'kind': '접수', 'cells': public_cells(by_id[identity])}]}
                for identity in case['provided_ids']]
        baseline['results'].append({'id': case['id'], 'question': case['question'],
                                   'methods': {'retrieved': {'status': 'ok', 'hits': hits}}})
    calls = []

    def complete(**kwargs):
        calls.append(deepcopy(kwargs))
        text = json.dumps(kwargs, ensure_ascii=False)
        assert all(key not in text for key in ('DO_NOT_SEND', 'expected_ids', 'successful_action_ids', 'target_action'))
        source = json.loads(kwargs['messages'][1]['content'].split('원문 자료:\n')[1])[0]
        payload = {'claims': [{'text': source['cells']['조치'], 'evidence': [{'id': source['id'], 'column': '조치'}]}]}
        return SimpleNamespace(model='mock', usage=None, choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)), close=lambda: None)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = SimpleNamespace(baseline=root / 'baseline.json', config=root / 'config.json', output=root / 'result')
        args.baseline.write_text(json.dumps(baseline, ensure_ascii=False))
        args.config.write_text(json.dumps(config))
        with patch.object(experiment, 'OpenAI', return_value=client), redirect_stdout(io.StringIO()):
            experiment.run(args)
            before = (args.output / 'results.json').read_bytes()
            count = len(calls)
            experiment.run(args)
            assert len(calls) == count and (args.output / 'results.json').read_bytes() == before
            args.config.write_text(json.dumps(config | {'reasoning_effort': 'low'}))
            try:
                experiment.run(args)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed conditions accepted')
        result = json.loads(before)
        assert result['complete'] and result['completed_answers'] == result['total_answers'] == 56
        assert len(calls) == 56
        assert [c['messages'][0]['content'] for c in calls[:4]] == [original_prompt, english, english, original_prompt]
        for left, right in zip(calls[::2], calls[1::2]):
            assert left['messages'][1] == right['messages'][1]
        assert llm.CLAIM_PROMPT == original_prompt
    print('PASS: fixed sources, prompt isolation/order, English repair, label exclusion, resume, frozen conditions')


if __name__ == '__main__':
    main()
