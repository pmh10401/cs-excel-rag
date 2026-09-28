"""네트워크 없이 평가 자료·원문 경계·재사용을 확인합니다."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from experiments import answer_baseline as evaluation
from ragkit.input_formats import HISTORY_RULES, HISTORY_SCHEMA


def main():
    data = json.loads(Path('experiments/action_cases.json').read_text())
    rows, mapping, dataset = evaluation.prepare(data)
    assert len(dataset['cases']) == 24 and len(mapping) == 48
    assert len(rows) < len(mapping)
    assert mapping['ACT-01-1'] == mapping['ACT-01-2']  # 서로 다른 조치, 같은 원문
    negative = next(c for c in dataset['cases'] if c['id'] == 'ACT-05')
    assert negative['expected_ids'] and not negative['successful_action_ids']
    public = json.dumps(rows, ensure_ascii=False)
    assert all(k not in public for k in ('expected', 'target_action', 'successful_action_ids'))
    unsafe = json.loads(json.dumps(data))
    unsafe['cases'][0]['id'] = '../outside'
    try:
        evaluation.prepare(unsafe)
    except ValueError:
        pass
    else:
        raise AssertionError('Checkpoint path traversal accepted')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db_path = root / 'fixture.sqlite3'
        db = sqlite3.connect(db_path)
        evaluation.storage.ingest(db, rows, HISTORY_RULES, schema=HISTORY_SCHEMA)
        db.close()
        before = db_path.read_bytes()
        db = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
        db.execute('PRAGMA query_only=ON')
        case = dataset['cases'][0]
        calls = []

        def answer(client, config, question, hits, *, details):
            assert question == case['question']
            assert len(hits) == 1
            encoded = json.dumps(hits, ensure_ascii=False)
            assert all(k not in encoded for k in ('expected', 'successful_action_ids', 'target_action'))
            assert '커넥터 압착 후 정상화.' in encoded and '센서 렌즈 세척 후에도 동일 증상 발생.' in encoded
            calls.append(question)
            details.update(answer_text='MODEL_TEXT', claims=[], answer_status='ok')
            return 'MODEL_TEXT\nRAW_SUPPLEMENT', ''

        with patch.object(evaluation.llm, 'answer', side_effect=answer), \
                patch.object(evaluation.semantic, 'search_cases', side_effect=AssertionError('provided must not search')):
            path = root / 'provided.json'
            result = evaluation.evaluate_case(db, case, 'provided', {}, None, None, path, 'frozen')
            assert result['answer'] == 'MODEL_TEXT' and 'RAW_SUPPLEMENT' not in result['answer']
            assert result['answer_diagnostics']['reply_with_supplements'].endswith('RAW_SUPPLEMENT')
            assert result['answer_checks']['passed'] is None  # 의미 정답을 자동으로 인정하지 않습니다.
            assert evaluation.evaluate_case(db, case, 'provided', {}, None, None, path, 'frozen') == result
            assert len(calls) == 1
            interrupted = root / 'interrupted.json'
            pending = json.loads(path.read_text())
            pending.update(complete=False, results=[])
            evaluation.atomic_report(interrupted, pending)
            recovered = evaluation.evaluate_case(db, case, 'provided', {}, None, None, interrupted, 'frozen')
            assert recovered['answer'] == 'MODEL_TEXT' and len(calls) == 2
            assert len(list(root.glob('interrupted.interrupted-*.json'))) == 1
            try:
                evaluation.evaluate_case(db, case, 'provided', {}, None, None, path, 'changed')
            except ValueError:
                pass
            else:
                raise AssertionError('Changed protocol accepted')
        with patch.object(evaluation.llm, 'answer', side_effect=RuntimeError('secret must not appear')), \
                patch.object(evaluation.semantic, 'search_cases', return_value=evaluation.source_hits(db, case['provided_ids'])) as search:
            result = evaluation.evaluate_case(db, case, 'retrieved', {}, None, None, root / 'retrieved.json', 'frozen')
            assert result['answer_error'] and 'secret' not in json.dumps(result)
            assert 'answer' not in result and search.call_args.kwargs['limit'] == 3
        # 실제 답변 함수까지 통과시키되 네트워크 경계만 가짜 응답으로 바꿉니다.
        def completion(**request):
            text = json.dumps(request, ensure_ascii=False)
            assert all(k not in text for k in ('expected_ids', 'successful_action_ids', 'target_action', 'ACT-01'))
            ref = evaluation.source_hits(db, case['provided_ids'])[0]['evidence'][0]['id']
            payload = {'claims': [{'evidence': [{'id': ref, 'column': '조치'}],
                                   'text': '커넥터 압착 후 정상화했습니다.'}]}
            return SimpleNamespace(model='mock', usage=None, choices=[SimpleNamespace(
                finish_reason='stop', message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        result = evaluation.evaluate_case(db, case, 'provided', {'model': 'mock', 'reasoning_effort': 'high'},
                                          None, client, root / 'request.json', 'frozen')
        assert not result.get('answer_error') and result['answer_diagnostics']['claims']
        assert result['answer_diagnostics']['calls'][0]['model'] == 'mock'
        db.close()
        assert db_path.read_bytes() == before
    print('PASS: deduplication, relevance vs success, label isolation, source preservation, model text, checkpoints, read-only DB, errors')


if __name__ == '__main__':
    main()
