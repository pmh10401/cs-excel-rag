"""Offline checks only. Run: python -m tests.test_retrieval_evaluation."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from ragkit import retrieval_evaluation as evaluation


def hit(identity, action='', *, title='펌프 진동', score=0.03):
    return {'id': identity + '/문서', 'record_id': identity, 'score': score, 'content': title,
            'evidence_ids': [identity + '/작업1'],
            'evidence': [{'id': identity + '/작업1', 'kind': '작업1',
                          'cells': {'제목': title, '조치내용(1)': action}, 'accepted': {}}]}


def main(root=None):
    dataset = {'provenance': 'SYNTHETIC test', 'cases': [
        {'id': 'q1', 'question': '펌프 진동 조치?', 'expected_ids': ['LABEL_SECRET'],
         'required': ['REQUIRED_SECRET'], 'forbidden': ['FORBIDDEN_SECRET']},
        {'id': 'q2', 'question': '없는 기록?', 'expected_ids': []}]}
    good = hit('a', '재체결 후 진동 재발함. 교체 예정.')
    empty = hit('b')
    plan = hit('c', '교체 예정. 정상 여부 점검 필요.')
    failed = hit('d', '교체 후에도 진동 지속.')
    before = copy.deepcopy([empty, plan, good, failed])
    assert evaluation.action_availability(plan)['nonempty']
    assert not evaluation.action_availability(plan)['actual_or_observation']
    for text in ('정상', '정상 ' * 100, '교체하지 않음', '교체했는지 확인 불가', '점검 미실시'):
        assert not evaluation.action_availability(hit('x', text))['actual_or_observation']
    assert evaluation.action_availability(good)['actual_or_observation']
    assert evaluation.action_availability(failed)['actual_or_observation']
    ranked = evaluation.prefer_action('펌프 진동', before)
    assert ranked == [good, empty, plan, failed] and before == [empty, plan, good, failed]
    missing_action = [empty, plan, hit('e'), good]
    assert evaluation.prefer_action('펌프 진동', missing_action) == missing_action
    assert evaluation.score({'expected_ids': ['a']}, missing_action)['action_actual_at_3'] is False
    assert evaluation.prefer_action('진공이 안 잡힙니다', before) == before
    assert evaluation.prefer_action('CS-DEMO-001 펌프 진동', before) == before
    assert evaluation.prefer_action('펌프 진동', [empty, hit('x', '교체함', score=0.02)]) == [empty, hit('x', '교체함', score=0.02)]
    unrelated = hit('x', '교체함', title='팬 누수')
    assert evaluation.prefer_action('펌프 진동', [empty, unrelated]) == [empty, unrelated]
    for identity in ('a', 'a/문서', 'a/작업1'):
        value = evaluation.score({'expected_ids': [identity]}, [empty, good])
        assert value['mrr_at_3'] == 0.5 and value['precision_at_3'] == 0.5
        assert value['recall_at_3'] == 1 and value['relevant_action_at_3']
    value = evaluation.score({'expected_ids': ['a', 'd']}, [good])
    assert value['recall_at_3'] == 0.5 and value['missing_expected_ids'] == ['d']
    assert not evaluation.score({'expected_ids': ['a']}, [empty]*3 + [good])['passed']
    assert evaluation.score({'expected_ids': []}, [])['passed']
    assert not evaluation.score({'expected_ids': []}, [good])['passed']
    assert evaluation.score({'expected_ids': ['a']}, [])['missing_relevant']
    assert evaluation.score({'expected_templates': ['T']}, [good], {'T': '펌프 진동'})['passed']
    assert evaluation.score({'expected_ids': ['a']}, [{'id': 'a'}])['action_evidence_missing']
    frozen, digest = evaluation.freeze(dataset)
    assert evaluation.freeze(dict(reversed(list(dataset.items()))))[1] == digest
    changed = copy.deepcopy(dataset)
    changed['cases'][0]['required'] = ['changed']
    assert evaluation.freeze(changed)[1] != digest
    for bad in ({'cases': []}, {'cases': [dataset['cases'][0]]*2},
                {'cases': [dataset['cases'][0] | {'required': ['[']}]},
                {'cases': [dataset['cases'][0] | {'expected_ids': 'a'}]}):
        try:
            evaluation.freeze(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid dataset accepted')

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / 'report.json'
        calls, answers = [], []

        def search(question, method):
            calls.append((question, method))
            # The preceding method must already be durably reported before the next callback.
            saved = json.loads(output.read_text())
            assert saved['freeze_hash'] == digest and not saved['complete']
            if question == '없는 기록?':
                if method == 'korean_hybrid':
                    raise RuntimeError('offline simulated failure')
                return {'hits': []}
            return {'hits': [good | {'expected_ids': ['LABEL_SECRET']}], 'seconds': 0.25}

        def answer(question, hits):
            answers.append((question, hits))
            assert all('expected_ids' not in h for h in hits)
            return 'REQUIRED_SECRET'

        report = evaluation.run_comparison(dataset, search, output, answer_fn=answer, experiment=True,
                                           translation={'status': 'available'})
        assert report['complete'] and len(calls) == 4
        assert 'LABEL_SECRET' not in json.dumps(answers, ensure_ascii=False)
        assert all('SECRET' not in q for q, _ in calls)
        first = report['results'][0]['methods']
        assert first['korean_hybrid']['answer_checks']['passed']
        assert not first['korean_hybrid']['score']['passed']
        assert report['results'][1]['methods']['korean_hybrid']['status'] == 'error'
        assert report['results'][1]['methods']['bilingual_hybrid']['score']['passed']
        assert report['summary']['paired_cases'] == 1
        assert json.loads(output.read_text()) == report and not list(temporary_root.glob('*.tmp'))

        # Real SDK exception types, fake credentials, no transport or model calls.
        from openai import APIConnectionError, APIStatusError
        answer_key, embedding_key = 'FAKE_ANSWER_KEY_SECRET', 'FAKE_EMBEDDING_KEY_SECRET'
        request = SimpleNamespace(headers={'Authorization': 'Bearer ' + embedding_key})
        message = answer_key + ' ' + embedding_key
        def transport_search(question, method):
            if method == 'korean_hybrid':
                raise APIConnectionError(message=message, request=request)
            return {'hits': [good]}
        def transport_answer(question, hits):
            raise APIStatusError(message, response=SimpleNamespace(status_code=401, request=request, headers={}),
                                 body={'api_key': answer_key})
        sanitized = evaluation.run_comparison(dataset, transport_search, temporary_root/'sanitized.json',
            answer_fn=transport_answer, translation={'status': 'available'})
        saved = (temporary_root/'sanitized.json').read_text()
        assert answer_key not in saved and embedding_key not in saved
        methods = sanitized['results'][0]['methods']
        assert methods['korean_hybrid']['error'].startswith('APIConnectionError:')
        assert methods['bilingual_hybrid']['answer_error'].startswith('APIStatusError:')
        assert methods['bilingual_hybrid']['hits'] == [good]

        replay = {'freeze_hash': digest, 'results': [{'id': 'q1', 'question': '펌프 진동 조치?',
                  'methods': {'korean_hybrid': {'hits': [], 'answer': 'REQUIRED_SECRET'}}}]}
        with patch('openai.OpenAI', side_effect=AssertionError('Offline must never create a client')):
            replayed = evaluation.run_comparison(dataset, evaluation.replay_search(replay, dataset), temporary_root/'replay.json')
        first = replayed['results'][0]['methods']
        assert first['korean_hybrid']['seconds'] is None and first['korean_hybrid']['latency_missing']
        assert first['bilingual_hybrid']['status'] == 'missing'
        assert replayed['summary']['bilingual_minus_korean'] is None
        assert not replayed['results'][1]['methods']['bilingual_hybrid'].get('score')
        try:
            evaluation.replay_search(replay | {'freeze_hash': 'wrong'}, dataset)
        except ValueError:
            pass
        else:
            raise AssertionError('Replay mismatch accepted')
        # A partial JSON write must leave the preceding valid report in place.
        with patch.object(evaluation.json, 'dump', side_effect=OSError('interrupted')):
            try:
                evaluation.atomic_report(output, {'new': True})
            except OSError:
                pass
        assert json.loads(output.read_text()) == report and not list(temporary_root.glob('*.tmp'))

        # Only disposable fixtures are written; availability inspection is read-only.
        path = temporary_root/'source.sqlite3'
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE rag_documents (record_id TEXT)')
        db.execute("INSERT INTO rag_documents VALUES ('a')")
        db.commit()
        db.close()
        original = path.read_bytes()
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        try:
            info = evaluation.translation_availability(db)
            assert info['status'] == 'unavailable' and info['generated'] == 0
            try:
                db.execute("INSERT INTO rag_documents VALUES ('b')")
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError('Read-only DB was writable')
        finally:
            db.close()
        assert path.read_bytes() == original

        cases_path = temporary_root/'cases.json'
        cases_path.write_text(json.dumps(dataset))
        assert evaluation.main(['--cases', str(cases_path), '--output', str(temporary_root/'frozen.json'), '--freeze-only']) == 0
        assert json.loads((temporary_root/'frozen.json').read_text())['freeze_hash'] == digest

        # Exercise the live CLI wiring with all network entrypoints mocked. No live calls.
        from ragkit import english, semantic
        config_path = temporary_root/'config.json'
        config_path.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'model': 'test',
            'reasoning_effort': 'high', 'embedding': {'base_url': 'http://127.0.0.1:1/v1', 'model': 'test'}}))
        settings_before = config_path.read_bytes()
        client = SimpleNamespace(close=lambda: None)
        def mock_search(db, question, config, **kwargs):
            assert kwargs['embedding_client'] is client
            assert config['reasoning_effort'] == 'high'
            assert db.execute('PRAGMA query_only').fetchone()[0] == 1
            assert db.in_transaction
            return []
        with patch('openai.OpenAI', return_value=client) as factory, \
                patch.object(semantic, 'search_cases', side_effect=mock_search) as search_mock, \
                patch.object(semantic, 'vector_count', return_value=1), \
                patch.object(semantic, 'client_for', side_effect=AssertionError('auto-start forbidden')), \
                patch.object(english, 'build_english', side_effect=AssertionError('implicit translation forbidden')):
            assert evaluation.main(['--cases', str(cases_path), '--output', str(temporary_root/'mock-live.json'),
                '--live', '--db', str(path), '--config', str(config_path)]) == 1
            assert factory.call_count == 1 and search_mock.call_count == 2
        mock_live = json.loads((temporary_root/'mock-live.json').read_text())
        assert mock_live['translation']['indexed_records'] == 0
        assert mock_live['results'][0]['methods']['bilingual_hybrid']['status'] == 'missing'
        assert mock_live['summary']['bilingual_minus_korean'] is None
        assert path.read_bytes() == original and config_path.read_bytes() == settings_before

        # Existing translations are counted, never generated or silently represented as full coverage.
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE english_translations (record_id TEXT)')
        db.execute('CREATE TABLE rag_english_documents (record_id TEXT)')
        assert evaluation.translation_availability(db)['status'] == 'unavailable'
        db.execute("INSERT INTO english_translations VALUES ('a')")
        db.execute("INSERT INTO rag_english_documents VALUES ('a')")
        assert evaluation.translation_availability(db)['status'] == 'available'
        db.execute("INSERT INTO rag_documents VALUES ('b')")
        assert evaluation.translation_availability(db)['status'] == 'partial'
        db.close()

    if root is None:
        root = Path(__file__).resolve().parents[1]
        if (root/'windows-demo').is_dir():
            root = root/'windows-demo'
    drafts = json.loads((Path(root)/'workflow_cases.json').read_text(encoding='utf-8'))
    evaluation.freeze(drafts)
    assert len(drafts['cases']) == 30 and drafts['synthetic'] and 'SYNTHETIC' in drafts['provenance']
    assert all(c['synthetic'] and c['status'] == 'DRAFT' and c['source_case_id'] and c['expected_templates']
               and set(c['expected_templates']) <= drafts['template_titles'].keys() for c in drafts['cases'])
    for label, title in drafts['template_titles'].items():
        source_hit = hit('synthetic', title=title)
        assert evaluation.score({'expected_templates': [label]}, [source_hit], drafts['template_titles'])['passed']
    for case in drafts['cases']:
        # Operational scoring wiring for all 30 drafts; no actual search or field accuracy measurement.
        source_hit = hit('synthetic', title=drafts['template_titles'][case['expected_templates'][0]])
        assert evaluation.score(case, [source_hit], drafts['template_titles'])['passed']
    print('PASS: frozen cases, scoring, label isolation, offline errors/empty, atomic reports, read-only DB, action experiment, 30 synthetic mappings')


if __name__ == '__main__':
    main()
