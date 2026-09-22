"""조치별 결과·원문 연결과 답변 보정을 네트워크 없이 검증합니다."""
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

from ragkit import cslist, llm, records, search, storage


def main(root):
    extract = getattr(cslist, 'action_outcomes', None)
    assert callable(extract), '원문에서 조치별 결과를 연결하는 기능이 없습니다.'
    original = '센서 교체 후에도 동일 증상 발생. 배선 접촉 불량 확인하여 수정 후 정상화.'
    steps = extract({'조치내용(1)': original})
    assert [item['status'] for item in steps] == ['증상 지속', '정상화'], steps
    assert [item['order'] for item in steps] == [1, 2]
    assert steps[0]['action_quote'] == '센서 교체'
    assert steps[1]['action_quote'] == '배선 접촉 불량 확인하여 수정'
    for item in steps:
        assert item['column'] == '조치내용(1)'
        assert item['action_quote'] in original and item['outcome_quote'] in original
        assert original.index(item['action_quote']) < original.index(item['outcome_quote'])

    cases = [
        ('필터 세척 후 일부 개선.', '일부 개선'),
        ('센서 교체 후 동일 증상 지속.', '증상 지속'),
        ('센서 교체 후에도 재발함.', '증상 지속'),
        ('센서 교체.', '결과 미확인'),
        ('센서 교체 후 정상화되지 않음.', '결과 미확인'),
        ('센서 교체 후 정상화 실패.', '결과 미확인'),
        ('센서 교체 후 정상화 여부 미확인.', '결과 미확인'),
        ('센서 교체 후 정상화 예상.', '결과 미확인'),
        ('센서 교체 후 정상화?', '결과 미확인'),
        ('센서 교체 예정 후 정상화 예상.', '결과 미확인'),
        ('센서 교체 후 증상 지속 안함.', '결과 미확인'),
        ('센서 교체 후 정상화. 장기 재발 미확인.', '결과 미확인'),
        ('센서 교체 후 정상화했으나 재발함.', '결과 미확인'),
        ('센서 교체 후 알람 지속되어 배선 수정 후 정상화.', '결과 미확인'),
        ('스트레이너 세척 후 11.8 L/min.', '결과 미확인'),
        ('1.SCW132 11.8 STEP Thinner Attack 불량 점검', '결과 미확인'),
    ]
    for value, expected in cases:
        parsed = extract({'조치내용(1)': value})
        assert len(parsed) == 1 and parsed[0]['status'] == expected, (value, parsed)
        assert parsed[0]['action_quote'] in value
        assert parsed[0]['outcome_quote'] in value
    assert extract({'조치내용(1)': '', '원인(1)': '센서 교체 후 정상화.'}) == []
    grouped = extract({'조치내용(1)': '센서 교체 및 배선 수정 후 정상화.'})
    assert len(grouped) == 1 and grouped[0]['action_quote'] == '센서 교체 및 배선 수정'

    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    row = dict.fromkeys(schema['columns'], '')
    row.update({'CS 접수번호': 'CS-ACTION-001', '제목': '센서 신호 불량', '현상': '센서 신호 불량',
                '조치내용(1)': original, '조치내용(2)': '센서 청소.',
                '_id_column': 'CS 접수번호', '_source_key': 'fictional-action-outcomes',
                '_cell_locations': {'조치내용(1)': 'CSList!AZ2', '조치내용(2)': 'CSList!BZ2'}})
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, [row], rules, schema=schema)
        saved = db.execute('SELECT original_json FROM normalized_records').fetchone()[0]
        for table in ('rag_documents', 'rag_original_documents'):
            sections = json.loads(db.execute(f'SELECT evidence FROM {table}').fetchone()[0])
            by_kind = {section['kind']: section for section in sections}
            assert by_kind['작업1']['action_outcomes'] == steps
            assert by_kind['작업2']['action_outcomes'][0]['status'] == '결과 미확인'
            assert by_kind['작업1']['cells']['조치내용(1)'] == original
        # 검토·재색인 후에도 결과 분류의 근거는 채택문이 아닌 원문입니다.
        entry = {'source_key': row['_source_key'], 'id': row['CS 접수번호'],
                 'revision': records.row_revision(row), 'column': '조치내용(2)',
                 'action': 'edit', 'value': '센서를 청소했습니다.'}
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
        db.execute("UPDATE rag_search_meta SET signature='before-action-outcomes'")
        assert cslist.ensure_search_index(db, rules, schema)
        assert not cslist.ensure_search_index(db, rules, schema)
        assert db.execute('SELECT original_json FROM normalized_records').fetchone()[0] == saved
        assert db.execute('SELECT COUNT(*) FROM cell_reviews').fetchone()[0] == 1
        hits = search.retrieve(db, 'CS-ACTION-001의 조치 결과는?')
        context, messages = llm.case_answer_messages('센서 신호 불량의 과거 조치는?', hits)
        work = next(section for section in json.loads(context) if section['kind'] == '작업1')
        assert work['action_outcomes'] == steps
        assert '센서를 청소했습니다.' not in context
        assert 'action_outcomes' in messages[0]['content']

        reference = {'id': 'CS-ACTION-001/작업1', 'column': '조치내용(1)'}
        bad = {'claims': [{'text': '센서 교체 후 정상화되었습니다.', 'evidence': [reference]}]}
        good = {'claims': [{'text': '센서 교체 후에도 동일 증상이 발생했습니다. '
                                   '배선 접촉 불량 확인하여 수정 후 정상화되었습니다.',
                            'evidence': [reference]}]}
        try:
            llm.validate_claims(bad, hits)
        except llm.ClaimValidationError as error:
            assert error.category == 'action_outcome'
        else:
            raise AssertionError('배선 수정 결과를 실패한 센서 교체의 효과로 사용했습니다.')
        assert '동일 증상' in llm.validate_claims(good, hits)

        def response(payload):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])
        create = Mock(side_effect=[response(bad), response(good)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        details = {}
        llm.answer(client, {'model': 'offline-test', 'reasoning_effort': 'high',
                           'max_context_chars': 40000, 'max_tokens': 3000},
                   '센서 신호 불량의 과거 조치는?', hits, details=details)
        assert create.call_count == 2 and details['attempts'] == 2
        assert details['validation_events'][0]['category'] == 'action_outcome'
        assert '센서 교체' in create.call_args.kwargs['messages'][-1]['content']
        from ragkit.web import pack_answer
        packed = pack_answer('질문', '', hits, None, details, 1)
        evidence = packed['claims'][0]['evidence'][0]
        assert evidence['quote'] == original and evidence['location'] == 'CSList!AZ2'
        assert evidence['action_outcomes'] == steps
    print('PASS: 조치별 결과·부정/예정/복합/누락·원문 보존·기존 DB 갱신·답변 재보정·웹 근거')
