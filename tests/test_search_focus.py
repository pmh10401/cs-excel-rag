"""증상 구별·검색 가중치·기존 색인의 안전한 갱신을 실제 SQLite로 검사합니다."""
import sqlite3
import unicodedata
from pathlib import Path
from unittest.mock import Mock, patch
from ragkit import cslist, records, storage, search, llm


def contrast_and_duplicates(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    question = '냉각 팬 소음이 아니라 이송 벨트가 끊어집니다.'
    assert search.focused_query(question) == '이송 벨트가 끊어집니다.'
    assert search.focused_query('온도는 11.8 C입니다. ' + question) == '온도는 11.8 C입니다. 이송 벨트가 끊어집니다.'
    assert search.focused_query('The issue is not fan noise but a broken belt.') == 'The issue is a broken belt.'
    assert search.focused_query(question + '\n고온에서만 발생하며 11.8 C까지 상승합니다.') == (
        '이송 벨트가 끊어집니다.\n고온에서만 발생하며 11.8 C까지 상승합니다.')
    assert search.focused_query('Not noise but vibration.\nOnly at 20 rpm.') == 'vibration.\nOnly at 20 rpm.'
    assert search.focused_query('야간 고온 운전 중\n' + question) == '야간 고온 운전 중\n이송 벨트가 끊어집니다.'
    for value in ('압력이 내려가지 않습니다', '누수가 없음', '팬뿐만 아니라 펌프도 고장',
                  '팬만이 아니라 펌프도 고장', 'Not only fan noise but also vibration.',
                  '팬 소음이 아니라', '팬 소음이 아니라 .', '팬 소음이 아니라면 점검',
                  'The fan is not running. The belt is worn but no leak was found.',
                  'CS-FOCUS-001 ' + question, '2026-09-01 ' + question,
                  '팬이 아니라 펌프가 아니라 벨트 문제입니다.'):
        assert search.focused_query(value) == value, value
    values = [
        {'현상': '냉각 팬 소음 발생', '조치내용(1)': '팬 교체'},
        {'현상': '이송 벨트가 끊어짐', '조치내용(1)': '벨트 장력 20 N으로 조정'},
        {'현상': '이송 벨트가 끊어짐', '조치내용(1)': '벨트 장력 20 N으로 조정'},
        {'현상': '이송 벨트가 끊어짐', '조치내용(1)': '벨트 장력 30 N으로 조정'},
        {'현상': '이송 벨트가 끊어짐', '조치내용(1)': '벨트 교체'},
    ]
    rows = [{key: '' for key in schema['columns']} | value | {
        schema['id_column']: f'CS-CONTRAST-{i:03}', '_id_column': schema['id_column']}
        for i, value in enumerate(values, 1)]
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        before = db.total_changes
        for options in ({}, {'original_only': True}):
            hits = search.retrieve(db, question, limit=3, **options)
            identities = {h['record_id'] for h in hits}
            assert len(hits) == 3 and 'CS-CONTRAST-001' not in identities, identities
            assert {'CS-CONTRAST-004', 'CS-CONTRAST-005'} <= identities, '수치·조치 차이를 지웠습니다.'
            assert len(identities & {'CS-CONTRAST-002', 'CS-CONTRAST-003'}) == 1
            for h in hits:
                source = rows[int(h['record_id'].rsplit('-', 1)[1])-1]
                assert all(source[k] == v for s in h['evidence'] for k, v in s['cells'].items())
        # 접수번호로 조회할 때는 내용이 같아도 두 건 모두 돌려줍니다.
        hits = search.retrieve(db, 'CS-CONTRAST-002 CS-CONTRAST-003', limit=3)
        assert {h['record_id'] for h in hits} == {'CS-CONTRAST-002', 'CS-CONTRAST-003'}
        assert len(search.distinct_hits(hits)) == 1
        # 원문이 같아도 채택한 조치가 달라지면 별개 후보입니다.
        hits[1]['evidence'][-1]['accepted']['조치내용(1)'] = '벨트 교체 후 정상화'
        assert len(search.distinct_hits(hits)) == 2
        assert db.total_changes == before
    print('PASS: 대조 질문·부정/추가/식별자 보존·동일 셀 중복 제외·수치/해결책/근거 보존')


def phrase_boundary_regression(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    # 셀 사이를 이어 붙여 생긴 '압력 정상'은 실제 구절이 아닙니다.
    cases = [{'현상': '압력 측정값 정상'}, {'문제현상': '불안정 압력', '현상': '정상 온도 확인'}]
    rows = [{key: values.get(key, '') for key in schema['columns']} |
            {schema['id_column']: f'CS-SCOPE-{i:03}', '_id_column': schema['id_column']}
            for i, values in enumerate(cases, 1)]
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        before = db.total_changes
        for options in ({}, {'original_only': True}):
            hits = search.retrieve(db, '압력 정상', **options)
            assert hits[0]['record_id'] == 'CS-SCOPE-001', (options, [h['id'] for h in hits])
        assert db.total_changes == before


def query_term_limit_regression(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    question = ('온도 압력 유량 습도 진공 배기 챔버 로봇 웨이퍼 필터 송풍기 전압 전류 저항 주파수 '
                '회전 속도 가속도 감도 조도 밝기 진동 소음 누수 누설 냉각수 가스 질소 산소 수소 오일 '
                '윤활 베어링 기어 모터 펌프 밸브 센서 배관 튜브 호스 케이블 배선 단자 접점 스위치 퓨즈 '
                '히터 열판 냉각기 전원 통신 신호 화면 표시 경보 알람 패널 차단기 나사 볼트 너트 패킹 '
                '씰 실린더 피스톤 벨트 체인 축 부하 전력 토크 풍량 풍속 밀도 농도 수위 잔량 용량 출력 '
                '입력 배출 복귀 상승 하강')
    with sqlite3.connect(':memory:') as db:
        row = {key: '' for key in schema['columns']} | {
            schema['id_column']: 'CS-LIMIT-001', '_id_column': schema['id_column'], '현상': '압력 정상'}
        storage.ingest(db, [row], rules, schema=schema)
        # 파생 구절 개수를 더해 기존 128개 단어 한도 안의 질문을 거절하면 안 됩니다.
        assert search.retrieve(db, question) == []


def query_alias_regression(root):
    for ending in ('복귀', '돌아오는', '돌아갑니다', '돌아와요', '되돌아왔습니다'):
        question = '대기압으로 ' + ending
        for text in (question, unicodedata.normalize('NFD', question)):
            assert search.query_aliases(text) == {'vent', '벤트'}, text
    for text in ('상대기압 복귀', '진공으로 복귀', '이벤트 복귀', '선풍기'):
        assert search.query_aliases(text) == set(), text
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    examples = [('송풍기가 느려집니다', 'Air fan RPM low.'),
                ('너무 뜨거워집니다', '고온 경보. 관찰 결과 미기재.'),
                ('대기압으로 돌아오는 동작이 늦습니다', 'Vent valve response delayed.')]
    rows = [{key: '' for key in schema['columns']} | {
        schema['id_column']: f'CS-ALIAS-{i:03}', '_id_column': schema['id_column'],
        '현상': value, '접수일자': f'2026-09-{i:02}', 'MTTR': '0' if i == 1 else ''}
        for i, (_, value) in enumerate(examples, 1)]
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        before = db.total_changes
        for i, (question, original) in enumerate(examples, 1):
            for options in ({}, {'original_only': True}):
                hits = search.retrieve(db, question, expand_query=True, **options)
                assert hits and hits[0]['record_id'] == f'CS-ALIAS-{i:03}', (question, options)
                assert original in hits[0]['content']
                assert hits[0]['evidence'][0]['cells']['현상'] == original
                assert hits[0]['evidence'][0]['cells']['MTTR'] == ('0' if i == 1 else '')
        with patch.object(search, 'query_aliases', side_effect=AssertionError('정확 조회에서 어휘 확장')):
            assert search.retrieve(db, 'CS-ALIAS-001 송풍기', expand_query=True)[0]['record_id'] == 'CS-ALIAS-001'
            assert search.retrieve(db, 'CS-ALIAS-999 송풍기', expand_query=True) == []
            assert search.retrieve(db, '2026-09-01 송풍기', expand_query=True)[0]['record_id'] == 'CS-ALIAS-001'
            search.retrieve(db, '송풍기가 느려집니다')  # 키워드 단독 경로는 확장하지 않습니다.
        # 파생어를 더하면 129개가 되는 질문은 원래 128개 조건으로 실행합니다.
        words = ('송풍기',) + tuple(f'q{i}' for i in range(127))
        with patch.object(search, 'word_tokens', return_value=words):
            assert search.retrieve(db, '송풍기', expand_query=True) == []
        with patch.object(search, 'word_tokens', return_value=words + ('q128',)):
            try:
                search.retrieve(db, '송풍기', expand_query=True)
            except ValueError as error:
                assert '질문이 너무' in str(error)
            else:
                raise AssertionError('원래 단어가 128개를 넘는 질문을 허용했습니다.')
        assert db.total_changes == before
    assert search.concepts('송풍기 뜨겁 대기압 복귀') == set(), '검색 확장이 사실 검사에 섞였습니다.'


def redundant_index_cleanup(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    row = {key: '' for key in schema['columns']} | {
        schema['id_column']: 'CS-CLEAN-001', '_id_column': schema['id_column'], '현상': '펌프 누수'}
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, [row], rules, schema=schema)
        # 이전 버전의 중복 FTS와 내부 테이블을 재현합니다.
        with db:
            db.execute('CREATE VIRTUAL TABLE rag_chunks USING fts5(content)')
            db.execute("INSERT INTO rag_chunks VALUES ('복구 가능한 구형 검색 색인')")
        protected = ('normalized_records', 'cell_reviews', 'rag_documents', 'rag_original_documents', 'rag_document_ids', 'rag_search_meta')
        before = {table: db.execute(f'SELECT * FROM {table}').fetchall() for table in protected}
        hits = search.retrieve(db, '펌프 누수')
        with patch.object(cslist, 'rebuild_search', side_effect=AssertionError('정리 중 불필요한 재색인')):
            db.execute('BEGIN')
            assert cslist.ensure_search_index(db, rules, schema)
            assert not db.execute("SELECT 1 FROM sqlite_master WHERE name GLOB 'rag_chunks*'").fetchone()
            db.rollback()
            assert db.execute('SELECT content FROM rag_chunks').fetchone() == ('복구 가능한 구형 검색 색인',)
            assert cslist.ensure_search_index(db, rules, schema)
            changed = db.total_changes
            assert not cslist.ensure_search_index(db, rules, schema)
            assert db.total_changes == changed
        assert search.retrieve(db, '펌프 누수') == hits
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name GLOB 'rag_chunks*'").fetchone()
        assert {table: db.execute(f'SELECT * FROM {table}').fetchall() for table in protected} == before
    print('PASS: 구형 중복 색인 제거·FTS 내부 테이블 정리·원문/검토/검색 보존·롤백·반복 실행')


def main(root):
    contrast_and_duplicates(root)
    redundant_index_cleanup(root)
    query_alias_regression(root)
    phrase_boundary_regression(root)
    query_term_limit_regression(root)
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
                              required_columns=schema['columns'], id_column=schema['id_column'])
    with sqlite3.connect(':memory:') as db:
        prepared = storage.ingest(db, rows, rules, schema=schema)
        for original_only in (True, False):
            for question in ('펌프 소음 과거 조치', '펌프 진동 과거 조치', '팬 누설 사례'):
                assert search.retrieve(db, question, original_only=original_only) == [], question
        assert {h['record_id'] for h in search.retrieve(db, '펌프 누수')} == {'CS-DEMO-003'}
        assert {h['record_id'] for h in search.retrieve(db, '펌프 누수 소음')} == {'CS-DEMO-003'}
        assert search.retrieve(db, '소음')[0]['record_id'] == 'CS-DEMO-005', '이상음도 찾아야 합니다.'
        assert search.retrieve(db, 'pump leakage')[0]['record_id'] == 'CS-DEMO-003'
        assert search.retrieve(db, 'CS-DEMO-003 소음 기록은?')[0]['record_id'] == 'CS-DEMO-003'
        leak_hits = search.retrieve(db, '펌프 누수')
        claim = {'text': '펌프 누수 시 교체용 씰을 발주했으며 교체는 아직 미실시입니다.',
                 'evidence': [{'id': 'CS-DEMO-003/작업2', 'column': '조치내용(2)'}]}
        try:
            llm.validate_claims({'claims': [claim]}, leak_hits)
        except ValueError as error:
            assert '증상' in str(error)
        else:
            raise AssertionError('누수 근거 없이 누수 증상을 설명했습니다.')
        claim['evidence'].append({'id': 'CS-DEMO-003/접수', 'column': '현상'})
        assert '미실시' in llm.validate_claims({'claims': [claim]}, leak_hits)
        for procurement in ('발주', '주문'):
            try:
                llm.validate_claims({'claims': [claim | {'text': f'씰을 {procurement}했습니다.'}]}, leak_hits)
            except ValueError as error:
                assert '미실시' in str(error)
            else:
                raise AssertionError('발주만 설명하고 교체 미실시를 생략했습니다.')
        for state in ('교체하지 않았습니다.', '교체는 안 했습니다.', '아직 교체 전입니다.', '미교체 상태입니다.'):
            assert llm.validate_claims({'claims': [claim | {'text': '씰을 발주했으며 ' + state}]}, leak_hits)
        client = Mock()
        reply, _ = llm.answer(client, {}, '펌프 소음', search.retrieve(db, '펌프 소음'))
        assert '없' in reply and not client.mock_calls, '없는 사례를 모델이 만들어서는 안 됩니다.'
        assert not search.compare_search(db, [{'question': '펌프 소음', 'expect_no_match': True}])[0]['accepted_hits']
        # 이미 저장된 원문과 채택 이력이 있는 구형 DB를 재현합니다.
        entry = dict(source_key=rows[2]['_source_key'], id='CS-DEMO-003', revision=records.row_revision(rows[2]),
                     column='현상', action='edit', value=rows[2]['현상'] + ' qzxjvkwp.')
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
        prepared = [storage.prepared_record(db, row, records.public_cells(row), rules) for row in rows]
        saved = {table: db.execute(f'SELECT * FROM {table}').fetchall()
                 for table in ('normalized_records', 'cell_reviews')}
        with db:
            for table in ('rag_documents', 'rag_original_documents'):
                db.execute(f'DROP TABLE {table}')
                db.execute(f'CREATE VIRTUAL TABLE {table} USING fts5(document_id UNINDEXED, record_id UNINDEXED, content UNINDEXED, evidence UNINDEXED, keywords)')
                db.execute(f'INSERT INTO {table} VALUES (?,?,?,?,?)', ('old', 'old', 'old', '[]', 'old'))
        old = db.execute('SELECT * FROM rag_documents').fetchall()
        try:
            cslist.rebuild_search(db, prepared + [{}], schema)
        except KeyError:
            pass
        else:
            raise AssertionError('손상된 입력을 성공으로 처리했습니다.')
        assert db.execute('SELECT * FROM rag_documents').fetchall() == old, '실패한 마이그레이션은 구형 색인까지 복구해야 합니다.'
        cslist.rebuild_search(db, prepared, schema)
        for table, values in saved.items():
            assert db.execute(f'SELECT * FROM {table}').fetchall() == values, table
        assert len(db.execute('PRAGMA table_info(rag_documents)').fetchall()) == 10
        assert search.retrieve(db, 'qzxjvkwp')[0]['record_id'] == 'CS-DEMO-003'
        assert not search.retrieve(db, 'qzxjvkwp', original_only=True)
        # 외부 검토 트랜잭션도 내부 SAVEPOINT가 먼저 확정하면 안 됩니다.
        before = db.execute('SELECT * FROM rag_documents').fetchall()
        db.execute('BEGIN')
        cslist.rebuild_search(db, prepared[:1], schema)
        db.rollback()
        assert db.execute('SELECT * FROM rag_documents').fetchall() == before
        # 버전 변경은 한 번만 이관하며 읽기 경로에서는 DB를 쓰지 않습니다.
        from unittest.mock import patch
        with patch.object(cslist, 'SEARCH_VERSION', cslist.SEARCH_VERSION + 1):
            assert cslist.ensure_search_index(db, rules, schema)
            changed = db.total_changes
            assert not cslist.ensure_search_index(db, rules, schema)
            assert db.total_changes == changed
            assert db.execute('SELECT COUNT(*) FROM rag_document_ids').fetchone()[0] == len(rows)
            assert search.retrieve(db, 'qzxjvkwp')[0]['record_id'] == 'CS-DEMO-003'
        for table, values in saved.items():
            assert db.execute(f'SELECT * FROM {table}').fetchall() == values

    # 같은 단어가 조치에만 있는 기록보다 현상/알람에 있는 기록을 먼저 찾습니다.
    cases = [
        {'현상': '온도 상승', '알람코드': 'E202', '조치내용(1)': '전원 차단'},
        {'현상': '통신 끊김', '조치내용(1)': '온도 상승 확인. E202 기록 확인.'},
        {'현상': '펌프 소음 없음. 진동 있음.', '조치내용(1)': '점검 예정'},
        {'현상': '팬 이상음', '조치내용(1)': '베어링 확인'},
    ]
    synthetic = [{key: values.get(key, '') for key in schema['columns']} |
                 {schema['id_column']: f'CS-FOCUS-{index:03}', '_id_column': schema['id_column']}
                 for index, values in enumerate(cases, 1)]
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, synthetic, rules, schema=schema)
        for question in ('온도 상승', 'E202'):
            assert search.retrieve(db, question)[0]['record_id'] == 'CS-FOCUS-001', question
        hits = search.retrieve(db, '펌프 소음')
        assert [h['record_id'] for h in hits] == ['CS-FOCUS-003']
        assert '소음 없음' in hits[0]['content'], '검색 일치는 긍정·해결 판정이 아닙니다.'
        assert search.retrieve(db, '팬 소음')[0]['record_id'] == 'CS-FOCUS-004'
    print('PASS: 증상·설비 구별, 동의어, 현상·알람 가중치, 무응답 시 API 생략, 색인 마이그레이션·롤백·검토 보존')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
