"""작은 SQLite 원문으로 관계 후보·상세 경계·안전한 우회와 갱신을 검사합니다."""
import sqlite3
import tempfile
from pathlib import Path
from contextlib import closing
from unittest.mock import patch
from ragkit import cslist, relations, semantic, storage
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_SHEET
from ragkit.records import public_cells, record_id, read_excel


def history(cs, **values):
    return {column: '' for column in HISTORY_COLUMNS} | {'CS NO': cs, '_id_column': 'CS NO',
        '_source_sheet': HISTORY_SHEET, '_history_detail': True,
        '_cell_locations': {'기인 Unit(부품)': '샘플!N2'}} | values


def main(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    a = history('CS-SAME', **{'설비': '세정장비', '기인 모듈': '회전모듈', '기인 Unit(부품)': '회전척',
                             '현상': '회전 중 진동', '조치': '축 정렬 후 정상화'})
    b = history('CS-SAME', **{'설비': '세정장비', '기인 모듈': '배수모듈', '기인 Unit(부품)': '배수밸브',
                             '현상': '누수 발생', '조치': '배수밸브 교체 예정'})
    c = history('CS-OTHER', **{'설비': '도포장비', '기인 모듈': '회전모듈', '기인 Unit(부품)': '회전척',
                              '현상': '회전 중 진동', '조치': '베어링 교체 후에도 재발'})
    query = '회전모듈의 회전척 진동 과거 조치'
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, [a, b, c], rules, schema=schema)
        original = db.execute('SELECT * FROM normalized_records').fetchall()
        assert relations.candidates(db, query)[0] == []  # 읽기 전용 구형 DB도 기존 경로 유지
        assert relations.ensure(db) and not relations.ensure(db)
        hits, info = relations.candidates(db, query)
        assert {h['record_id'] for h in hits} == {record_id(a), record_id(c)}, info
        assert all('배수밸브' not in h['content'] for h in hits)
        assert len(info['paths']) == 2 and all(len(p['cells']) == 2 for p in info['paths'])
        assert {p['receipt'] for p in info['paths']} == {'CS-SAME', 'CS-OTHER'}
        assert all(c['entity'] == relations.key(c['name']) and c['kind'] == relations.COLUMNS[c['column']]
                   for p in info['paths'] for c in p['cells'])
        assert any(cell['location'] == '샘플!N2' for p in info['paths'] for cell in p['cells'])
        assert relations.candidates(db, '세정장비 회전척 누수')[0] == []
        for q in ('회전척 진동', '펌프 누수', '회전모듈 회전척 11.8 bar 진동',
                  '회전모듈 회전척 진동이 아닌 누수', '회전모듈 회전척 정상',
                  '회전모듈 회전척 교체 예정', '회전모듈 또는 회전척 진동'):
            assert not relations.candidates(db, q)[0], q
        for q in ('세정장비 회전척 누수가 없다', '세정장비 회전척 점검 계획',
                  '세정장비 회전척 누수 미확인', '세정장비 회전척 안 움직임',
                  '회전모듈의 회전척 진동은 없습니다'):
            assert not relations.candidates(db, q)[0], q
        with patch.object(semantic, 'vector_hits', return_value=hits):
            details = {}
            selected = semantic.search_cases(db, query, {}, mode='graph', details=details)
            assert details['vector_search']['relations']['candidate_ids']
            assert details['vector_search']['hybrid_ids'] == [h['id'] for h in selected]
            assert all(h['record_id'] != record_id(b) for h in selected)
            for q in ('펌프 누수', '회전모듈 회전척 11.8 bar 진동', '회전모듈 회전척 정상'):
                assert semantic.search_cases(db, q, {}, mode='graph') == semantic.search_cases(db, q, {}, mode='hybrid')
        # 키워드·벡터 양쪽에 없었던 상세가 관계 후보만으로 들어오는지를 확인합니다.
        with patch.object(semantic, 'vector_hits', return_value=[]):
            assert semantic.search_cases(db, query, {}, mode='hybrid', keyword_hits=[]) == []
            graph_only = semantic.search_cases(db, query, {}, mode='graph', keyword_hits=[])
            assert {h['record_id'] for h in graph_only} == {record_id(a), record_id(c)}
        with patch.object(relations, 'candidates', side_effect=AssertionError('고정 조회를 확장했습니다.')):
            with patch.object(semantic, 'vector_hits', side_effect=AssertionError('정확 조회에 임베딩 호출')):
                exact = semantic.search_cases(db, 'CS NO=CS-SAME', {}, mode='graph')
                assert {h['record_id'] for h in exact} == {record_id(a), record_id(b)}
                assert semantic.search_cases(db, query, {}, mode='graph', record_ids=[record_id(b)])[0]['record_id'] == record_id(b)
        assert db.execute('SELECT * FROM normalized_records').fetchall() == original
        # 관계 경로가 있어도 벡터 서버 오류를 숨기지 않으며 키워드·관계 조회는 가능합니다.
        with patch.object(semantic, 'vector_hits', side_effect=RuntimeError('offline')):
            details = {}
            assert semantic.search_cases(db, query, {}, mode='graph', details=details)
            assert details['vector_search']['warning'] and details['vector_search']['relations']['status'] == 'used'
        # 같은 작업 안에서만 두 이름을 연결합니다. 서로 다른 작업의 부품을 한 경로로 합치지 않습니다.
        split = {column: '' for column in schema['columns']} | {'CS 접수번호': 'CS-SPLIT',
                 '_id_column': 'CS 접수번호', 'UNIT명': '회전모듈', '설비코드': '세정장비',
                 '문제현상': '진동 발생', 'Part명(1)': '회전척', 'Part명(2)': '배수밸브'}
        storage.ingest(db, [split], rules, schema=schema, append=True)
        assert relations.candidates(db, '회전척 배수밸브')[0] == []
        work_hits, work_info = relations.candidates(db, '회전모듈 배수밸브 진동')
        assert [h['record_id'] for h in work_hits] == ['CS-SPLIT'], work_info
        assert [p['evidence_id'] for p in work_info['paths']] == ['CS-SPLIT/작업2']
        assert work_info['paths'][0]['receipt'] == 'CS-SPLIT'
        assert {c['column'] for c in work_info['paths'][0]['cells']} == {'UNIT명', 'Part명(2)'}
        before = db.execute('SELECT * FROM rag_relations ORDER BY 1,2,3').fetchall()
        prepared = storage.prepared_record(db, split, public_cells(split), rules)
        with patch.object(relations, 'sync', side_effect=RuntimeError('failure')):
            try:
                cslist.rebuild_search(db, [prepared], schema, rules=rules, incremental=True)
            except RuntimeError:
                pass
            else:
                raise AssertionError('관계 갱신 실패가 원자적으로 처리되지 않았습니다.')
        assert db.execute('SELECT * FROM rag_relations ORDER BY 1,2,3').fetchall() == before
        changed = split | {'Part명(1)': '흡착척'}
        storage.ingest(db, [changed], rules, schema=schema, append=True)
        assert not db.execute("SELECT 1 FROM rag_relations WHERE record_id='CS-SPLIT' AND entity='회전척'").fetchone()
        assert db.execute("SELECT 1 FROM rag_relations WHERE record_id='CS-SPLIT' AND entity='흡착척'").fetchone()
        storage.ingest(db, [b], rules, schema=schema)
        assert not relations.candidates(db, query)[0] and db.execute('SELECT COUNT(DISTINCT record_id) FROM rag_relations').fetchone()[0] == 1
        broad = [history(f'CS-BROAD-{i}', **{'설비': '세정장비', '기인 Unit(부품)': '회전척', '현상': '진동'}) for i in range(101)]
        storage.ingest(db, broad, rules, schema=schema)
        assert relations.candidates(db, '세정장비 회전척 진동')[1]['status'] == 'skipped'
    # 첫 생성의 바깥 SAVEPOINT가 커밋되므로 새 연결에서도 자료가 유지됩니다.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'cases.sqlite3'
        with closing(sqlite3.connect(path)) as db:
            storage.ingest(db, [a], rules, schema=schema)
            assert not db.in_transaction
            relations.ensure(db)
        with closing(sqlite3.connect(path)) as db:
            assert not relations.ensure(db) and relations.candidates(db, query)[0]
    # 배포한 가상 Excel과 안내 질문으로 같은 CS의 서로 다른 부품 상세를 실제로 찾습니다.
    examples = read_excel(root / 'relation_sample.xlsx', HISTORY_SHEET, required_columns=HISTORY_COLUMNS, id_column='CS NO')
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, examples, rules, schema=schema)
        relations.ensure(db)
        hits, info = relations.candidates(db, '세정장비 회전모듈 진동 과거 조치')
        assert len(examples) == 4 and len(hits) == len(info['paths']) == 3
        same_cs = [h for h in hits if h['evidence'][0]['cells']['CS NO'] == 'CS-DEMO-001']
        assert len(same_cs) == 2 and {h['evidence'][0]['cells']['기인 Unit(부품)'] for h in same_cs} == {'구동벨트', '회전척'}
    # 등록된 코드의 숫자만 허용하며 측정값·다른 코드나 정확한 CS 조건을 넓히지 않습니다.
    coded = [history(f'CS-CODE-{i}', **{'설비': code, '기인 모듈': module,
             '기인 Unit(부품)': '회전척', '현상': '진동 발생', '조치': '축 정렬'})
             for i, (code, module) in enumerate((('SIM-ETCH-001', 'IFEP2'),
                 ('SIM-ETCH-010', 'IFEP2'), ('SIM-ETCH-001', '다른모듈')), 1)]
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, coded, rules, schema=schema)
        relations.ensure(db)
        q = 'sim-etch-001에서 IFEP2의 진동 과거 조치'
        related, info = relations.candidates(db, q)
        assert [h['record_id'] for h in related] == [record_id(coded[0])], info
        assert len(relations.candidates(db, 'IFEP2 회전척 진동')[0]) == 2
        for bad in ('SIM-ETCH-0010 IFEP2 진동', 'XSIM-ETCH-001 IFEP2 진동',
                    'SIM-ETCH-001_2 IFEP2 진동', 'SIM-ETCH-001 IFEP20 진동',
                    'SIM-ETCH-001 IFEP2 11.8bar 진동', 'SIM-ETCH-001 IFEP2 -0.5 진동',
                    'SIM-ETCH-001 IFEP2 2026-09 진동', 'SIM-ETCH-001 IFEP2 1건 진동',
                    'SIM-ETCH-001 IFEP2 진동 없음', 'SIM-ETCH-001 IFEP2 정상',
                    'SIM-ETCH-001 IFEP2 교체 예정'):
            assert not relations.candidates(db, bad)[0], bad
        with patch.object(semantic, 'vector_hits', side_effect=AssertionError('코드 정확 조회에 임베딩 호출')):
            detail = {}
            chosen = semantic.search_cases(db, q, {}, mode='graph', details=detail)
            assert chosen[0]['record_id'] == record_id(coded[0])
            assert detail['vector_search']['relations']['status'] == 'used'
            assert {h['record_id'] for h in chosen} <= {record_id(coded[0]), record_id(coded[2])}
            # 정확 검색이 허용한 후보 범위 밖의 관계 경로도 표시하거나 추가하지 않습니다.
            only_other = [h for h in semantic.retrieve(db, q, limit=20) if h['record_id'] == record_id(coded[2])]
            detail = {}
            assert semantic.search_cases(db, q, {}, mode='graph', keyword_hits=only_other, details=detail) == only_other
            assert not detail['vector_search']['relations']['paths']
            for bad in ('SIM-ETCH-001 IFEP2 11.8bar 진동', 'SIM-ETCH-001 IFEP2 정상'):
                assert semantic.search_cases(db, bad, {}, mode='graph') == semantic.search_cases(db, bad, {}, mode='hybrid')
            with patch.object(relations, 'candidates', side_effect=AssertionError('CS·날짜 조회를 확장했습니다.')):
                for exact in ('CS NO=CS-CODE-1', 'SIM-ETCH-001 IFEP2 2026-09-01 진동'):
                    assert semantic.search_cases(db, exact, {}, mode='graph') == semantic.search_cases(db, exact, {}, mode='hybrid')
    print('PASS: 관계 조회·상세/작업 분리·숫자 코드/측정값 구분·정확 후보 보존·안전 조건·원자적 갱신·배포 예제 질문')
