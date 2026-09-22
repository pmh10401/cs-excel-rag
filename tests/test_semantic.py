"""실제 SQLite·FAISS와 모의 임베딩 서버로 검색·재사용·오래된 벡터 제외를 검사합니다."""
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ragkit import cslist, records, search, semantic, storage


def action_variants(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    values = [
        {'현상': '펌프 진동', '조치내용(1)': '베어링 교체 후 정상화.'},
        {'현상': '펌프 진동', '조치내용(1)': '베어링 교체 후 정상화.'},
        {'현상': '펌프 누수', '조치내용(1)': '씰 교체 후 정상화.'},
        {'현상': '펌프 진동', '조치내용(1)': '축 정렬 후 정상화.'},
        {'현상': '펌프 진동', '조치내용(1)': '축 정렬 후에도 재발함.'},
    ]
    rows = [{key: '' for key in schema['columns']} | value | {
        schema['id_column']:f'CS-ACTION-{i}', '_id_column':schema['id_column'], '접수일자':f'2026-09-{i:02}'}
        for i, value in enumerate(values, 1)]
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        hits = [search.retrieve(db, row[schema['id_column']])[0] for row in rows]
        before = json.dumps(hits, ensure_ascii=False)
        changed = semantic.prefer_action_variants(hits, '펌프 진동 과거 조치')
        assert [h['record_id'] for h in changed] == ['CS-ACTION-1','CS-ACTION-4','CS-ACTION-3','CS-ACTION-2','CS-ACTION-5']
        assert json.dumps(hits, ensure_ascii=False) == before, '원본 후보·인용 셀을 수정했습니다.'
        assert sorted(h['id'] for h in changed) == sorted(h['id'] for h in hits)
        # UI/CLI의 실제 공통 진입점에서도 반환 후보와 검색 상세가 같은 순서여야 합니다.
        with patch.object(semantic, 'vector_hits', return_value=hits), patch.object(semantic, 'vector_count', return_value=5):
            for mode in ('hybrid', 'vector'):
                details = {}
                selected = semantic.search_cases(db, '펌프 진동 과거 조치', {}, mode=mode,
                                                 keyword_hits=[], details=details, limit=3)
                assert [h['id'] for h in selected] == [h['id'] for h in changed[:3]]
                assert details['vector_search'][mode+'_ids'] == [h['id'] for h in selected]
                assert details['vector_search']['action_diversity']['changed_positions'] == 1
        for question in ('CS-ACTION-2', '펌프 11.8 bar 진동', 'SCW132 진동', '펌프 베어링 교체 조치'):
            assert semantic.prefer_action_variants(hits, question) == hits, question
        # 누락된 근거나 조치가 없는 구형 후보도 버리거나 임의로 묶지 않습니다.
        unknown = [{'id':'no-evidence'}, {'id':'other','evidence':[]}]
        assert semantic.prefer_action_variants(unknown, '진동') == unknown
    print('PASS: 동일 증상 안의 조치 다양성·첫 후보/상태/원문 보존·숫자/구체 조치 조건 보호')


def main(root):
    action_variants(root)
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = records.read_excel(root / 'cslist_sample.xlsx', 'CSList', required_columns=schema['columns'], id_column=schema['id_column'])
    config = {'embedding': {'base_url': 'http://127.0.0.1:1/v1', 'model': 'sample', 'batch_size': 2, 'min_score': 0.5}}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        # 질문에 없는 단어로 기록된 진공 사례를 의미 검색으로 찾는 상황입니다.
        values = [[1., 0., 0.] if '압력이 내려가지' in t or '진공' in t else [0., 1., 0.]
                  for t in kwargs['input']]
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in reversed(list(enumerate(values)))])
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create), close=lambda: None)
    with tempfile.TemporaryDirectory() as temporary, closing(sqlite3.connect(Path(temporary) / 'vectors.sqlite3')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        assert db.execute('SELECT COUNT(*) FROM rag_document_ids').fetchone()[0] == len(rows)
        assert cslist.ensure_search_index(db, rules, schema) is False
        before = db.execute('SELECT * FROM normalized_records').fetchall()
        with patch.object(semantic, 'client_for', return_value=client):
            result = semantic.build_vectors(db, client, config)
            assert result['indexed'] == len(rows) and not result['failures']
            assert db.execute('SELECT * FROM normalized_records').fetchall() == before
            count = len(calls)
            assert semantic.build_vectors(db, client, config)['generated'] == 0 and len(calls) == count
            statements = []
            db.set_trace_callback(statements.append)
            hits = semantic.search_cases(db, '압력이 내려가지 않습니다',
                                         {'embedding': config['embedding'] | {'min_score': -1}}, mode='vector')
            db.set_trace_callback(None)
            assert sum('SELECT document_id,record_id,content,evidence FROM rag_documents WHERE' in sql
                       for sql in statements) == 1, '벡터 후보마다 원문 전체를 다시 조회했습니다.'
            lookup = next(sql for sql in statements if 'SELECT document_id,record_id,content,evidence FROM rag_documents WHERE' in sql)
            plan = str(db.execute('EXPLAIN QUERY PLAN ' + lookup).fetchall())
            assert 'INDEX 0:=' in plan and 'sqlite_autoindex_rag_document_ids' in plan, plan
            assert not any('SELECT id FROM normalized_records' in sql for sql in statements)
            assert hits[0]['record_id'] == 'CS-DEMO-001'
            assert '영점 조정함' in hits[0]['content'] and hits[0]['evidence'][0]['locations']
            info = {}
            assert semantic.search_cases(db, '압력이 내려가지 않습니다', config, details=info)
            assert info['vector_search']['hybrid_ids'] and info['vector_search']['vector_records'] == len(rows)
            assert {'keyword', 'index_load', 'embedding', 'vector_search', 'source_lookup', 'fusion'} <= info['timings'].keys()
            # 제외한 현상은 임베딩 입력에서 빼지만, 조회 추적에는 원래 질문도 남깁니다.
            contrast = '팬 소음이 아니라 압력이 내려가지 않습니다'
            details = {}
            semantic.search_cases(db, contrast, config, mode='vector', details=details)
            assert calls[-1]['input'] == ['압력이 내려가지 않습니다']
            assert details['vector_search']['query'] == contrast
            assert details['vector_search']['focused_query'] == '압력이 내려가지 않습니다'
            # 평가용 외부 클라이언트는 서버 자동 준비·종료를 호출하지 않습니다. 순위는 같습니다.
            expected = semantic.search_cases(db, '압력이 내려가지 않습니다', config)
            with patch.object(semantic, 'client_for', side_effect=AssertionError('server preparation')):
                assert semantic.search_cases(db, '압력이 내려가지 않습니다', config,
                                             embedding_client=client) == expected
            # 보완어는 결합 검색에만 사용하며 벡터 장애 시 기존 키워드로 돌아갑니다.
            with patch.object(semantic, 'retrieve', wraps=search.retrieve) as lookup, \
                    patch.object(semantic, 'vector_hits', return_value=[]):
                for mode in ('keyword', 'vector', 'hybrid'):
                    lookup.reset_mock()
                    semantic.search_cases(db, '뜨거운 냉각수', config, mode=mode)
                    assert lookup.call_args.kwargs['expand_query'] == (mode == 'hybrid')
            with patch.object(semantic, 'retrieve', wraps=search.retrieve) as lookup, \
                    patch.object(semantic, 'vector_hits', side_effect=RuntimeError('offline')):
                info = {}
                fallback = semantic.search_cases(db, '뜨거운 냉각수', config, details=info)
                assert fallback == search.retrieve(db, '뜨거운 냉각수', limit=20)[:3]
                assert not lookup.call_args.kwargs.get('expand_query', False)
                assert info['vector_search']['keyword_ids'] == [h['id'] for h in fallback]
            # 각 검색기의 1위 점수가 같아도 키워드 입력 순서로 먼저 선택하지 않습니다.
            with patch.object(semantic, 'vector_hits', return_value=[{'id': 'v1'}, {'id': 'v2'}]):
                fused = semantic.search_cases(db, '다른 표현', config, keyword_hits=[{'id': 'k1'}, {'id': 'k2'}])
                assert [hit['id'] for hit in fused] == ['v1', 'k1', 'v2']
                fused = semantic.search_cases(db, '다른 표현', config, keyword_hits=[{'id': 'k1'}, {'id': 'v2'}])
                assert fused[0]['id'] == 'v2', '두 검색기에 공통으로 나온 후보의 RRF 점수를 보존해야 합니다.'
            count = len(calls)
            # 수동 1/5/100건 설정이 결합·벡터·정확검색·장애 대체 경로까지 전달됩니다.
            many = [{'id': f'candidate-{i}'} for i in range(100)]
            with patch.object(semantic, 'retrieve', return_value=many) as lookup, \
                    patch.object(semantic, 'vector_hits', return_value=many) as vectors:
                for limit in (1, 5, 100):
                    for mode in ('keyword', 'vector', 'hybrid'):
                        result = semantic.search_cases(db, '압력 현상', config, limit=limit, mode=mode)
                        assert len(result) == limit
                        assert lookup.call_args.kwargs['limit'] == max(20, limit)
                    assert vectors.call_args.kwargs['limit'] == max(20, limit)
                    assert len(semantic.search_cases(db, 'CS-DEMO-001', config, limit=limit)) == limit
                vectors.side_effect = RuntimeError('offline')
                assert len(semantic.search_cases(db, '압력 현상', config, limit=5)) == 5
                for invalid in (0, 101, True, '3', 2.5):
                    try:
                        semantic.search_cases(db, '현상', config, limit=invalid)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('잘못된 참고 사례 수를 허용했습니다.')
            assert semantic.search_cases(db, 'CS-DEMO-001의 기록', config)[0]['record_id'] == 'CS-DEMO-001'
            assert not semantic.search_cases(db, 'CS-DEMO-999의 기록', config)
            assert len(calls) == count, '정확한 접수번호 조회에 임베딩을 요청했습니다.'
            # 채택한 증상만 벡터 입력에 추가하고 해당 접수의 이전 벡터를 즉시 제외합니다.
            entry = {'source_key': rows[0]['_source_key'], 'id': 'CS-DEMO-001', 'revision': records.row_revision(rows[0]),
                     'column': '현상', 'action': 'edit', 'value': rows[0]['현상'] + ' 압력이 내려가지 않음.'}
            storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
            assert semantic.vector_count(db, config) == len(rows) - 1
            assert 'CS-DEMO-001' not in semantic.load_index(db, config)[0]
            assert semantic.build_vectors(db, client, config)['generated'] == 1
            assert '압력이 내려가지' in calls[-1]['input'][0]
            other = {'embedding': config['embedding'] | {'model': 'new-model'}}
            assert semantic.vector_count(db, other) == 0
            details = {}
            semantic.search_cases(db, '진공 알람', other, details=details)
            assert details['vector_search']['warning']
            try:
                semantic.search_cases(db, '진공 알람', other, mode='vector')
            except ValueError:
                pass
            else:
                raise AssertionError('벡터 전용 검색의 실패를 숨겼습니다.')
        # 잘못된 응답은 저장하지 않고 다음 배치를 처리합니다. 다시 실행하면 실패분만 생성합니다.
        with patch.object(semantic, 'client_for', side_effect=RuntimeError('LM Studio 자동 시작 실패')):
            detail = {}
            assert semantic.search_cases(db, '진공 알람', config, details=detail)
            assert '키워드' in detail['vector_search']['warning']
        changed = {'embedding': config['embedding'] | {'revision': '2'}}
        bad = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float('nan'), 0, 0]) for i in range(len(kw['input']))])))
        result = semantic.build_vectors(db, bad, changed, limit=2)
        assert result['indexed'] == 0 and len(result['failures']) == 2
        assert semantic.build_vectors(db, client, changed, limit=2)['generated'] == 2
        assert semantic.build_vectors(db, client, changed)['indexed'] == len(rows)
        # 원문 삭제 이후 캐시에서도 이전 접수가 사라져야 합니다.
        storage.ingest(db, rows[1:], rules, schema=schema)
        assert 'CS-DEMO-001' not in semantic.load_index(db, config)[0]
        assert semantic.vector_count(db, config) == len(rows)-1
    # 실제 FAISS에서 동일한 원문 두 건 때문에 후보가 줄어들면 후순위를 보충합니다.
    with closing(sqlite3.connect(':memory:')) as db:
        shared = {key: '' for key in schema['columns']} | {'현상': '압력 저하', '조치내용(1)': '설정값 20으로 복원'}
        cases = [shared, shared, shared | {'현상': '압력 저하 센서 단선', '조치내용(1)': '센서 교체'}]
        duplicates = [v | {schema['id_column']: f'CS-VECTOR-{i}', '_id_column': schema['id_column']}
                      for i, v in enumerate(cases, 1)]
        storage.ingest(db, duplicates, rules, schema=schema)
        source_vectors = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[.9, .1, 0.] if '센서' in text else [1., 0., 0.])
            for i, text in enumerate(kw['input'])])))
        semantic.build_vectors(db, source_vectors, config)
        count = len(calls)
        result = semantic.vector_hits(db, '압력', config, limit=2, embedding_client=SimpleNamespace(
            embeddings=SimpleNamespace(create=lambda **kw: (calls.append(kw) or SimpleNamespace(data=[
                SimpleNamespace(index=0, embedding=[1., 0., 0.])])))))
        assert len(result) == 2 and result[1]['record_id'] == 'CS-VECTOR-3', [h['id'] for h in result]
        assert len(calls) == count + 1, '중복 보충을 위해 질문을 다시 임베딩했습니다.'
    for values in ([[0., 0.]], [[1., 0.], [1.]], [[float('inf'), 0.]]):
        bad = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=v) for i, v in enumerate(values)])))
        try:
            semantic.embed(bad, ['x']*len(values), semantic.settings(config))
        except ValueError:
            pass
        else:
            raise AssertionError('잘못된 임베딩을 허용했습니다.')
    print('PASS: FAISS 의미 검색·결합 검색·원문 인용·숫자 조건·증분 저장·검토 후 무효화')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
