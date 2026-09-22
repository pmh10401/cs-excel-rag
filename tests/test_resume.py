"""LLM 처리 중 실패·재시작·설정 변경에도 검토 DB와 완료한 작업을 보존하는지 검사합니다."""
import copy
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from ragkit import cslist, records, storage


def main(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
                              required_columns=schema['columns'], id_column=schema['id_column'])[:2]
    config = {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock-model', 'reasoning_effort': 'high'}
    client = object()

    def snapshot(db):
        return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY 1').fetchall()
                for table in ('normalized_records', 'cell_reviews', 'rag_documents', 'rag_original_documents')}

    def failure_after_first(db, progress):
        calls = []

        def normalize(_client, _config, row, _rules, schema=None):
            calls.append(records.record_id(row))
            if len(calls) == 2:
                raise ValueError('의도한 두 번째 행 실패')
            return records.public_cells(row)

        with patch.object(storage, 'normalize', side_effect=normalize):
            try:
                storage.ingest(db, rows, rules, client, config, schema=schema, progress=progress.append)
            except ValueError as error:
                assert '의도한' in str(error)
            else:
                raise AssertionError('실패를 정상 완료로 처리했습니다.')
        assert calls == [records.record_id(row) for row in rows]
        assert progress[-1]['completed'] == 1 and progress[-1]['failed_id'] == records.record_id(rows[1])

    def resume(db, current_rows=rows, settings=config, current_rules=rules, current_schema=schema):
        calls, progress = [], []

        def normalize(_client, _config, row, _rules, schema=None):
            calls.append(records.record_id(row))
            return records.public_cells(row)

        with patch.object(storage, 'normalize', side_effect=normalize):
            prepared = storage.ingest(db, current_rows, current_rules, client, settings,
                                      schema=current_schema, progress=progress.append)
        assert progress[-1]['completed'] == progress[-1]['total'] == len(current_rows)
        return calls, prepared, progress

    # SQLite의 with는 트랜잭션만 끝냅니다. closing으로 연결도 닫아 Windows의 임시 DB 삭제를 허용합니다.
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        path = folder / 'resume.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            storage.ingest(db, rows, rules, schema=schema)
            column = '원인(1)'
            edited = rows[0][column] + ' 검토 표시.'
            storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [{
                'source_key': rows[0]['_source_key'], 'id': records.record_id(rows[0]),
                'revision': records.row_revision(rows[0]), 'column': column, 'action': 'edit', 'value': edited,
            }]}, rules, schema=schema)
            before = snapshot(db)
            failure_after_first(db, [])
            assert snapshot(db) == before, '실패한 중간 결과가 기존 검색 자료나 검토에 반영됐습니다.'
        # 연결을 새로 열어도 완료한 첫 행은 모델에 다시 보내지 않습니다.
        with closing(sqlite3.connect(path)) as db, db:
            calls, prepared, progress = resume(db)
            assert calls == [records.record_id(rows[1])]
            assert progress[-1]['reused'] == 1
            assert prepared[0]['effective'][column] == edited
            calls, _, progress = resume(db)
            assert calls == [] and progress[-1]['reused'] == 2

        # 캐시의 문맥이 달라졌다면 새로 처리해야 합니다. 이전 버전을 재사용하면 안 됩니다.
        for kind in ('text', 'typed', 'source', 'model', 'server', 'reasoning', 'sampling', 'rules', 'schema', 'prompt'):
            case_path = folder / (kind + '.sqlite3')
            with closing(sqlite3.connect(case_path)) as db, db:
                failure_after_first(db, [])
            current = copy.deepcopy(rows)
            settings, current_rules, current_schema = dict(config), dict(rules), copy.deepcopy(schema)
            if kind == 'text':
                current[0]['제목'] += ' 변경'
            elif kind == 'typed':
                current[0]['_typed_cells']['접수일시']['number_format'] = 'yyyy/mm/dd'
            elif kind == 'source':
                for row in current:
                    row['_source_key'] = 'different-workbook'
            elif kind == 'model':
                settings['model'] = 'another-model'
            elif kind == 'server':
                settings['base_url'] = 'http://127.0.0.1:2/v1'
            elif kind == 'reasoning':
                settings['reasoning_effort'] = 'medium'
            elif kind == 'sampling':
                settings.update(max_tokens=30000, temperature=1, top_p=1)
            elif kind == 'rules':
                current_rules['원인(1)'] = 'preserve'
            elif kind == 'schema':
                current_schema['name'] = 'updated schema'
            from ragkit import llm
            prompt = llm.NORMAL_PROMPT + ('\nchanged' if kind == 'prompt' else '')
            with closing(sqlite3.connect(case_path)) as db, db, patch.object(llm, 'NORMAL_PROMPT', prompt):
                calls, _, progress = resume(db, current, settings, current_rules, current_schema)
            assert len(calls) == 2 and progress[-1]['reused'] == 0, kind

        # 색인 갱신이 실패해도 공개된 데이터는 되돌리고, 중간 저장은 재사용합니다.
        with closing(sqlite3.connect(folder / 'promote.sqlite3')) as db, db:
            storage.ingest(db, rows, rules, schema=schema)
            before = snapshot(db)
            with patch.object(cslist, 'rebuild_search', side_effect=RuntimeError('색인 실패')):
                try:
                    resume(db)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('색인 실패를 숨겼습니다.')
            assert snapshot(db) == before
        with closing(sqlite3.connect(folder / 'promote.sqlite3')) as db, db:
            calls, _, progress = resume(db)
            assert calls == [] and progress[-1]['reused'] == 2
    print('PASS: 중단 후 재개·원문/검토/색인 보존·캐시 문맥 변경·공개 단계 롤백')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
