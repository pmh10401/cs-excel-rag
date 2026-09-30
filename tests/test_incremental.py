"""누적 등록, 기존 파생 자료 보존, Excel 삭제 후 DB 단독 처리를 실제 SQLite로 검사합니다."""
from contextlib import closing
import copy
import json
import re
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook
from ragkit import cli, cslist, english, records, semantic, storage, web


def main(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    config = {'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY', 'model': 'mock',
              'timeout_seconds': 1, 'reasoning_effort': 'high', 'normalization_workers': 1,
              'embedding': {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock-vector', 'auto_index': False}}
    requests = []
    def embed(**kwargs):
        requests.extend(kwargs['input'])
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1., .1]) for i, _ in enumerate(kwargs['input'])])
    client = SimpleNamespace(embeddings=SimpleNamespace(create=embed),
                             chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)), close=lambda: None)

    def normalized(_client, _config, row, _rules, schema=None, previous_error=''):
        return records.public_cells(row)

    def translated(_client, _config, cells, previous=''):
        return {key: re.sub('[가-힣]+', ' maintenance ', value) for key, value in cells.items()}

    def finished(app, status='done'):
        deadline = time.monotonic() + 10
        while app.busy() and time.monotonic() < deadline:
            time.sleep(.01)
        result = app.job_snapshot()
        assert result['status'] == status, result
        return result['result'] if status == 'done' else result

    def snapshot(db):
        return {name: db.execute('SELECT * FROM ' + name + ' ORDER BY 1').fetchall() for name in (
            'normalized_records', 'cell_reviews', 'rag_documents', 'rag_original_documents', 'rag_document_ids',
            'english_translations', 'rag_english_documents', 'vector_sources', 'vector_embeddings', 'vector_meta')}

    with TemporaryDirectory() as directory:
        folder = Path(directory)
        cfg = folder / 'config.json'
        cfg.write_text(json.dumps(config), encoding='utf-8')
        app = web.App(root, cfg, folder / 'output', root / 'cslist_schema.json')
        paths = [folder / 'first.xlsx', folder / 'second.xlsx']
        for i, path in enumerate(paths, 1):
            book = Workbook()
            sheet = book.active
            sheet.title = 'CSList'
            sheet.append(schema['columns'])
            values = {schema['id_column']: f'CS-ADD-{i:03}', '현상': '펌프 누수' if i == 1 else '팬 진동',
                      '원인(1)': '부품 마모', '조치내용(1)': '부품 교체 후 정상화'}
            sheet.append([values.get(key) for key in schema['columns']])
            book.save(path)
            book.close()
        first, second = (app.read_rows(path, 'openpyxl') for path in paths)
        with patch.object(cli, 'OpenAI', side_effect=AssertionError('원문 등록에서 모델 호출')):
            cli.main(root, ['--schema', str(root / 'cslist_schema.json'), '--config', str(folder / 'missing.json'),
                           '--output', str(folder / 'cli-import'), '--excel', str(paths[0]), '--import-only'])
        with closing(sqlite3.connect(folder / 'cli-import' / 'demo.sqlite3')) as db:
            assert db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0] == 1
        app.import_input({'path': str(paths[0])})
        assert finished(app)['import'] == {'added': 1, 'updated': 0, 'unchanged': 0, 'conflicts': [], 'total': 1}
        paths[0].unlink()  # 사용자 원본이 아닌 이번 검사에서 만든 임시 Excel만 삭제합니다.
        with patch.object(app, 'read_rows', side_effect=AssertionError('등록 후 Excel을 읽었습니다.')), \
                patch.object(web, 'OpenAI', return_value=client), patch.object(storage, 'normalize', side_effect=normalized), \
                patch.object(english, 'translate_cells', side_effect=translated), patch.object(semantic, 'client_for', return_value=client):
            app.normalize({})
            finished(app)
            app.accept_pending({})
            finished(app)
            app.translate_english({'limit': 0})
            assert finished(app)['indexed'] == 1
            app.vectors({'limit': 0})
            assert finished(app)['indexed'] == 1
            app.ask({'question': '전체 접수 건수는?'})
            assert finished(app)['aggregate']['count'] == 1
            app.start_overnight({})
            assert finished(app)['ready']
        with closing(app.connect()) as db:
            before = snapshot(db)
            requests.clear()
            summary = {}
            storage.ingest(db, second, rules, schema=schema, append=True, import_result=summary)
            assert summary['added'] == 1 and summary['total'] == 2
            for name, previous in before.items():
                if name != 'vector_meta':
                    assert all(row in snapshot(db)[name] for row in previous), name
            assert semantic.vector_count(db, config) == english.english_count(db) == 1
            after = snapshot(db)
            # 다른 경로의 복사본도 내용이 같으면 원래 출처·검토·제안·벡터를 보존합니다.
            copied = copy.deepcopy(first)
            copied[0]['_source_key'] = 'copied-workbook'
            storage.ingest(db, copied, rules, schema=schema, append=True)
            assert snapshot(db) == after
            with patch.object(storage, 'normalize', side_effect=AssertionError('동일 자료 재요청')):
                storage.ingest(db, copied, rules, client, config, schema=schema, append=True)
            assert snapshot(db) == after
            assert semantic.build_vectors(db, client, config)['generated'] == 1 and len(requests) == 1
            with patch.object(english, 'translate_cells', side_effect=translated):
                assert english.build_english(db, client, config, schema, limit=0)['processed'] == 1
            before = snapshot(db)
            # 검토 후 파생 자료 동기화가 실패해도 원문·검토·양쪽 색인·벡터가 함께 복구됩니다.
            entry = {'source_key': first[0]['_source_key'], 'id': records.record_id(first[0]),
                     'revision': records.row_revision(first[0]), 'column': '현상', 'action': 'edit', 'value': '펌프 누수가 발생했습니다.'}
            with patch.object(semantic, 'sync_sources', side_effect=RuntimeError('의도한 벡터 동기화 실패')):
                try:
                    storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, rules, schema=schema)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('검토 동기화 실패를 숨겼습니다.')
            assert snapshot(db) == before, '검토 실패로 기존 검색 자료가 달라졌습니다.'
            conflict = copy.deepcopy(copied)
            conflict[0]['현상'] = '다른 원문'
            try:
                storage.ingest(db, second + conflict, rules, schema=schema, append=True)
            except ValueError as error:
                assert '같은 접수번호' in str(error)
            else:
                raise AssertionError('다른 파일의 충돌을 덮어썼습니다.')
            assert snapshot(db) == before
            changed = copy.deepcopy(first)
            changed[0]['현상'] = '펌프 진동'
            with patch.object(cslist, 'rebuild_search', side_effect=RuntimeError('색인 저장 실패')):
                try:
                    storage.ingest(db, changed, rules, schema=schema, append=True)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('저장 오류를 무시했습니다.')
            assert snapshot(db) == before, '등록과 색인이 함께 롤백되지 않았습니다.'
            storage.ingest(db, changed, rules, schema=schema, append=True, import_result=summary)
            assert summary['updated'] == 1 and summary['total'] == 2
            assert semantic.vector_count(db, config) == english.english_count(db) == 1
            saved = storage.prepared_record(db, changed[0], None, rules)
            assert saved['reviews']['현상']['stale']
            assert db.execute('SELECT COUNT(*) FROM rag_documents').fetchone()[0] == 2
            assert db.execute('SELECT COUNT(*) FROM rag_document_ids').fetchone()[0] == 2
            assert list(english.translation_documents(db, ['CS-ADD-002']))
            extra = copy.deepcopy(second)
            extra[0][schema['id_column']] = 'CS-ADD-003'
            def concurrent_write(info):
                if info['completed'] == 1:
                    with closing(sqlite3.connect(app.db_path)) as other, other:
                        other.execute("UPDATE vector_meta SET revision=revision || '-other' WHERE id=1")
            try:
                storage.ingest(db, extra, rules, schema=schema, append=True, progress=concurrent_write)
            except ValueError as error:
                assert '다른 연결' in str(error)
            else:
                raise AssertionError('다른 연결의 변경을 무시하고 저장했습니다.')
            assert db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0] == 2
        # 실제 UI 경로로 재등록해도 중복이 생기지 않으며 마지막 Excel까지 삭제해도 이어집니다.
        app.import_input({'path': str(paths[1])})
        assert finished(app)['import']['unchanged'] == 1
        paths[1].unlink()
        with patch.object(app, 'read_rows', side_effect=AssertionError('삭제된 파일 재읽기')), \
                patch.object(web, 'OpenAI', return_value=client), patch.object(storage, 'normalize', side_effect=normalized):
            def partly_failed(client, config, row, rules, schema=None):
                if records.record_id(row) == 'CS-ADD-002':
                    raise ValueError('의도한 두 번째 접수 실패')
                return records.public_cells(row)
            with patch.object(storage, 'normalize', side_effect=partly_failed):
                app.normalize({})
                assert finished(app, 'error')['progress']['completed'] == 1
            assert app.db_summary()['records'] == 2, '다듬기 실패로 등록 원문을 잃었습니다.'
            app.normalize({})
            result = finished(app)
            assert result['rows'] == 2 and result['reused'] == 1
            app.ask({'question': '전체 접수 건수는?'})
            assert finished(app)['aggregate']['count'] == 2
        # 콘솔 --saved 역시 파일을 재조회하지 않고 같은 DB를 사용합니다.
        with patch.object(cli, 'read_excel', side_effect=AssertionError('CLI가 Excel을 읽었습니다.')):
            cli.main(root, ['--schema', str(root / 'cslist_schema.json'), '--config', str(cfg),
                           '--output', str(app.output), '--saved', '--question', '전체 접수 건수는?'])
        # 재다듬기는 제안이 깨졌더라도 저장된 원문으로 복구할 수 있어야 합니다.
        with closing(app.connect()) as db, db:
            db.execute("UPDATE normalized_records SET normalized_json='{broken' WHERE id='CS-ADD-001'")
        with patch.object(cli, 'read_excel', side_effect=AssertionError('CLI가 Excel을 읽었습니다.')), \
                patch.object(cli, 'OpenAI', return_value=client), patch.object(storage, 'normalize', side_effect=normalized):
            cli.main(root, ['--schema', str(root / 'cslist_schema.json'), '--config', str(cfg),
                           '--output', str(app.output), '--saved', '--normalize-only'])
        with closing(app.connect()) as db:
            assert isinstance(json.loads(db.execute("SELECT normalized_json FROM normalized_records WHERE id='CS-ADD-001'").fetchone()[0]), dict)
    print('PASS: 누적 추가·중복 재사용·출처 충돌 차단·변경분 색인·트랜잭션 롤백·Excel 삭제 후 UI/CLI/야간 처리')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
