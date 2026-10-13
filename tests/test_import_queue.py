"""여러 실제 임시 Excel/캐시를 순차 등록하고 충돌·실패·중복 경로를 검사합니다."""
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from unittest.mock import patch

from openpyxl import Workbook
from ragkit import records, source_cache, web
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_SHEET


def main(root):
    with TemporaryDirectory() as directory:
        folder = Path(directory).resolve()
        cfg = folder / 'config.json'
        cfg.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY',
            'model': 'mock', 'reasoning_effort': 'high', 'timeout_seconds': 1,
            'embedding': {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock'}}), encoding='utf-8')
        app = web.App(root, cfg, folder / 'output', root / 'cslist_schema.json')

        def write(name, values, *, history=False):
            path = folder / name
            book = Workbook()
            sheet = book.active
            sheet.title = HISTORY_SHEET if history else 'CSList'
            columns = HISTORY_COLUMNS if history else app.schema['columns']
            sheet.append(columns)
            for row in values:
                sheet.append([row.get(column) for column in columns])
            book.save(path)
            book.close()
            return path

        def finish():
            deadline = time.monotonic() + 10
            while app.busy() and time.monotonic() < deadline:
                time.sleep(.01)
            job = app.job_snapshot()
            assert job['status'] == 'done', job
            return job['result']

        row = {'CS 접수번호': 'QUEUE-1', '현상': '펌프 누수'}
        first = write('first.xlsx', [row])
        copy = write('copy.xlsx', [row])
        conflict = write('conflict.xlsx', [row | {'CS 접수번호': 'REJECT-NEW'}, row | {'현상': '내용 충돌'}])
        missing = folder / 'missing.xlsx'
        invalid = folder / 'invalid.xlsx'
        invalid.write_bytes(b'not an Excel file')
        history = write('history.xlsx', [{'CS NO': 'QUEUE-1', '현상': '冷却泵漏水'}], history=True)
        cache = folder / 'history.rag.jsonl.gz'
        source_cache.convert_excel(history, cache, 'CSList', required_columns=app.schema['columns'],
                                   id_column=app.schema['id_column'])
        last = write('last.xlsx', [row | {'CS 접수번호': 'QUEUE-2'}])
        paths = [first, copy, first, conflict, missing, invalid, history, cache, last]

        entered, release = threading.Event(), threading.Event()
        original_import = app.import_file
        def slow_import(path, reader):
            if path == first:
                entered.set()
                assert release.wait(5)
            return original_import(path, reader)

        with patch.object(app, 'import_file', side_effect=slow_import), \
                patch.object(app, 'update_vectors', return_value={'indexed': 3}) as vectors:
            app.import_many({'paths': [str(path) for path in paths], 'reader': 'openpyxl'})
            try:
                assert entered.wait(5)
                pending = app.job_snapshot()['progress']
                assert [item['status'] for item in pending['files']] == ['running'] + ['waiting'] * 8
                for action in (lambda: app.import_many({'paths': [str(last)]}),
                               lambda: app.import_input({'path': str(last)})):
                    try:
                        action()
                    except web.ApiError as error:
                        assert error.status == 409
                    else:
                        raise AssertionError('다른 쓰기 작업이 대기열 작업에 끼어들었습니다.')
            finally:
                release.set()
                result = finish()
            assert [item['status'] for item in pending['files']] == ['running'] + ['waiting'] * 8, '이전 스냅샷 변경'
            assert (result['completed'], result['succeeded'], result['failed'], result['skipped']) == (9, 5, 3, 1), result
            assert result['import'] == {'added': 3, 'updated': 0, 'unchanged': 2}
            assert '같은 접수번호' in result['files'][3]['error']
            assert result['files'][-1]['status'] == 'done'
            vectors.assert_called_once_with(input_changed=False)
        with closing(app.connect()) as db:
            originals = {identity: json.loads(raw) for identity, raw in db.execute('SELECT id,original_json FROM normalized_records')}
            assert set(originals) == {'QUEUE-1', records.record_id(app.read_rows(history, 'openpyxl')[0]), 'QUEUE-2'}
            assert originals['QUEUE-1']['_excel_location'].startswith(first.name + ' / ')

        # 오류만 다시 제출할 수 있으며 임베딩 실패도 등록 성공을 취소하지 않습니다.
        write('missing.xlsx', [row | {'CS 접수번호': 'QUEUE-3'}])
        with patch.object(app, 'update_vectors', side_effect=ValueError('시험용 임베딩 실패')):
            app.import_many({'paths': [str(missing)]})
            retried = finish()
            assert retried['succeeded'] == 1 and retried['embedding_warning'] == '시험용 임베딩 실패'
        assert app.db_summary()['records'] == 4
        with patch.object(app, 'update_vectors', side_effect=AssertionError('성공 파일 없이 임베딩 호출')):
            app.import_many({'paths': [str(invalid)]})
            assert finish()['failed'] == 1
        with patch.object(app, 'update_vectors', return_value={'indexed': 4}):
            app.import_many({'paths': [str(history), str(conflict)]})
            assert finish()['failed'] == 1
            assert app.input_sheet == HISTORY_SHEET and app.excel == history
        for body in ({}, {'paths': []}, {'paths': [str(first)] * 101}, {'paths': [None]},
                     {'paths': ['']}, {'paths': ['bad\x00.xlsx']}, {'paths': [str(first)], 'reader': 'bad'}):
            try:
                app.import_many(body)
            except web.ApiError as error:
                assert error.status == 400
            else:
                raise AssertionError('잘못된 대기열 요청 수락')
    print('PASS: 여러 파일 순차 등록·실패 후 계속·충돌 파일 롤백·중복 재사용·혼합 양식·대기 상태·작업 잠금·벡터 1회 갱신')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
