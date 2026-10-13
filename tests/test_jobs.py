"""작업 상태 저장과 실제 프로세스 종료 후 실행 잠금 해제를 확인합니다."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil
import time
from unittest.mock import patch


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('ragkit.jobs'), '영속 작업 저장소가 없습니다.')
        self.jobs = importlib.import_module('ragkit.jobs')
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.store = self.jobs.JobStore(self.root)

    def test_reopen_and_completed_retention(self):
        identity = self.store.save('import_many', {'paths': ['a.xlsx']}, {'status': 'running'})
        reopened = self.jobs.JobStore(self.root)
        self.assertEqual(reopened.interrupted()['id'], identity)
        for _ in range(25):
            reopened.save('overnight', {}, {'status': 'done'})
        self.assertEqual(reopened.interrupted()['id'], identity)
        reopened.save('import_many', {}, {'status': 'done'}, identity)
        self.assertIsNone(reopened.interrupted())
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(self.root / 'jobs.sqlite3')) as db:
            self.assertEqual(db.execute('select count(*) from jobs').fetchone()[0], 20)

    def test_live_owner_and_real_process_death(self):
        code = '''
import sys, time
from pathlib import Path
from ragkit.jobs import JobStore
store = JobStore(Path(sys.argv[1]))
handle = store.acquire()
assert handle is not None
identity = store.save('overnight', {}, {'status':'running'})
print(identity, flush=True)
time.sleep(60)
'''
        child = subprocess.Popen([sys.executable, '-u', '-c', code, str(self.root)], stdout=subprocess.PIPE, text=True)
        try:
            identity = child.stdout.readline().strip()
            self.assertTrue(identity)
            self.assertIsNone(self.store.acquire())
            self.assertIsNone(self.store.interrupted(), '살아 있는 작업을 가져왔습니다.')
            child.kill()
            child.wait(timeout=5)
            self.assertEqual(self.store.interrupted()['id'], identity)
            handle = self.store.acquire()
            self.assertIsNotNone(handle)
            self.store.release(handle)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()

    def test_corrupt_store_is_not_replaced(self):
        path = self.root / 'jobs.sqlite3'
        path.write_bytes(b'broken database')
        with self.assertRaises(self.jobs.JobStorageError):
            self.store.latest()
        self.assertEqual(path.read_bytes(), b'broken database')

    def test_unfinished_sqlite_journal_recovers_committed_snapshot(self):
        import sqlite3
        from contextlib import closing
        identity = self.store.save('overnight', {}, {'status': 'running', 'pad': 'a' * 500000})
        crashed = self.root / 'crashed'
        crashed.mkdir()
        with closing(sqlite3.connect(self.store.path)) as db:
            db.execute('PRAGMA cache_size=5')
            db.execute('BEGIN IMMEDIATE')
            db.execute('UPDATE jobs SET snapshot=? WHERE id=?', (json.dumps({'status': 'running', 'pad': 'b' * 500000}), identity))
            journal = self.root / 'jobs.sqlite3-journal'
            self.assertEqual(journal.read_bytes()[:8], bytes.fromhex('d9d505f920a163d7'))
            shutil.copyfile(self.store.path, crashed / 'jobs.sqlite3')
            shutil.copyfile(journal, crashed / journal.name)
            db.rollback()
        restored = self.jobs.JobStore(crashed).latest()
        self.assertEqual(restored['snapshot']['pad'], 'a' * 500000)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        from ragkit import web
        self.web = web
        self.assertTrue(callable(getattr(web.App, 'recover_jobs', None)), 'App 재시작 복구가 없습니다.')
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(__file__).resolve().parents[1]
        if (self.root / 'windows-demo').is_dir():
            self.root /= 'windows-demo'
        self.folder_path = Path(self.folder.name).resolve()
        self.cfg = self.folder_path / 'config.json'
        self.cfg.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'SECRET',
            'model': 'test', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
        self.output = self.folder_path / 'output'

    def app(self):
        return self.web.App(self.root, self.cfg, self.output, self.root / 'cslist_schema.json')

    def finish(self, app):
        deadline = time.monotonic() + 15
        while app.busy() and time.monotonic() < deadline:
            time.sleep(.01)
        job = app.job_snapshot()
        self.assertEqual(job['status'], 'done', job)
        return job['result']

    def kill_import(self, *, after_commit=False):
        paths = [self.folder_path / name for name in ('first.xlsx', 'second.xlsx')]
        for path in paths:
            shutil.copyfile(self.root / 'cslist_sample.xlsx', path)
        code = '''
import sys, time
from pathlib import Path
from ragkit.web import App
root, cfg, output, first, second, mode = sys.argv[1:]
app = App(root, cfg, output, Path(root) / 'cslist_schema.json')
original = app.import_file
def work(path, reader):
    if mode == 'boundary' and path.name == 'second.xlsx':
        print('KILL_READY', flush=True)
        time.sleep(60)
    value = original(path, reader)
    if mode == 'commit':
        print('KILL_READY', flush=True)
        time.sleep(60)
    return value
app.import_file = work
app.import_many({'paths': [first, second], 'reader': 'openpyxl'})
while app.busy(): time.sleep(.01)
'''
        child = subprocess.Popen([sys.executable, '-u', '-c', code, str(self.root), str(self.cfg), str(self.output),
                                  *(str(p) for p in paths), 'commit' if after_commit else 'boundary'],
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'KILL_READY')
            other = self.app()
            self.assertFalse(other.recover_jobs(), '실행 중인 다른 프로세스의 작업을 회수했습니다.')
            with self.assertRaises(self.web.ApiError):
                other.import_many({'paths': [str(paths[0])]})
        finally:
            child.kill()
            child.wait(timeout=5)
            child.stdout.close()
        return paths

    def test_completed_file_can_be_deleted_before_restart(self):
        paths = self.kill_import()
        paths[0].unlink()
        app = self.app()
        self.assertTrue(app.recover_jobs())
        result = self.finish(app)
        self.assertEqual((result['completed'], result['succeeded']), (2, 2))
        self.assertTrue(all(item['status'] == 'done' for item in result['files']))
        self.assertIsNone(app.jobs.interrupted())

    def test_commit_before_journal_replays_idempotently(self):
        self.kill_import(after_commit=True)
        app = self.app()
        from contextlib import closing
        from ragkit import records, storage
        with closing(app.connect()) as db:
            before = list(db.execute('select id, original_json from normalized_records order by id'))
            row = json.loads(before[0][1])
            entry = {'source_key': row['_source_key'], 'id': records.record_id(row),
                     'revision': records.row_revision(row), 'column': '현상', 'action': 'original', 'value': ''}
            storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': [entry]}, app.rules, schema=app.schema)
            reviews = list(db.execute('select * from cell_reviews'))
        self.assertTrue(app.recover_jobs())
        result = self.finish(app)
        self.assertEqual(result['import']['added'], 0)
        with closing(app.connect()) as db:
            self.assertEqual(list(db.execute('select id, original_json from normalized_records order by id')), before)
            self.assertEqual(list(db.execute('select * from cell_reviews')), reviews)

    def test_overnight_restart_reuses_saved_normalization_without_excel(self):
        from types import SimpleNamespace
        path = self.folder_path / 'first.xlsx'
        shutil.copyfile(self.root / 'cslist_sample.xlsx', path)
        app = self.app()
        calls = []
        def create(**request):
            cells = json.loads(request['messages'][1]['content'])
            calls.append(cells)
            answer = {key: cells['원본 셀'][key] for key in cells['수정할 열']}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(answer)))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
        app.import_file(path, 'openpyxl')
        with patch.object(self.web, 'OpenAI', return_value=client):
            app.normalize_rows(app.stored_rows(), None, 'openpyxl')
        self.assertTrue(calls)
        calls.clear()
        app.jobs.save('overnight', {'path': str(path), 'reader': 'openpyxl'},
                      {'status': 'running', 'overnight': {'input': {'rows': 3}, 'normalize': {'rows': 3}}})
        path.unlink()
        resumed = self.app()
        with patch.object(self.web, 'OpenAI', return_value=client), patch('ragkit.overnight.build_english', return_value={'failures': []}):
            self.assertTrue(resumed.recover_jobs())
            result = self.finish(resumed)
        self.assertTrue(result['ready'])
        self.assertFalse(calls, '성공한 다듬기를 모델에 다시 보냈습니다.')

    def test_overnight_reuses_input_and_stop_survives_restart(self):
        app = self.app()
        saved = {'status': 'running', 'overnight': {'input': {'rows': 2}}, 'stop_requested': True}
        identity = app.jobs.save('overnight', {'path': str(self.folder_path / 'deleted.xlsx'), 'reader': 'openpyxl'}, saved)
        with patch('ragkit.overnight.run', return_value={'ready': True}) as run:
            self.assertTrue(app.recover_jobs())
            run.assert_not_called()
            self.assertEqual(app.job_snapshot()['recovery']['id'], identity)
            with self.assertRaises(self.web.ApiError):
                app.resume_job({'id': 'wrong'})
            app.resume_job({'id': identity})
            self.finish(app)
            self.assertIsNone(run.call_args.args[1], '등록 완료 Excel을 다시 요구합니다.')

    def test_dismiss_keeps_source_and_checks_identity(self):
        app = self.app()
        identity = app.jobs.save('overnight', {}, {'status': 'running', 'stop_requested': True})
        app.recover_jobs()
        with self.assertRaises(self.web.ApiError):
            app.dismiss_job({'id': 'stale'})
        app.dismiss_job({'id': identity})
        self.assertIsNone(app.jobs.interrupted())
        self.assertIsNone(app.job_snapshot()['recovery'])

    def test_failed_journal_write_prevents_untracked_work(self):
        from ragkit.jobs import JobStorageError
        app = self.app()
        with patch.object(app.jobs, 'save', side_effect=JobStorageError('저장 실패')), patch.object(app, 'import_file') as work:
            with self.assertRaises(JobStorageError):
                app.import_many({'paths': [str(self.root / 'cslist_sample.xlsx')]})
            work.assert_not_called()
        handle = app.jobs.acquire()
        self.assertIsNotNone(handle)
        app.jobs.release(handle)
        self.assertFalse(app.db_path.exists())

    def test_external_completion_refreshes_locked_ui(self):
        first, second = self.app(), self.app()
        with patch('ragkit.web.threading.Thread') as thread:
            first.import_many({'paths': [str(self.folder_path / 'missing.xlsx')]})
            second.recover_jobs()
            self.assertTrue(second.job_snapshot()['recovery']['active'])
            thread.call_args.kwargs['target']()
        job = second.job_snapshot()
        self.assertIsNone(job['recovery'])
        self.assertEqual(job['result']['failed'], 1)

    def test_completed_failures_remain_visible_during_startup_embedding(self):
        app = self.app()
        app.import_many({'paths': [str(self.folder_path / 'missing.xlsx')]})
        self.finish(app)
        reopened = self.app()
        reopened.recover_jobs()
        self.assertEqual(reopened.job_snapshot()['result']['failed'], 1)
        reopened.config['embedding'] = {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock'}
        with patch.object(reopened, 'update_vectors', return_value={'generated': 0}):
            reopened.start_embedding()
            self.finish(reopened)
        self.assertEqual(reopened.job_snapshot()['preparation']['result']['failed'], 1)

    def test_resume_survives_another_crash_before_worker_starts(self):
        app = self.app()
        identity = app.jobs.save('overnight', {'path': str(self.folder_path / 'deleted.xlsx')},
                                 {'status': 'running', 'overnight': {'input': {'rows': 2}}, 'stop_requested': True})
        # 시작 직후 실행 스레드로 넘어가기 전 저장한 복구 정보도 입력 완료를 유지해야 합니다.
        with patch('ragkit.overnight.run', return_value={'ready': True}), patch('ragkit.web.threading.Thread') as thread:
            app.resume_job({'id': identity})
            self.assertIn('input', app.jobs.latest()['snapshot']['overnight'])
            thread.call_args.kwargs['target']()

    def test_recovery_http_uses_session_origin_and_exact_id(self):
        from http.cookiejar import CookieJar
        from urllib.request import build_opener, HTTPCookieProcessor, Request
        from urllib.error import HTTPError
        import re
        import threading
        app = self.app()
        identity = app.jobs.save('overnight', {}, {'status': 'running', 'stop_requested': True})
        app.recover_jobs()
        server = self.web.ThreadingHTTPServer(('127.0.0.1', 0), self.web.make_handler(app))
        app.port = server.server_port
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{app.port}'
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        try:
            with opener.open(base) as response:
                token = re.search(r'name="csrf-token" content="([^"]+)"', response.read().decode())[1]
            for origin, body, expected in [('https://attacker.example', {'id': identity}, 403),
                                           (base, {'id': 'stale'}, 409), (base, {'id': identity, 'extra': 1}, 400),
                                           (base, {'id': identity}, 200)]:
                request = Request(base + '/api/job/dismiss', data=json.dumps(body).encode(),
                                  headers={'Origin': origin, 'X-CSRF-Token': token, 'Content-Type': 'application/json'})
                try:
                    response = opener.open(request, timeout=5)
                except HTTPError as error:
                    response = error
                with response:
                    self.assertEqual(response.status, expected, response.read())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


def main(root=None):
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    assert result.wasSuccessful(), '작업 복구 검사 실패'


if __name__ == '__main__':
    unittest.main()
