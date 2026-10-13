"""자료 갱신 중 질문의 원문 사본과 모델 실행 순서를 실제 스레드·SQLite로 확인합니다."""
from contextlib import closing
import copy
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ragkit import records, storage, web, semantic


def until(predicate):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('동시 작업이 제한 시간 안에 진행하지 않았습니다.')


class ConcurrentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        root = Path(__file__).resolve().parents[1] / 'windows-demo'
        if not root.exists():
            root = Path(__file__).resolve().parents[1]
        config = folder / 'config.json'
        config.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY',
                                     'model': 'test', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
        self.app = web.App(root, config, folder / 'out', root / 'cslist_schema.json')
        self.rows = records.read_excel(root / 'cslist_sample.xlsx', 'CSList',
            required_columns=self.app.schema['columns'], id_column=self.app.schema['id_column'])
        with closing(self.app.connect()) as db:
            storage.ingest(db, self.rows, self.app.rules, schema=self.app.schema)

    def test_question_keeps_original_while_writer_commits_and_job_is_preserved(self):
        app = self.app
        entered, release, write, written = (threading.Event() for _ in range(4))
        old = []

        def prepare():
            self.assertTrue(write.wait(10))
            changed = copy.deepcopy(self.rows)
            for row in changed:
                row['조치내용(1)'] = '새로운 조치 기록'
            with closing(app.connect()) as db:
                storage.ingest(db, changed, app.rules, schema=app.schema)
            app.set_stage('자료 저장 완료')
            app.checkpoint_job()
            written.set()
            return {'succeeded': 1}

        def answer(client, config, question, hits, *, details):
            section = next(s for hit in hits for s in hit['evidence'] if s['cells'].get('조치내용(1)'))
            old.append(section['cells']['조치내용(1)'])
            entered.set()
            self.assertTrue(release.wait(10))
            self.assertEqual(section['cells']['조치내용(1)'], old[0])
            details['claims'] = [{'text': old[0], 'evidence': [
                {'id': section['id'], 'column': '조치내용(1)', 'quote': old[0]}]}]
            return old[0], ''

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)), close=lambda: None)
        try:
            with patch.object(web, 'OpenAI', return_value=fake), patch.object(web, 'answer', side_effect=answer):
                app.start_job('import_many', prepare, request={'files': [], 'reader': 'openpyxl'})
                app.ask({'question': '펌프 누수 조치'}, owner=app.token)
                self.assertTrue(entered.wait(5))
                during = app.job_snapshot(app.token)
                self.assertTrue(during['maintenance_running'])
                self.assertEqual(during['preparation']['kind'], 'import_many')
                write.set()
                self.assertTrue(written.wait(5), '모델 대기 중 DB 쓰기를 막았습니다.')
                until(lambda: app.job['status'] == 'done')
                saved = app.jobs.latest()
                self.assertEqual(saved['kind'], 'import_many')
                self.assertNotIn('question', saved['snapshot'])
                release.set()
                until(lambda: not app.busy())
                done = app.job_snapshot(app.token)
                self.assertEqual(done['kind'], 'ask')
                self.assertEqual(done['result']['claims'][0]['evidence'][0]['quote'], old[0])
                self.assertNotEqual(old[0], '새로운 조치 기록')
                self.assertEqual(done['preparation']['status'], 'done')
        finally:
            write.set()
            release.set()
            until(lambda: not app.busy())

    def test_question_gets_next_model_turn_without_losing_background_trace_owner(self):
        app = self.app
        entered, release = threading.Event(), threading.Event()
        order = []
        def create(**request):
            label = request['messages'][0]['content']
            order.append(label)
            if label == 'background-1':
                entered.set()
                self.assertTrue(release.wait(10))
            return {'ok': True}
        fake = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
        def prepare():
            with app.llm_client() as client:
                for i in (1, 2):
                    client.chat.completions.create(messages=[{'role': 'user', 'content': f'background-{i}'}])
            return {}
        def question():
            app.set_stage('내 질문 단계')
            with app.llm_client() as client:
                client.chat.completions.create(messages=[{'role': 'user', 'content': 'question'}])
            return {}
        try:
            with patch.object(web, 'OpenAI', side_effect=fake):
                app.start_job('normalize', prepare)
                self.assertTrue(entered.wait(5))
                app.queue_question(app.token, question)
                until(lambda: app.job_snapshot(app.token)['status'] == 'running')
                release.set()
                until(lambda: not app.busy())
            self.assertEqual(order, ['background-1', 'question', 'background-2'])
            lines = [json.loads(line) for line in app.llm_trace.path.read_text().splitlines()]
            kinds = [(item['kind'], item.get('owner')) for item in lines]
            self.assertEqual(sum(kind == 'ask' and bool(owner) for kind, owner in kinds), 1)
            self.assertEqual(sum(kind == 'normalize' and owner is None for kind, owner in kinds), 2)
        finally:
            release.set()
            until(lambda: not app.busy())

    def test_all_retrieval_reads_share_a_read_only_snapshot(self):
        app = self.app
        selected, release = threading.Event(), threading.Event()
        from ragkit.search import retrieve
        with closing(app.connect()) as db:
            db.execute('PRAGMA journal_mode=WAL')
        observed = []
        def lookup(db, question, **options):
            before = db.execute('SELECT search_text FROM normalized_records ORDER BY id').fetchall()
            selected.set()
            self.assertTrue(release.wait(10))
            after = db.execute('SELECT search_text FROM normalized_records ORDER BY id').fetchall()
            observed.append((db.in_transaction, db.execute('PRAGMA query_only').fetchone()[0], before == after))
            return retrieve(db, question, **options)
        try:
            with patch.object(web, 'retrieve', side_effect=lookup):
                app.ask({'question': '없는현상가나다'})
                self.assertTrue(selected.wait(5))
                with closing(app.connect()) as db, db:
                    db.execute("UPDATE normalized_records SET search_text='다른 버전'")
                release.set()
                until(lambda: not app.busy())
            self.assertEqual(observed, [(True, 1, True)])
        finally:
            release.set()
            until(lambda: not app.busy())

    def test_preparation_can_start_after_question_and_does_not_hide_its_result(self):
        app = self.app
        entered, release = threading.Event(), threading.Event()
        def question():
            entered.set()
            self.assertTrue(release.wait(10))
            return {'answer': '이전 근거로 만든 답변'}
        try:
            app.queue_question(app.token, question)
            self.assertTrue(entered.wait(5))
            app.start_job('vectors', lambda: {'generated': 2})
            until(lambda: app.job['status'] == 'done')
            release.set()
            until(lambda: not app.busy())
            status = app.job_snapshot(app.token)
            self.assertEqual(status['kind'], 'ask')
            self.assertEqual(status['result']['answer'], '이전 근거로 만든 답변')
            self.assertEqual(status['maintenance']['result']['generated'], 2)
        finally:
            release.set()
            until(lambda: not app.busy())

    def test_translation_embedding_and_answer_do_not_hold_a_db_snapshot(self):
        app = self.app
        app.config['embedding'] = {'base_url': 'http://127.0.0.1:1/v1', 'model': 'fixture',
                                   'auto_start': False, 'min_score': -1}
        def vector(**request):
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1., 0.])
                                         for i, _ in enumerate(request['input'])])
        with closing(app.connect()) as db:
            semantic.build_vectors(db, SimpleNamespace(embeddings=SimpleNamespace(create=vector)), app.config)
        opened, calls = [], []
        connect = app.connect
        def tracked_connect():
            db = connect()
            opened.append(db)
            return db
        def no_snapshot(stage):
            calls.append(stage)
            for db in opened:
                try:
                    active = db.in_transaction
                except Exception as error:
                    self.assertIn('closed', str(error))
                else:
                    self.assertFalse(active, stage + ' 호출 중 읽기 트랜잭션이 열려 있습니다.')
        def translate(**request):
            no_snapshot('translation')
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content=json.dumps({'query': 'pump leakage action'})))])
        def embedding(**request):
            no_snapshot('embedding')
            return vector(**request)
        def answer(client, config, question, hits, *, details):
            no_snapshot('answer')
            self.assertTrue(hits)
            self.assertEqual(details['search']['english_query'], 'pump leakage action')
            return '확인', ''
        client = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=translate)), close=lambda: None)
        emb = SimpleNamespace(embeddings=SimpleNamespace(create=embedding), close=lambda: None)
        with patch.object(app, 'connect', side_effect=tracked_connect), patch.object(web, 'OpenAI', side_effect=client), \
                patch.object(semantic, 'client_for', return_value=emb), patch.object(web, 'answer', side_effect=answer):
            app.ask({'question': '펌프 누수 조치', 'bilingual': True, 'search_mode': 'hybrid'})
            until(lambda: not app.busy())
        result = app.job_snapshot(app.token)
        self.assertEqual(result['status'], 'done', result)
        self.assertEqual(calls, ['translation', 'embedding', 'answer'])
        self.assertEqual(len(result['result']['evidence_snapshot']['sha256']), 64)


class ModelTurnTests(unittest.TestCase):
    def test_parallel_background_requests_drain_before_question(self):
        from ragkit.workload import ModelTurns
        turns = ModelTurns()
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]
        asked = threading.Event()
        def background(i):
            with turns.background():
                entered[i].set()
                release[i].wait(10)
            finished[i].set()
        def question():
            with turns.question():
                asked.set()
        workers = [threading.Thread(target=background, args=(i,)) for i in range(2)]
        workers.append(threading.Thread(target=question))
        try:
            for worker in workers[:2]:
                worker.start()
            self.assertTrue(all(event.wait(5) for event in entered), '기존 병렬 처리를 직렬로 바꿨습니다.')
            turns.enqueue_question()
            workers[2].start()
            release[0].set()
            self.assertTrue(finished[0].wait(5))
            self.assertFalse(asked.is_set(), '다른 자료 모델 요청이 아직 실행 중입니다.')
            release[1].set()
            self.assertTrue(asked.wait(5))
        finally:
            for event in release:
                event.set()
            for worker in workers:
                if worker.ident:
                    worker.join(10)

    def test_background_gets_a_turn_between_questions_and_exception_releases_it(self):
        from ragkit.workload import ModelTurns
        turns = ModelTurns()
        order = []
        background_entered, background_release = threading.Event(), threading.Event()
        def background():
            with turns.background():
                order.append('background')
                background_entered.set()
                background_release.wait(10)
        def question():
            try:
                with turns.question():
                    order.append('question')
                    raise ValueError('의도한 모델 오류')
            except ValueError:
                pass
        turns.enqueue_question()
        workers = [threading.Thread(target=background), threading.Thread(target=question)]
        try:
            with turns.question():
                workers[0].start()
                until(lambda: turns.background_waiting == 1)
                turns.enqueue_question()
                workers[1].start()
            self.assertTrue(background_entered.wait(5))
            self.assertEqual(order, ['background'])
            background_release.set()
            for worker in workers:
                worker.join(5)
                self.assertFalse(worker.is_alive())
            with turns.background(exclusive=True):
                order.append('after-error')
            self.assertEqual(order, ['background', 'question', 'after-error'])
        finally:
            background_release.set()
            for worker in workers:
                if worker.ident:
                    worker.join(10)


def main(root=None):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (ConcurrentTests, ModelTurnTests))
    result = unittest.TextTestRunner().run(suite)
    assert result.wasSuccessful()
    node = shutil.which('node')
    if node:
        subprocess.run([node, str(Path(__file__).with_name('test_concurrent_ui.cjs'))], check=True)
    else:
        print('SKIP: Node가 없어 선택 화면 완료 처리 검사는 생략했습니다.')


if __name__ == '__main__':
    main()
