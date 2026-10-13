"""작은 실제 SQLite와 모의 Clef로 재정렬·실패 대체·원문/읽기 전용 경계를 확인합니다."""
import copy
from contextlib import closing, nullcontext
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from experiments.clef_rag import ReadOnlyApp
from ragkit import clef, cslist, storage, web
from ragkit.input_formats import HISTORY_COLUMNS


def runtime_checks():
    config = {'base_url': 'http://127.0.0.1:1234/v1', 'model': 'configured',
              'clef': clef.options({'manage_lm_studio': True})}
    idle = {'identifier': 'configured', 'type': 'llm', 'status': 'idle', 'queued': 0}
    other = idle | {'identifier': 'other', 'status': 'predicting', 'queued': 1}
    with patch.object(clef, 'wait_gpu'), patch.object(clef, '_lms', return_value='lms'), patch.object(clef, '_run', side_effect=[
            json.dumps([idle, other]), '', '', json.dumps([idle, other]), '']) as command:
        with clef.chat_model(config, lambda text: None):
            assert command.call_args_list[-1].args[1:] == (
                'load', 'configured', '--identifier', 'configured', '--context-length', '16384', '--parallel', '1')
        assert [call.args[1:] for call in command.call_args_list if call.args[1] == 'unload'] == [
            ('unload', 'configured'), ('unload', 'configured')]
    with patch.object(clef, '_lms', return_value='lms'), patch.object(clef, '_run', return_value=json.dumps([idle | {'queued': 1}])) as command:
        try:
            clef.release_chat(config, lambda text: None)
        except ValueError:
            pass
        else:
            raise AssertionError('사용 중인 답변 모델을 해제했습니다.')
        assert command.call_count == 1
    with patch.object(clef, 'wait_gpu'), patch.object(clef, '_lms', return_value='lms'), patch.object(clef, '_run', side_effect=[
            '[]', subprocess.TimeoutExpired('load', 180), json.dumps([idle]), '']) as command:
        try:
            with clef.chat_model(config, lambda text: None):
                raise AssertionError('적재 실패 이후 실행했습니다.')
        except subprocess.TimeoutExpired:
            pass
        assert command.call_args.args[1:] == ('unload', 'configured')
    with patch.object(clef, '_run', side_effect=AssertionError('관리하지 않는 서버를 조작했습니다.')):
        with clef.chat_model(config | {'clef': {}}, lambda text: None):
            pass
        try:
            clef.release_chat(config | {'base_url': 'http://remote.example:1234/v1'}, lambda text: None)
        except ValueError:
            pass
        else:
            raise AssertionError('원격 서버의 모델 관리를 허용했습니다.')
    with patch.object(clef, '_health', side_effect=[{'busy': True}, {'busy': False}]) as health, \
            patch.object(clef.time, 'sleep') as pause:
        clef.wait_gpu(config, lambda text: None)
        assert health.call_count == 2 and pause.call_count == 1
    with patch.object(clef, '_health', return_value={'busy': True}), \
            patch.object(clef.time, 'monotonic', side_effect=[0, 311]), \
            patch.object(clef, '_lms', return_value='lms'), patch.object(clef, '_run') as command:
        try:
            with clef.chat_model(config, lambda text: None):
                raise AssertionError('Clef 실행 중에 답변 모델을 적재했습니다.')
        except ValueError:
            pass
        command.assert_not_called()
    with patch.object(clef, '_health', side_effect=URLError('상태 확인 불가')), \
            patch.object(clef, '_lms', return_value='lms'), patch.object(clef, '_run') as command:
        try:
            with clef.chat_model(config, lambda text: None):
                raise AssertionError('GPU 상태를 모르는데 답변 모델을 적재했습니다.')
        except URLError:
            pass
        command.assert_not_called()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        python = path / 'python'
        python.touch()
        cfg = clef.options({'runtime_python': str(python), 'model_path': str(path)})
        health = {'model': 'clef:flash', 'backend': 'mlx', 'device': 'gpu', 'busy': False}
        def response(info):
            return io.BytesIO(json.dumps(info).encode())
        with patch.object(clef, 'urlopen', return_value=response(health)), patch.object(clef.subprocess, 'Popen') as start:
            clef.prepare(cfg, lambda text: None, path / 'log')
            start.assert_not_called()
        with patch.object(clef, 'urlopen', side_effect=[URLError(ConnectionRefusedError()), response(health)]), \
                patch.object(clef.subprocess, 'Popen') as start:
            clef.prepare(cfg, lambda text: None, path / 'log')
            start.assert_called_once()
            assert start.call_args.args[0][:4] == [str(python), '-u', '-m', 'experiments.clef_mlx_server']
        with patch.object(clef, 'urlopen', return_value=response(health | {'device': 'cpu'})), \
                patch.object(clef.subprocess, 'Popen') as start:
            try:
                clef.prepare(cfg, lambda text: None, path / 'log')
            except ValueError:
                pass
            else:
                raise AssertionError('다른 기존 서버를 GPU로 채택했습니다.')
            start.assert_not_called()
        with patch.object(clef, 'urlopen', side_effect=URLError(ConnectionRefusedError())), \
                patch.object(clef.subprocess, 'Popen') as start:
            try:
                clef.prepare(cfg | {'runtime_python': str(path / 'missing')}, lambda text: None, path / 'log')
            except ValueError:
                pass
            else:
                raise AssertionError('없는 실행 환경을 시작했습니다.')
            start.assert_not_called()


def main(root):
    runtime_checks()
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = [dict.fromkeys(HISTORY_COLUMNS, '') | {'CS NO': f'CS-CLEF-{i}', '현상': '진공 알람',
            '조치': text, '_id_column': 'CS NO'}
            for i, text in enumerate(('배기 튜브 교체 예정', '배기 튜브 교체 후 알람 없음'), 1)]
    with closing(sqlite3.connect(':memory:')) as db:
        storage.ingest(db, rows, rules, schema=schema)
        from ragkit.search import retrieve
        hits = retrieve(db, '진공 알람 과거 조치', limit=3)
        hits[0]['evidence'][0]['accepted'] = {'현상': '미채택 문장'}
        original = copy.deepcopy(hits)
        result = {'model': 'clef:flash', 'backend': 'mlx', 'device': 'gpu', 'state_truncated': False, 'done_reason': 'decide',
            'usage': {'input_tokens': 150, 'output_tokens': 0}, 'answers': {
                'ranking': {'type': 'choice', 'choice': 'case_1', 'probabilities': {'case_0': .1, 'case_1': .9}}}}
        cfg = {'candidates': 2}
        def response(value):
            return io.BytesIO(json.dumps(value).encode())
        with patch.object(clef, 'urlopen', return_value=response(result)) as request:
            details = {}
            ranked = clef.rerank(db, '진공 알람 과거 조치', hits, cfg, details=details)
            assert [h['id'] for h in ranked] == [hits[1]['id'], hits[0]['id']]
            assert hits == original and details['clef_search']['status'] == 'used'
            sent = json.loads(request.call_args.args[0].data)
            assert '미채택 문장' not in json.dumps(sent, ensure_ascii=False) and sent['keep_alive'] == 0
            assert {s['id'] for sections in sent['state']['cases'].values() for s in sections} == {
                s['id'] for h in hits for s in h['evidence']}
            assert web.pack_answer('질문', '', ranked, None, details, 0)['clef_search'] == details['clef_search']
        for invalid in (result | {'state_truncated': True}, result | {'model': 'other'}, result | {'device': 'cpu'},
                        result | {'usage': {'output_tokens': False}}, result | {'answers': {}},
                        result | {'answers': {'ranking': {'type': 'choice', 'choice': 'case_0',
                                                        'probabilities': ['case_0', 'case_1']}}},
                        result | {'answers': {'ranking': {
                            'type': 'choice', 'choice': 'case_0', 'probabilities': {'case_0': float('nan')}}}}):
            with patch.object(clef, 'urlopen', return_value=response(invalid)):
                details = {}
                assert clef.rerank(db, '진공 알람', hits, cfg, details=details) == hits
                assert details['clef_search']['status'] == 'error'
        with patch.object(clef, 'urlopen', side_effect=TimeoutError()):
            details = {}
            assert clef.rerank(db, '진공 알람', hits, cfg, details=details) == hits
            assert details['clef_search']['status'] == 'error'
        with patch.object(clef, 'urlopen', side_effect=AssertionError('정확 검색에 Clef를 호출했습니다.')):
            for q in ('CS NO=CS-CLEF-1', 'SIM-ETCH-001 진공 알람', '2026-09-01 진공 알람'):
                assert clef.rerank(db, q, hits, cfg, details={}) == hits
            assert clef.rerank(db, '진공 알람', hits[:1], cfg, details={}) == hits[:1]
    for value in ({'base_url': 'https://external.example'}, {'base_url': 'http://user:secret@localhost'},
                  {'candidates': True}, {'candidates': 21}, {'timeout_seconds': 0},
                  {'enabled': 'true'}, {'manage_lm_studio': 1}, {'runtime_python': '/missing'},
                  {'model_path': '/missing'}, {'runtime_python': '', 'model_path': ''}):
        try:
            clef.options(value)
        except ValueError:
            pass
        else:
            raise AssertionError('잘못된 Clef 설정을 허용했습니다.')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        with closing(sqlite3.connect(path / 'demo.sqlite3')) as db:
            storage.ingest(db, rows, rules, schema=schema)
        (path / 'config.json').write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY',
            'model': 'mock', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
        app = ReadOnlyApp(root, path / 'config.json', path / 'ui', root / 'cslist_schema.json')
        app.db_path = path / 'demo.sqlite3'
        with closing(app.connect()) as db:
            try:
                db.execute('DELETE FROM normalized_records')
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError('읽기 전용 DB 쓰기를 허용했습니다.')
            assert db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0] == 2
        with patch.object(web.App, 'guard', return_value='owner'):
            assert app.guard(SimpleNamespace(path='/api/ask'), True) == 'owner'
            try:
                app.guard(SimpleNamespace(path='/api/import'), True)
            except web.ApiError as error:
                assert error.status == 403
            else:
                raise AssertionError('자료 변경 경로를 허용했습니다.')
        assert app.update_settings({'reference_limit': 5})['reference_limit'] == 5
        try:
            app.update_settings({'auto_english': True})
        except web.ApiError as error:
            assert error.status == 403
        else:
            raise AssertionError('원문 처리 설정 변경을 허용했습니다.')
        app.lms, app.answer_source_model = 'lms', 'mock-source-model'
        with patch('experiments.clef_rag.subprocess.run', return_value=SimpleNamespace(returncode=0)) as command, \
                patch.object(web.App, 'llm_client', return_value=nullcontext(object())):
            try:
                with app.llm_client():
                    raise ValueError('답변 실패 표본')
            except ValueError:
                pass
            assert command.call_args_list[0].args[0][:3] == ['lms', 'load', 'mock-source-model']
            assert command.call_args_list[-1].args[0] == ['lms', 'unload', 'clef-rag-answer']
        # 본 시스템은 저장한 설정을 복원하고, 참고 수로 자르기 전에 후보를 재정렬합니다.
        config_path = path / 'main-config.json'
        payload = json.loads((path / 'config.json').read_text()) | {'clef': cfg}
        config_path.write_text(json.dumps(payload))
        main_app = web.App(root, config_path, path / 'main-ui', root / 'cslist_schema.json')
        main_app.db_path = path / 'demo.sqlite3'
        assert main_app.clef_enabled and main_app.public_config()['clef_configured']
        assert 'runtime_python' not in main_app.public_config()
        main_app.update_settings({'clef_enabled': False})
        assert not web.App(root, config_path, main_app.output, root / 'cslist_schema.json').clef_enabled
        main_app.config['clef']['manage_lm_studio'] = True
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **request: None)), close=lambda: None)
        with patch.object(web, 'OpenAI', return_value=fake), \
                patch.object(clef, '_run', side_effect=AssertionError('꺼진 Clef에서 답변 모델을 관리했습니다.')):
            with main_app.llm_client():
                pass
        main_app.config['clef']['manage_lm_studio'] = False
        with patch.object(main_app, 'save_settings', side_effect=OSError('저장 실패')):
            try:
                main_app.update_settings({'clef_enabled': True})
            except OSError:
                pass
            assert not main_app.clef_enabled
        for bad in ('true', 1, None):
            try:
                main_app.update_settings({'clef_enabled': bad})
            except web.ApiError as error:
                assert error.status == 400
            else:
                raise AssertionError('잘못된 Clef 사용 설정을 허용했습니다.')
        def compare():
            main_app.ask({'question': '진공 알람', 'reference_limit': 1}, comparison=True)
            deadline = time.monotonic() + 5
            while main_app.question_worker and time.monotonic() < deadline:
                time.sleep(.01)
            job = main_app.job_snapshot(main_app.token)
            assert job['status'] == 'done', job
            return job['result']
        with patch.object(web.semantic, 'search_cases', return_value=copy.deepcopy(original)), \
                patch.object(clef, 'urlopen', side_effect=AssertionError('꺼진 Clef를 호출했습니다.')):
            baseline = compare()
            assert baseline['clef_search'] is None
        main_app.update_settings({'clef_enabled': True})
        with patch.object(web.semantic, 'search_cases', return_value=copy.deepcopy(original)) as search, \
                patch.object(clef, 'urlopen', return_value=response(result)):
            final = compare()
            assert search.call_args.kwargs['limit'] == 2
            assert final['clef_search']['status'] == 'used'
            assert final['clef_search']['reranked_ids'] == [original[1]['id'], original[0]['id']]
            assert final['references']['selected'] == 1
        assert web.App(root, config_path, main_app.output, root / 'cslist_schema.json').clef_enabled
        with patch('experiments.clef_rag.subprocess.run', side_effect=[
                subprocess.TimeoutExpired('load', 180), SimpleNamespace(returncode=0)]) as command:
            try:
                with app.llm_client():
                    raise AssertionError('적재 실패 이후 답변 모델을 사용했습니다.')
            except subprocess.TimeoutExpired:
                pass
            assert command.call_args_list[-1].args[0] == ['lms', 'unload', 'clef-rag-answer']
    print('PASS: Clef 재정렬·오류 대체·정확 검색/원문 보존·본 시스템 설정/참고 수·자동 시작·모델 순차 실행·읽기 전용 경계')


if __name__ == '__main__':
    main(Path('windows-demo'))
