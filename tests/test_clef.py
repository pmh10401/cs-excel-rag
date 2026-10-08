"""작은 실제 SQLite와 모의 Clef로 재정렬·실패 대체·원문/읽기 전용 경계를 확인합니다."""
import copy
from contextlib import closing, nullcontext
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from experiments.clef_rag import ReadOnlyApp
from ragkit import clef, cslist, storage, web
from ragkit.input_formats import HISTORY_COLUMNS


def main(root):
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
                  {'candidates': True}, {'candidates': 21}, {'timeout_seconds': 0}):
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
        with patch('experiments.clef_rag.subprocess.run', side_effect=[
                subprocess.TimeoutExpired('load', 180), SimpleNamespace(returncode=0)]) as command:
            try:
                with app.llm_client():
                    raise AssertionError('적재 실패 이후 답변 모델을 사용했습니다.')
            except subprocess.TimeoutExpired:
                pass
            assert command.call_args_list[-1].args[0] == ['lms', 'unload', 'clef-rag-answer']
    print('PASS: Clef 재정렬·잘림/분포/시간 오류 대체·정확 검색 보존·원문 입력·읽기 전용 DB/HTTP 경계')


if __name__ == '__main__':
    main(Path('windows-demo'))
