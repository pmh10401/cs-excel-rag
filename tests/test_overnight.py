"""야간 실행을 실제 SQLite·다듬기·검토·검색 경로로 검증합니다. 모델 통신만 대체합니다."""
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
from types import SimpleNamespace
from unittest.mock import patch

from openai import APIConnectionError, AuthenticationError
from ragkit import web, semantic


def main(root):
    assert callable(getattr(web.App, 'start_overnight', None)), '야간 자동 재시도 진입점이 없습니다.'
    from ragkit import overnight
    check_cancellation(root)
    config = {'base_url':'http://127.0.0.1:1/v1','api_key':'LOCAL-SECRET','model':'test',
              'reasoning_effort':'high','timeout_seconds':1,'normalization_workers':1,
              'embedding':{'base_url':'http://127.0.0.1:1/v1','model':'test-embedding','batch_size':1}}
    calls = []
    first_normal = first_english = None
    def response(value, finish='stop'):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=json.dumps(value,ensure_ascii=False)))])
    def create(**request):
        nonlocal first_normal, first_english
        cells = json.loads(request['messages'][1]['content'])
        calls.append(cells)
        if '원본 셀' in cells:
            key = json.dumps(cells,ensure_ascii=False)
            if first_normal is None:
                first_normal = key
                return response({}, 'length')
            if key == first_normal:
                assert '이전 실행' in json.dumps(request['messages'],ensure_ascii=False)
            return response({k:cells['원본 셀'][k] for k in cells['수정할 열']})
        key = json.dumps(cells,ensure_ascii=False)
        if first_english is None:
            first_english = key
            raise APIConnectionError(request=SimpleNamespace())
        return response({k:v+' maintenance' for k,v in cells.items()})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=lambda:None)
    vector_calls = []
    def embed(**request):
        vector_calls.extend(request['input'])
        return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[1.,.1]) for i,_ in enumerate(request['input'])])
    vector_client = SimpleNamespace(embeddings=SimpleNamespace(create=embed),close=lambda:None)
    def finished(app):
        deadline=time.monotonic()+20
        while app.busy() and time.monotonic()<deadline:
            time.sleep(.01)
        assert not app.busy(),app.job_snapshot()
        return app.job_snapshot()
    with TemporaryDirectory() as folder:
        folder=Path(folder);cfg=folder/'config.json';cfg.write_text(json.dumps(config))
        app=web.App(root,cfg,folder/'output',root/'cslist_schema.json')
        selection={'path':str(root/'cslist_sample.xlsx'),'reader':'openpyxl'}
        with patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client), \
                patch.object(overnight,'RETRY_SECONDS',.01):
            app.start_overnight(selection)
            job=finished(app)
        assert job['status']=='done' and job['result']['ready'],job
        assert job['result']['attempts']['normalize']==2 and job['result']['attempts']['english']==2,job
        assert sum(json.dumps(c,ensure_ascii=False)==first_normal for c in calls)==2
        assert sum(json.dumps(c,ensure_ascii=False)==first_english for c in calls)==2
        with closing(app.connect()) as db:
            assert db.execute("SELECT COUNT(*) FROM cell_reviews WHERE action='accept'").fetchone()[0]>0
            assert db.execute('SELECT COUNT(*) FROM english_failures').fetchone()[0]==0
            assert semantic.vector_count(db,config)==db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0]
            assert db.execute('SELECT COUNT(*) FROM rag_english_documents').fetchone()[0]>0
        report=json.loads((app.output/'overnight_result.json').read_text())
        assert report['ready'] and 'LOCAL-SECRET' not in json.dumps(report)
        # 2회차 실행은 성공한 다듬기·번역·벡터를 모델에 다시 보내지 않습니다.
        calls.clear();vector_calls.clear()
        with patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client):
            app.start_overnight(selection);job=finished(app)
        assert job['result']['ready'] and not calls and not vector_calls,job
        # 재시도 대기 중 중지하면 다음 요청을 보내지 않고 검색 준비 완료와 구별합니다.
        with patch.object(app,'normalize_rows',side_effect=web.NormalizationBatchError([{'id':'x','error':'LOCAL-SECRET'}],1)):
            app.start_overnight(selection)
            deadline=time.monotonic()+5
            while not (app.job_snapshot().get('overnight') or {}).get('retry_at') and time.monotonic()<deadline:
                time.sleep(.01)
            app.stop_overnight({})
            job=finished(app)
        assert job['status']=='done' and job['result']['stopped'] and not job['result']['ready'],job
        # 인증 오류를 반복하거나 준비 완료로 표시하지 않습니다.
        auth=AuthenticationError('LOCAL-SECRET',response=SimpleNamespace(status_code=401,request=SimpleNamespace(),headers={}),body={})
        with patch.object(app,'normalize_rows',side_effect=auth) as request:
            app.start_overnight(selection);job=finished(app)
        assert job['status']=='error' and request.call_count==1,job
        assert not json.loads((app.output/'overnight_result.json').read_text())['ready']
    print('PASS: 야간 실패분 재시도·자동 채택·벡터/영어 준비·재실행 재사용·중지·인증 오류 구분')


def check_cancellation(root):
    from ragkit import storage, english, records, cslist
    schema,rules=cslist.load_schema(root/'cslist_schema.json')
    rows=records.read_excel(root/'cslist_sample.xlsx','CSList',required_columns=schema['columns'],id_column=schema['id_column'])[:2]
    config={'model':'test','reasoning_effort':'high','normalization_workers':1,
            'embedding':{'base_url':'http://test/v1','model':'test','batch_size':1}}
    stop=threading.Event()
    def create(**request):
        stop.set()
        cells=json.loads(request['messages'][1]['content'])
        value=({k:cells['원본 셀'][k] for k in cells['수정할 열']} if '원본 셀' in cells else
               {k:v+' maintenance' for k,v in cells.items()})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(value,ensure_ascii=False)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def cancelled(operation):
        try:operation()
        except storage.ProcessingCancelled:pass
        else:raise AssertionError('중지 요청을 무시했습니다.')
    import sqlite3
    with closing(sqlite3.connect(':memory:')) as db:
        cancelled(lambda:storage.ingest(db,rows,rules,client,config,schema=schema,cancelled=stop.is_set))
        assert db.execute('SELECT COUNT(*) FROM normalization_checkpoints').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0]==0
        storage.ingest(db,rows,rules,schema=schema)
        stop.clear()
        cancelled(lambda:english.build_english(db,client,config,schema,limit=0,cancelled=stop.is_set))
        assert db.execute('SELECT COUNT(*) FROM english_translations').fetchone()[0]==1
        assert english.english_count(db)==1
        def embed(**request):
            stop.set()
            return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[1.,.1]) for i in range(len(request['input']))])
        stop.clear()
        vector_client=SimpleNamespace(embeddings=SimpleNamespace(create=embed))
        cancelled(lambda:semantic.build_vectors(db,vector_client,config,cancelled=stop.is_set))
        assert semantic.vector_count(db,config)==1
    print('PASS: 진행 중 요청 저장 후 다듬기·영어·벡터 중지 및 다음 요청 차단')


if __name__=='__main__':
    main(Path(__file__).resolve().parents[1]/'windows-demo')
