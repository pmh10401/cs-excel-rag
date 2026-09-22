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
        with patch.object(app,'normalize_rows',side_effect=web.NormalizationBatchError([{'id':'x','error':'LOCAL-SECRET','status_code':401}],1)):
            app.start_overnight(selection)
            deadline=time.monotonic()+5
            while not (app.job_snapshot().get('overnight') or {}).get('retry_at') and time.monotonic()<deadline:
                time.sleep(.01)
            app.stop_overnight({})
            job=finished(app)
        assert job['status']=='done' and job['result']['stopped'] and not job['result']['ready'],job
        # 인증 오류도 제한 없이 재시도하고, 서버가 복구된 뒤에만 준비 완료로 표시합니다.
        auth=AuthenticationError('LOCAL-SECRET',response=SimpleNamespace(status_code=401,request=SimpleNamespace(),headers={}),body={})
        original_normalize = app.normalize_rows
        auth_attempts = []
        def recover_auth(*args, **kwargs):
            auth_attempts.append(1)
            if len(auth_attempts) < 3:
                raise auth
            return original_normalize(*args, **kwargs)
        with patch.object(app,'normalize_rows',side_effect=recover_auth), \
                patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client), \
                patch.object(overnight,'RETRY_SECONDS',.01):
            app.start_overnight(selection);job=finished(app)
        assert job['result']['ready'] and len(auth_attempts)==3,job
        assert 'LOCAL-SECRET' not in (app.output/'overnight_result.json').read_text()

        # 자료 읽기·채택·벡터·번역·최종 색인의 예외도 재시도합니다. 완료 단계는 다시 실행하지 않습니다.
        def fail_once(operation, error):
            attempts = []
            def call(*args, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise error
                return operation(*args, **kwargs)
            return call
        with patch.object(app,'stored_rows',side_effect=fail_once(app.stored_rows,OSError('자료 읽기 오류'))), \
                patch.object(overnight,'accept_pending_reviews',side_effect=fail_once(overnight.accept_pending_reviews,ValueError('채택 저장 오류'))), \
                patch.object(semantic,'build_vectors',side_effect=fail_once(semantic.build_vectors,ValueError('벡터 응답 오류'))), \
                patch.object(overnight,'build_english',side_effect=fail_once(overnight.build_english,RuntimeError('영어 처리 오류'))), \
                patch.object(overnight,'ensure_search_index',side_effect=fail_once(overnight.ensure_search_index,OSError('검색 색인 오류'))), \
                patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client), \
                patch.object(overnight,'RETRY_SECONDS',.01):
            app.start_overnight({});job=finished(app)
        assert job['result']['ready'] and job['result']['attempts']=={
            'input':2,'normalize':1,'review':2,'embedding':2,'english':2,'index':2},job

        # 자동 채택에서 거절한 제안은 기존 캐시를 맹신하지 않고 그 접수만 LLM으로 보완합니다.
        original_review = overnight.accept_pending_reviews
        review_calls = []
        review_target = []
        def corrupt_then_review(db, *args, **kwargs):
            review_calls.append(1)
            if len(review_calls) == 1:
                identity = db.execute('SELECT id FROM normalized_records ORDER BY id LIMIT 1').fetchone()[0]
                review_target.append(identity)
                db.execute("UPDATE normalized_records SET normalized_json='{}' WHERE id=?",(identity,))
                db.execute('DELETE FROM cell_reviews WHERE record_id=?',(identity,))
                db.commit()
            return original_review(db, *args, **kwargs)
        calls.clear()
        with patch.object(overnight,'accept_pending_reviews',side_effect=corrupt_then_review), \
                patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client), \
                patch.object(overnight,'RETRY_SECONDS',.01):
            app.start_overnight({});job=finished(app)
        assert job['result']['ready'] and len(review_calls)==2 and job['result']['attempts']['normalize']==2,job
        normal_calls = [c for c in calls if '원본 셀' in c]
        assert normal_calls and all(c['원본 셀']['CS 접수번호']==review_target[0] for c in normal_calls), '성공 접수도 재요청했습니다.'

        # 검토 전 발견한 손상 제안도 다시 생성합니다. 정상 캐시는 모델을 호출하지 않습니다.
        with closing(app.connect()) as db, db:
            db.execute("UPDATE normalized_records SET normalized_json='{' WHERE id=?", (review_target[0],))
        calls.clear()
        with patch.object(web,'OpenAI',return_value=client), patch.object(semantic,'client_for',return_value=vector_client):
            app.start_overnight({});job=finished(app)
        assert job['result']['ready'] and job['result']['attempts']['normalize']==1,job
        normal_calls = [c for c in calls if '원본 셀' in c]
        assert normal_calls and all(c['원본 셀']['CS 접수번호']==review_target[0] for c in normal_calls)

        # 보고 파일을 쓸 수 없어도 DB 작업과 메모리 진행 상태를 유지합니다.
        write_text = Path.write_text
        def no_report(path, *args, **kwargs):
            if path.name == 'overnight_result.tmp':
                raise PermissionError('시험용 상태 파일 잠금')
            return write_text(path,*args,**kwargs)
        with patch.object(Path,'write_text',no_report), patch.object(web,'OpenAI',return_value=client), \
                patch.object(semantic,'client_for',return_value=vector_client):
            app.start_overnight({});job=finished(app)
        assert job['result']['ready'] and '상태 파일 저장 실패' in job['result']['report_warning'],job
    print('PASS: 야간 모든 단계 재시도·인증 복구·검토 실패 보완·성공 재사용·상태 파일 오류 격리·중지')


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
