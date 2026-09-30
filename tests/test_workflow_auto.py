"""검토→영어 갱신, 실패분 재시도와 접수별 상태를 실제 SQLite로 확인합니다."""
from contextlib import closing
import json
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading
from ragkit import cslist, english, records, storage, web


def check_llm_trace():
    from ragkit.llm_trace import LLMTrace
    secret = 'PRIVATE-KEY-한국어'
    with TemporaryDirectory() as directory:
        output = Path(directory)
        trace = LLMTrace(output, {'api_key': secret})
        barrier = threading.Barrier(2)
        def create(**request):
            barrier.wait(timeout=5)
            if request['messages'][0]['content'] == 'error':
                raise ValueError(secret)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='<script>' + secret, reasoning_content='생각'),finish_reason='stop')])
        with ThreadPoolExecutor(max_workers=2) as pool:
            ok = pool.submit(trace.capture, create, 'normalize', {'model':'test','messages':[{'role':'user','content':'hello '+secret}]})
            bad = pool.submit(trace.capture, create, 'english', {'model':'test','messages':[{'role':'user','content':'error'}]})
            assert secret in ok.result().choices[0].message.content, '실제 응답까지 변경했습니다.'
            try: bad.result()
            except ValueError: pass
            else: raise AssertionError('모델 오류를 숨겼습니다.')
        items = trace.read()['items']
        assert len(items) == 2 and {item['status'] for item in items} == {'done','error'}
        assert all('request' not in item for item in items)
        assert secret not in trace.path.read_text()
        assert secret not in json.dumps([trace.read(item['id']) for item in items], ensure_ascii=False)
        assert len(LLMTrace(output, {'api_key':secret}).read()['items']) == 2
        assert trace.read('unknown') is None
        response = SimpleNamespace(content='x'*200000+secret)
        trace.capture(lambda **kw:response, 'ask', {'model':'test','messages':[]})
        record = trace.read(trace.read()['items'][0]['id'])
        assert record['response_truncated'] and len(record['response'])==200000
        with patch.object(Path, 'open', side_effect=PermissionError('blocked')):
            assert trace.capture(lambda **kw:response, 'ask', {'messages':[]}) is response
        assert trace.warning
        for _ in range(32): trace.capture(lambda **kw:SimpleNamespace(content='ok'), 'ask', {'messages':[]})
        assert len(trace.read()['items']) == 30
        trace.path.write_text('x'*10_000_000)
        trace.capture(lambda **kw:SimpleNamespace(content='ok'), 'ask', {'messages':[]})
        assert trace.path.with_suffix('.jsonl.1').exists() and trace.path.stat().st_size < 10000
    print('PASS: 병렬 요청·응답 기록·API 키 마스킹·재시작·길이 제한·회전·기록 실패 격리')


def main(root):
    check_llm_trace()
    check_vector_timing(root)
    schema, rules = cslist.load_schema(root/'cslist_schema.json')
    rows = records.read_excel(root/'cslist_sample.xlsx', 'CSList', required_columns=schema['columns'], id_column=schema['id_column'])[:3]
    config = {'base_url':'http://127.0.0.1:1/v1','api_key':'LOCAL-SECRET','model':'test','reasoning_effort':'high','timeout_seconds':1}
    calls=[]
    failing=set()
    def create(**kwargs):
        cells=json.loads(kwargs['messages'][1]['content']);calls.append(cells)
        if any(v in failing or v in {english.protect_numbers(text)[0] for text in failing} for v in cells.values()):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content=''))])
        values = {k:re.sub('[가-힣]+', ' maintenance ', v.replace('재발 미확인', 'recurrence unconfirmed')) for k,v in cells.items()}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(values,ensure_ascii=False)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=lambda:None)
    with TemporaryDirectory() as folder:
        folder=Path(folder); cfg=folder/'config.json';cfg.write_text(json.dumps(config))
        app=web.App(root,cfg,folder/'output',root/'cslist_schema.json')
        assert app.auto_english
        with closing(app.connect()) as db:
            storage.ingest(db,rows,rules,schema=schema)
            before=db.execute('SELECT * FROM normalized_records').fetchall()
            ids=[r[schema['id_column']] for r in rows]
            # 부분 대상만 처리하며, 다음 접수는 첫 실패와 관계없이 처리합니다.
            failing.add(rows[0]['현상'])
            result=english.build_english(db,client,config,schema,limit=0,record_ids=ids[:2])
            assert len(result['failures'])==1 and result['generated']==1,result
            assert result['rows']==3 and result['selected']==2 and result['indexed']==1
            evidence=dict(english.translation_documents(db,ids))
            assert english.translation_status(db,ids[0],evidence[ids[0]],schema,config)['status']=='failed'
            assert english.translation_status(db,ids[1],evidence[ids[1]],schema,config)['status']=='ready'
            assert english.translation_status(db,ids[2],evidence[ids[2]],schema,config)['status']=='pending'
            db.execute('DELETE FROM rag_english_documents WHERE record_id=?',(ids[1],))
            assert english.translation_status(db,ids[1],evidence[ids[1]],schema,config)['status']=='pending'
            english.rebuild_english(db,schema)
            # 문서 재생성 직후의 오래된 rowid 대응표로 현재 실패를 지우면 안 됩니다.
            saved_rowid=db.execute('SELECT document_rowid FROM rag_document_ids WHERE record_id=?',(ids[0],)).fetchone()[0]
            db.execute('UPDATE rag_document_ids SET document_rowid=-100 WHERE record_id=?',(ids[0],))
            english.rebuild_english(db,schema)
            assert db.execute('SELECT 1 FROM english_failures WHERE record_id=?',(ids[0],)).fetchone()
            db.execute('UPDATE rag_document_ids SET document_rowid=? WHERE record_id=?',(saved_rowid,ids[0]))
            # 다른 모델 프로필의 실패는 현재 프로필의 재시도 대상이 아닙니다.
            count=len(calls)
            other=english.build_english(db,client,config|{'model':'other'},schema,limit=0,failed_only=True)
            assert other['processed']==0 and len(calls)==count
            failing.clear();count=len(calls)
            result=english.build_english(db,client,config,schema,limit=0,failed_only=True)
            assert len(calls)==count+1 and result['generated']==1
            assert db.execute('SELECT COUNT(*) FROM english_failures').fetchone()[0]==0
            assert db.execute('SELECT * FROM normalized_records').fetchall()==before
            with patch.object(english,'request_json',side_effect=ValueError('x'*295+'LOCAL-SECRET detail')):
                failed=english.build_english(db,client,config,schema,limit=0,record_ids=[ids[2]])
            assert 'LOCAL' not in failed['failures'][0]['error'] and len(failed['failures'][0]['error']) <= 300
            count=len(calls);english.build_english(db,client,config,schema,limit=0,record_ids=ids[:2]);assert len(calls)==count
            status=app.pipeline_status(db,rows[0],None)
            assert status['english']['status']=='ready' and status['vector']['status']=='disabled'
        with patch.object(web,'OpenAI',return_value=client):
            result=app.update_search([ids[2]])
            assert result['english']['generated']==1
        # 실제 검토 저장 진입점이 변경한 접수만 자동 처리하는지 확인합니다.
        app.auto_english = True
        entry = {'source_key': rows[0]['_source_key'], 'id': ids[0], 'revision': records.row_revision(rows[0]),
                 'column':'현상', 'action':'edit', 'value':rows[0]['현상']+' 추가 기록.'}
        count=len(calls)
        with patch.object(web,'OpenAI',return_value=client):
            assert app.reviews({'format':'cell-reviews-v1','entries':[entry]})['saved']==1
            deadline=time.monotonic()+10
            while app.busy() and time.monotonic()<deadline:time.sleep(.01)
            assert not app.busy()
            job=app.job_snapshot()
            assert job['kind']=='auto_search' and job['status']=='done',job
            assert job['result']['english']['generated']==1 and len(calls)==count+1
        with closing(app.connect()) as db:
            assert db.execute('SELECT value FROM cell_reviews WHERE record_id=? AND column_name=?',(ids[0],'현상')).fetchone()[0]==entry['value']
            translated = json.loads(db.execute('SELECT translated_json FROM english_translations WHERE record_id=?', (ids[0],)).fetchone()[0])
            assert translated['현상'] == re.sub('[가-힣]+', ' maintenance ', entry['value'])
        # 후속 연결 실패도 저장된 원문과 검토를 취소하지 않습니다.
        with patch.object(web,'OpenAI',side_effect=RuntimeError('LOCAL-SECRET connection')):
            assert app.update_search([ids[0]])['english_warning']=='*** connection'
        assert app.update_settings({'auto_english':False})['auto_english'] is False
        assert not web.App(root,cfg,folder/'output',root/'cslist_schema.json').auto_english
        with patch.object(web,'OpenAI',side_effect=AssertionError('disabled')):
            assert app.update_search(ids)=={}
        try:app.update_settings({'auto_english':'yes'})
        except web.ApiError:pass
        else:raise AssertionError('문자열 설정을 허용했습니다.')
        app.save_answer_diagnostics('실제 질문',[],{'validation_events':[{'error':'LOCAL-SECRET'}]},'error')
        text=(app.output/'answer_diagnostics.jsonl').read_text()
        assert 'LOCAL-SECRET' not in text and json.loads(text)['question']=='실제 질문'
    print('PASS: 변경 접수만 자동 번역·실패 분리·재시도·원문 보존·상태·설정·진단 기록')

def check_vector_timing(root):
    """실제 다듬기·검토 진입점에서 외부 모델만 대체해 갱신 시점과 저장 순서를 확인합니다."""
    source = root / 'cslist_sample.xlsx'
    config = {'base_url':'http://127.0.0.1:1/v1','api_key':'LOCAL-SECRET','model':'test',
              'reasoning_effort':'high','timeout_seconds':1,
              'embedding':{'base_url':'http://127.0.0.1:1/v1','model':'embedding-test'}}
    def create(**kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        values = {key:payload['원본 셀'][key] + ' 기록.' for key in payload['수정할 열']}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                               message=SimpleNamespace(content=json.dumps(values,ensure_ascii=False)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=lambda:None)
    events, failures = [], set()
    english_started, release = threading.Event(), threading.Event()
    def wait_job(app):
        deadline = time.monotonic() + 10
        while app.busy() and time.monotonic() < deadline:
            time.sleep(.01)
        assert not app.busy() and app.job_snapshot()['status'] == 'done', app.job_snapshot()
        return app.job_snapshot()['result']
    with TemporaryDirectory() as folder:
        folder=Path(folder);cfg=folder/'config.json';cfg.write_text(json.dumps(config))
        app=web.App(root,cfg,folder/'output',root/'cslist_schema.json')
        row = records.read_excel(source,'CSList',required_columns=app.schema['columns'],id_column=app.schema['id_column'])[0]
        identity=row[app.schema['id_column']]
        chosen=row['현상']+' 검토 반영.'
        def vectors(**kwargs):
            events.append(('vector',kwargs))
            if 'vector' in failures:
                raise RuntimeError('LOCAL-SECRET vector failure')
            if not kwargs.get('input_changed'):
                with closing(app.connect()) as db:
                    evidence = next(english.translation_documents(db,[identity]))[1]
                    assert chosen in web.semantic.document_text(evidence), '검토 저장 전에 벡터를 생성했습니다.'
            return {'generated':1,'failures':[]}
        def translate(*args, **kwargs):
            assert events[-1][0] == 'vector', events
            events.append(('english',{}));english_started.set()
            assert release.wait(5)
            if 'english' in failures:
                raise RuntimeError('LOCAL-SECRET english failure')
            return {'generated':1,'failures':[]}
        with patch.object(web,'OpenAI',return_value=client), patch.object(app,'update_vectors',side_effect=vectors), \
                patch.object(web,'build_english',side_effect=translate):
            app.normalize({'path':str(source),'reader':'openpyxl'})
            wait_job(app)
            assert events == [], '다듬기 직후에 벡터나 영어를 요청했습니다.'
            entry={'source_key':row['_source_key'],'id':identity,'revision':records.row_revision(row),
                   'column':'현상','action':'edit','value':chosen}
            try:
                app.reviews({'format':'cell-reviews-v1','entries':[entry]})
                assert english_started.wait(5)
                assert [event[0] for event in events] == ['vector','english']
                assert app.busy(), '영어 요청 대기 상황을 재현하지 못했습니다.'
            finally:
                release.set()
            result=wait_job(app)
            assert result['embedding']['generated']==result['english']['generated']==1
            events.clear()
            app.accept_pending({'path':str(source),'reader':'openpyxl'})
            wait_job(app)
            assert [event[0] for event in events]==['vector','english'], '일괄 검토 갱신 순서가 다릅니다.'
            for failed in ('vector','english'):
                events.clear();failures.add(failed)
                result=app.update_search([identity])
                assert [event[0] for event in events]==['vector','english']
                key='embedding' if failed=='vector' else 'english'
                assert result[key+'_warning'].startswith('***')
                assert result['english' if failed=='vector' else 'embedding']['generated']==1
                failures.clear()
            events.clear()
            app.convert({'path':str(source),'reader':'openpyxl'})
            wait_job(app)
            assert events == [('vector',{'input_changed':True})], '원문 변환 후 벡터 준비가 사라졌습니다.'
        with closing(app.connect()) as db:
            assert db.execute('SELECT value FROM cell_reviews WHERE record_id=? AND column_name=?',
                              (identity,'현상')).fetchone()[0]==chosen
    print('PASS: 다듬기 후 벡터 생략·개별/일괄 검토 저장 후 벡터 우선·영어 지연/실패 격리·변환 경로 유지')


if __name__=='__main__':main(Path(__file__).resolve().parents[1]/'windows-demo')
