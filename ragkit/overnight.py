"""단일 사용자의 야간 실행. 기존 체크포인트·검토·색인을 재사용합니다."""
from contextlib import closing
import json
import time

from openai import APIError
from . import semantic
from .english import build_english
from .storage import NormalizationBatchError, ProcessingCancelled, accept_pending_reviews
from .cslist import ensure_search_index

RETRY_SECONDS = 60
PERMANENT_STATUS = {400, 401, 403, 404, 405, 413, 422}


def run(app, path, reader):
    stop = app.overnight_stop
    result = {'ready':False, 'stopped':False, 'attempts':{}, 'pending':{}, 'failures':[], 'phase':'입력 읽기',
              'retry_at':None, 'started_at':time.time(), 'path':str(path)}
    report = app.output / 'overnight_result.json'
    previous_errors = {}
    failed_stages = {}

    def publish():
        # 한 작업만 실행하므로 최신 상태 파일 하나면 충분합니다. 성공 DB 기록은 별도로 유지합니다.
        temporary = report.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(report)
        with app.lock:
            app.job['overnight'] = json.loads(json.dumps(result))

    def check_stop():
        if stop.is_set():
            raise ProcessingCancelled()

    def attempt(name, operation):
        check_stop()
        result['phase'] = name
        result['retry_at'] = None
        result['attempts'][name] = result['attempts'].get(name,0)+1
        with app.lock:
            app.job['progress'] = None
        publish()
        app.set_stage(f'야간 실행 · {name} · {result["attempts"][name]}회차')
        try:
            value = operation()
            failures = value.get('failures',[])
        except NormalizationBatchError as error:
            failures = error.failures
            previous_errors.update({item['id']:app.redact(item['error']) for item in failures})
        except APIError as error:
            if getattr(error,'status_code',None) in PERMANENT_STATUS:
                raise
            failures = [{'error':app.redact(error)}]
        check_stop()
        if not failures:
            result[name] = value
            result['pending'].pop(name,None)
            failed_stages.pop(name,None)
            result['failures'] = [item for group in failed_stages.values() for item in group][:20]
            publish()
            return value
        failed_stages[name] = [{**item,'phase':name,'error':app.redact(item['error'])} for item in failures[:20]]
        result['pending'][name] = len(failures)
        result['failures'] = [item for group in failed_stages.values() for item in group][:20]
        publish()
        if any(item.get('status_code') in PERMANENT_STATUS for item in failures):
            raise ValueError('서버 인증·요청 설정을 확인해야 합니다: '+result['failures'][0]['error'])
        return None

    def pause():
        attempt_number = max(result['attempts'].values())
        delay = min(300, RETRY_SECONDS * min(attempt_number,5))
        result['retry_at'] = time.time()+delay
        publish()
        app.set_stage(f'실패분 자동 재시도 대기 · {delay:g}초 · 중지 가능')
        if stop.wait(delay):
            raise ProcessingCancelled()

    try:
        publish()
        rows = app.read_rows(path,reader) if path else app.stored_rows()
        while attempt('normalize',lambda:app.normalize_rows(rows,path,reader,
                previous_errors=previous_errors,cancelled=stop.is_set)) is None:
            pause()
        check_stop()
        result['phase'] = '자동 채택'
        publish()
        with closing(app.connect()) as db, db:
            def review_progress(done,total):
                check_stop()
                app.set_stage(f'검사를 통과한 제안 자동 채택 {done}/{total}건')
            result['review'] = accept_pending_reviews(db,app.rules,schema=app.schema,progress=review_progress)
        if result['review']['failures'] or result['review']['missing_rows']:
            raise ValueError('자동 채택하지 못한 제안이 있습니다. 검토 화면을 확인하세요.')
        check_stop()
        configured = bool((app.config.get('embedding') or {}).get('model'))
        if not configured:
            result['embedding'] = {'skipped':True,'reason':'임베딩 모델 미설정 · 키워드 검색 준비'}

        def vectors():
            with closing(app.connect()) as db, closing(semantic.client_for(app.config,progress=app.set_stage)) as client:
                return semantic.build_vectors(db,client,app.config,progress=app.set_stage,cancelled=stop.is_set)

        def english():
            with closing(app.connect()) as db, app.llm_client() as client:
                return build_english(db,client,app.config|{'normalization_workers':app.normalization_workers},
                    app.schema,limit=0,progress=app.set_stage,cancelled=stop.is_set)

        while 'embedding' not in result or 'english' not in result:
            # 한쪽 실패가 다른 쪽의 성공 결과 저장을 막지 않습니다.
            if 'embedding' not in result:
                attempt('embedding',vectors)
                with app.lock:
                    app.embedding_status = {'status':'ready' if 'embedding' in result else 'error',
                                            'message':'야간 벡터 준비 완료' if 'embedding' in result else '실패 벡터 자동 재시도 예정'}
            if 'english' not in result:
                attempt('english',english)
            if 'embedding' not in result or 'english' not in result:
                pause()
        check_stop()
        with closing(app.connect()) as db:
            ensure_search_index(db,app.rules,app.schema)
        result.update(ready=True,phase='검색 준비 완료',failures=[])
    except ProcessingCancelled:
        result.update(stopped=True,phase='중지됨 · 저장한 성공분은 다음 실행에서 재사용')
    except Exception as error:
        result.update(phase='확인 필요',error=app.redact(error))
        raise
    finally:
        result.update(retry_at=None,finished_at=time.time())
        publish()
    return result
