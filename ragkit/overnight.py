"""야간 실행: 성공분은 재사용하고 미완료 단계는 중지 요청까지 반복합니다."""
from contextlib import closing
import json
import time

from . import semantic
from .english import build_english
from .storage import NormalizationBatchError, ProcessingCancelled, accept_pending_reviews
from .cslist import ensure_search_index

RETRY_SECONDS = 60
STAGE_LABELS = {'input': '자료 준비', 'normalize': '문장 다듬기', 'review': '자동 채택',
                'embedding': '벡터 생성', 'english': '영어 번역', 'index': '검색 확인'}


def run(app, path, reader, *, previous=None):
    stop = app.overnight_stop
    result = {'ready': False, 'stopped': False, 'attempts': {}, 'pending': {}, 'failures': [],
              'phase': 'input', 'status': 'running', 'retry_at': None, 'started_at': time.time(),
              'path': str(path) if path else None}
    if previous:
        # 단계 완료 표시 대신 기존 행별 캐시를 다시 검사하므로 설정 변경도 반영됩니다.
        result['attempts'] = dict(previous.get('attempts', {}))
        result['started_at'] = previous.get('started_at', result['started_at'])
        if 'input' in previous:
            result['input'] = previous['input']
    report = app.output / 'overnight_result.json'
    previous_errors, failed_stages = {}, {}
    rows = []

    def publish():
        result['stop_requested'] = stop.is_set()
        result.pop('report_warning', None)
        try:
            temporary = report.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(report)
        except OSError as error:
            # 상태 파일 오류가 DB에 저장한 성공 결과나 재시도 루프를 중단하지 않습니다.
            result['report_warning'] = '상태 파일 저장 실패: ' + app.redact(error)
        with app.lock:
            app.job['overnight'] = json.loads(json.dumps(result))
        app.checkpoint_job()

    def check_stop():
        if stop.is_set():
            raise ProcessingCancelled()

    def attempt(name, operation):
        check_stop()
        result.update(phase=name, status='running', retry_at=None)
        result['attempts'][name] = result['attempts'].get(name, 0) + 1
        with app.lock:
            app.job['progress'] = None
        publish()
        app.set_stage(f'야간 실행 · {STAGE_LABELS[name]} · {result["attempts"][name]}회차')
        try:
            value = operation()
            failures = value.get('failures', [])
        except ProcessingCancelled:
            raise
        except NormalizationBatchError as error:
            failures = error.failures
        except Exception as error:
            # 인증·요청·저장·색인 오류도 다음 회차에 다시 확인합니다. 성공으로 간주하지 않습니다.
            failures = [{'error': app.redact(error)}]
            if getattr(error, 'status_code', None) is not None:
                failures[0]['status_code'] = error.status_code
        check_stop()
        if name in ('normalize', 'review'):
            previous_errors.clear()
            previous_errors.update({item['id']: app.redact(item['error']) for item in failures if item.get('id')})
        if not failures:
            result[name] = value
            result['pending'].pop(name, None)
            failed_stages.pop(name, None)
        else:
            failed_stages[name] = [{**item, 'phase': name, 'error': app.redact(item['error'])} for item in failures[:20]]
            result['pending'][name] = len(failures)
        result['failures'] = [item for group in failed_stages.values() for item in group][:20]
        publish()
        return None if failures else value

    def pause():
        check_stop()
        delay = min(300, RETRY_SECONDS * min(max(result['attempts'].values()), 5))
        result.update(status='waiting', retry_at=time.time() + delay)
        publish()
        app.set_stage(f'미완료 항목 자동 재시도 대기 · {delay:g}초 · 중지 가능')
        if stop.wait(delay):
            raise ProcessingCancelled()

    def retry(name, operation):
        while attempt(name, operation) is None:
            pause()

    def prepare():
        nonlocal rows
        if path:
            app.import_file(path, reader)
        rows = app.stored_rows()
        return {'rows': len(rows)}

    def review():
        with closing(app.connect()) as db, db:
            def on_progress(done, total):
                check_stop()
                app.set_stage(f'검사를 통과한 제안 자동 채택 {done}/{total}건')
            value = accept_pending_reviews(db, app.rules, schema=app.schema, progress=on_progress)
        value['failures'].extend({'id': identity, 'error': '수정 제안이 없어 다시 생성해야 합니다.'}
                                 for identity in value.pop('missing_ids'))
        return value

    def vectors():
        with closing(app.connect()) as db, app.embedding_client() as client:
            return semantic.build_vectors(db, client, app.config, progress=app.set_stage, cancelled=stop.is_set)

    def english():
        with closing(app.connect()) as db, app.llm_client() as client:
            return build_english(db, client, app.config | {'normalization_workers': app.normalization_workers},
                                 app.schema, limit=0, progress=app.set_stage, cancelled=stop.is_set)

    def index():
        with closing(app.connect()) as db:
            ensure_search_index(db, app.rules, app.schema)
        return {'checked': True}

    try:
        retry('input', prepare)
        while 'review' not in result:
            if 'normalize' not in result:
                retry('normalize', lambda: app.normalize_rows(rows, None, reader,
                      previous_errors=previous_errors, cancelled=stop.is_set))
            if attempt('review', review) is None:
                if previous_errors:
                    # 검토 검사를 통과하지 못한 접수만 캐시 재사용을 건너뛰고 보완 요청합니다.
                    result.pop('normalize', None)
                pause()
        if not (app.config.get('embedding') or {}).get('model'):
            result['embedding'] = {'skipped': True, 'reason': '임베딩 모델 미설정 · 키워드 검색 준비'}
        while 'embedding' not in result or 'english' not in result:
            if 'embedding' not in result:
                attempt('embedding', vectors)
                with app.lock:
                    app.embedding_status = {'status': 'ready' if 'embedding' in result else 'error',
                        'message': '야간 벡터 준비 완료' if 'embedding' in result else '실패 벡터 자동 재시도 예정'}
            if 'english' not in result:
                attempt('english', english)
            if 'embedding' not in result or 'english' not in result:
                pause()
        retry('index', index)
        check_stop()
        result.update(ready=True, status='ready', phase='검색 준비 완료', failures=[])
    except ProcessingCancelled:
        result.update(stopped=True, status='stopped', phase='중지됨 · 저장한 성공분은 다음 실행에서 재사용')
    finally:
        result.update(retry_at=None, finished_at=time.time())
        publish()
    return result
