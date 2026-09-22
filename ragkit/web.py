"""로컬 CS RAG 웹 UI. 기존 입수·검토·검색·답변 함수를 재사용합니다."""
import ctypes
import hashlib
from collections import deque
from ipaddress import IPv4Address
import hmac
import json
import secrets
import sqlite3
import sys
import threading
import time
import webbrowser
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from openai import APIError, APITimeoutError, OpenAI

from .cslist import column_group, load_schema, ensure_search_index
from .input_formats import row_schema, row_rules, is_history, HISTORY_SHEET, HISTORY_SCHEMA
from .llm import answer, editable, select_answer_cases
from .records import (public_cells, read_excel, record_id, row_revision,
                      HeaderMappingError, load_header_aliases, save_header_aliases,
                      validate_header_aliases)
from .search import count_records, retrieve
from .english import (build_english, bilingual_retrieve, english_count,
                      translation_documents, translation_status, translation_profile)
from .source_cache import convert_excel, SUFFIX
from . import semantic
from .llm_trace import LLMTrace
from .storage import apply_reviews, accept_pending_reviews, ingest, import_summary, prepared_record, validate_workers, NormalizationBatchError

ASSETS = Path(__file__).resolve().parent / 'assets'
STATIC = {
    '/': ('app.html', 'text/html; charset=utf-8'),
    '/app.html': ('app.html', 'text/html; charset=utf-8'),
    '/app.css': ('app.css', 'text/css; charset=utf-8'),
    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
}
GET_API = {'/api/state', '/api/job', '/api/records', '/api/record', '/api/llm-messages'}
POST_API = {'/api/inspect', '/api/import', '/api/import-many', '/api/convert', '/api/normalize', '/api/english', '/api/vectors', '/api/compare-search', '/api/ask', '/api/reviews', '/api/reviews/accept-pending', '/api/header-mappings', '/api/settings', '/api/overnight', '/api/overnight/stop'}
PUBLIC_CONFIG = ('model', 'base_url', 'reasoning_effort', 'timeout_seconds',
                 'max_context_chars', 'max_tokens', 'context_window', 'temperature', 'top_p',
                 'answer_max_tokens', 'answer_temperature', 'answer_top_p')
MAX_BODY = 2_000_000
MAX_QUESTION = 2000
MAX_PATH = 4096
MAX_QUERY = 200
MAX_ID = 200
MAX_ENTRIES = 4000
PAGE_SIZE_MAX = 50
LIST_FIELDS = ('제목', '고객사명', '설비코드', '접수일자')
LIST_KEYS = {'제목': 'title', '고객사명': 'customer', '설비코드': 'equipment', '접수일자': 'date'}
SHEET = 'CSList'
SOURCE_CHANGED = '등록되지 않았거나 변경된 원문이 있습니다. DB에 추가·갱신을 실행하면 반영됩니다. 질문에는 현재 DB에 저장된 원문을 사용합니다.'
NO_DATA = '저장된 자료가 없습니다. 자료 화면에서 DB에 추가·갱신을 실행하세요.'


class ApiError(Exception):
    def __init__(self, status, message, code=None, extra=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.extra = extra or {}


def com_enter():
    if sys.platform != 'win32':
        return False
    hr = ctypes.windll.ole32.CoInitializeEx(None, 0x2)
    if hr in (0, 1):
        return True
    raise ApiError(500, 'Windows Excel 연결을 초기화하지 못했습니다.')


def com_leave(entered):
    if entered:
        ctypes.windll.ole32.CoUninitialize()


def cookies(handler):
    found = {}
    for part in handler.headers.get('Cookie', '').split(';'):
        if '=' in part:
            key, _, value = part.strip().partition('=')
            found[key] = value
    return found


def origin_of(handler, port, hosts=('127.0.0.1', 'localhost')):
    allowed = {f'http://{host}:{port}' for host in hosts}
    origin = handler.headers.get('Origin')
    if origin:
        return origin in allowed
    referer = handler.headers.get('Referer', '')
    return any(referer.startswith(item + '/') or referer == item for item in allowed)


def host_allowed(handler, port, hosts=('127.0.0.1', 'localhost')):
    host = handler.headers.get('Host', '')
    name, _, extra = host.partition(':')
    if name not in hosts:
        return False
    return extra == str(port) or (not extra and port in (80, 443))


class App:
    def __init__(self, root, config_path, output_dir, schema_path, excel=None, reader='openpyxl'):
        self.root = Path(root)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output / 'demo.sqlite3'
        self.settings_path = self.output / 'cslist_settings.json'
        self.schema, self.rules = load_schema(Path(schema_path))
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        if not isinstance(self.config, dict):
            raise ValueError('모델 설정은 JSON 객체여야 합니다.')
        for key in ('base_url', 'api_key', 'model', 'reasoning_effort', 'timeout_seconds'):
            if key not in self.config:
                raise ValueError(f'설정 파일에 {key}가 필요합니다.')
        if self.config.get('embedding') is not None and not isinstance(self.config['embedding'], dict):
            raise ValueError('embedding 설정은 JSON 객체여야 합니다.')
        for flag in ('auto_start', 'auto_index'):
            if type((self.config.get('embedding') or {}).get(flag, True)) is not bool:
                raise ValueError(f'embedding.{flag}는 참 또는 거짓이어야 합니다.')
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()  # 공유 DB 갱신과 순차 질문 실행의 접수 잠금입니다.
        self.llm_trace = LLMTrace(self.output, self.config)
        self.sync_busy = False
        self.port = None
        self.reader = reader if reader in ('openpyxl', 'xlwings') else 'openpyxl'
        self.excel = None
        self.excel_text = ''
        self.input_sheet = SHEET
        self.job = self._idle_job()
        self.overnight_stop = threading.Event()
        self.embedding_status = {'status': 'idle', 'message': ''}
        self.normalization_workers = validate_workers(self.config.get('normalization_workers', 200))
        self.auto_english = True
        self.reference_limit = 3
        self._load_settings()
        self.sessions = {self.token: {'seen': time.monotonic(), 'job': self._idle_job(), 'reference_limit': self.reference_limit}}
        self.questions = deque()
        self.question_worker = False
        self.allowed_hosts = {'127.0.0.1', 'localhost'}
        self.header_aliases = load_header_aliases(
            self.output / 'column_aliases.json', self.schema['columns'],
            self.schema['id_column'], self.schema.get('name'))
        if excel is not None:
            self.excel_text = str(excel)
            try:
                self.excel = self.resolve_excel(str(excel))
                self.excel_text = str(self.excel)
            except ApiError:
                self.excel = Path(str(excel))

    @contextmanager
    def llm_client(self):
        with closing(OpenAI(base_url=self.config['base_url'], api_key=self.config['api_key'],
                            timeout=self.config['timeout_seconds'], max_retries=0)) as client:
            create = client.chat.completions.create
            owner = self.job.get('owner')
            owner = hashlib.sha256(owner.encode()).hexdigest() if owner else None
            client.chat.completions.create = lambda **request: self.llm_trace.capture(create, self.job.get('kind'), request, owner=owner)
            try:
                yield client
            finally:
                client.chat.completions.create = create

    def _idle_job(self):
        return {'status': 'idle', 'kind': None, 'stage': '', 'started': None,
                'finished': None, 'error': None, 'code': None, 'result': None, 'progress': None}

    def redact(self, text):
        text = str(text)
        for key in (self.config.get('api_key'), (self.config.get('embedding') or {}).get('api_key')):
            if key and key in text:
                text = text.replace(key, '***')
        return text[:500]

    def public_error(self, error):
        if isinstance(error, NormalizationBatchError):
            return self.redact(str(error).split('\n', 1)[0]), 'normalization_partial'
        if isinstance(error, ApiError):
            return error.message, error.code
        if isinstance(error, APITimeoutError):
            return '서버 응답 시간 초과입니다. 다른 모델 요청이 끝난 뒤 다시 질문하세요.', 'timeout'
        if isinstance(error, APIError):
            return '모델 서버에 연결하지 못했습니다. 서버 주소와 실행 상태를 확인하세요.', 'model'
        if isinstance(error, (ValueError, RuntimeError, OSError)):
            return self.redact(error), None
        return '작업을 완료하지 못했습니다. 입력과 서버 상태를 확인하세요.', None

    def public_config(self):
        result = {key: self.config[key] for key in PUBLIC_CONFIG if key in self.config}
        result['embedding_model'] = (self.config.get('embedding') or {}).get('model', '')
        result['embedding_auto_index'] = self.auto_index_enabled()
        return result

    def auto_index_enabled(self):
        cfg = self.config.get('embedding') or {}
        return bool(cfg.get('base_url') and cfg.get('model') and cfg.get('auto_index', True))

    def start_embedding(self):
        cfg = self.config.get('embedding') or {}
        if cfg.get('base_url') and cfg.get('model'):
            self.start_job('auto_vectors', self.update_vectors)

    def update_vectors(self, *, input_changed=False):
        with self.lock:
            self.embedding_status = {'status': 'running', 'message': '임베딩 자동 준비·갱신 중'}
        try:
            if input_changed:
                result = self.index_input(self.excel, self.reader, 0)
            else:
                with closing(semantic.client_for(self.config, progress=self.set_stage)) as client:
                    self.set_stage('임베딩 모델 응답 확인 중')
                    semantic.embed(client, ['임베딩 연결 확인'], semantic.settings(self.config))
                    result = {'rows': 0, 'indexed': 0, 'generated': 0, 'failures': []}
                    if self.auto_index_enabled() and self.db_summary()['records']:
                        with closing(self.connect()) as db:
                            result = semantic.build_vectors(db, client, self.config, progress=self.set_stage)
            if result['failures']:
                raise ValueError(f"벡터 {len(result['failures'])}개 생성 실패. 성공분은 저장했습니다. 다음 실행에서 이어서 처리합니다. "
                                 + self.redact(result['failures'][0]['error']))
            with self.lock:
                self.embedding_status = {'status': 'ready', 'message': '임베딩 준비 완료' + (' · 필요한 벡터 갱신 완료' if self.auto_index_enabled() else '')}
            return result
        except Exception as error:
            with self.lock:
                self.embedding_status = {'status': 'error', 'message': self.redact(error)}
            raise

    def update_search(self, record_ids):
        """검토 저장은 이미 완료됐습니다. 후속 실패로 저장된 검토를 취소하지 않습니다."""
        result = {}
        # 채택문 벡터를 먼저 준비해 영어 번역의 지연·실패와 분리합니다.
        if self.auto_index_enabled() and record_ids:
            try:
                result['embedding'] = self.update_vectors()
            except Exception as error:
                result['embedding_warning'] = self.redact(error)
        if self.auto_english and record_ids:
            try:
                self.set_stage('검토한 접수의 영어 검색 자료 갱신 중')
                with closing(self.connect()) as db, self.llm_client() as client:
                    result['english'] = build_english(db, client, self.config | {'normalization_workers': self.normalization_workers}, self.schema,
                        limit=0, record_ids=record_ids, progress=self.set_stage)
            except Exception as error:
                result['english_warning'] = self.redact(error)
        return result

    def pipeline_status(self, db, original, proposal, prepared=None):
        prepared = prepared or prepared_record(db, original, proposal, self.rules)
        pending = sum(editable(original, key, self.rules) and
                      (r['action'] == 'pending' or r['stale']) for key, r in prepared['reviews'].items())
        identity = prepared['id']
        evidence = next(iter(translation_documents(db, [identity])), None)
        english = translation_status(db, identity, evidence[1], self.schema, self.config) if evidence else {'status': 'pending', 'error': ''}
        english['error'] = self.redact(english['error'])
        vector = 'disabled'
        if (self.config.get('embedding') or {}).get('model'):
            vector = 'pending'
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='vector_embeddings'").fetchone():
                try:
                    found = db.execute('SELECT 1 FROM vector_sources s JOIN vector_embeddings e ON s.source_hash=e.source_hash '
                                       "WHERE s.record_id=? AND s.content!='' AND e.profile=?", (identity, semantic.profile(self.config))).fetchone()
                    if found:
                        vector = 'ready'
                except ValueError:
                    pass
        return {'review': {'status': 'pending' if pending else 'complete', 'pending_cells': pending},
                'english': english, 'vector': {'status': vector}}

    def save_answer_diagnostics(self, question, hits, details, status):
        """실제 질문은 로컬에만 기록합니다. 평가의 기대 정답은 사람이 별도로 지정합니다."""
        payload = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'question': question,
                   'model': self.config['model'], 'status': status,
                   'candidate_ids': [hit.get('record_id', hit['id']) for hit in hits],
                   'validation_events': details.get('validation_events', []),
                   'attempts': details.get('attempts', 0)}
        text = json.dumps(payload, ensure_ascii=False)
        for secret in (self.config.get('api_key'), (self.config.get('embedding') or {}).get('api_key')):
            if secret:
                text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], '***')
        try:
            # ponytail: 개인용 로컬 JSONL입니다. 장기 운영으로 파일이 커지면 보관 기간을 추가합니다.
            with (self.output / 'answer_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(text+'\n')
        except OSError:
            details['note'] = (details.get('note') or '') + ' 진단 기록을 저장하지 못했습니다.'

    def resolve_excel(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, 'Excel 경로를 입력하세요.')
        text = value.strip().strip('"\'')
        if not text or '\x00' in text or len(text) > MAX_PATH:
            raise ApiError(400, '파일 경로가 올바르지 않습니다.')
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.root / path
        try:
            path = path.resolve()
        except OSError as error:
            raise ApiError(400, '파일 경로가 올바르지 않습니다.') from error
        if path.suffix.lower() != '.xlsx' and not path.name.lower().endswith(SUFFIX):
            raise ApiError(400, '.xlsx 또는 .rag.jsonl.gz 파일을 지정하세요.')
        return path

    def _load_settings(self):
        if not self.settings_path.exists():
            return
        try:
            saved = json.loads(self.settings_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict):
            return
        if type(saved.get('auto_english')) is bool:
            self.auto_english = saved['auto_english']
        if type(saved.get('reference_limit')) is int and 0 <= saved['reference_limit'] <= 100:
            self.reference_limit = saved['reference_limit']
        if 'normalization_workers' in saved:
            self.normalization_workers = validate_workers(saved['normalization_workers'])
        excel = saved.get('excel')
        reader = saved.get('reader')
        if saved.get('sheet') in (SHEET, HISTORY_SHEET):
            self.input_sheet = saved['sheet']
        if reader in ('openpyxl', 'xlwings'):
            self.reader = reader
        if isinstance(excel, str) and excel.strip():
            self.excel_text = excel.strip()
            try:
                self.excel = self.resolve_excel(excel)
                self.excel_text = str(self.excel)
            except ApiError:
                self.excel = None

    def save_settings(self, workers=None, auto_english=None, reference_limit=None):
        payload = {'excel': self.excel_text or (str(self.excel) if self.excel else ''),
                   'sheet': self.input_sheet, 'reader': self.reader,
                   'normalization_workers': self.normalization_workers if workers is None else workers,
                   'auto_english': self.auto_english if auto_english is None else auto_english,
                   'reference_limit': self.reference_limit if reference_limit is None else reference_limit}
        temporary = self.settings_path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(self.settings_path)
        finally:
            temporary.unlink(missing_ok=True)

    def update_settings(self, body):
        self._begin_sync()
        try:
            if not isinstance(body, dict) or not body or set(body) - {'normalization_workers', 'auto_english', 'reference_limit'}:
                raise ApiError(400, '동시 처리 수·영어 자동 갱신·참고 사례 수만 설정할 수 있습니다.')
            reference_limit = body.get('reference_limit', self.reference_limit)
            if type(reference_limit) is not int or not 0 <= reference_limit <= 100:
                raise ApiError(400, '참고 사례 수는 0(자동) 또는 1~100 사이 정수여야 합니다.')
            auto_english = body.get('auto_english', self.auto_english)
            if type(auto_english) is not bool:
                raise ApiError(400, '영어 자동 갱신 여부는 참 또는 거짓이어야 합니다.')
            try:
                workers = validate_workers(body.get('normalization_workers', self.normalization_workers))
            except ValueError as error:
                raise ApiError(400, str(error)) from error
            self.save_settings(workers=workers, auto_english=auto_english, reference_limit=reference_limit)
            self.normalization_workers = workers
            self.auto_english = auto_english
            self.reference_limit = reference_limit
            return {'normalization_workers': workers, 'auto_english': auto_english, 'reference_limit': reference_limit}
        finally:
            self._end_sync()

    def connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def db_summary(self):
        if not self.db_path.exists():
            return {'exists': False, 'records': 0, 'has_proposals': False}
        try:
            with closing(self.connect()) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='normalized_records'").fetchone():
                    return {'exists': False, 'records': 0, 'has_proposals': False}
                records = db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0]
                proposals = db.execute(
                    "SELECT 1 FROM normalized_records WHERE normalized_json != 'null' LIMIT 1").fetchone()
                return {'exists': True, 'records': records, 'has_proposals': bool(proposals),
                        'english_records': english_count(db), 'vector_records': semantic.vector_count(db, self.config),
                        'english_failed': db.execute('SELECT COUNT(*) FROM english_failures f JOIN normalized_records n ON n.id=f.record_id WHERE f.profile=?', (translation_profile(self.config),)).fetchone()[0]
                        if db.execute("SELECT 1 FROM sqlite_master WHERE name='english_failures'").fetchone() else 0}
        except sqlite3.Error:
            return {'exists': False, 'records': 0, 'has_proposals': False}

    def require_excel(self, body=None):
        body = body or {}
        raw_path = body.get('path')
        if raw_path is not None and not isinstance(raw_path, str):
            raise ApiError(400, 'Excel 경로를 입력하세요.')
        path_value = raw_path if isinstance(raw_path, str) and raw_path.strip() else self.excel_text
        reader = body.get('reader', self.reader)
        if 'reader' in body and not isinstance(body['reader'], str):
            raise ApiError(400, '읽기 방식은 일반 파일 또는 Windows Excel에서 열린 파일이어야 합니다.')
        if reader not in ('openpyxl', 'xlwings'):
            raise ApiError(400, '읽기 방식은 일반 파일 또는 Windows Excel에서 열린 파일이어야 합니다.')
        path = self.resolve_excel(path_value or '')
        if path.name.lower().endswith(SUFFIX):
            reader = 'openpyxl'  # 변환 파일에는 Excel 연결과 DRM 재인증이 필요하지 않습니다.
        if reader == 'openpyxl' and not path.is_file():
            raise ApiError(400, 'Excel 파일을 찾을 수 없습니다. 전체 경로를 확인하세요.')
        return path, reader

    def read_rows(self, path, reader):
        def load():
            rows = read_excel(path, SHEET, required_columns=self.schema['columns'],
                              id_column=self.schema['id_column'], reader=reader,
                              header_aliases=self.header_aliases)
            if rows:
                self.input_sheet = rows[0]['_source_sheet']
            return rows
        if reader != 'xlwings':
            return load()
        entered = False
        try:
            entered = com_enter()
            return load()
        finally:
            com_leave(entered)

    def stored_rows(self):
        """등록 후에는 Excel 연결 없이 DB의 원문을 사용합니다."""
        if not self.db_summary()['records']:
            raise ApiError(400, NO_DATA, 'no_data')
        with closing(self.connect()) as db:
            return [json.loads(raw) for (raw,) in db.execute('SELECT original_json FROM normalized_records ORDER BY id')]

    def session(self, handler, *, create=False):
        token = cookies(handler).get('csrf_token', '')
        now = time.monotonic()
        with self.lock:
            for key, value in list(self.sessions.items()):
                if now - value['seen'] > 86400 and value['job']['status'] not in ('queued', 'running'):
                    del self.sessions[key]
            if token not in self.sessions:
                if not create:
                    raise ApiError(403, '세션이 만료되었습니다. 화면을 새로고침하세요.')
                if len(self.sessions) >= 128:
                    raise ApiError(503, '사용자 세션이 가득 찼습니다. 나중에 접속하세요.')
                token = secrets.token_urlsafe(32)
                self.sessions[token] = {'seen': now, 'job': self._idle_job(), 'reference_limit': self.reference_limit}
            self.sessions[token]['seen'] = now
        return token

    def job_snapshot(self, owner=None):
        with self.lock:
            server_busy = self.job['status'] == 'running' or self.sync_busy or self.question_worker
            maintenance = (self.job['status'] == 'running' and self.job['kind'] != 'ask') or self.sync_busy
            job = self.job
            if owner is not None:
                own = self.sessions[owner]['job']
                if job['kind'] == 'ask' or own['status'] in ('queued', 'running') or (not maintenance and (own.get('started') or 0) > (job.get('started') or 0)):
                    job = own
            job = dict(job)
            position = next((i for i, (queued, _) in enumerate(self.questions, 1) if queued.get('owner') == owner), 0) if owner else 0
            started, finished = job.get('started'), job.get('finished')
            progress = dict(job['progress']) if isinstance(job.get('progress'), dict) else None
        elapsed = round((finished if finished is not None else time.monotonic()) - started, 1) if started else 0
        return {
            'status': job['status'], 'kind': job['kind'],
            'stage': self.redact(job['stage']) if job['stage'] else '',
            'elapsed_seconds': elapsed, 'error': job['error'], 'code': job['code'],
            'result': job['result'] if job['status'] not in ('running', 'queued') else None,
            'question': job.get('question'),
            'progress': progress, 'overnight': job.get('overnight'),
            'queue_position': position, 'server_busy': server_busy, 'maintenance_running': maintenance,
        }

    def queue_question(self, owner, func, *, question=None):
        with self.lock:
            if self.sync_busy or (self.job['status'] == 'running' and self.job['kind'] != 'ask'):
                raise ApiError(409, '자료 갱신 작업 중입니다. 완료 후 질문하세요.', 'busy')
            if self.sessions[owner]['job']['status'] in ('queued', 'running'):
                raise ApiError(409, '내 질문이 이미 대기 또는 처리 중입니다.', 'busy')
            if len(self.questions) >= 50:
                raise ApiError(429, '질문 대기열이 가득 찼습니다. 잠시 후 다시 제출하세요.', 'queue_full')
            job = self._idle_job() | {'kind': 'ask', 'status': 'queued', 'stage': '질문 대기 중', 'owner': owner, 'question': question}
            self.sessions[owner]['job'] = job
            self.questions.append((job, func))
            if self.question_worker:
                return
            self.question_worker = True
        # ponytail: 단일 프로세스 메모리 대기열. 재시작 복구가 필요하면 영속 작업 테이블로 바꿉니다.
        threading.Thread(target=self.run_questions, name='cs-rag-questions', daemon=True).start()

    def run_questions(self):
        while True:
            with self.lock:
                if not self.questions:
                    self.question_worker = False
                    return
                job, func = self.questions.popleft()
                self.job = job
                job.update(status='running', stage='준비 중', started=time.monotonic())
            try:
                result = func()
                update = dict(status='done', stage='완료', result=result, error=None, code=None)
            except Exception as error:
                message, code = self.public_error(error)
                update = dict(status='error', stage='실패', error=message, code=code,
                              result=getattr(error, 'extra', {}).get('answer_result'))
            with self.lock:
                job.update(update, finished=time.monotonic())

    def busy(self):
        with self.lock:
            return self.job['status'] == 'running' or self.sync_busy or self.question_worker

    def start_job(self, kind, func, *, from_sync=False):
        with self.lock:
            if self.job['status'] == 'running' or self.question_worker or (self.sync_busy and not from_sync):
                raise ApiError(409, '다른 작업이 진행 중입니다. 끝난 뒤에 다시 실행하세요.', 'busy')
            self.job = {'status': 'running', 'kind': kind, 'stage': '준비 중',
                        'started': time.monotonic(), 'finished': None,
                        'error': None, 'code': None, 'result': None, 'progress': None}
            if kind == 'overnight':
                self.overnight_stop.clear()

        def target():
            try:
                result = func()
                if kind == 'bulk_review':
                    result.update(self.update_search(result.pop('_review_ids', [])))
                if self.auto_index_enabled() and (kind in ('convert', 'import') or
                                                  (kind == 'import_many' and result['succeeded'])):
                    try:
                        result['embedding'] = self.update_vectors(input_changed=kind == 'convert')
                    except Exception as error:
                        # 등록·변환 결과는 보관하고 임베딩 실패만 별도로 알립니다.
                        result['embedding_warning'] = self.redact(error)
                finished = time.monotonic()
                with self.lock:
                    self.job.update(status='done', stage='완료', result=result,
                                    error=None, code=None, finished=finished)
            except Exception as error:
                message, code = self.public_error(error)
                finished = time.monotonic()
                with self.lock:
                    self.job.update(status='error', stage='실패', error=message,
                                    code=code, result=getattr(error, 'extra', {}).get('answer_result'), finished=finished)

        threading.Thread(target=target, name='cs-rag-job', daemon=True).start()

    def _begin_sync(self):
        with self.lock:
            if self.job['status'] == 'running' or self.question_worker or self.sync_busy:
                raise ApiError(409, '다른 작업이 진행 중입니다. 끝난 뒤에 다시 실행하세요.', 'busy')
            self.sync_busy = True

    def _end_sync(self):
        with self.lock:
            self.sync_busy = False

    def set_stage(self, stage):
        with self.lock:
            if self.job['status'] == 'running':
                self.job['stage'] = self.redact(stage)

    def state(self, owner=None):
        summary = self.db_summary()
        schema = HISTORY_SCHEMA if self.input_sheet == HISTORY_SHEET else self.schema
        return {
            'excel': self.excel_text,
            'reader': self.reader,
            'sheet': self.input_sheet,
            'output_dir': str(self.output.resolve()),
            'schema_name': schema.get('name', 'CSList'),
            'id_column': schema['id_column'],
            'columns': len(schema['columns']),
            'schema_columns': list(schema['columns']),
            'header_aliases': dict(self.header_aliases),
            'normalization_workers': self.normalization_workers,
            'settings': {'auto_english': self.auto_english, 'reference_limit': self.sessions[owner]['reference_limit'] if owner else self.reference_limit},
            'db': summary,
            'config': self.public_config(),
            'embedding_status': dict(self.embedding_status),
            'job': self.job_snapshot(owner),
            'platform': sys.platform,
        }

    def inspect(self, body):
        self._begin_sync()
        try:
            path, reader = self.require_excel(body)
            try:
                rows = self.read_rows(path, reader)
            except HeaderMappingError as error:
                if error.id_column == 'CS NO':
                    raise ApiError(400, self.redact(error)) from error
                raise ApiError(400, self.redact(error), 'header_mapping_required', extra={
                    'mapping': {
                        'missing': error.missing,
                        'unknown': error.unknown,
                        'observed': error.observed,
                        'aliases': dict(self.header_aliases),
                        'schema_columns': list(self.schema['columns']),
                        'id_column': self.schema['id_column'],
                        'path': str(path),
                    }
                }) from error
            self.excel = path
            self.excel_text = str(path)
            self.reader = reader
            self.save_settings()
            summary = self.db_summary()
            with closing(self.connect() if self.db_path.exists() else sqlite3.connect(':memory:')) as db:
                incoming = import_summary(db, rows)
            matches = incoming['unchanged'] == len(rows) if summary['records'] else None
            result = {
                'path': str(path),
                'sheet': rows[0]['_source_sheet'],
                'rows': len(rows),
                'columns': len(public_cells(rows[0])),
                'id_column': rows[0]['_id_column'],
                'schema_name': row_schema(rows[0], self.schema)['name'],
                'reader': reader,
                'matches_stored': matches,
                'db': summary,
                'import': incoming,
            }
            if incoming['conflicts']:
                result['warning'] = '다른 파일의 같은 접수번호와 내용이 다릅니다. 등록 전에 접수번호를 확인하세요: ' + ', '.join(incoming['conflicts'][:20])
            elif matches is False:
                result['warning'] = SOURCE_CHANGED
            return result
        finally:
            self._end_sync()

    def convert(self, body):
        path, reader = self.require_excel(body)
        if path.suffix.lower() != '.xlsx':
            raise ApiError(400, '이미 변환된 파일입니다. 원본 Excel을 선택해 다시 변환하세요.')
        output = self.output / 'inputs' / (path.stem + SUFFIX)

        def work():
            def on_progress(info):
                with self.lock:
                    self.job['stage'] = info['stage']
                    self.job['progress'] = dict(info)
            entered = com_enter() if reader == 'xlwings' else False
            try:
                result = convert_excel(path, output, SHEET, required_columns=self.schema['columns'],
                    id_column=self.schema['id_column'], reader=reader, header_aliases=self.header_aliases,
                    progress=on_progress)
            finally:
                com_leave(entered)
            with self.lock:
                self.excel = output.resolve()
                self.excel_text = str(self.excel)
                self.reader = 'openpyxl'
                self.input_sheet = result['sheet']
            self.save_settings()
            return result

        self.start_job('convert', work)
        return self.job_snapshot()

    def import_input(self, body):
        path, reader = self.require_excel(body)
        self.start_job('import', lambda: self.import_file(path, reader))
        return {'accepted': True, 'kind': 'import'}

    def import_file(self, path, reader):
        """한 파일의 검사·저장을 끝냅니다. 단일 등록과 대기열이 같은 보존 규칙을 씁니다."""
        self.set_stage(f'{path.name} · 원문을 DB에 누적 등록 중')
        previous_sheet = self.input_sheet
        try:
            rows = self.read_rows(path, reader)
            incoming = {}
            with closing(self.connect()) as db:
                ingest(db, rows, self.rules, schema=self.schema, append=True, import_result=incoming)
        except Exception:
            self.input_sheet = previous_sheet  # 실패한 다른 양식을 선택된 파일의 시트로 표시하지 않습니다.
            raise
        with self.lock:
            self.excel, self.excel_text, self.reader = path, str(path), reader
        self.save_settings()
        return {'rows': len(rows), 'import': incoming, 'path': str(path),
                'reader': reader, 'sheet': rows[0]['_source_sheet']}

    def import_many(self, body):
        paths = body.get('paths')
        reader = body.get('reader', self.reader)
        if set(body) - {'paths', 'reader'} or not isinstance(paths, list) or not 1 <= len(paths) <= 100:
            raise ApiError(400, '파일 경로를 1~100개 지정하세요. 읽기 방식도 함께 지정할 수 있습니다.')
        if any(not isinstance(path, str) or not path.strip() or '\x00' in path or len(path) > MAX_PATH for path in paths):
            raise ApiError(400, '각 파일의 전체 경로를 올바르게 입력하세요.')
        if reader not in ('openpyxl', 'xlwings'):
            raise ApiError(400, '읽기 방식은 일반 파일 또는 Windows Excel에서 열린 파일이어야 합니다.')

        def work():
            # ponytail: 최대 100파일의 메모리 대기열. 서버 재시작 복구가 필요하면 작업 목록을 영속화합니다.
            files = [{'path': path, 'status': 'waiting'} for path in paths]
            result = {'files': files, 'reader': reader, 'total': len(files), 'completed': 0,
                      'succeeded': 0, 'failed': 0, 'skipped': 0, 'rows': 0,
                      'import': {'added': 0, 'updated': 0, 'unchanged': 0}}

            def publish():
                with self.lock:
                    # 조회 스레드가 처리 도중 바뀌는 목록을 읽지 않도록 스냅샷을 교체합니다.
                    self.job['progress'] = result | {'files': [dict(item) for item in files],
                                                     'import': dict(result['import'])}

            seen = set()
            publish()
            for item in files:
                item['status'] = 'running'
                publish()
                try:
                    path = self.resolve_excel(item['path'])
                    item['path'] = str(path)
                    if path in seen:
                        item.update(status='skipped', reason='대기열에 같은 파일 경로가 있어 한 번만 처리합니다.')
                        result['skipped'] += 1
                    else:
                        seen.add(path)
                        path, mode = self.require_excel({'path': str(path), 'reader': reader})
                        item.update(self.import_file(path, mode), status='done')
                        result['succeeded'] += 1
                        result['rows'] += item['rows']
                        for key in result['import']:
                            result['import'][key] += item['import'][key]
                except Exception as error:
                    message, code = self.public_error(error)
                    item.update(status='error', error=message, code=code)
                    result['failed'] += 1
                result['completed'] += 1
                publish()
            return result

        # 기존 작업 잠금을 사용합니다. Excel/SQLite는 파일별로 순차 처리하고 벡터는 마지막에 갱신합니다.
        self.start_job('import_many', work)
        return {'accepted': True, 'kind': 'import_many'}

    def normalize(self, body):
        # 경로를 명시한 기존 API 요청도 지원합니다. 새 UI는 등록한 DB 전체를 처리합니다.
        path, reader = self.require_excel(body) if body.get('path') else (None, self.reader)

        self.start_job('normalize', lambda: self.normalize_rows(
            self.read_rows(path, reader) if path else self.stored_rows(), path, reader))
        return {'accepted': True, 'kind': 'normalize'}

    def start_overnight(self, body):
        if not isinstance(body, dict) or set(body)-{'path','reader'}:
            raise ApiError(400,'야간 실행에는 입력 경로와 읽기 방식만 지정하세요.')
        path, reader = self.require_excel(body) if body.get('path') else (None, self.reader)
        if self.config.get('embedding'):
            semantic.settings(self.config)
        from .overnight import run
        self.start_job('overnight',lambda:run(self,path,reader))
        return {'accepted':True,'kind':'overnight'}

    def stop_overnight(self, body):
        if body:
            raise ApiError(400,'중지 요청에는 추가 값이 필요하지 않습니다.')
        with self.lock:
            if self.job['kind'] != 'overnight' or self.job['status'] != 'running':
                raise ApiError(409,'실행 중인 야간 작업이 없습니다.')
            self.overnight_stop.set()
            if self.job.get('overnight'):
                self.job['overnight']['stop_requested'] = True
            self.job['stage'] = '중지 요청됨 · 진행 중인 행·묶음을 저장한 뒤 중지합니다.'
        return {'stopping':True}

    def normalize_rows(self, rows, path, reader, *, previous_errors=None, cancelled=None):
        if path:
            with self.lock:
                self.excel = path
                self.excel_text = str(path)
                self.reader = reader
            self.save_settings()
        self.set_stage('문장 다듬기 실행 중')
        config = self.config | {'normalization_workers': self.normalization_workers}
        def on_progress(info):
            info = info if isinstance(info, dict) else {}
            failed = info.get('failed_id')
            copied = {
                'total': info.get('total'),
                'completed': info.get('completed'),
                'reused': info.get('reused'),
                'fallback_cells': info.get('fallback_cells', 0),
                'attempted': info.get('attempted'),
                'failed': info.get('failed'),
                'failures': [{'id': self.redact(item['id']), 'error': self.redact(item['error'])}
                             for item in info.get('failures', [])],
                'current_id': info.get('current_id') if isinstance(info.get('current_id'), str) else '',
                'stage': info.get('stage') if isinstance(info.get('stage'), str) else '',
                'failed_id': failed if isinstance(failed, str) or failed is None else None,
                'workers': info.get('workers'),
                'active': info.get('active'),
            }
            with self.lock:
                if self.job['status'] == 'running':
                    self.job['progress'] = copied
                    if copied['stage']:
                        self.job['stage'] = self.redact(copied['stage'])
        incoming = {}
        with self.llm_client() as client:
            with closing(self.connect()) as db:
                prepared = ingest(db, rows, self.rules, client, config, schema=self.schema,
                                  progress=on_progress, previous_errors=previous_errors, cancelled=cancelled,
                                  append=True, import_result=incoming)
        result = {'rows': len(prepared), 'path': str(path) if path else None, 'output_dir': str(self.output.resolve()),
                  'fallback_cells': sum(len(record['fallbacks']) for record in prepared), 'import': incoming}
        with self.lock:
            progress = self.job.get('progress') if isinstance(self.job.get('progress'), dict) else None
            reused = progress.get('reused') if progress else None
        if reused is not None:
            result['reused'] = reused
        return result


    def vectors(self, body):
        if set(body) - {'path', 'reader', 'limit'}:
            raise ApiError(400, '벡터 생성에는 입력 경로·읽기 방식·처리 건수만 지정하세요.')
        limit = body.get('limit', 20)
        if type(limit) is not int or limit < 0:
            raise ApiError(400, '처리 건수는 0(전체) 또는 양의 정수여야 합니다.')
        try:
            semantic.settings(self.config)
        except ValueError as error:
            raise ApiError(400, str(error)) from error
        path, reader = (None, self.reader) if self.db_summary()['records'] else self.require_excel(body)

        def work():
            result = self.index_input(path, reader, limit)
            with self.lock:
                self.embedding_status = {'status': 'error' if result['failures'] else 'ready',
                                         'message': '일부 벡터 생성 실패' if result['failures'] else '임베딩 준비 완료 · 요청한 벡터 생성 완료'}
                if path:
                    self.excel, self.excel_text, self.reader = path, str(path), reader
            if path:
                self.save_settings()
            result['failures'] = [{'error': self.redact(item['error'])} for item in result['failures']]
            return result

        self.start_job('vectors', work)
        return {'accepted': True, 'kind': 'vectors'}

    def index_input(self, path, reader, limit):
        self.set_stage('DB 검색 자료 확인 중')
        with closing(self.connect()) as db:
            if path:
                rows = self.read_rows(path, reader)
                if import_summary(db, rows)['updated']:
                    raise ApiError(409, SOURCE_CHANGED, 'source_changed')
                self.set_stage('새 원문 누적 등록 중 · 문장 다듬기 호출 없음')
                ingest(db, rows, self.rules, schema=self.schema, append=True)
            else:
                ensure_search_index(db, self.rules, self.schema)
            with closing(semantic.client_for(self.config, progress=self.set_stage)) as client:
                return semantic.build_vectors(db, client, self.config, limit=limit, progress=self.set_stage)

    def ask(self, body, *, comparison=False, owner=None):
        if not isinstance(body, dict) or not isinstance(body.get('question'), str):
            raise ApiError(400, '질문을 입력하세요.')
        question = body['question'].strip()
        if not question:
            raise ApiError(400, '질문을 입력하세요.')
        if len(question) > MAX_QUESTION:
            raise ApiError(400, '질문이 너무 깁니다. 한 가지 질문으로 나누어 입력하세요.')
        bilingual = body.get('bilingual', False)
        if type(bilingual) is not bool:
            raise ApiError(400, '영어 검색 사용 여부는 참 또는 거짓이어야 합니다.')
        search_mode = 'hybrid' if comparison else body.get('search_mode', 'keyword')
        if search_mode not in ('keyword', 'vector', 'hybrid'):
            raise ApiError(400, '검색 방식은 keyword, vector, hybrid 중 하나여야 합니다.')
        reference_limit = body.get('reference_limit', self.sessions[owner]['reference_limit'] if owner else self.reference_limit)
        if type(reference_limit) is not int or not 0 <= reference_limit <= 100:
            raise ApiError(400, '참고 사례 수는 0(자동) 또는 1~100 사이 정수여야 합니다.')
        retrieval_limit = reference_limit or 100
        if comparison:
            bilingual = False
        if not self.db_path.exists():
            raise ApiError(400, NO_DATA, 'no_data')

        def work():
            started = time.monotonic()
            details = {'timings': {}}
            timings = details['timings']
            self.set_stage('DB 원문 확인 중')
            with closing(self.connect()) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='normalized_records'").fetchone():
                    raise ApiError(400, NO_DATA, 'no_data')
                details['source_db'] = True
                timings['source_check'] = round(time.monotonic() - started, 4)
                self.set_stage('검색 자료 버전 확인 중')
                stage_started = time.monotonic()
                details['index_rebuilt'] = ensure_search_index(db, self.rules, self.schema)
                timings['index_prepare'] = round(time.monotonic() - stage_started, 4)
                self.set_stage('검색 중')
                stage_started = time.monotonic()
                aggregate = count_records(db, question)
                timings['aggregate_check'] = round(time.monotonic() - stage_started, 4)
                search_started = time.monotonic()
                hits = []
                if bilingual and not aggregate:
                    self.set_stage('한국어·영어로 검색 중')
                    with self.llm_client() as client:
                        hits = bilingual_retrieve(db, question, client, self.config, details=details,
                                                  limit=max(20, retrieval_limit) if search_mode == 'hybrid' else retrieval_limit,
                                                  search_english=search_mode != 'vector')
                    timings['bilingual'] = round(time.monotonic() - search_started, 4)
                if not aggregate and (search_mode != 'keyword' or comparison):
                    hits = semantic.search_cases(db, question, self.config, mode=search_mode,
                        keyword_hits=hits if bilingual else None, details=details, limit=retrieval_limit)
                elif not aggregate and not bilingual:
                    hits = retrieve(db, question, limit=retrieval_limit)
                    timings['keyword'] = round(time.monotonic() - search_started, 4)
                timings['retrieval_total'] = round(time.monotonic() - search_started, 4)
                details['references'] = {'mode': 'auto' if reference_limit == 0 else 'manual',
                    'requested_limit': reference_limit, 'candidates': len(hits), 'selected': len(hits),
                    'applied_to': 'comparison' if comparison else 'aggregate' if aggregate else 'answer'}
                stage_started = time.monotonic()
                if comparison and not aggregate:
                    reply, code = '검색 방식별 후보를 비교했습니다. 답변 모델은 호출하지 않았습니다.', 'ok'
                    details['comparison'] = True
                    details['note'] = '키워드·벡터·결합 검색 후보는 아래 검색 방식과 후보 비교에서 확인하세요.'
                elif aggregate:
                    reply, code = aggregate['answer'], 'aggregate'
                elif not hits:
                    reply, code = '해당 질문에 맞는 기록이 없어 확인할 수 없습니다.', 'no_match'
                else:
                    self.set_stage('답변 생성 중')
                    with self.llm_client() as client:
                        try:
                            answer_config = self.config
                            if reference_limit == 0:
                                self.set_stage('답변 모델의 문맥 한도·참고 사례 분량 확인 중')
                                hits, answer_config = select_answer_cases(self.config, question, hits, details=details)
                            self.set_stage('답변 생성 중')
                            reply, _context = answer(client, answer_config, question, hits, details=details)
                            code = details.get('answer_status', 'ok')
                        except Exception as error:
                            text, failure_code = self.public_error(error)
                            failure_code = ('validation_failed' if isinstance(error, ValueError) and
                                            ('근거 검사' in text or 'claims' in text) else failure_code or 'error')
                            summary = details | {'claims': [], 'source_supplements': [],
                                'validation_events': [{k: self.redact(v) if isinstance(v, str) else v
                                    for k, v in event.items() if k != 'response'}
                                    for event in details.get('validation_events', [])]}
                            result = pack_answer(question, text, hits, None, summary, time.monotonic()-started, code=failure_code)
                            raise ApiError(422 if failure_code == 'validation_failed' else 500, text, failure_code,
                                           extra={'answer_result': result}) from error
                        finally:
                            self.save_answer_diagnostics(question, hits, details, details.get('answer_status', 'error'))
                timings['answer'] = round(time.monotonic() - stage_started, 4)
            timings['total'] = round(time.monotonic() - started, 4)
            if not comparison and (aggregate or not hits):
                self.save_answer_diagnostics(question, hits, details, code)
            return pack_answer(question, reply, hits, aggregate, details, time.monotonic() - started, code=code)

        self.queue_question(owner or self.token, work, question=question)
        return {'accepted': True, 'kind': 'ask'}

    def translate_english(self, body):
        if not isinstance(body, dict) or set(body) - {'path', 'reader', 'limit', 'failed_only'}:
            raise ApiError(400, '영어 변환에는 경로·읽기 방식·건수·실패 건 여부만 지정하세요.')
        limit = body.get('limit', 20)
        if type(limit) is not int or limit < 0:
            raise ApiError(400, '처리 건수는 0(전체) 또는 양의 정수여야 합니다.')
        failed_only = body.get('failed_only', False)
        if type(failed_only) is not bool:
            raise ApiError(400, '실패한 번역만 실행 여부는 참 또는 거짓이어야 합니다.')
        if not self.db_summary()['records']:
            raise ApiError(400, NO_DATA, 'no_data')

        def work():
            self.set_stage('DB 원문 확인 중')
            with closing(self.connect()) as db:
                ensure_search_index(db, self.rules, self.schema)
                with self.llm_client() as client:
                    result = build_english(db, client, self.config | {'normalization_workers': self.normalization_workers}, self.schema, limit=limit, progress=self.set_stage, failed_only=failed_only)
            result['failures'] = [{'id': self.redact(item['id']), 'error': self.redact(item['error'])}
                                  for item in result['failures']]
            return result

        self.start_job('english', work)
        return {'accepted': True, 'kind': 'english'}

    def records(self, query):
        q = (query.get('q') or [''])[0].strip()
        if len(q) > MAX_QUERY:
            raise ApiError(400, '검색어가 너무 깁니다.')
        try:
            page = int((query.get('page') or ['1'])[0])
            size = int((query.get('page_size') or ['20'])[0])
        except ValueError as error:
            raise ApiError(400, '페이지 값이 올바르지 않습니다.') from error
        if page < 1 or page > 1_000_000 or size < 1 or size > PAGE_SIZE_MAX:
            raise ApiError(400, f'페이지는 1부터, 한 페이지는 최대 {PAGE_SIZE_MAX}건입니다.')
        summary = self.db_summary()
        if not summary['exists']:
            return {'page': page, 'page_size': size, 'total': 0, 'items': [], 'warning': NO_DATA}
        where, params = '', []
        if q:
            where = ' WHERE id LIKE ? OR instr(original_json, ?) > 0'
            params = [f'%{q}%', q]
        with closing(self.connect()) as db:
            total = db.execute('SELECT COUNT(*) FROM normalized_records' + where, params).fetchone()[0]
            rows = db.execute(
                'SELECT id, original_json, normalized_json FROM normalized_records' + where + ' ORDER BY id LIMIT ? OFFSET ?',
                (*params, size, (page - 1) * size)).fetchall()
            pipelines = {identity: self.pipeline_status(db, json.loads(raw), json.loads(proposal))
                         for identity, raw, proposal in rows}
        items = []
        for identity, raw, _proposal in rows:
            original = json.loads(raw)
            item = {'id': identity, 'pipeline': pipelines[identity],
                    'case_number': original.get(original.get('_id_column', 'id'), identity),
                    'source_sheet': original.get('_source_sheet', '')}
            for field in LIST_FIELDS:
                item[LIST_KEYS[field]] = original.get(field, '')
            if is_history(original):
                item.update(customer=original.get('Site', ''), equipment=original.get('설비', ''), date=original.get('불량발생일', ''))
            items.append(item)
        return {'page': page, 'page_size': size, 'total': total, 'query': q, 'items': items}

    def record(self, query):
        identity = (query.get('id') or [''])[0].strip()
        if not identity or len(identity) > MAX_ID:
            raise ApiError(400, '접수번호를 선택하세요.')
        if not self.db_path.exists():
            raise ApiError(400, NO_DATA, 'no_data')
        with closing(self.connect()) as db:
            row = db.execute('SELECT original_json, normalized_json FROM normalized_records WHERE id=?',
                             (identity,)).fetchone()
            if row is None:
                raise ApiError(404, '해당 접수를 찾을 수 없습니다.')
            original, proposal = json.loads(row[0]), json.loads(row[1])
            prepared = prepared_record(db, original, proposal, self.rules)
            pipeline = self.pipeline_status(db, original, proposal, prepared)
        cells = public_cells(original)
        schema, rules = row_schema(original, self.schema), row_rules(original, self.rules)
        grouped = {}
        for key in schema['columns']:
            grouped.setdefault(column_group(key), []).append(key)
        header = {field: cells.get(field, '') for field in (schema['id_column'],) + LIST_FIELDS if field in cells}
        groups = []
        for name, keys in grouped.items():
            block = []
            for key in keys:
                review = prepared['reviews'].get(key, {'action': 'pending', 'value': cells.get(key, ''), 'stale': False})
                proposed = None if proposal is None else proposal.get(key)
                item = {
                    'column': key,
                    'original': cells.get(key, ''),
                    'proposal': proposed,
                    'value': review.get('value', cells.get(key, '')),
                    'action': review.get('action', 'pending'),
                    'editable': editable(original, key, self.rules),
                    'stale': bool(review.get('stale')),
                    'fallback': prepared['fallbacks'].get(key),
                    'location': original.get('_cell_locations', {}).get(key, ''),
                    'preserve': rules.get(key, 'preserve') != 'rewrite',
                }
                if review.get('stale') and 'previous' in review:
                    item['previous'] = review['previous']
                block.append(item)
            groups.append({'name': name, 'cells': block})
        return {
            'id': prepared['id'],
            'revision': prepared['revision'],
            'source_key': original.get('_source_key', ''),
            'has_proposal': proposal is not None,
            'pipeline': pipeline,
            'header': header,
            'groups': groups,
        }

    def reviews(self, body):
        self._begin_sync()
        try:
            if not isinstance(body, dict) or body.get('format') != 'cell-reviews-v1' or not isinstance(body.get('entries'), list):
                raise ApiError(400, '올바른 검토 형식이 아닙니다.')
            entries = body['entries']
            if not entries:
                raise ApiError(400, '저장할 검토 항목이 없습니다. 변경한 셀만 저장합니다.')
            if len(entries) > MAX_ENTRIES:
                raise ApiError(400, '한 번에 저장할 검토가 너무 많습니다.')
            if not self.db_path.exists():
                raise ApiError(400, NO_DATA, 'no_data')
            with closing(self.connect()) as db:
                try:
                    apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': entries}, self.rules, schema=self.schema)
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
            identity = entries[0].get('id') if isinstance(entries[0], dict) else None
            if self.auto_english or self.auto_index_enabled():
                identities = sorted({entry['id'] for entry in entries})
                # 저장한 검토만 번역하고 같은 작업 잠금에서 벡터 갱신까지 마칩니다.
                self.start_job('auto_search', lambda: self.update_search(identities), from_sync=True)
            return {'saved': len(entries), 'id': identity}
        finally:
            self._end_sync()

    def accept_pending(self, body):
        if not isinstance(body, dict) or set(body) - {'path', 'reader'}:
            raise ApiError(400, '일괄 채택에는 현재 Excel 경로와 읽기 방식만 지정하세요.')
        if not self.db_summary()['records']:
            raise ApiError(400, NO_DATA, 'no_data')

        def work():
            self.set_stage('DB 원문 확인 중')
            with closing(self.connect()) as db, db:
                # 확인한 DB와 저장할 DB가 다른 작업에 의해 바뀌지 않도록 쓰기 잠금을 잡습니다.
                db.execute('BEGIN IMMEDIATE')
                def progress(checked, total):
                    self.set_stage(f'미검토 제안 확인 {checked}/{total}건' if checked < total else '검토 저장 및 검색 반영 중')
                changed = set()
                result = accept_pending_reviews(db, self.rules, schema=self.schema, progress=progress, changed_ids=changed)
                result['_review_ids'] = sorted(changed)
            result['failures'] = [{'id': self.redact(item['id']), 'error': self.redact(item['error'])}
                                  for item in result['failures']]
            return result

        self.start_job('bulk_review', work)
        return {'accepted': True, 'kind': 'bulk_review'}

    def header_mappings(self, body):
        self._begin_sync()
        try:
            aliases = body.get('aliases') if isinstance(body, dict) else None
            if not isinstance(aliases, dict):
                raise ApiError(400, '열 이름 대응은 객체로 보내 주세요.')
            try:
                validated = validate_header_aliases(
                    aliases, self.schema['columns'], self.schema['id_column'])
                save_header_aliases(
                    self.output / 'column_aliases.json', validated,
                    self.schema['columns'], self.schema['id_column'],
                    self.schema.get('name'))
            except ValueError as error:
                raise ApiError(400, self.redact(error)) from error
            except OSError as error:
                raise ApiError(500, '열 이름 대응을 저장하지 못했습니다.') from error
            self.header_aliases = validated
            return {
                'saved': True,
                'aliases': dict(self.header_aliases),
                'schema_columns': list(self.schema['columns']),
                'id_column': self.schema['id_column'],
            }
        finally:
            self._end_sync()

    def guard(self, handler, mutating):
        if not host_allowed(handler, self.port, self.allowed_hosts):
            raise ApiError(403, '허용되지 않은 호스트입니다.')
        token = self.session(handler)
        header = handler.headers.get('X-CSRF-Token', '')
        cookie = cookies(handler).get('csrf_token', '')
        if not hmac.compare_digest(header, token) or not hmac.compare_digest(cookie, token):
            raise ApiError(403, '세션이 유효하지 않습니다. 이 프로그램이 연 주소에서 다시 여세요.')
        if mutating and not origin_of(handler, self.port, self.allowed_hosts):
            raise ApiError(403, '허용되지 않은 요청 출처입니다.')
        return token

    def read_json(self, handler):
        length = handler.headers.get('Content-Length')
        try:
            size = int(length)
        except (TypeError, ValueError) as error:
            raise ApiError(411, '요청 크기를 확인할 수 없습니다.') from error
        if size < 0 or size > MAX_BODY:
            raise ApiError(413, '요청이 너무 큽니다.')
        ctype = handler.headers.get('Content-Type', '').split(';')[0].strip().lower()
        if ctype != 'application/json':
            raise ApiError(415, 'JSON 요청만 받습니다.')
        raw = handler.rfile.read(size)
        try:
            body = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(400, 'JSON을 읽지 못했습니다.') from error
        if not isinstance(body, dict):
            raise ApiError(400, 'JSON 객체가 필요합니다.')
        return body

    def handle(self, handler, method):
        parsed = urlparse(handler.path)
        path = parsed.path
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        try:
            if '..' in path or '\\' in path or '\x00' in path:
                raise ApiError(404, '페이지를 찾을 수 없습니다.')
            if not host_allowed(handler, self.port, self.allowed_hosts):
                raise ApiError(403, '허용되지 않은 호스트입니다.')
            if path in STATIC:
                if method != 'GET':
                    raise ApiError(405, '허용되지 않은 방법입니다.')
                self.send_asset(handler, path)
                return
            if path not in GET_API and path not in POST_API:
                raise ApiError(404, '지원하지 않는 요청입니다.')
            owner = self.guard(handler, mutating=method != 'GET')
            query = parse_qs(parsed.query)
            if method == 'GET' and path in GET_API:
                if path == '/api/state':
                    send_json(handler, 200, self.state(owner))
                elif path == '/api/job':
                    send_json(handler, 200, self.job_snapshot(owner))
                elif path == '/api/llm-messages':
                    if set(query) - {'id'} or len(query.get('id', [])) > 1:
                        raise ApiError(400, '요청 번호만 지정하세요.')
                    data = self.llm_trace.read(query.get('id', [None])[0], owner=hashlib.sha256(owner.encode()).hexdigest())
                    if data is None:
                        raise ApiError(404, '최근 기록에 없는 요청입니다.')
                    send_json(handler, 200, data)
                elif path == '/api/records':
                    send_json(handler, 200, self.records(query))
                else:
                    send_json(handler, 200, self.record(query))
            elif method == 'POST' and path in POST_API:
                body = self.read_json(handler)
                if path == '/api/inspect':
                    send_json(handler, 200, self.inspect(body))
                elif path == '/api/import':
                    send_json(handler, 202, self.import_input(body))
                elif path == '/api/import-many':
                    send_json(handler, 202, self.import_many(body))
                elif path == '/api/convert':
                    send_json(handler, 202, self.convert(body))
                elif path == '/api/normalize':
                    send_json(handler, 202, self.normalize(body))
                elif path == '/api/overnight':
                    send_json(handler, 202, self.start_overnight(body))
                elif path == '/api/overnight/stop':
                    send_json(handler, 200, self.stop_overnight(body))
                elif path == '/api/english':
                    send_json(handler, 202, self.translate_english(body))
                elif path == '/api/vectors':
                    send_json(handler, 202, self.vectors(body))
                elif path in ('/api/ask', '/api/compare-search'):
                    send_json(handler, 202, self.ask(body, comparison=path == '/api/compare-search', owner=owner))
                elif path == '/api/reviews':
                    send_json(handler, 200, self.reviews(body))
                elif path == '/api/reviews/accept-pending':
                    send_json(handler, 202, self.accept_pending(body))
                elif path == '/api/settings':
                    if set(body) == {'reference_limit'}:
                        limit = body['reference_limit']
                        if type(limit) is not int or not 0 <= limit <= 100:
                            raise ApiError(400, '참고 사례 수는 0(자동) 또는 1~100 사이 정수여야 합니다.')
                        with self.lock:
                            self.sessions[owner]['reference_limit'] = limit
                        send_json(handler, 200, {'reference_limit': limit})
                    else:
                        send_json(handler, 200, self.update_settings(body))
                else:
                    send_json(handler, 200, self.header_mappings(body))
            else:
                raise ApiError(405, '허용되지 않은 방법입니다.')
        except ApiError as error:
            payload = {'error': error.message, 'code': error.code}
            if error.extra:
                payload.update(error.extra)
            send_json(handler, error.status, payload)
        except Exception as error:
            message, code = self.public_error(error)
            send_json(handler, 500, {'error': message, 'code': code})

    def send_asset(self, handler, path):
        name, content_type = STATIC[path]
        data = (ASSETS / name).read_bytes()
        extra = {}
        if name == 'app.html':
            token = self.session(handler, create=True)
            html = data.decode('utf-8').replace('__SESSION_TOKEN__', token)
            data = html.encode('utf-8')
            extra['Set-Cookie'] = f'csrf_token={token}; Path=/; SameSite=Strict; HttpOnly'
        handler.send_response(200)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Content-Length', str(len(data)))
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('X-Frame-Options', 'DENY')
        handler.send_header('Content-Security-Policy', "frame-ancestors 'none'")
        for key, value in extra.items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(data)


def pack_answer(question, reply, hits, aggregate, details, elapsed, code='ok'):
    sources = {section['id']: section for hit in (hits or []) for section in hit.get('evidence', [])}
    seen = set()
    def pack_claims(claims):
        packed = []
        for claim in claims:
            items = []
            for evidence in claim.get('evidence', []):
                source = sources.get(evidence['id'], {})
                receipt, _, work = evidence['id'].partition('/')
                items.append({
                    'id': evidence['id'], 'receipt': receipt,
                    'work': work or source.get('kind', ''), 'column': evidence['column'],
                    'quote': evidence.get('quote', ''),
                    'location': (source.get('locations') or {}).get(evidence['column'], ''),
                    'action_outcomes': [step for step in source.get('action_outcomes', [])
                                        if step['column'] == evidence['column']],
                })
            packed.append({'text': claim.get('text', ''), 'evidence': items})
        return packed
    claims = pack_claims((details or {}).get('claims', []))
    seen.update((e['id'], e['column']) for claim in claims for e in claim['evidence'])
    supplements = []
    for item in pack_claims((details or {}).get('source_supplements', [])):
        additional = []
        for evidence in item['evidence']:
            key = (evidence['id'], evidence['column'])
            if key not in seen:
                seen.add(key)
                additional.append(evidence)
        if additional:
            supplements.append({'evidence': additional})
    ok = code in ('ok', 'aggregate', 'no_match', 'insufficient_evidence')
    return {
        'question': question,
        'status': code,
        'answer': (details or {}).get('answer_text', reply) if code == 'ok' else reply,
        'elapsed_seconds': round(elapsed, 1),
        'aggregate': aggregate,
        'claims': claims,
        'source_supplements': supplements,
        'note': (details or {}).get('note'),
        'validation_events': [{k: v for k, v in event.items() if k not in ('response', 'response_text')}
                              for event in (details or {}).get('validation_events', [])],
        'search': (details or {}).get('search'),
        'vector_search': (details or {}).get('vector_search'),
        'references': (details or {}).get('references'),
        'timings': (details or {}).get('timings', {}),
        'source_db': (details or {}).get('source_db', False),
        'index_rebuilt': (details or {}).get('index_rebuilt', False),
        'comparison': (details or {}).get('comparison', False),
        'error': None if ok else reply,
        'code': code,
    }


def send_json(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.send_header('X-Content-Type-Options', 'nosniff')
    handler.send_header('X-Frame-Options', 'DENY')
    handler.send_header('Content-Security-Policy', "frame-ancestors 'none'")
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        pass


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            app.handle(self, 'GET')

        def do_POST(self):
            app.handle(self, 'POST')

        def log_message(self, fmt, *args):
            sys.stderr.write('[ui] %s\n' % (fmt % args))

    return Handler


def serve(root, *, config_path, output_dir, schema_path, port=8765, excel=None, reader='openpyxl',
          open_browser=True, host=None):
    app = App(root, config_path, output_dir, schema_path, excel=excel, reader=reader)
    host = host or app.config.get('web_host', '127.0.0.1')
    try:
        address = IPv4Address(host)
        if address.is_unspecified or address.is_multicast:
            raise ValueError()
    except (ValueError, TypeError) as error:
        raise ValueError('웹 접속 주소에는 127.0.0.1 또는 서버 PC의 구체적인 IPv4 주소를 지정하세요.') from error
    app.allowed_hosts.add(str(address))
    server = ThreadingHTTPServer((str(address), port), make_handler(app))
    server.allow_reuse_address = True
    app.port = server.server_address[1]
    url = f'http://{address}:{app.port}/'
    print(f'CS RAG UI: {url}', flush=True)
    print(f'출력 폴더: {Path(output_dir).resolve()}', flush=True)
    print('종료하려면 Ctrl+C를 누르세요.', flush=True)
    app.start_embedding()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nUI를 종료합니다.', flush=True)
    finally:
        server.server_close()
