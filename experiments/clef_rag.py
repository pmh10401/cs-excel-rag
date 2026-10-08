"""기존 DB를 읽기 전용으로 사용하는 독립 Clef 테스트 웹 화면입니다."""
import argparse
from contextlib import closing, contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import sqlite3
import subprocess
from urllib.parse import urlparse

from ragkit import web
from ragkit.clef import options
from ragkit.cslist import ensure_search_index


class ReadOnlyApp(web.App):
    def connect(self):
        return sqlite3.connect(self.db_path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)

    def guard(self, handler, mutating):
        owner = super().guard(handler, mutating)
        if mutating and urlparse(handler.path).path not in ('/api/ask', '/api/compare-search', '/api/settings'):
            raise web.ApiError(403, 'Clef 테스트 화면은 질문·검색 비교만 허용합니다. 원문 DB는 읽기 전용입니다.')
        return owner

    def update_settings(self, body):
        if not isinstance(body, dict) or set(body) != {'reference_limit'}:
            raise web.ApiError(403, '테스트에서는 참고 사례 수만 저장할 수 있습니다.')
        return super().update_settings(body)

    def ask(self, body, **kwargs):
        # ponytail: 테스트는 새 질문만 받습니다. LLM을 먼저 적재하는 번역/후속 질문은 메모리 검증 후 확장합니다.
        if isinstance(body, dict) and (body.get('bilingual') or body.get('follow_up')):
            raise web.ApiError(400, 'Clef 테스트는 새 질문만 지원합니다. 영어 번역 질문 함께 사용을 끄세요.')
        return super().ask(body, **kwargs)

    @contextmanager
    def llm_client(self):
        # 테스트 별칭만 관리합니다. Clef와 답변 모델을 함께 상주시켜 메모리를 압박하지 않습니다.
        try:
            if getattr(self, 'lms', None):
                self.set_stage('테스트 답변 모델 적재 중 · 16K 문맥')
                subprocess.run([self.lms, 'load', self.answer_source_model, '--context-length', '16384',
                                '--parallel', '1', '--identifier', 'clef-rag-answer', '--yes'],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
            with super().llm_client() as client:
                yield client
        finally:
            if getattr(self, 'lms', None):
                cleanup = subprocess.run([self.lms, 'unload', 'clef-rag-answer'], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
                if cleanup.returncode:
                    print('테스트 답변 별칭 해제를 확인하지 못했습니다. lms ps로 확인하세요.', flush=True)


def run(args):
    if not args.db.is_file():
        raise ValueError('등록된 원문 DB 파일을 지정하세요.')
    if args.output.resolve() == args.db.resolve().parent:
        raise ValueError('테스트 결과 폴더는 원문 DB 폴더와 분리하세요.')
    app = ReadOnlyApp(Path('windows-demo'), args.config, args.output, args.schema)
    app.db_path = args.db.absolute()
    app.config['clef'] = options({'base_url': args.clef_url, 'model': args.clef_model,
                                 'candidates': args.candidates, 'timeout_seconds': args.timeout})
    app.clef_enabled = True
    if args.answer_model:
        app.config['model'] = args.answer_model
    else:
        app.lms = shutil.which('lms')
        if app.lms:
            app.answer_source_model = app.config['model']
            app.config['model'] = 'clef-rag-answer'
    # 외부 서버에 원문을 보내지 않습니다. 기존 파일 설정을 수정하지 않고 이 프로세스에만 적용합니다.
    for cfg in (app.config, app.config.get('embedding') or {}):
        if cfg.get('base_url') and urlparse(cfg['base_url']).hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Clef 테스트의 답변·임베딩 서버도 로컬 주소여야 합니다.')
    if app.config.get('embedding'):
        app.config['embedding'] = app.config['embedding'] | {'auto_start': False, 'auto_index': False}
    with closing(app.connect()) as db:
        ensure_search_index(db, app.rules, app.schema)  # 구형 색인은 쓰기를 시도하면 RO 연결이 거절합니다.
    assets = args.output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for name, _ in set(web.STATIC.values()):
        shutil.copy2(web.ASSETS / name, assets / name)
    html = assets / 'app.html'
    html.write_text(html.read_text().replace('CS RAG', 'Clef 테스트 RAG').replace('id="bilingual" checked', 'id="bilingual"').replace(
        '<h1>질문</h1>', '<h1>Clef 테스트 질문</h1><p class="muted">원문 DB는 읽기 전용입니다. '
        'Clef는 후보 관련성만 재정렬하고 기존 LLM과 인용 검사가 답변을 처리합니다. 실제 업무 품질은 미검증입니다.</p>'))
    with (assets / 'app.css').open('a') as f:
        f.write('\n.nav-btn[data-nav="source"],.nav-btn[data-nav="review"],#open-settings,#follow-up-btn,'
                'label:has(#bilingual),label:has(#bilingual)+p{display:none}\n')
    with (assets / 'app.js').open('a') as f:
        f.write('\ndocument.querySelector(\'.nav-btn[data-nav="ask"]\').click();\n')
    web.ASSETS = assets  # 독립 테스트 프로세스의 화면 복사본만 사용합니다.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), web.make_handler(app))
    app.port = server.server_address[1]
    print(f'Clef 테스트 RAG: http://127.0.0.1:{app.port}/ · 원문 DB 읽기 전용', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('outputs/mac-cslist-ui/demo.sqlite3'))
    parser.add_argument('--config', type=Path, default=Path('mac_config.json'))
    parser.add_argument('--schema', type=Path, default=Path('windows-demo/cslist_schema.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/clef-rag-20261008/ui'))
    parser.add_argument('--clef-url', default='http://127.0.0.1:11436')
    parser.add_argument('--clef-model', default='clef:flash')
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--answer-model', help='별도로 적재한 테스트 답변 모델의 LM Studio 식별자')
    parser.add_argument('--port', type=int, default=8780)
    run(parser.parse_args())
