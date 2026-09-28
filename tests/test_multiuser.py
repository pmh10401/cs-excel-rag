"""독립 브라우저 세션의 실제 HTTP 질문 대기열·결과·로그 격리를 검사합니다."""
from contextlib import closing
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPCookieProcessor, Request

from ragkit import records, storage, web, llm


def until(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('질문 처리 상태가 제한 시간 내에 바뀌지 않았습니다.')


def main(root):
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        cfg = folder / 'config.json'
        cfg.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY',
            'model': 'test', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
        excel = root / 'cslist_sample.xlsx'
        app = web.App(root, cfg, folder / 'out', root / 'cslist_schema.json', excel=excel)
        rows = records.read_excel(excel, 'CSList', required_columns=app.schema['columns'], id_column=app.schema['id_column'])
        with closing(app.connect()) as db:
            storage.ingest(db, rows, app.rules, schema=app.schema)
        server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.make_handler(app))
        app.port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f'http://127.0.0.1:{app.port}'
        entered = {key: threading.Event() for key in 'ABC'}
        release = {key: threading.Event() for key in 'ABC'}
        calls = []

        class Browser:
            def __init__(self):
                self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
                self.token = ''
                self.reload()

            def request(self, path, body=None, *, status=200, extra=None):
                headers = {'X-CSRF-Token': self.token, 'Origin': origin, 'Content-Type': 'application/json'}
                headers.update(extra or {})
                req = Request(origin + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
                try:
                    response = self.opener.open(req, timeout=10)
                except HTTPError as error:
                    response = error
                with response:
                    data = response.read().decode()
                    assert response.code == status, (path, response.code, data)
                    return json.loads(data) if response.headers.get_content_type() == 'application/json' else data

            def reload(self):
                html = self.request('/')
                self.token = re.search(r'name="csrf-token" content="([^"]+)"', html)[1]

        def create(**request):
            content = request['messages'][1]['content']
            key = next(key for key in 'ABC' if f'사용자{key}' in content)
            calls.append(key)
            entered[key].set()
            assert release[key].wait(10), '모의 LLM 해제 시간 초과'
            if key == 'B':
                raise RuntimeError('사용자B 전용 실패')
            sources = json.loads(content.split('원문 자료:\n')[1])
            source = next(s for s in sources if s['cells'].get('조치내용(1)'))
            data = {'claims': [{'text': source['cells']['조치내용(1)'],
                'evidence': [{'id': source['id'], 'column': '조치내용(1)'}]}]}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))])

        def client(**options):
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)

        def completed(browser):
            job = browser.request('/api/job')
            return job if job['status'] in ('done', 'error') else None

        try:
            with patch.object(web, 'OpenAI', side_effect=client):
                a, b, c = Browser(), Browser(), Browser()
                assert len({a.token, b.token, c.token, app.token}) == 4
                question = lambda key: {'question': f'사용자{key}: 펌프 누수 조치', 'search_mode': 'keyword'}
                a.request('/api/ask', question('A'), status=202)
                assert entered['A'].wait(5)
                assert b.request('/api/job')['status'] == 'idle'
                b.request('/api/settings', {'reference_limit': 7})
                assert b.request('/api/state')['settings']['reference_limit'] == 7
                assert a.request('/api/state')['settings']['reference_limit'] == 3
                b.request('/api/ask', question('B'), status=202)
                c.request('/api/ask', question('C'), status=202)
                assert b.request('/api/job')['queue_position'] == 1
                assert c.request('/api/job')['queue_position'] == 2
                assert c.request('/api/job')['result'] is None
                a.request('/api/ask', question('A'), status=409)
                a.request('/api/conversation/reset', {}, status=409)
                b.request('/api/ask', question('B'), status=409)
                c.request('/api/settings', {'normalization_workers': 2}, status=409)
                c.request('/api/inspect', {'path': str(excel)}, status=409)
                c.request('/api/state', extra={'X-CSRF-Token': a.token}, status=403)
                c.request('/api/settings', {'reference_limit': 4}, extra={'Origin': 'http://evil.test'}, status=403)
                token = b.token
                b.reload()
                assert b.token == token and b.request('/api/job')['status'] == 'queued'
                assert b.request('/api/job')['question'] == question('B')['question']
                assert calls == ['A']
                release['A'].set()
                assert entered['B'].wait(5)
                done_a = until(lambda: completed(a))
                assert done_a['status'] == 'done' and '사용자A' in done_a['result']['question']
                assert c.request('/api/job')['queue_position'] == 1
                assert '사용자A' not in json.dumps(b.request('/api/state'), ensure_ascii=False)
                logs_a = a.request('/api/llm-messages')['items']
                assert len(logs_a) == 1 and 'owner' not in logs_a[0]
                identity = logs_a[0]['id']
                detail = a.request('/api/llm-messages?id=' + identity)
                assert '사용자A' in detail['request'] and 'owner' not in detail
                b.request('/api/llm-messages?id=' + identity, status=404)
                assert all(item['id'] != identity for item in b.request('/api/llm-messages')['items'])
                release['B'].set()
                assert entered['C'].wait(5)
                failed_b = until(lambda: completed(b))
                assert failed_b['status'] == 'error'
                assert '사용자B' not in json.dumps(a.request('/api/job'), ensure_ascii=False)
                release['C'].set()
                assert until(lambda: completed(c))['status'] == 'done'
                until(lambda: not app.busy())
                assert calls == ['A', 'B', 'C'], calls
                a.reload()
                assert a.request('/api/job')['result'] == done_a['result']
                assert a.token not in app.llm_trace.path.read_text()
                # 동기식 입력 검사도 마지막 질문을 다른 브라우저에 노출하면 안 됩니다.
                app._begin_sync()
                try:
                    during_sync = a.request('/api/job')
                    assert during_sync['maintenance_running']
                    assert during_sync['result'] == done_a['result']
                finally:
                    app._end_sync()
                # 예전 소유자 없는 질문 기록은 숨기고 공유 자료 작업 로그만 공개합니다.
                app.llm_trace.capture(lambda **kw: {}, 'ask', {'messages': []})
                legacy_id = app.llm_trace.read()['items'][0]['id']
                a.request('/api/llm-messages?id=' + legacy_id, status=404)
                app.llm_trace.capture(lambda **kw: {}, 'normalize', {'messages': []})
                shared_id = app.llm_trace.read()['items'][0]['id']
                c.request('/api/llm-messages?id=' + shared_id)

                maintenance_entered, maintenance_release = threading.Event(), threading.Event()
                def maintenance():
                    maintenance_entered.set()
                    assert maintenance_release.wait(10)
                    return {'maintenance': True}
                try:
                    app.start_job('normalize', maintenance)
                    assert maintenance_entered.wait(5)
                    assert a.request('/api/job')['maintenance_running']
                    a.request('/api/ask', question('A'), status=409)
                finally:
                    maintenance_release.set()
                    until(lambda: not app.busy())
                assert a.request('/api/job')['kind'] == 'normalize'

                # 명시적으로 허용한 LAN Host/Origin만 통과합니다. 실제 LAN 연결은 별도 확인 대상입니다.
                app.allowed_hosts.add('192.168.0.50')
                lan = {'Host': f'192.168.0.50:{app.port}', 'Origin': f'http://192.168.0.50:{app.port}'}
                a.request('/api/settings', {'reference_limit': 2}, extra=lan)
                a.request('/api/state', extra={'Host': f'192.168.0.51:{app.port}'}, status=403)
                a.request('/api/settings', {'reference_limit': 2}, extra=lan | {'Origin': 'http://192.168.0.51'}, status=403)

                # 서버에 보관한 자신의 대화만 사용해 질문을 풀고, 같은 DB를 다시 검색합니다.
                history = a.request('/api/job')['conversation']
                assert len(history) == 1 and history[0]['question'] == question('A')['question']
                assert b.request('/api/job')['conversation'] == []  # 실패 응답은 문맥에 넣지 않습니다.
                b.request('/api/ask', {'question': '그 조치는?', 'follow_up': True}, status=400)
                a.request('/api/ask', {'question': '조치는?', 'instruction_language': 'other'}, status=400)
                a.request('/api/ask', {'question': '조치는?', 'follow_up': 'true'}, status=400)
                a.request('/api/compare-search', {'question': '조치는?', 'follow_up': True}, status=400)
                receipt = history[0]['answer_claims'][0]['receipts'][0]
                resolved_question = receipt + '의 조치내용(1)은?'
                def follow_response(**request):
                    prompt = request['messages'][0]['content']
                    user = request['messages'][1]['content']
                    if prompt == llm.FOLLOW_UP_PROMPTS['en']:
                        payload = json.loads(user)
                        assert len(payload['history']) <= 3
                        assert '사용자C' not in user and '사용자B' not in user
                        assert 'PRIVATE-CLIENT-HISTORY' not in user
                        data = ({'clarification': '어떤 사례를 뜻하나요?'} if payload['question'] == '모호함'
                                else {'question': resolved_question})
                    else:
                        assert prompt == llm.answer_prompt('en')
                        assert user.startswith('질문: ' + resolved_question + '\n')
                        assert '"history"' not in user
                        sources = json.loads(user.split('원문 자료:\n')[1])
                        source = next(s for s in sources if s['cells'].get('조치내용(1)'))
                        data = {'claims': [{'text': source['cells']['조치내용(1)'],
                            'evidence': [{'id': source['id'], 'column': '조치내용(1)'}]}]}
                    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                        message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))])
                follow = {'question': '첫 번째 사례의 조치는?', 'follow_up': True, 'instruction_language': 'en',
                          'reference_limit': 0, 'history': 'PRIVATE-CLIENT-HISTORY'}
                follow_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=follow_response)), close=lambda: None)
                with patch.object(web, 'OpenAI', return_value=follow_client), patch.object(web, 'retrieve', wraps=web.retrieve) as retrieval:
                    for _ in range(5):
                        a.request('/api/ask', follow, status=202)
                        assert until(lambda: completed(a))['status'] == 'done'
                        until(lambda: not app.busy())
                    result = a.request('/api/job')['result']
                    assert result['question'] == follow['question'] and result['resolved_question'] == resolved_question
                    assert result['instruction_language'] == 'en' and result['follow_up']
                    assert retrieval.call_count == 5 and retrieval.call_args.args[1] == resolved_question
                    assert len(a.request('/api/job')['conversation']) == 5
                    assert len(c.request('/api/job')['conversation']) == 1
                    before = a.request('/api/job')['conversation']
                    a.request('/api/ask', follow | {'question': '모호함'}, status=202)
                    assert until(lambda: completed(a))['result']['status'] == 'clarification'
                    until(lambda: not app.busy())
                    assert retrieval.call_count == 5 and a.request('/api/job')['conversation'] == before
                a.reload()
                assert a.request('/api/job')['conversation'] == before
                a.request('/api/conversation/reset', {}, extra={'Origin': 'http://evil.test'}, status=403)
                a.request('/api/conversation/reset', {'unexpected': True}, status=400)
                a.request('/api/conversation/reset', {})
                assert a.request('/api/job')['conversation'] == [] and a.request('/api/job')['status'] == 'idle'
                assert len(c.request('/api/job')['conversation']) == 1

                # 작업자가 시작하기 직전에도 대기 상태를 표시하고 50건에서 접수를 제한합니다.
                with patch.object(threading.Thread, 'start'):
                    for index in range(50):
                        owner = f'queue-{index}'
                        app.sessions[owner] = {'seen': time.monotonic(), 'job': app._idle_job(), 'reference_limit': 3}
                        app.queue_question(owner, lambda: {})
                    assert app.job_snapshot('queue-0')['status'] == 'queued'
                    try:
                        app.queue_question(a.token, lambda: {})
                    except web.ApiError as error:
                        assert error.status == 429
                    else:
                        raise AssertionError('대기열 한도 초과를 허용했습니다.')
                app.run_questions()
                assert not app.busy()
        finally:
            for event in release.values():
                event.set()
            until(lambda: not app.busy())
            server.shutdown()
            server.server_close()
            thread.join(5)
    print('PASS: 다중 세션 FIFO·실패 후 계속·새로고침·답변/로그 격리·대기 한도·자료 잠금·LAN 출처 검사')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
