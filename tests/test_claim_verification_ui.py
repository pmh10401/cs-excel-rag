"""영어 근거 검사의 설정·준비·답변 결과를 실제 HTTP와 모의 모델로 확인합니다."""
from contextlib import closing
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ragkit import records, storage, web, nli, claim_verification as guard

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'cslist_schema.json').exists():
    ROOT /= 'windows-demo'


class ClaimVerificationUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.config = self.folder / 'config.json'
        self.config.write_text(json.dumps({'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'EMPTY',
            'model': 'mock', 'reasoning_effort': 'high', 'timeout_seconds': 1}))
        self.app = web.App(ROOT, self.config, self.folder / 'out', ROOT / 'cslist_schema.json')
        rows = records.read_excel(ROOT / 'cslist_sample.xlsx', 'CSList', required_columns=self.app.schema['columns'],
                                  id_column=self.app.schema['id_column'])[:1]
        rows[0].update({'현상': '펌프 누수', '조치내용(1)': 'Seal replacement is planned.'})
        with closing(self.app.connect()) as db:
            storage.ingest(db, rows, self.app.rules, schema=self.app.schema)
        self.app.auto_english = False
        self.server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.make_handler(self.app))
        self.app.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.temp.cleanup()

    def request(self, path, body=None, *, auth=True, status=200):
        headers = {'Origin': f'http://127.0.0.1:{self.app.port}', 'Content-Type': 'application/json'}
        if auth:
            headers.update({'X-CSRF-Token': self.app.token, 'Cookie': f'csrf_token={self.app.token}'})
        connection = http.client.HTTPConnection('127.0.0.1', self.app.port, timeout=10)
        try:
            connection.request('POST' if body is not None else 'GET', path,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            result = json.loads(response.read())
            self.assertEqual(response.status, status, result)
            return result
        finally:
            connection.close()

    def completed(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = self.request('/api/job')
            if result['status'] in ('done', 'error'):
                return result
            time.sleep(.01)
        self.fail('HTTP 작업이 제한 시간에 끝나지 않았습니다.')

    def model_client(self, **options):
        def create(**request):
            sources = json.loads(request['messages'][1]['content'].split('원문 자료:\n')[1])
            source = next(s for s in sources if s['cells'].get('조치내용(1)'))
            payload = {'claims': [{'text': 'Seal replacement is planned.',
                'evidence': [{'id': source['id'], 'column': '조치내용(1)'}]}]}
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(payload)))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)

    def test_mode_is_validated_saved_and_restored(self):
        self.request('/api/settings', {'claim_guard_mode': 'typo'}, status=400)
        self.request('/api/settings', {'claim_guard_mode': 'observe'})
        self.assertEqual(self.request('/api/state')['settings']['claim_guard_mode'], 'observe')
        reopened = web.App(ROOT, self.config, self.folder / 'out', ROOT / 'cslist_schema.json')
        self.assertEqual(reopened.claim_guard_mode, 'observe')

    def test_prepare_requires_csrf_and_strict_input_and_refreshes_status(self):
        self.request('/api/claim-guard/prepare', {'download': True}, auth=False, status=403)
        self.request('/api/claim-guard/prepare', {'download': 'yes'}, status=400)
        with patch.object(nli, 'prepare_model', return_value={'ready': True, 'loaded': True}) as preparing:
            self.request('/api/claim-guard/prepare', {'download': False}, status=202)
            job = self.completed()
        self.assertEqual(job['kind'], 'claim_guard_prepare')
        self.assertEqual(job['status'], 'done')
        preparing.assert_called_once_with(download=False)

    def test_enforce_unavailable_is_error_without_public_answer(self):
        self.request('/api/settings', {'claim_guard_mode': 'enforce'})
        error = {'mode': 'enforce', 'status': 'error', 'claims': [], 'code': 'verification_unavailable',
                 'message': '선택 설치 필요'}
        with patch.object(web, 'OpenAI', side_effect=self.model_client), patch.object(guard, 'verify_claims', return_value=error):
            self.request('/api/ask', {'question': '펌프 누수 조치', 'search_mode': 'keyword'}, status=202)
            job = self.completed()
        self.assertEqual(job['status'], 'error')
        self.assertEqual(job['code'], 'verification_unavailable')
        self.assertEqual(job['result']['claims'], [])
        self.assertEqual(job['result']['claim_verification']['status'], 'error')

    def test_observe_unsupported_remains_visible_with_warning(self):
        self.request('/api/settings', {'claim_guard_mode': 'observe'})
        result = {'mode': 'observe', 'status': 'unsupported', 'claims': [], 'message': '지지 부족'}
        with patch.object(web, 'OpenAI', side_effect=self.model_client), patch.object(guard, 'verify_claims', return_value=result):
            self.request('/api/ask', {'question': '펌프 누수 조치', 'search_mode': 'keyword'}, status=202)
            job = self.completed()
        self.assertEqual(job['status'], 'done')
        self.assertEqual(job['result']['claim_verification']['status'], 'unsupported')
        self.assertTrue(job['result']['claims'])
        self.assertIn('기록 모드', job['result']['note'])


if __name__ == '__main__':
    unittest.main()
