"""로컬 요청·응답 기록. DB와 분리하고 병렬 요청마다 고유 번호를 붙입니다."""
from collections import deque
from datetime import datetime, timezone
import json
import threading
import time
import uuid


class LLMTrace:
    def __init__(self, output, config):
        self.path = output / 'llm_messages.jsonl'
        self.keys = [key for key in (config.get('api_key'), (config.get('embedding') or {}).get('api_key')) if key]
        self.lock = threading.Lock()
        self.entries = deque(maxlen=30)
        self.warning = ''
        for path in (self.path.with_suffix('.jsonl.1'), self.path):
            if path.exists():
                try:
                    with path.open(encoding='utf-8') as stream:
                        for line in stream:
                            try:
                                entry = json.loads(self.clean(line))
                                if isinstance(entry, dict) and all(key in entry for key in ('id', 'time', 'status', 'request', 'response')):
                                    self.entries.append(entry)
                            except (ValueError, TypeError):
                                continue
                except OSError:
                    self.warning = '이전 요청·응답 기록을 읽지 못했습니다.'

    def clean(self, text):
        for key in self.keys:
            text = text.replace(json.dumps(key, ensure_ascii=False)[1:-1], '***').replace(key, '***')
        return text

    def capture(self, create, kind, request, *, owner=None):
        request_text = self.clean(json.dumps({key: request[key] for key in
            ('model', 'messages', 'extra_body', 'max_tokens', 'temperature', 'top_p', 'response_format') if key in request},
            ensure_ascii=False, indent=2, default=str))
        entry = {'id': uuid.uuid4().hex, 'time': datetime.now(timezone.utc).isoformat(),
                 'kind': kind or 'llm', 'model': self.clean(str(request.get('model', ''))), 'status': 'running',
                 'request': request_text[:100000], 'response': '', 'seconds': None,
                 'request_truncated': len(request_text) > 100000, 'response_truncated': False, 'owner': owner}
        with self.lock:
            self.entries.append(entry)
        started = time.monotonic()
        try:
            result = create(**request)
            # SDK의 전체 응답에는 최종 문장뿐 아니라 서버가 제공하는 추론·사용량도 들어 있습니다.
            try:
                value = result.model_dump() if hasattr(result, 'model_dump') else result
                encoded = self.clean(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: vars(v)))
            except (TypeError, ValueError, AttributeError):
                encoded = '응답을 받았으나 기록용 JSON으로 변환하지 못했습니다.'
            with self.lock:
                entry.update(status='done', response=encoded[:200000], response_truncated=len(encoded) > 200000)
            return result
        except Exception as error:
            message = self.clean(type(error).__name__ + ': ' + str(error))
            with self.lock:
                entry.update(status='error', response=message[:200000], response_truncated=len(message) > 200000)
            raise
        finally:
            with self.lock:
                entry['seconds'] = round(time.monotonic()-started, 3)
                try:
                    self.entries.remove(entry)
                except ValueError:
                    pass
                self.entries.append(entry)
                try:
                    # 최근 파일 2개만 보관합니다. 모델 요청과 달리 기록 실패는 작업을 실패시키지 않습니다.
                    if self.path.exists() and self.path.stat().st_size >= 10_000_000:
                        self.path.replace(self.path.with_suffix('.jsonl.1'))
                    with self.path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                except OSError:
                    self.warning = '요청·응답 파일 저장에 실패했습니다. 현재 실행의 최근 기록만 볼 수 있습니다.'

    def read(self, identity=None, *, owner=None):
        with self.lock:
            entries = [entry for entry in self.entries if owner is None or entry.get('owner') == owner
                       or (entry.get('owner') is None and entry.get('kind') != 'ask')]
            if identity is not None:
                return next(({k: v for k, v in entry.items() if k != 'owner'} for entry in entries if entry['id'] == identity), None)
            return {'items': [{key: value for key, value in entry.items() if key not in ('request', 'response', 'owner')}
                              for entry in reversed(entries)], 'warning': self.warning}
