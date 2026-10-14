"""격리 합성 실험 전용 AGY 어댑터. 운영 클라이언트와 설정에는 연결하지 않습니다."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace

MODEL = 'gemini-3.8-flash-high'
PROFILE = '(version 1) (allow default) (deny file-read* (subpath "/Users/mac/Documents") (subpath "/Users/mac/Library/CloudStorage") (subpath "/Users/mac/.codex"))'
EVIDENCE = {'type': 'object', 'properties': {k: {'type': 'string'} for k in ('id', 'column', 'quote')},
            'required': ['id', 'column'], 'additionalProperties': False}
CLAIM = {'type': 'object', 'properties': {'text': {'type': 'string'}, 'evidence': {'type': 'array', 'minItems': 1, 'items': EVIDENCE}},
         'required': ['text', 'evidence'], 'additionalProperties': False}
ANSWER_SCHEMA = {'type':'object','properties':{
    'claims':{'type':'array','maxItems':5,'items':CLAIM},
    'status':{'type':'string','enum':['insufficient_evidence']}},'required':['claims'],'additionalProperties':False}
FOLLOW_SCHEMA = {'type':'object','properties':{
    'question':{'type':'string'},'clarification':{'type':'string'},
    'source_ids':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':4}},'additionalProperties':False}
# CLI 지원 범위만 제한합니다. status/claims 배타성과 후속 질문 형식은 기존 RAG 검사가 판정합니다.


def decode_events(text):
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    initial = [e['init'] for e in events if e.get('event') == 'init']
    if len(initial) != 1 or initial[0].get('model') != MODEL:
        raise ValueError('AGY actual model was not confirmed')
    for event in events:
        step = event.get('step_update', {})
        finalization = step.get('step_type') == 'tool' and step.get('tool_name') == 'finish'
        if event.get('event') == 'step_update' and step.get('step_type') not in ('user_input', 'agent_response', 'finish') and not finalization:
            raise ValueError('AGY attempted an external tool/action; only structured finalization is allowed')
        if event.get('event') not in ('init', 'step_update', 'result'):
            raise ValueError('Unknown AGY event; retained for inspection')
    final = [e['result'] for e in events if e.get('event') == 'result']
    if len(final) != 1 or final[0].get('status') != 'SUCCESS' or not isinstance(final[0].get('structured_output'), dict):
        raise ValueError('AGY did not complete a structured response')
    return final[0]


class AgyClient:
    def __init__(self, directory, key, *, schema=ANSWER_SCHEMA, timeout=180, deadline=None):
        self.directory, self.key, self.schema, self.timeout = Path(directory), key, schema, timeout
        self.deadline=deadline
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        if request['model'] != MODEL or set(request) - {'model', 'messages', 'extra_body', 'max_tokens'}:
            raise ValueError('AGY adapter cannot promise temperature, token-cap or other unsupported controls')
        if request.get('extra_body', {}).get('reasoning_effort') != 'high':
            raise ValueError('Frozen AGY effort is high')
        timeout=self.timeout if self.deadline is None else min(self.timeout,int(self.deadline-time.time())-15)
        if timeout<1:raise TimeoutError('Original model deadline reached; no new CLI request or repair started')
        index = len(self.calls)
        stem = self.directory / 'agy-traces' / f'{self.key}-{index}'
        stem.parent.mkdir(exist_ok=True, parents=True)
        if stem.with_suffix('.request.json').exists():
            raise FileExistsError('Existing raw call retained; use a new key for an explicitly separate probe/run')
        prompt = ('Use only the enclosed messages and source cells. Do not use tools, files, browser, web, memory, '
                  'subagents or other context. Finish with the requested JSON only. Message roles below define the RAG task.\n'
                  + json.dumps(request['messages'], ensure_ascii=False))
        record = {'unmapped_options': {k: request[k] for k in ('max_tokens',) if k in request}, 'request': request, 'requested_model': MODEL, 'requested_effort': 'high', 'schema': self.schema,'cli_timeout_seconds':timeout}
        stem.with_suffix('.request.json').write_text(json.dumps(record, ensure_ascii=False))
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix='cs-rag-agy-', dir='/private/tmp') as workspace:
                command = ['/usr/bin/sandbox-exec', '-p', PROFILE, '/Users/mac/.local/bin/agy', '--sandbox',
                           '--disable-slash-commands', '--model', MODEL, '--effort', 'high', '--output-format', 'stream-json',
                           '--json-schema', json.dumps(self.schema, ensure_ascii=False), '--print-timeout', f'{timeout}s', '--print', prompt]
                process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, start_new_session=True)
                stdout = stderr = ''
                try:
                    stdout, stderr = process.communicate(timeout=timeout + 15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        stdout, stderr = process.communicate()
                    record['error'] = 'timeout'
                    raise TimeoutError('Owned AGY request timed out; no automatic transport retry')
                finally:
                    stem.with_suffix('.stdout.jsonl').write_text(stdout)
                    stem.with_suffix('.stderr.log').write_text(stderr)
                record['exit'] = process.returncode
                if process.returncode:
                    raise RuntimeError(f'AGY exit {process.returncode}; local trace preserved')
                final = decode_events(stdout)
                record.update(confirmed_model=MODEL, provider_duration=final.get('duration_seconds'), usage=final.get('usage'),
                              actual_output=final['structured_output'], conversation_id=final.get('conversation_id'))
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                    content=json.dumps(final['structured_output'], ensure_ascii=False)))])
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            record['wall_seconds'] = time.perf_counter() - started
            self.calls.append(record)
            stem.with_suffix('.metadata.json').write_text(json.dumps(record, ensure_ascii=False))
