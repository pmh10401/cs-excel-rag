"""준비된 로컬 LM Studio만 자동 시작합니다. 모델 다운로드·다른 모델 종료는 하지 않습니다."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from openai import APIConnectionError


def prepare(client, cfg, progress=None):
    address = urlparse(cfg['base_url'])
    if not cfg.get('auto_start', True) or address.scheme != 'http' or address.hostname not in ('localhost', '127.0.0.1') or address.path.rstrip('/') != '/v1':
        return
    def report(message):
        if progress:
            progress(message)
    report('임베딩 서버 연결 확인 중')
    offline = False
    try:
        models = client.models.list(timeout=5)
        if cfg['model'] in {item.id for item in models.data}:
            return
    except APIConnectionError:
        offline = True
    # 인증 실패·HTTP 오류는 실행 중인 서버의 문제이므로 자동 시작으로 덮지 않습니다.
    executable = shutil.which('lms')
    if not executable:
        bundled = Path.home() / '.lmstudio' / 'bin' / ('lms.exe' if sys.platform == 'win32' else 'lms')
        if bundled.is_file():
            executable = str(bundled)
    if not executable:
        raise RuntimeError('LM Studio의 lms 실행 파일을 찾지 못했습니다. 설치 경로와 CLI 등록 상태를 확인하세요.')

    def run(*args):
        try:
            result = subprocess.run([executable, *args], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120,
                creationflags=0x08000000 if sys.platform == 'win32' else 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f'LM Studio 자동 준비 중 {args[0]} 실행을 완료하지 못했습니다: {error}') from error
        if result.returncode:
            raise RuntimeError(f'LM Studio {args[0]} 실패: {(result.stderr or result.stdout)[-500:]}')
        return result.stdout

    if offline:
        report('LM Studio 임베딩 서버 자동 시작 중')
        run('daemon', 'up')
        run('server', 'start', '--port', str(address.port or 80), '--bind', '127.0.0.1')
        if cfg['model'] in {item.id for item in client.models.list(timeout=10).data}:
            return  # 서버의 자동 로드 기능이 이미 이 모델을 제공하는 경우
    report('설치된 임베딩 모델 확인 중')
    downloaded = json.loads(run('ls', '--embedding', '--json'))
    if not isinstance(downloaded, list) or any(not isinstance(item, dict) for item in downloaded):
        raise RuntimeError('LM Studio의 모델 목록 형식을 확인할 수 없습니다.')
    key = 'text-embedding-qwen3-embedding-0.6b' if cfg['model'] == 'cs-embedding' else cfg['model']
    matches = [item for item in downloaded if item.get('type') == 'embedding'
               and key in (item.get('modelKey'), item.get('path'))]
    if len(matches) != 1 or not isinstance(matches[0].get('modelKey'), str) or matches[0]['modelKey'].startswith('-'):
        raise RuntimeError(f"설정한 임베딩 모델을 설치 목록에서 찾지 못했습니다: {cfg['model']}")
    report('임베딩 모델 자동 로드 중')
    # 메모리 확보를 위한 다른 모델 종료 등의 확인을 자동 승인하지 않습니다.
    run('load', matches[0]['modelKey'], '--identifier', cfg['model'], '--context-length', '8192')
    if cfg['model'] not in {item.id for item in client.models.list(timeout=10).data}:
        raise RuntimeError('모델 로드 후에도 임베딩 API에서 모델을 찾지 못했습니다.')
