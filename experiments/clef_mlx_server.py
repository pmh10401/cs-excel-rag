"""Clef 전용 MLX GPU 로더의 로컬 HTTP 연결. 요청 후 GPU 프로세스를 해제합니다."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = 'mlx-community/clef-flash-8bit'
REVISION = '5a195dc147926b63b1d139691ed4480ca9f92c0e'
MODEL = 'clef:flash'
LOADER_SHA256 = '852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5'


def validate_checkpoint(path):
    if path.name != REVISION or not (path / 'clef_mlx.py').is_file():
        raise ValueError('검증한 revision의 Hugging Face 스냅샷 폴더가 필요합니다.')
    if hashlib.sha256((path / 'clef_mlx.py').read_bytes()).hexdigest() != LOADER_SHA256:
        raise ValueError('공개 Clef MLX 로더의 해시가 검증한 버전과 다릅니다.')


def infer(path, request):
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError('Metal GPU를 사용할 수 없습니다. CPU로 대체하지 않습니다.')
    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError('MLX 기본 장치가 GPU가 아닙니다.')
    sys.path.insert(0, str(path))
    import clef_mlx
    config = json.loads((path / 'config.json').read_text())
    if config.get('quantization', {}).get('bits') != 8:
        raise ValueError('검증 대상은 Clef-flash MLX 8비트입니다.')
    started = time.monotonic()
    model = clef_mlx.load(path, head_dtype=mx.float32, backend='lm')
    loaded = time.monotonic()
    mx.reset_peak_memory()
    # 공개 로더의 엄격한 길이 검사로 잘린 원문은 추론·채택하지 않습니다.
    result = model.systemone(request, max_length=16384, truncate=False)
    result.update(state_truncated=False, done_reason='decide', backend='mlx', device='gpu',
                  model_repo=REPO, model_revision=REVISION, quantization_bits=8,
                  peak_memory_bytes=mx.get_peak_memory(), load_seconds=round(loaded-started, 3),
                  inference_seconds=round(time.monotonic()-loaded, 3))
    return result


def serve(path, port):
    lock = threading.Lock()  # ponytail: GPU 요청 하나씩 처리합니다. 동시 모델 상주는 메모리 검증 후 확장합니다.

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path not in ('/health', '/api/version'):
                self.send(404, {'error': '지원하지 않는 경로입니다.'})
                return
            self.send(200, {'backend': 'mlx', 'device': 'gpu', 'model': MODEL, 'busy': lock.locked(),
                            'repo': REPO, 'revision': REVISION, 'context_length': 16384})

        def do_POST(self):
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if (host not in (f'127.0.0.1:{port}', f'localhost:{port}')
                    or origin not in (None, 'http://127.0.0.1:8780', 'http://localhost:8780')):
                self.send(403, {'error': '로컬 테스트 요청만 허용합니다.'})
                return
            if self.path not in ('/api/decide', '/v1/systemone'):
                self.send(404, {'error': '지원하지 않는 경로입니다.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('64KiB 이하의 JSON 요청이 필요합니다.')
                body = json.loads(self.rfile.read(size))
                if (not isinstance(body, dict) or body.get('model') != MODEL or 'state' not in body
                        or not isinstance(body.get('questions'), dict) or not 1 <= len(body['questions']) <= 64
                        or body.get('images') or body.get('videos')):
                    raise ValueError('Clef-flash의 텍스트 상태·질문만 지원합니다.')
            except (ValueError, TypeError):
                self.send(400, {'error': '요청 형식·모델·입력 크기를 확인하세요.'})
                return
            try:
                with lock:
                    env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                    # 별도 프로세스가 끝나야 응답합니다. 답변 모델 적재 전에 GPU 메모리가 해제됩니다.
                    result = subprocess.run([sys.executable, '-u', '-m', 'experiments.clef_mlx_server',
                        '--model-path', str(path), '--worker'], input=json.dumps(body, ensure_ascii=False),
                        capture_output=True, text=True, timeout=300, env=env)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-1000:])
                    data = json.loads(result.stdout)
                self.send(200, data)
            except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as error:
                self.send(503, {'error': 'Clef MLX GPU 실행에 실패했습니다.', 'detail': str(error)[:1000]})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Clef MLX GPU: http://127.0.0.1:{port}/ · 요청별 GPU 해제', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11436)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_checkpoint(args.model_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not 1 <= args.port <= 65535:
        parser.error('포트는 1~65535여야 합니다.')
    if args.worker:
        print(json.dumps(infer(args.model_path, json.load(sys.stdin)), ensure_ascii=False))
    else:
        serve(args.model_path.absolute(), args.port)
