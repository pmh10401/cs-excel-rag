"""선택 설치한 영어 근거 검사 모델. 질문 중에는 네트워크에 연결하지 않습니다."""
import importlib.util
import math
import multiprocessing
import threading
import time

MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
MODEL_REV = 'b5113eb38ab63efdd7f280f8c144ea8b13f978ce'
THRESHOLD = 0.8
CPU_THREADS = 4
LABELS = {'entailment', 'neutral', 'contradiction'}
CACHE_FILES = ('config.json', 'tokenizer_config.json', 'spm.model', 'model.safetensors')
_lock = threading.RLock()
_worker = None


class NLIError(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def missing_dependencies():
    return [name for name in ('torch', 'transformers', 'sentencepiece') if importlib.util.find_spec(name) is None]


def cache_present():
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(MODEL_ID, name, revision=MODEL_REV), str) for name in CACHE_FILES)
    except (ImportError, OSError, ValueError):
        return False


def status():
    """화면 조회만으로 큰 라이브러리나 모델을 적재하지 않습니다."""
    missing = missing_dependencies()
    cached = cache_present()
    worker = _worker
    loaded = bool(worker and worker['prepared'] and worker['process'].is_alive())
    return {'ready': not missing and cached, 'loaded': loaded,
            'dependencies_ready': not missing, 'model_cached': cached,
            'message': ('선택 라이브러리를 설치하세요: setup.bat guard 또는 requirements-claim-verification.txt'
                        if missing else '모델을 다운로드·준비하세요.' if not cached else
                        '영어 근거 검사 모델 적재 완료' if loaded else
                        '모델 파일 준비됨 · 첫 검사에서 모델을 적재합니다.')}


def label_map(id2label):
    try:
        mapping = {str(label).strip().lower(): int(index) for index, label in id2label.items()}
    except (TypeError, ValueError, AttributeError) as error:
        raise NLIError('검사 모델의 라벨 구성이 올바르지 않습니다.', 'labels') from error
    if set(mapping) != LABELS or set(mapping.values()) != {0, 1, 2} or len(id2label) != 3:
        raise NLIError('검사 모델은 entailment/neutral/contradiction의 서로 다른 라벨 ID가 필요합니다.', 'labels')
    return mapping


def sequence_limit(tokenizer, model):
    values = [getattr(tokenizer, 'model_max_length', None),
              getattr(model.config, 'max_position_embeddings', None)]
    values = [value for value in values if type(value) is int and 1 < value < 100000]
    if not values:
        raise NLIError('검사 모델의 입력 토큰 한도를 확인할 수 없습니다.', 'limit')
    return min(values)


def score_pair(premise, hypothesis, *, tokenizer, model, device='cpu', limit=None):
    """실험과 운영이 공유하는 계산입니다. 전제와 주장을 자르지 않습니다."""
    if not isinstance(premise, str) or not premise.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
        raise NLIError('검사할 원문 또는 주장이 비어 있습니다.', 'empty')
    import torch
    import torch.nn.functional as F
    mapping = label_map(model.config.id2label)
    limit = sequence_limit(tokenizer, model) if limit is None else limit
    ids = tokenizer(premise, hypothesis, truncation=False, add_special_tokens=True)['input_ids']
    count = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
    if count > limit:
        raise NLIError(f'영어 근거 검사 입력 {count}토큰이 모델 한도 {limit}토큰을 넘습니다. 원문을 자르지 않았습니다.', 'overlength')
    if count < 1:
        raise NLIError('검사 입력 토큰이 없습니다.', 'tokens')
    encoded = tokenizer(premise, hypothesis, truncation=False, return_tensors='pt')
    if encoded['input_ids'].shape[-1] != count:
        raise NLIError('검사 토크나이저의 입력 길이가 달라졌습니다.', 'tokens')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
    if logits.ndim != 2 or tuple(logits.shape) != (1, 3) or not torch.isfinite(logits).all():
        raise NLIError('검사 모델의 출력 모양이나 숫자에 오류가 있습니다.', 'classifier')
    probs = F.softmax(logits.float(), dim=-1)[0]
    values = {name: float(probs[index].item()) for name, index in mapping.items()}
    if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values())
            or abs(sum(values.values()) - 1) > 1e-6):
        raise NLIError('검사 모델의 확률이 유한하게 정규화되지 않았습니다.', 'classifier')
    support = values['entailment']
    return {'label': max(values, key=values.get), 'probabilities': values,
            'support_probability': support, 'threshold': THRESHOLD, 'passed': support >= THRESHOLD,
            'tokens': count, 'limit': limit}


def load_model():
    """고정된 로컬 파일만 읽습니다. pickle 가중치나 원격 Python 코드를 실행하지 않습니다."""
    if missing_dependencies():
        raise NLIError('영어 근거 검사 선택 라이브러리를 먼저 설치하세요: setup.bat guard 또는 requirements-claim-verification.txt', 'dependencies')
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(CPU_THREADS)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV,
            local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REV,
            local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32)
        label_map(model.config.id2label)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters() if parameter.is_floating_point()):
            raise NLIError('검사 모델 가중치가 float32가 아닙니다.', 'dtype')
        model.to('cpu')
        model.eval()
        sequence_limit(tokenizer, model)
        return tokenizer, model
    except NLIError:
        raise
    except Exception as error:
        raise NLIError('영어 근거 검사 모델을 로컬에서 적재하지 못했습니다. 모델 다운로드·준비를 실행하세요.', 'model_load') from error


def _worker_main(connection):
    """모델만 소유하는 자식 프로세스입니다. DB·원문·API 키는 받지 않습니다."""
    try:
        tokenizer, model = load_model()
        connection.send({'ok': True, 'value': {'ready': True, 'loaded': True, 'model': MODEL_ID,
            'revision': MODEL_REV, 'device': 'cpu', 'dtype': 'float32',
            'limit': sequence_limit(tokenizer, model), 'threshold': THRESHOLD}})
        while True:
            premise, hypothesis = connection.recv()
            try:
                value = score_pair(premise, hypothesis, tokenizer=tokenizer, model=model)
                connection.send({'ok': True, 'value': value})
            except NLIError as error:
                connection.send({'ok': False, 'message': str(error), 'code': error.code})
            except Exception:
                connection.send({'ok': False, 'message': '영어 근거 검사 모델의 계산을 완료하지 못했습니다.', 'code': 'inference'})
    except NLIError as error:
        connection.send({'ok': False, 'message': str(error), 'code': error.code})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def _stop_worker():
    global _worker
    worker, _worker = _worker, None
    if worker:
        process = worker['process']
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        worker['connection'].close()


def _time_left(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise NLIError('영어 근거 검사의 시간 한도를 넘었습니다. 이 질문의 검사를 완료하지 않았습니다.', 'verification_timeout')
    return remaining


def _receive(connection, deadline):
    if not connection.poll(_time_left(deadline)):
        _time_left(deadline)
        raise NLIError('영어 근거 검사 응답 시간 한도를 넘었습니다.', 'verification_timeout')
    message = connection.recv()
    _time_left(deadline)
    if not message['ok']:
        raise NLIError(message['message'], message['code'])
    return message['value']


def _request(pair=None, *, deadline=None):
    global _worker
    deadline = time.monotonic() + 120 if deadline is None else deadline
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise NLIError('검사 시간 한도가 올바르지 않습니다.', 'option')
    if not _lock.acquire(timeout=_time_left(deadline)):
        raise NLIError('영어 검사 대기 중 시간 한도를 넘었습니다.', 'verification_timeout')
    try:
        # ponytail: 한 CPU 프로세스에서 한 쌍씩 처리합니다. 동시 질문 처리량을 늘릴 때만 확장합니다.
        if not _worker or not _worker['process'].is_alive():
            _stop_worker()
            context = multiprocessing.get_context('spawn')  # Windows에서도 같은 실행 경계를 사용합니다.
            connection, child = context.Pipe()
            process = context.Process(target=_worker_main, args=(child,), daemon=True)
            _worker = {'process': process, 'connection': connection, 'prepared': None}
            try:
                process.start()
            finally:
                child.close()
            _worker['prepared'] = _receive(connection, deadline)
        if pair is None:
            return dict(_worker['prepared'])
        _worker['connection'].send(pair)
        return _receive(_worker['connection'], deadline)
    except NLIError as error:
        if _worker and (_worker['prepared'] is None or error.code == 'verification_timeout'):
            _stop_worker()
        raise
    except (EOFError, BrokenPipeError, OSError) as error:
        _stop_worker()
        raise NLIError('영어 검사 프로세스와 통신하지 못했습니다. 다음 요청에서 다시 준비합니다.', 'worker') from error
    finally:
        _lock.release()


def prepare_model(download=False, *, deadline=None):
    if type(download) is not bool:
        raise NLIError('download는 참 또는 거짓이어야 합니다.', 'option')
    if missing_dependencies():
        raise NLIError('선택 라이브러리를 먼저 설치하세요: setup.bat guard 또는 requirements-claim-verification.txt', 'dependencies')
    if download:
        with _lock:
            if not cache_present():
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(MODEL_ID, revision=MODEL_REV, allow_patterns=[*CACHE_FILES,
                        'tokenizer.json', 'special_tokens_map.json', 'added_tokens.json'])
                except Exception as error:
                    raise NLIError('고정 검사 모델 다운로드에 실패했습니다. 네트워크와 저장 공간을 확인하세요.', 'download') from error
    return _request(deadline=deadline)


def score(premise, hypothesis, *, deadline=None):
    if not isinstance(premise, str) or not premise.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
        raise NLIError('검사할 원문 또는 주장이 비어 있습니다.', 'empty')
    if missing_dependencies():
        raise NLIError('선택 라이브러리를 먼저 설치하세요: setup.bat guard 또는 requirements-claim-verification.txt', 'dependencies')
    return _request((premise, hypothesis), deadline=deadline)
