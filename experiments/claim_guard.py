"""창작 주장으로 규칙 검사와 로컬 NLI·MiniCheck를 비교하는 독립 실험입니다."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

from ragkit.llm import ClaimValidationError, validate_claims
from ragkit.retrieval_evaluation import atomic_report

MDEBERTA_ID = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
MDEBERTA_REV = 'b5113eb38ab63efdd7f280f8c144ea8b13f978ce'
MINICHECK_ID = 'lytang/MiniCheck-Flan-T5-Large'
MINICHECK_REV = '96eafd01cee2d16cf81aaa2fb226b14f422a37b3'
# MiniCheck 공식 점수: 첫 디코더 스텝에서 토큰 3=미지지, 209=지지. 숫자 문자 '0'의 vocab id로 바꾸지 않습니다.
MINICHECK_TOKEN_UNSUPPORTED = 3
MINICHECK_TOKEN_SUPPORTED = 209
MINICHECK_LIMIT = 2048
NLI_LABELS = ('entailment', 'neutral', 'contradiction')
SEMANTIC = ('supported', 'contradicted', 'not_mentioned')
DEFAULT_THRESHOLD = 0.8
DEFAULT_OUTPUT_ROOT = Path('outputs/claim-guard-20261001')
CPU_THREADS = 4
TORCH_DTYPE = 'float32'
DECISION_STATUSES = {'ok', 'rejected', 'empty_evidence'}
ERROR_STATUSES = {'error', 'overlength', 'classifier_error', 'model_load_failed', 'device_unavailable',
                  'verification_failed'}
REQUIRED_CATEGORIES = {
    'planned_vs_completed', 'recurrence_unconfirmed', 'failed_vs_successful',
    'action_outcome_linkage', 'numbers_signs_units', 'part_swap_same_cs',
    'missing_support', 'paraphrase_list_numbering', 'question_conditions',
}


class ProbeError(Exception):
    """분류기·길이 오류입니다. 사실 거절과 구별합니다."""
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class OverlengthError(ProbeError):
    def __init__(self, message):
        super().__init__(message, 'overlength')


class ClassifierError(ProbeError):
    def __init__(self, message):
        super().__init__(message, 'classifier_error')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_digest(case):
    payload = json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def softmax(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ClassifierError('로짓이 유한한 숫자가 아닙니다.')
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        raise ClassifierError('확률을 정규화할 수 없습니다.')
    probs = [x / total for x in exps]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-6:
        raise ClassifierError('확률이 유한하게 정규화되지 않았습니다.')
    return probs


def nli_label_map(id2label):
    if not isinstance(id2label, dict) or not id2label:
        raise ClassifierError('모델 config id2label이 없습니다.')
    mapping = {}
    for key, label in id2label.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as error:
            raise ClassifierError(f'id2label 키가 정수가 아닙니다: {key!r}') from error
        mapping[str(label).strip().lower()] = index
    if set(mapping) != set(NLI_LABELS) or set(mapping.values()) != {0, 1, 2}:
        raise ClassifierError(f'NLI 라벨 ID는 0,1,2의 순열이고 이름은 entailment/neutral/contradiction이어야 합니다: {id2label}')
    return mapping


def sequence_limit(backend, tokenizer=None, model=None):
    if backend == 'minicheck':
        return MINICHECK_LIMIT
    values = []
    config = getattr(model, 'config', None)
    for obj, attr in ((tokenizer, 'model_max_length'), (config, 'max_position_embeddings')):
        value = getattr(obj, attr, None) if obj is not None else None
        if type(value) is int and 1 < value < 100000:
            values.append(value)
    if not values:
        raise ClassifierError('mDeBERTa 입력 한도를 확인할 수 없습니다.')
    return min(values)


def reject_if_overlong(token_count, limit):
    if type(token_count) is not int or token_count < 1:
        raise ClassifierError('토큰 수가 올바르지 않습니다.')
    if token_count > limit:
        raise OverlengthError(f'입력 길이 {token_count}가 한도 {limit}를 넘습니다. 자른 입력은 평가하지 않습니다.')


def decide_from_probabilities(probabilities, *, support_key, threshold, nli_label):
    if not isinstance(probabilities, dict) or not probabilities:
        raise ClassifierError('확률 객체가 없습니다.')
    values = []
    for name, value in probabilities.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ClassifierError(f'확률 {name}이 유한한 [0, 1] 값이 아닙니다.')
        values.append(value)
    if abs(sum(values) - 1) > 1e-6:
        raise ClassifierError('확률 합이 1이 아닙니다.')
    support = probabilities.get(support_key)
    if type(support) not in (int, float) or not math.isfinite(support):
        raise ClassifierError('지지 확률이 없습니다.')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ClassifierError('임계값은 라벨에 맞춰 바꾸지 않는 고정 진단값입니다.')
    return {
        'nli_label': nli_label,
        'probabilities': {key: float(value) for key, value in probabilities.items()},
        'support_probability': float(support),
        'threshold': float(threshold),
        'threshold_pass': float(support) >= float(threshold),
        'class_pass': nli_label in ('entailment', 'supported'),
    }


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_cases(data)
    return data


def validate_cases(data):
    if not isinstance(data, dict) or data.get('format') != 'claim-guard-synthetic-v1':
        raise ValueError('claim-guard-synthetic-v1 자료가 필요합니다.')
    for key in ('scope', 'segmentation_note', 'english_note', 'ground_truth_note', 'probe_note'):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f'{key} 설명이 필요합니다.')
    cases = data.get('cases')
    if not isinstance(cases, list) or not 36 <= len(cases) <= 60:
        raise ValueError('손수 작성한 사례는 36~60건이어야 합니다.')
    seen, categories, chinese = set(), set(), 0
    for case in cases:
        identity = case.get('id') if isinstance(case, dict) else None
        if not isinstance(identity, str) or not identity.isascii() or not identity.replace('_', '').replace('-', '').isalnum():
            raise ValueError('사례 ID는 경로가 아닌 영문·숫자·하이픈·밑줄이어야 합니다.')
        if identity in seen:
            raise ValueError(f'사례 ID가 중복입니다: {identity}')
        seen.add(identity)
        if not isinstance(case.get('category'), str) or not case['category'].strip():
            raise ValueError(f'{identity}: 범주가 필요합니다.')
        categories.add(case['category'])
        if case.get('expected_semantic') not in SEMANTIC:
            raise ValueError(f'{identity}: expected_semantic이 올바르지 않습니다.')
        if not isinstance(case.get('rationale'), str) or not case['rationale'].strip():
            raise ValueError(f'{identity}: 근거 설명이 필요합니다.')
        question, claim = case.get('question'), case.get('claim')
        if not isinstance(question, dict) or not isinstance(claim, dict):
            raise ValueError(f'{identity}: question과 claim의 한국어·영어 짝이 필요합니다.')
        for field in (question, claim):
            for language in ('ko', 'en'):
                if not isinstance(field.get(language), str) or not field[language].strip():
                    raise ValueError(f'{identity}: {language} 문장이 비어 있습니다.')
        sources, citations = case.get('sources'), case.get('citations')
        if not isinstance(sources, list) or not sources:
            raise ValueError(f'{identity}: 원문 출처가 필요합니다.')
        if not isinstance(citations, list) or any(not isinstance(item, dict) for item in citations):
            raise ValueError(f'{identity}: citations는 객체 목록이어야 합니다.')
        by_id, parents_by_cs = {}, {}
        for source in sources:
            sid, cells, cells_en = source.get('id'), source.get('cells'), source.get('cells_en')
            if not isinstance(sid, str) or not sid.startswith('누적28상세:') or '/' not in sid[len('누적28상세:'):]:
                raise ValueError(f'{identity}: 상세 ID는 누적28상세:부모/종류 형식이어야 합니다.')
            if sid in by_id:
                raise ValueError(f'{identity}: 출처 ID가 중복입니다.')
            if not isinstance(cells, dict) or not isinstance(cells_en, dict):
                raise ValueError(f'{identity}: cells와 cells_en이 필요합니다.')
            receipt = cells.get('CS NO') or cells.get('CS 접수번호')
            if not isinstance(receipt, str) or not receipt.strip():
                raise ValueError(f'{identity}: CS 번호가 필요합니다.')
            parents_by_cs.setdefault(receipt.casefold(), set()).add(sid.rsplit('/', 1)[0])
            for key, value in cells.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(f'{identity}: 원문 셀은 문자열이어야 합니다.')
                if any('\u4e00' <= ch <= '\u9fff' for ch in value):
                    chinese += 1
            for key, value in cells_en.items():
                if not isinstance(key, str) or not isinstance(value, str) or key not in cells:
                    raise ValueError(f'{identity}: 영어 대응 셀이 원문과 맞지 않습니다.')
            by_id[sid] = source
        for receipt, parents in parents_by_cs.items():
            count = sum(1 for source in sources
                        if (source['cells'].get('CS NO') or source['cells'].get('CS 접수번호')).casefold() == receipt)
            if len(parents) != count:
                raise ValueError(f'{identity}: 같은 CS의 상세는 서로 다른 누적28상세 부모를 써야 합니다.')
        for ref in citations:
            if set(ref) != {'id', 'column'} or any(not isinstance(ref[key], str) or not ref[key] for key in ref):
                raise ValueError(f'{identity}: 인용은 id와 column만 써야 합니다.')
            source = by_id.get(ref['id'])
            if source is None or ref['column'] not in source['cells'] or ref['column'] not in source['cells_en']:
                raise ValueError(f'{identity}: 인용이 원문·영어 대응 셀을 가리키지 않습니다.')
        for language in ('ko', 'en'):
            premise, hypothesis = cited_pair(case, language=language)
            if len(premise) + len(hypothesis) > 900:
                raise ValueError(f'{identity}: 비교 쌍이 모델 한도를 넘길 수 있어 거부합니다.')
            if case['question']['ko'] in premise or case['question']['en'] in premise:
                raise ValueError(f'{identity}: 질문을 전제에 넣지 않습니다.')
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        raise ValueError(f'필수 범주가 없습니다: {sorted(missing)}')
    if chinese < 2:
        raise ValueError('중국어가 섞인 원문 셀 사례가 더 필요합니다.')
    return data


def build_hits(case):
    """규칙 검사에는 검색된 상세 전체를 넣되, 모델 전제는 인용 셀만 씁니다."""
    return [{'evidence': [{'id': source['id'], 'kind': source.get('kind', '접수'),
                           'cells': dict(source['cells'])}]} for source in case['sources']]


def cited_pair(case, *, language):
    # 인용한 셀 값만 이어서 씁니다. 열 이름·질문·미인용 상세는 전제에 넣지 않습니다.
    key = 'cells' if language == 'ko' else 'cells_en'
    by_id = {source['id']: source for source in case['sources']}
    values = [by_id[ref['id']][key][ref['column']] for ref in case['citations']]
    return '\n'.join(values), case['claim'][language]


def empty_model_evidence(case, language):
    if not case.get('citations'):
        return True
    premise, _ = cited_pair(case, language=language)
    return not premise.strip()


def resolve_language(backend, language=None):
    if language is None:
        return 'en' if backend == 'minicheck' else 'ko'
    if language not in ('ko', 'en'):
        raise ValueError(f'language는 ko 또는 en이어야 합니다: {language}')
    if backend == 'baseline' and language != 'ko':
        raise ValueError('baseline/en은 사용할 수 없습니다. 규칙 검사는 한국어 주장과 원문 셀을 봅니다.')
    if backend == 'minicheck' and language != 'en':
        raise ValueError('minicheck/ko는 사용할 수 없습니다. MiniCheck는 영어 입력만 사용합니다.')
    return language


def rule_check(case):
    started = time.perf_counter()
    payload = {'claims': [{'text': case['claim']['ko'],
                           'evidence': [{'id': ref['id'], 'column': ref['column']} for ref in case['citations']]}]}
    copied = deepcopy(payload)
    try:
        validate_claims(copied, build_hits(case))
        result = {'status': 'ok', 'passed': True, 'error': None, 'category': None}
    except ClaimValidationError as error:
        result = {'status': 'rejected', 'passed': False, 'error': str(error)[:800],
                  'category': getattr(error, 'category', 'validation')}
    except Exception as error:
        result = {'status': 'error', 'passed': False, 'error': f'{type(error).__name__}: {error}'[:800],
                  'category': 'exception'}
    result['seconds'] = time.perf_counter() - started
    return result


def compared_pair(case, backend, language=None):
    language = resolve_language(backend, language)
    premise, hypothesis = cited_pair(case, language=language)
    return {
        'language': language,
        'premise': premise,
        'hypothesis': hypothesis,
        'cited': [{'id': ref['id'], 'column': ref['column']} for ref in case['citations']],
        'note': '합성 탐침의 비교 쌍입니다. 질문과 미인용 셀은 넣지 않았습니다.',
    }


def run_model(case, backend, model_fn, threshold, language=None):
    started = time.perf_counter()
    language = resolve_language(backend, language)
    if empty_model_evidence(case, language):
        return {'status': 'empty_evidence', 'passed': False, 'error': '빈 근거는 모델에 통과시키지 않습니다.',
                'seconds': time.perf_counter() - started, 'review': True}
    premise, hypothesis = cited_pair(case, language=language)
    try:
        scores = model_fn(premise, hypothesis)
        decided = decide_from_probabilities(
            scores['probabilities'],
            support_key='entailment' if backend == 'mdeberta' else 'supported',
            threshold=threshold,
            nli_label=scores['nli_label'])
        decided.update(status='ok', passed=decided['threshold_pass'], error=None,
                       seconds=time.perf_counter() - started)
        return decided
    except ProbeError as error:
        return {'status': error.status, 'passed': False, 'error': str(error)[:800],
                'seconds': time.perf_counter() - started, 'review': True}
    except Exception as error:
        return {'status': 'error', 'passed': False, 'error': f'{type(error).__name__}: {error}'[:800],
                'seconds': time.perf_counter() - started, 'review': True}


def combine(baseline, model, backend):
    if backend == 'baseline' or model.get('status') == 'skipped':
        return {'status': baseline['status'], 'passed': baseline['passed'],
                'note': '규칙 검사만 사용했습니다.'}
    if model['status'] in ERROR_STATUSES or baseline['status'] == 'error':
        return {'status': 'error', 'passed': False,
                'note': '모델 예외·길이 초과·비정상 확률은 올바른 거절로 세지 않습니다.'}
    if model['status'] == 'empty_evidence':
        return {'status': 'empty_evidence', 'passed': False,
                'note': '빈 근거는 모델 통과 전에 거절합니다.'}
    if baseline['status'] in DECISION_STATUSES and model['status'] in DECISION_STATUSES:
        passed = bool(baseline['passed'] and model['passed'])
        return {'status': 'ok' if passed else 'rejected', 'passed': passed}
    return {'status': 'error', 'passed': False, 'note': '결합 판단을 만들지 못했습니다.'}


def evaluate_case(case, *, backend, threshold, model_fn=None, language=None):
    started = time.perf_counter()
    language = resolve_language(backend, language)
    baseline = rule_check(case)
    if backend == 'baseline':
        model = {'status': 'skipped', 'passed': None, 'seconds': 0.0}
    else:
        if model_fn is None:
            raise ClassifierError('모델 추론 함수가 없습니다.')
        model = run_model(case, backend, model_fn, threshold, language=language)
    return {
        'id': case['id'],
        'category': case['category'],
        'expected_semantic': case['expected_semantic'],
        'compared_pair': compared_pair(case, backend, language=language),
        'baseline': baseline,
        'model': model,
        'combined': combine(baseline, model, backend),
        'seconds': time.perf_counter() - started,
    }


def arm_metrics(results, arm):
    rows = [(item, item.get(arm)) for item in results
            if item.get(arm) and item[arm].get('status') != 'skipped']
    false_accepts, false_rejects, errors, reviews = [], [], [], []
    decisions, seconds, by_category = 0, [], {}
    for item, value in rows:
        category = by_category.setdefault(item['category'], {
            'decisions': 0, 'errors': 0, 'false_accepts': 0, 'false_rejects': 0})
        seconds.append(value.get('seconds') or 0.0)
        if value['status'] in ERROR_STATUSES or value['status'] not in DECISION_STATUSES:
            errors.append(item['id'])
            category['errors'] += 1
            if value.get('review'):
                reviews.append(item['id'])
            continue
        decisions += 1
        category['decisions'] += 1
        passed = bool(value.get('passed'))
        if item['expected_semantic'] == 'supported' and not passed:
            false_rejects.append(item['id'])
            category['false_rejects'] += 1
        elif item['expected_semantic'] != 'supported' and passed:
            false_accepts.append(item['id'])
            category['false_accepts'] += 1
    return {
        'attempted': len(rows),
        'decisions': decisions,
        'coverage': (decisions / len(rows)) if rows else 0.0,
        'errors': len(errors),
        'error_ids': errors,
        'review_ids': reviews,
        'false_accepts': len(false_accepts),
        'false_accept_ids': false_accepts,
        'false_rejects': len(false_rejects),
        'false_reject_ids': false_rejects,
        'mean_seconds': (sum(seconds) / len(seconds)) if seconds else None,
        'categories': by_category,
        'note': '오류·길이 초과·비정상 확률은 올바른 거절로 세지 않습니다. 낮은 확률은 전역 보정된 진실이 아닙니다.',
    }


def nli_class_counts(results):
    counts = {}
    for item in results:
        model = item.get('model') or {}
        if model.get('status') != 'ok':
            continue
        key = f"{item['expected_semantic']}->{model.get('nli_label')}"
        counts[key] = counts.get(key, 0) + 1
    return {'pairs': counts, 'note': 'NLI argmax 클래스는 별도 진단입니다. 주 점수는 지지/미지지이며 보정된 3분류 정확도가 아닙니다.'}


def aggregate(results, *, backend, total, language=None):
    language = resolve_language(backend, language)
    baseline, model, combined = (arm_metrics(results, arm) for arm in ('baseline', 'model', 'combined'))
    error_ids = [item['id'] for item in results
                 if item['baseline']['status'] in ERROR_STATUSES
                 or item['model']['status'] in ERROR_STATUSES
                 or item['combined']['status'] in ERROR_STATUSES]
    completed = len(results)
    missing = max(total - completed, 0)
    return {
        'completed': completed,
        'total': total,
        'missing': missing,
        'errors': len(set(error_ids)),
        'error_ids': sorted(set(error_ids)),
        'complete': completed == total and missing == 0,
        'complete_with_errors': bool(error_ids) or missing > 0,
        'summary': {
            'backend': backend,
            'language': language,
            'baseline': baseline,
            'model': model,
            'combined': combined,
            'nli_class': nli_class_counts(results),
            'timings': {
                'mean_case_seconds': (sum(item['seconds'] for item in results) / len(results)) if results else None,
                'mean_baseline_seconds': baseline['mean_seconds'],
                'mean_model_seconds': model['mean_seconds'],
            },
            'probe_note': '고정 합성 개발 탐침이며 정확도 추정이나 채택 근거가 아닙니다.',
        },
    }


def package_versions(names):
    import importlib.metadata
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_paths():
    ragkit = Path(validate_claims.__code__.co_filename).resolve().parent
    return {
        'experiments/claim_guard.py': Path(__file__).resolve(),
        'ragkit/llm.py': ragkit / 'llm.py',
        'ragkit/cslist.py': ragkit / 'cslist.py',
        'ragkit/search.py': ragkit / 'search.py',
        'ragkit/records.py': ragkit / 'records.py',
        'ragkit/input_formats.py': ragkit / 'input_formats.py',
        'ragkit/retrieval_evaluation.py': Path(atomic_report.__code__.co_filename).resolve(),
        'ragkit/nli.py': ragkit / 'nli.py',
    }


def code_digests():
    return {name: digest(path) for name, path in code_paths().items()}


def verify_frozen_inputs(args, conditions):
    if digest(args.dataset) != conditions['dataset_sha256']:
        raise ValueError('실행 중 자료가 바뀌었습니다. 완료로 인증하지 않습니다.')
    if code_digests() != conditions['code_sha256']:
        raise ValueError('실행 중 코드가 바뀌었습니다. 완료로 인증하지 않습니다.')


def freeze_conditions(args, data):
    backend = args.backend
    model_id = revision = None
    packages = {'python': platform.python_version(), 'platform': platform.platform()}
    packages.update(package_versions(('openpyxl',)))
    if backend == 'mdeberta':
        model_id, revision = MDEBERTA_ID, MDEBERTA_REV
        packages.update(package_versions(('torch', 'transformers', 'tokenizers', 'sentencepiece')))
    elif backend == 'minicheck':
        model_id, revision = MINICHECK_ID, MINICHECK_REV
        packages.update(package_versions(('torch', 'transformers', 'tokenizers', 'sentencepiece')))
    return {
        'dataset_sha256': digest(args.dataset),
        'code_sha256': code_digests(),
        'case_ids': [case['id'] for case in data['cases']],
        'backend': backend,
        'language': resolve_language(backend, getattr(args, 'language', None)),
        'model_id': model_id,
        'revision': revision,
        'device': args.device,
        'cpu_threads': CPU_THREADS,
        'dtype': TORCH_DTYPE,
        'threshold': args.threshold,
        'local_files_only': bool(args.local_files_only),
        'runtime': packages,
        'scope': data['scope'],
        'segmentation_note': data['segmentation_note'],
        'decision_note': '임계값을 라벨에 맞춰 조정하거나 모델을 채택하지 않습니다.',
    }


def ensure_protocol(output, conditions):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('conditions') != conditions:
            raise ValueError('고정한 입력·코드·모델·언어·장치·임계값이 달라졌습니다. 새 출력 폴더를 사용하세요.')
        return saved, digest(path)
    if any(output.iterdir()):
        raise ValueError('새 실험은 빈 출력 폴더를 사용하세요.')
    protocol = {'frozen_at': datetime.now().astimezone().isoformat(), 'conditions': conditions}
    atomic_report(path, protocol)
    return protocol, digest(path)


def read_checkpoint(path, protocol_hash, case):
    saved = json.loads(path.read_text(encoding='utf-8'))
    if (saved.get('protocol_sha256') != protocol_hash or saved.get('case_sha256') != case_digest(case)
            or saved.get('case_id') != case['id'] or not isinstance(saved.get('result'), dict)):
        raise ValueError(f'{case["id"]}: 체크포인트 입력·구성이 달라졌습니다. 새 출력 폴더를 사용하세요.')
    return saved['result']


def encode_length(tokenizer, *texts, pair=False):
    encoded = tokenizer(texts[0], texts[1], truncation=False, add_special_tokens=True) if pair else tokenizer(
        texts[0], truncation=False, add_special_tokens=True)
    ids = encoded['input_ids']
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def score_mdeberta(premise, hypothesis, *, tokenizer, model, device, limit):
    from ragkit.nli import NLIError, score_pair
    try:
        scored = score_pair(premise, hypothesis, tokenizer=tokenizer, model=model, device=device, limit=limit)
    except NLIError as error:
        if error.code == 'overlength':
            raise OverlengthError(str(error)) from error
        raise ClassifierError(str(error)) from error
    return {'nli_label': scored['label'], 'probabilities': scored['probabilities'],
            'support_probability': scored['support_probability']}


def score_minicheck(premise, hypothesis, *, tokenizer, model, device, limit):
    import torch
    if tokenizer.eos_token is None:
        raise ClassifierError('MiniCheck 토크나이저에 eos_token이 없습니다.')
    text = 'predict: ' + premise + tokenizer.eos_token + hypothesis
    token_count = encode_length(tokenizer, text)
    reject_if_overlong(token_count, limit)
    encoded = tokenizer(text, truncation=False, return_tensors='pt')
    if encoded['input_ids'].shape[-1] != token_count:
        raise ClassifierError('토크나이저가 입력을 잘랐습니다.')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                       decoder_input_ids=decoder_input_ids).logits
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[-1] <= MINICHECK_TOKEN_SUPPORTED:
        raise ClassifierError(f'MiniCheck 로짓 모양이 올바르지 않습니다: {tuple(logits.shape)}')
    selected = logits[0, 0, [MINICHECK_TOKEN_UNSUPPORTED, MINICHECK_TOKEN_SUPPORTED]].float()
    if not torch.isfinite(selected).all():
        raise ClassifierError('MiniCheck 선택 로짓이 유한하지 않습니다.')
    probs = softmax([float(selected[0].item()), float(selected[1].item())])
    label = 'supported' if probs[1] >= probs[0] else 'unsupported'
    return {'nli_label': label, 'probabilities': {'unsupported': probs[0], 'supported': probs[1]},
            'support_probability': probs[1]}


def resolve_device(name):
    if name == 'cpu':
        return 'cpu'
    if name == 'mps':
        import torch
        mps = getattr(torch.backends, 'mps', None)
        if mps is None or not mps.is_available():
            raise RuntimeError('MPS를 요청했지만 이 환경에서 사용할 수 없습니다. CPU로 바꾸지 않습니다.')
        return 'mps'
    raise ValueError(f'장치 {name}은 지원하지 않습니다.')


def load_backend(backend, device, local_files_only):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(CPU_THREADS)
    if backend == 'mdeberta':
        tokenizer = AutoTokenizer.from_pretrained(MDEBERTA_ID, revision=MDEBERTA_REV,
                                                  local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            MDEBERTA_ID, revision=MDEBERTA_REV, local_files_only=local_files_only, torch_dtype=torch.float32)
        nli_label_map(model.config.id2label)
        scorer = score_mdeberta
    elif backend == 'minicheck':
        tokenizer = AutoTokenizer.from_pretrained(MINICHECK_ID, revision=MINICHECK_REV,
                                                  local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MINICHECK_ID, revision=MINICHECK_REV, local_files_only=local_files_only, torch_dtype=torch.float32)
        scorer = score_minicheck
    else:
        raise ValueError(f'backend {backend}에는 모델이 없습니다.')
    floating = {param.dtype for param in model.parameters() if param.is_floating_point()}
    if floating - {torch.float32}:
        raise ClassifierError(f'float32가 아닌 가중치가 있습니다: {floating}')
    model.to(device)
    model.eval()
    limit = sequence_limit(backend, tokenizer, model)
    return lambda premise, hypothesis: scorer(
        premise, hypothesis, tokenizer=tokenizer, model=model, device=device, limit=limit)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('baseline', 'mdeberta', 'minicheck'), default='baseline')
    parser.add_argument('--language', choices=('ko', 'en'), default=None,
                        help='모델 입력 언어: mDeBERTa 기본 ko, MiniCheck는 en만 지원합니다.')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).with_name('claim_guard_cases.json'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='라벨에 맞춰 조정하지 않는 고정 진단 임계값입니다.')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args(argv)
    args.language = resolve_language(args.backend, args.language)
    if args.output is None:
        if args.backend == 'mdeberta' and args.language == 'en':
            args.output = DEFAULT_OUTPUT_ROOT / 'mdeberta-en'
        else:
            args.output = DEFAULT_OUTPUT_ROOT / args.backend
    if type(args.threshold) is not float or not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        raise ValueError('threshold는 0과 1 사이의 고정 진단값이어야 합니다.')
    return args


def empty_report(protocol_hash, total, *, status, language=None):
    return {
        'format': 'claim-guard-probe-v1',
        'status': status,
        'language': language,
        'progress': f'0/{total}',
        'completed': 0,
        'total': total,
        'complete': False,
        'complete_with_errors': False,
        'missing': total,
        'errors': 0,
        'error_ids': [],
        'protocol_sha256': protocol_hash,
        'results': [],
        'summary': None,
    }


def persist_progress(path, report, stats, *, certify=False):
    report.update(stats)
    report['progress'] = f"{stats['completed']}/{stats['total']}"
    if certify:
        report['complete'] = stats['complete']
        report['complete_with_errors'] = stats['complete_with_errors']
        if stats['complete']:
            report['status'] = 'complete_with_errors' if stats['complete_with_errors'] else 'complete'
        else:
            report['status'] = 'running'
    else:
        report['complete'] = False
        if report.get('status') not in ('model_load_failed', 'verification_failed'):
            report['status'] = 'running'
    atomic_report(path, report)


def read_existing_checkpoints(cases, checkpoints, protocol_hash):
    completed, pending = {}, []
    for case in cases:
        path = checkpoints / f'{case["id"]}.json'
        if path.exists():
            completed[case['id']] = read_checkpoint(path, protocol_hash, case)
        else:
            pending.append(case)
    return completed, pending


def run(args, model_fn=None):
    data = load_cases(args.dataset)
    cases = data['cases']
    language = resolve_language(args.backend, getattr(args, 'language', None))
    conditions = freeze_conditions(args, data)
    _, protocol_hash = ensure_protocol(args.output, conditions)
    target = args.output / 'results.json'
    checkpoints = args.output / 'checkpoints'
    print(f"탐침 시작 backend={args.backend} language={language} cases={len(cases)} output={args.output}",
          flush=True)
    completed, pending = read_existing_checkpoints(cases, checkpoints, protocol_hash)
    report = empty_report(protocol_hash, len(cases), status='starting', language=language)
    report['results'] = [completed[case['id']] for case in cases if case['id'] in completed]
    persist_progress(target, report, aggregate(report['results'], backend=args.backend, total=len(cases),
                                               language=language), certify=False)
    loaded_fn = model_fn
    if pending and args.backend != 'baseline' and loaded_fn is None:
        report['status'] = 'loading_model'
        atomic_report(target, report)
        print('모델 로드를 시작합니다. 다운로드가 있으면 이 이후에 진행됩니다.', flush=True)
        try:
            device = resolve_device(args.device)
            loaded_fn = load_backend(args.backend, device, args.local_files_only)
        except Exception as error:
            stats = aggregate(report['results'], backend=args.backend, total=len(cases), language=language)
            report.update(stats)
            report.update(status='model_load_failed', complete=False, complete_with_errors=True,
                          missing=len(pending),
                          load_error=f'{type(error).__name__}: {error}'[:1000],
                          load_traceback=traceback.format_exc()[-1500:])
            atomic_report(target, report)
            print(f"모델 로드 실패: {report['load_error']}", flush=True)
            return report
    report['results'] = []
    # ponytail: 최대 60건의 체크포인트만 재집계합니다. 더 큰 평가부터 누적 집계를 사용합니다.
    for case in cases:
        if case['id'] in completed:
            result = completed[case['id']]
        else:
            result = evaluate_case(case, backend=args.backend, threshold=args.threshold, model_fn=loaded_fn,
                                  language=language)
            atomic_report(checkpoints / f'{case["id"]}.json', {
                'protocol_sha256': protocol_hash, 'case_sha256': case_digest(case),
                'case_id': case['id'], 'result': result})
            completed[case['id']] = result
        report['results'].append(result)
        persist_progress(target, report, aggregate(report['results'], backend=args.backend, total=len(cases),
                                                   language=language), certify=False)
        print(f"{report['progress']} {case['id']} language={language} baseline={result['baseline']['status']} "
              f"model={result['model'].get('status')} passed={result['combined'].get('passed')} "
              f"seconds={result['seconds']:.3f}", flush=True)
    try:
        verify_frozen_inputs(args, conditions)
    except (ValueError, OSError) as error:
        report.update(status='verification_failed', complete=False, complete_with_errors=True,
                      verification_error=str(error))
        atomic_report(target, report)
        print(f"완료 인증 실패: {error}", flush=True)
        return report
    persist_progress(target, report, aggregate(report['results'], backend=args.backend, total=len(cases),
                                               language=language), certify=True)
    print(json.dumps({key: report[key] for key in
                      ('status', 'progress', 'language', 'complete', 'complete_with_errors', 'missing', 'errors')},
                     ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    report = run(parse_args())
    raise SystemExit(0 if report['complete'] and not report['complete_with_errors'] else 1)
