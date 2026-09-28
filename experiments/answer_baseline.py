"""같은 합성 질문을 지정 원문과 실제 검색 원문으로 답합니다. 운영 자료는 열지 않습니다."""
import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

from openai import OpenAI

from experiments.action_selection import validate_dataset
from ragkit import llm, semantic, storage
from ragkit.input_formats import HISTORY_COLUMNS, HISTORY_RULES, HISTORY_SCHEMA, HISTORY_SHEET
from ragkit.records import record_id
from ragkit.retrieval_evaluation import atomic_report, freeze, run_comparison, summary


ARMS = ('provided', 'retrieved')
RUBRIC = {
    'source_coverage': '해당 문항에 지정된 관련 원문이 상위 3개에 있는지. 전체 코퍼스의 의미적 적합률은 아님.',
    'supported_action': '성공 조치가 있는 질문에서는 조건에 맞는 조치와 그 결과를 설명한다.',
    'unsupported_success': '예정·실패·재발·부분 개선·미확인·다른 조건을 검증된 성공으로 추천하지 않는다.',
    'qualifiers': '답변에 포함한 조치의 제한, 순서, 수치·단위와 복합 조치의 한계를 보존한다.',
    'evidence': '본문의 사실이 인용한 원문 셀에 있고 관련 조건과 모순되지 않는다.',
    'abstention': '검증된 성공이 없어도 관련 기록의 제한을 설명할 수 있다. 검색 결과 없음과 구별한다.',
    'failure': '호출·JSON·인용 검사 실패를 정답이나 올바른 보류로 세지 않는다.',
    'review': '자동 검사 통과는 의미 정답이 아니다. 원문 대조 검토자와 판단 이유를 별도로 기록한다.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data):
    """조치 후보를 원문 문서로 바꿉니다. 동일 원문을 후보 수만큼 복제하지 않습니다."""
    validate_dataset(data)
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', c['id']) for c in data['cases']):
        raise ValueError('Question IDs must use letters, digits, underscores or hyphens, never paths.')
    rows, mapping, sources = [], {}, {}
    for case in data['cases']:
        for candidate in case['candidates']:
            key = (candidate['symptom']['ko'], candidate['record']['ko'])
            if key not in sources:
                row = dict.fromkeys(HISTORY_COLUMNS, '')
                row.update({'CS NO': f'BASE-{len(rows) + 1:03}', '현상': key[0], '조치': key[1],
                            '_id_column': 'CS NO', '_source_sheet': HISTORY_SHEET})
                sources[key] = record_id(row)
                rows.append(row)
            if candidate['id'] in mapping:
                raise ValueError('Candidate IDs must be globally unique.')
            mapping[candidate['id']] = sources[key]
    cases = []
    for case in data['cases']:
        # 성공 여부는 검색 관련성과 다릅니다. 예정·실패 원문도 관련 자료일 수 있습니다.
        ids = list(dict.fromkeys(mapping[c['id']] for c in case['candidates']))
        relevant = list(dict.fromkeys(mapping[c['id']] for c in case['candidates']
                                     if c['expected']['context'] == 'match'))
        cases.append({'id': case['id'], 'question': case['question']['ko'], 'expected_ids': relevant,
                      'provided_ids': ids, 'successful_action_ids': case['expected_ids']})
    dataset = {'provenance': data['scope'], 'synthetic': True, 'cases': cases}
    return rows, mapping, dataset


def source_hits(db, identities):
    result = []
    for identity in identities:
        row = db.execute('SELECT document_id,record_id,content,evidence FROM rag_documents WHERE record_id=?',
                         (identity,)).fetchone()
        if row is None:
            raise ValueError('Missing source document.')
        evidence = json.loads(row[3])
        result.append(dict(id=row[0], record_id=row[1], content=row[2], kind='문서', evidence=evidence,
                           evidence_ids=[s['id'] for s in evidence]))
    return result


def evaluate_case(db, case, arm, config, embedding, client, path, protocol_hash):
    """기존 평가기의 검색·답변·원자적 저장을 재사용하고 모델 진단만 함께 보관합니다."""
    dataset = {'provenance': 'Frozen synthetic answer baseline', 'cases': [case]}
    _, case_hash = freeze(dataset)
    metadata = {'protocol_sha256': protocol_hash, 'arm': arm}
    if path.exists():
        report = json.loads(path.read_text())
        if report['freeze_hash'] != case_hash or report['metadata'] != metadata:
            raise ValueError('Checkpoint conditions changed; use a new output directory.')
        if report['complete']:
            return report['results'][0]['methods']['korean_hybrid']
        # 중단 기록은 보존하고 이 문항만 다시 실행합니다. 완결된 실패 결과는 자동 재시도하지 않습니다.
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        path.rename(path.with_name(f'{path.stem}.interrupted-{stamp}.json'))
    diagnostics = {}

    def search(question, method):
        details = {}
        hits = (source_hits(db, case['provided_ids']) if arm == 'provided' else
                semantic.search_cases(db, question, config, embedding_client=embedding, details=details, limit=3))
        return {'hits': hits, 'details': details, 'answer_diagnostics': diagnostics}

    def answer(question, hits):
        calls = []
        diagnostics['calls'] = calls

        def completion(**request):
            # 창작 자료만 로컬 outputs에 기록합니다. API 키나 SDK 예외 본문은 저장하지 않습니다.
            call = {'request': request}
            calls.append(call)
            response = client.chat.completions.create(**request)
            call.update(model=response.model, usage=response.usage.model_dump() if response.usage else None)
            if response.model != config['model']:
                raise ValueError('Unexpected answer model.')
            return response

        observed = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        reply, _ = llm.answer(observed, config, question, hits, details=diagnostics)
        # 원문 보완을 붙인 문자열로 답변의 누락이 가려지지 않도록 모델 본문을 따로 평가합니다.
        diagnostics['reply_with_supplements'] = reply
        return diagnostics.get('answer_text', reply)

    report = run_comparison(dataset, search, path, answer_fn=answer, methods=('korean_hybrid',), metadata=metadata)
    return report['results'][0]['methods']['korean_hybrid']


def run(args):
    data = json.loads(args.dataset.read_text(encoding='utf-8'))
    rows, mapping, dataset = prepare(data)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    cfg = semantic.settings(config)
    args.output.mkdir(parents=True, exist_ok=True)
    contract, db_path = args.output / 'protocol.json', args.output / 'source.sqlite3'
    paths = [args.dataset, Path(__file__), Path('experiments/action_selection.py'), *sorted(Path('ragkit').glob('*.py'))]
    conditions = dict(dataset_sha256=digest(args.dataset), config_sha256=digest(args.config),
                      code_sha256={str(p): digest(p) for p in paths}, rubric=RUBRIC,
                      model=config['model'], reasoning_effort=config['reasoning_effort'],
                      answer_options={k: v for k, v in config.items() if k in
                                      ('answer_max_tokens', 'answer_temperature', 'max_context_chars', 'timeout_seconds')},
                      embedding_profile=semantic.profile(config), embedding_model=cfg['model'],
                      source_rows=rows, candidate_to_source=mapping, dataset=dataset,
                      arms=list(ARMS), retrieved_limit=3, order='odd: provided/retrieved; even: retrieved/provided',
                      scope='Previously exposed synthetic development cases; original Korean only. '
                            'No translation, training, prompt tuning, production DB or configuration changes.')
    with closing(OpenAI(base_url=cfg['base_url'], api_key=cfg['api_key'], timeout=cfg['timeout_seconds'],
                        max_retries=0)) as embedding:
        if contract.exists():
            protocol = json.loads(contract.read_text())
            if protocol['conditions'] != conditions or digest(db_path) != protocol['db_sha256']:
                raise ValueError('Frozen input/config/code/database changed; choose a new output directory.')
        else:
            if any(args.output.iterdir()):
                raise ValueError('Use an empty output directory for a new experiment.')
            # DB 생성은 여기의 고정 파일명만 사용합니다. 운영 DB를 인자로 받지 않습니다.
            with closing(sqlite3.connect(db_path)) as db:
                storage.ingest(db, rows, HISTORY_RULES, schema=HISTORY_SCHEMA)
                vectors = semantic.build_vectors(db, embedding, config, progress=print)
                if vectors['failures'] or vectors['indexed'] != len(rows):
                    raise ValueError('Incomplete fixture embeddings; answers were not started.')
            protocol = {'frozen_at': datetime.now().astimezone().isoformat(), 'conditions': conditions,
                        'vectors': vectors, 'db_sha256': digest(db_path)}
            atomic_report(contract, protocol)  # 첫 답변 요청 이전에 원문·기준·코드를 고정합니다.
        protocol_hash = digest(contract)
        report = {'complete': False, 'protocol_sha256': protocol_hash, 'results': []}
        target = args.output / 'results.json'
        with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db, \
                closing(OpenAI(base_url=config['base_url'], api_key=config.get('api_key', 'EMPTY'),
                               timeout=config.get('timeout_seconds', 120), max_retries=0)) as client:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            for index, case in enumerate(dataset['cases']):
                item = {'id': case['id'], 'question': case['question'], 'methods': {}}
                report['results'].append(item)
                for arm in ARMS if index % 2 == 0 else reversed(ARMS):
                    checkpoint = args.output / 'checkpoints' / f'{case["id"]}-{arm}.json'
                    entry = evaluate_case(db, case, arm, config, embedding, client, checkpoint, protocol_hash)
                    item['methods'][arm] = entry
                    report['completed_answers'] = sum(len(r['methods']) for r in report['results'])
                    report['total_answers'] = 2 * len(dataset['cases'])
                    atomic_report(target, report)
                    print(f'{report["completed_answers"]}/{report["total_answers"]} {case["id"]} {arm} '
                          f'answer_error={bool(entry.get("answer_error"))} '
                          f'seconds={entry.get("answer_seconds", 0):.2f}', flush=True)
        if digest(db_path) != protocol['db_sha256'] or any(digest(p) != h for p, h in conditions['code_sha256'].items()) \
                or digest(args.config) != conditions['config_sha256']:
            raise ValueError('Run inputs changed; completion not certified.')
        report.update(complete=True, summary=summary(report['results'], {'status': 'unavailable'}),
                      semantic_review='pending; automatic checks are not semantic correctness')
        atomic_report(target, report)
        print('Complete; semantic source review remains required.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('experiments/action_cases.json'))
    parser.add_argument('--config', type=Path, default=Path('mac_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
