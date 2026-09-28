"""저장된 합성 검색 후보로 한국어·영어 답변 지시문만 비교합니다. 운영 코드는 바꾸지 않습니다."""
import argparse
from contextlib import closing
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch

from openai import OpenAI

from experiments.answer_baseline import RUBRIC, digest, evaluate_case, prepare
from ragkit import llm
from ragkit.retrieval_evaluation import atomic_report


PROMPT = Path(llm.__file__).with_name('assets') / 'answer_prompt_en.txt'
REPAIR_KO, REPAIR_EN = llm.ANSWER_REPAIR['ko'], llm.ANSWER_REPAIR['en']
LENGTH_KO, LENGTH_EN = llm.ANSWER_LENGTH_RETRY['ko'], llm.ANSWER_LENGTH_RETRY['en']


def translate_request(request, config):
    request = deepcopy(request)
    for message in request['messages'][2:]:
        if message['role'] != 'user':
            continue
        content = message['content']
        if content == LENGTH_KO:
            message['content'] = LENGTH_EN
        elif content.startswith('검사 오류: ') and content.endswith(REPAIR_KO):
            # 검사기의 한국어 진단·원문 주소는 그대로 두고 보정 지시만 번역합니다.
            message['content'] = 'Validation error: ' + content[len('검사 오류: '):-len(REPAIR_KO)] + REPAIR_EN
        else:
            raise ValueError('Unrecognized repair instructions; translation must be reviewed again.')
    if sum(len(m['content']) for m in request['messages']) > config.get('max_context_chars', 24000):
        raise ValueError('Translated request exceeds the unchanged character limit.')
    request.update(llm.generation_options(config, for_answer=True, messages=request['messages']))
    return request


def edge_cases():
    """사용자가 알려 준 날짜·중복 인용 조건과 숫자 경계를 창작 자료로 확인합니다."""
    examples = [
        ('EDGE-DATE-01', '스핀에서 문제가 발생할 시 할 수 있는 조치들을 알려줘. 답변에 참고문헌의 날짜를 꼭 추가해줘. 인용이 중복되지 않게 해줘.', [
            {'불량발생일': '2026-09-01', '현상': '스핀 회전 알람이 발생했습니다.', '조치': '신호 배선을 재체결한 후 알람이 해소되었습니다.'},
            {'불량발생일': '2026.09.03', '현상': '스핀 회전 속도가 불안정했습니다.', '조치': '모터 커넥터를 재체결한 후 속도가 안정되었습니다.'}],
         '두 사례의 조치·결과·날짜와 각각의 근거 열을 포함하고 셀 주소를 중복하지 않는다.'),
        ('EDGE-DATE-02', '스핀 회전 알람의 과거 조치와 결과, 불량발생일을 알려줘. 날짜가 없으면 미기재로 표시해줘.', [
            {'불량발생일': '', '현상': '스핀 회전 알람이 발생했습니다.', '조치': '신호 배선을 재체결한 후 알람이 해소되었습니다.'}],
         '조치·결과를 답하고 빈 불량발생일을 인용해 미기재로 설명한다. 날짜를 만들거나 전체 답변을 보류하지 않는다.'),
        ('EDGE-NUM-01', '스핀 척 흡착 부족 사례의 조치, 조치 전후 압력과 관찰 결과를 알려줘.', [
            {'현상': '스핀 척 흡착 부족. 압력 -0.1 MPa.', '조치': '흡입 호스를 교체한 후 압력 -0.2 MPa. 30분 관찰 중 흡착 부족 없음. 장기 재발 미확인.'}],
         '교체 전 -0.1 MPa와 후 -0.2 MPa의 부호·단위·순서를 보존하고 30분 관찰과 장기 재발 미확인을 함께 설명한다.'),
        ('EDGE-NUM-02', '스핀 모터 교체가 완료되었나요? 부품비도 원문 값으로 알려줘.', [
            {'현상': '스핀 모터 회전 불가.', '조치': '모터 발주 완료. 교체 미실시. 부품비 1,000으로 기록했으며 금액 단위 미기재.'}],
         '발주와 교체를 구분하고 교체 미실시와 부품비 1,000을 답한다. 원문에 없는 원·달러 단위를 붙이지 않는다.'),
    ]
    result = []
    for identity, question, cells, review in examples:
        hits = []
        for index, original in enumerate(cells, 1):
            receipt = f'LANG-{identity}-{index}'
            source = {'id': receipt + '/접수', 'kind': '접수', 'cells': {'CS NO': receipt, **original}}
            hits.append({'id': receipt + '/문서', 'record_id': receipt, 'kind': '문서', 'evidence': [source]})
        result.append({'case': {'id': identity, 'question': question, 'expected_ids': [h['record_id'] for h in hits]},
                       'hits': hits, 'review': review, 'cohort': 'targeted_synthetic'})
    return result


def frozen_cases(baseline_path):
    baseline = json.loads(baseline_path.read_text())
    if baseline.get('complete') is not True:
        raise ValueError('The source baseline must be complete.')
    _, _, dataset = prepare(json.loads(Path('experiments/action_cases.json').read_text()))
    if [r['id'] for r in baseline['results']] != [c['id'] for c in dataset['cases']]:
        raise ValueError('Source baseline must contain exactly the existing 24 cases in order.')
    result = []
    for case, saved in zip(dataset['cases'], baseline['results'], strict=True):
        source = saved['methods']['retrieved']
        if saved['question'] != case['question'] or source['status'] != 'ok' or not source['hits']:
            raise ValueError('Missing or changed source retrieval.')
        hits = [{key: hit[key] for key in ('id', 'record_id', 'kind', 'content', 'evidence', 'evidence_ids') if key in hit}
                for hit in source['hits'][:3]]
        result.append({'case': case, 'hits': hits, 'cohort': 'existing_synthetic'})
    return result + edge_cases()


def run(args):
    cases = frozen_cases(args.baseline)
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    prompts = {'ko': llm.CLAIM_PROMPT, 'en': PROMPT.read_text().strip()}
    files = [args.baseline, args.config, Path(__file__), PROMPT, Path('experiments/answer_baseline.py'),
             Path('experiments/action_cases.json'), *sorted(Path('ragkit').glob('*.py'))]
    conditions = {'files_sha256': {str(p): digest(p) for p in files}, 'cases': cases, 'prompts': prompts,
                  'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
                  'rubric': RUBRIC, 'extra_rubric': 'Dates and numeric edge-case criteria are frozen in each case review.',
                  'decision_rule': 'English is a candidate only if source-reviewed complete answers increase without '
                                   'more unsupported successes, incorrect abstentions or final execution/validation failures. '
                                   'A single synthetic Mac run does not authorize automatic Windows 120B adoption.',
                  'order': 'odd: ko/en; even: en/ko; serial, one pass, no model tuning',
                  'prompt_tokens': {k: len(llm._answer_encoder().encode_ordinary(v)) for k, v in prompts.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text())['conditions'] != conditions:
            raise ValueError('Frozen conditions changed; use a new output directory.')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Use an empty output directory.')
        atomic_report(path, {'frozen_at': datetime.now().astimezone().isoformat(), 'conditions': conditions})
    protocol_hash = digest(path)
    report = {'complete': False, 'protocol_sha256': protocol_hash, 'results': [], 'total_answers': len(cases) * 2}
    with closing(OpenAI(base_url=config['base_url'], api_key=config.get('api_key', 'EMPTY'),
                        timeout=config.get('timeout_seconds', 240), max_retries=0)) as client:
        for index, item in enumerate(cases):
            case = item['case']
            row = {'id': case['id'], 'question': case['question'], 'cohort': item['cohort'], 'methods': {}}
            report['results'].append(row)
            for language in ('ko', 'en') if index % 2 == 0 else ('en', 'ko'):
                checkpoint = args.output / 'checkpoints' / f'{case["id"]}-{language}.json'
                # This process is serial. Restoring the module constant prevents the next Korean arm using English.
                with patch.object(llm, 'CLAIM_PROMPT', prompts[language]):
                    entry = evaluate_case(None, case, language, config, None, client, checkpoint, protocol_hash,
                                          fixed_hits=deepcopy(item['hits']),
                                          prepare_request=(lambda req: translate_request(req, config)) if language == 'en' else None)
                row['methods'][language] = entry
                report['completed_answers'] = sum(len(r['methods']) for r in report['results'])
                atomic_report(args.output / 'results.json', report)
                print(f'{report["completed_answers"]}/{report["total_answers"]} {case["id"]} {language} '
                      f'error={bool(entry.get("answer_error"))} seconds={entry.get("answer_seconds", 0):.2f}', flush=True)
    if any(digest(p) != h for p, h in conditions['files_sha256'].items()):
        raise ValueError('Inputs changed during evaluation; completion not certified.')
    report.update(complete=True, semantic_review='pending; automated validation is not semantic correctness')
    atomic_report(args.output / 'results.json', report)
    print('Complete; paired source review remains required.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('mac_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
