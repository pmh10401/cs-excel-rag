"""창작한 사례로 셀 수정 전후를 비교합니다. 기준 문장은 모델에 보내지 않습니다."""
import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path

from openai import OpenAI
from . import llm
from .records import text_for
from .report import highlight, ASSETS


def answer_checks(reply, case):
    """사람이 정한 필수·금지 표현으로 알려진 오류를 재검사합니다. 정답 기준은 모델에 보내지 않습니다."""
    for key in ('required', 'forbidden'):
        patterns = case.get(key, [])
        if not isinstance(patterns, list) or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError(f'{key}에는 비어 있지 않은 정규식 문자열 목록이 필요합니다.')
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f'{key} 정규식 오류: {error}') from error
    missing = [p for p in case.get('required', []) if not re.search(p, reply)]
    forbidden = [p for p in case.get('forbidden', []) if re.search(p, reply)]
    return {'passed': not (missing or forbidden) if case.get('required') or case.get('forbidden') else None,
            'missing': missing, 'forbidden': forbidden,
            'note': '지정한 표현의 회귀 검사입니다. 전체 의미 정확도 판정은 원문 대조가 필요합니다.'}


def render_results(report, destination):
    esc = lambda text: html.escape(str(text))
    items = []
    failed = sum('error' in item for item in report['results'])
    changed = sum(any(value != item['case']['input'].get(key)
                      for key, value in item.get('proposal', {}).items()) for item in report['results'])
    for item in report['results']:
        source = item['case']
        comparisons = []
        for column in source['input']:
            if report['column_rules'].get(column) != 'rewrite':
                continue
            before, after = highlight(source['input'][column], item.get('proposal', {}).get(column, ''))
            comparisons.append(f'<tr><th scope="row">{esc(column)}</th><td>{before}</td><td>{after}</td></tr>')
        state = '자동 검사 통과 · 의미 검토 필요' if not item.get('error') else '처리 실패: ' + item['error']
        items.append(f'<section><h2>{esc(source["id"])} · {esc(source["category"])}</h2><p>{esc(state)}</p><div class="scroll"><table><tr><th>열</th><th>원문</th><th>LLM 제안</th></tr>{"".join(comparisons)}</table></div><p>메모의 비교용 참고 문장: {esc(source["reference"])}</p><p>검토 기준: {esc(source["review"])}</p><p>판단할 내용: 가독성이 좋아졌는가 / 의미가 유지되었는가 / 누락 정보가 추가되지 않았는가</p></section>')
    theme = (ASSETS / 'theme.css').read_text(encoding='utf-8')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>셀 다듬기 평가</title>
<style>{theme}</style>
<main><header class="hero"><div class="eyebrow">QUALITY REVIEW</div><h1>문장이 좋아졌는지,<br>의미가 유지됐는지.</h1></header><p>{esc(report['provenance'])}</p><p>처리 {len(report['results'])}/{report['total']}건 · 모델 {esc(report['model'])} · 추론 {esc(report['reasoning_effort'])}</p><p>자동 검사 통과 {len(report['results']) - failed}건 · 처리 실패 {failed}건 · 수정한 사례 {changed}건</p><p>참고 문장은 정답 문자열이 아닙니다. 자동 검사 통과와 수정 여부는 의미 정확도나 가독성 향상을 뜻하지 않습니다.</p><p>프롬프트·열 규칙 버전: {esc(report['version'])}</p>{''.join(items)}</main></html>'''
    destination.write_text(page, encoding='utf-8')


def run_evaluation(client, config, dataset, output_dir, rules, limit=None):
    if not isinstance(dataset, dict) or not isinstance(dataset.get('cases'), list) or not isinstance(dataset.get('provenance'), str):
        raise ValueError('평가 자료에는 provenance와 cases가 필요합니다.')
    seen = set()
    for case in dataset['cases']:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) for k in ('id', 'category', 'reference', 'review')):
            raise ValueError('평가 항목에는 id, category, reference, review 문자열이 필요합니다.')
        text_for(case.get('input'))
        if not isinstance(case['input'].get('메모'), str):
            raise ValueError('평가 입력에는 메모 문자열이 필요합니다.')
        if case['id'] != case['input']['id'] or case['id'] in seen:
            raise ValueError('평가 ID가 중복되었거나 입력 ID와 다릅니다.')
        seen.add(case['id'])
    cases = dataset['cases'][:limit]
    if not cases:
        raise ValueError('평가 사례가 없습니다.')
    # 실행별 파일을 남겨 프롬프트 변경 전후의 결과를 비교할 수 있습니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns() % 1000000000)
    base = output_dir / ('evaluation-' + stamp)
    version = hashlib.sha256((llm.NORMAL_PROMPT + json.dumps(rules, sort_keys=True)).encode()).hexdigest()
    report = {'provenance': dataset['provenance'], 'model': config['model'],
              'reasoning_effort': config['reasoning_effort'], 'base_url': config['base_url'],
              'version': version, 'prompt': llm.NORMAL_PROMPT, 'column_rules': rules,
              'total': len(cases), 'results': []}
    for case in cases:
        print(f"평가 {len(report['results']) + 1}/{len(cases)}: {case['id']}", flush=True)
        item = {'case': case}
        started = time.monotonic()
        try:
            # reference, review는 여기에 전달하지 않습니다. 원문만 다듬습니다.
            item['proposal'] = llm.normalize(client, config, case['input'], rules)
        except Exception as error:
            item['error'] = str(error)
        item['seconds'] = time.monotonic() - started
        report['results'].append(item)
        base.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        render_results(report, base.with_suffix('.html'))
    print(f"완료: {base.with_suffix('.html')}")
    return report


def main(root, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='몇 건만 평가. 생략하면 선택한 시작 위치부터 전체')
    parser.add_argument('--start', type=int, default=1, help='시작 사례 순번. 추가 사례는 51')
    parser.add_argument('--config', type=Path, default=root / 'config.json', help='모델 서버 설정 JSON')
    parser.add_argument('--output', type=Path, help='평가 결과 저장 폴더')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit는 1 이상이어야 합니다.')
    if args.start < 1:
        parser.error('--start는 1 이상이어야 합니다.')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    rules = llm.load_rules(root / 'column_rules.json')
    dataset = json.loads((root / 'evaluation_cases.json').read_text(encoding='utf-8'))
    if args.start > len(dataset['cases']):
        parser.error('--start가 전체 사례 수를 넘었습니다.')
    dataset = dataset | {'cases': dataset['cases'][args.start - 1:]}
    with OpenAI(base_url=config['base_url'], api_key=config['api_key'], timeout=config['timeout_seconds'], max_retries=0) as client:
        report = run_evaluation(client, config, dataset, args.output or root / 'evaluations', rules, args.limit)
    failed = sum('error' in item for item in report['results'])
    if failed:
        raise SystemExit(f'처리 실패 {failed}건. 평가 결과의 오류 내용을 확인하세요.')
