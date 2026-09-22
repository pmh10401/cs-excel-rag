"""API나 DB를 호출하지 않고 비교·검토 화면을 만듭니다."""
import difflib
import html
import json
from pathlib import Path
from .records import public_cells
from .llm import editable
from .input_formats import row_rules

ASSETS = Path(__file__).resolve().parent / 'assets'


def highlight(original, proposed):
    # HTML 태그를 먼저 이스케이프합니다. 셀에 스크립트가 있어도 글자로만 표시합니다.
    left, right = [], []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, original, proposed, autojunk=False).get_opcodes():
        old, new = html.escape(original[a:b]), html.escape(proposed[c:d])
        left.append(f'<del>{old}</del>' if tag != 'equal' and old else old)
        right.append(f'<ins>{new}</ins>' if tag != 'equal' and new else new)
    return ''.join(left), ''.join(right)


def render(report, destination):
    # 입력과 답변의 HTML 기호를 변환해 코드가 아닌 글자로 표시합니다.
    esc = lambda value: html.escape(str(value))
    columns = tuple(dict.fromkeys(key for row in report['rows'] for key in public_cells(row)))
    is_cs = report.get('workflow') == 'cslist'
    input_label = 'xlwings · Excel의 현재 셀 값' if report.get('reader') == 'xlwings' else 'openpyxl · 파일 직접 읽기'
    apply_help = (f'다운로드한 JSON을 {report.get("menu_command", "start.bat")}의 3번 검토 저장 메뉴에서 적용하세요. 이후 4번 메뉴로 질문하세요.'
                  if is_cs else '프로그램 폴더에서 python demo.py --apply-reviews "검토 JSON 경로"로 적용하세요. .venv의 Python을 사용하고, 이후 같은 입력으로 답변을 다시 생성하세요.')
    table = '<tr>' + ''.join(f'<th>{esc(key)}</th>' for key in columns) + '</tr>'
    table += ''.join('<tr>' + ''.join(f'<td>{esc(row.get(key, "해당 양식에 없는 열"))}</td>' for key in columns) + '</tr>' for row in report['rows'])
    comparisons = ''
    for record in report['normalized_records']:
        cells, preserved, previous_group = '', '', None
        row = record['original_cells']
        for key, original in public_cells(row).items():
            if is_cs and row_rules(row, report['column_rules']).get(key) != 'rewrite':
                preserved += f'<tr><th>{esc(key)}</th><td>{esc(original)}</td><td>{esc(row.get("_cell_locations", {}).get(key, ""))}</td></tr>'
                continue
            if is_cs:
                from .cslist import column_group
                group = column_group(key)
                if group != previous_group:
                    cells += f'<tr class="group-heading"><th colspan="4">{esc(group)}</th></tr>'
                    previous_group = group
            value = (record['normalized'] or public_cells(row))[key]
            location = row.get('_cell_locations', {}).get(key, '')
            left, right = highlight(original, value)
            if record.get('fallbacks', {}).get(key):
                right += f'<p>원문 유지 · 자동 재보정 실패: {esc(record["fallbacks"][key])}</p>'
            state = '빈 셀 유지' if not original.strip() else '원문 보존'
            control, attrs = '', ''
            if editable(row, key, report['column_rules']):
                review = record['reviews'][key]
                state = '원문 변경 · 재검토 필요' if review['stale'] else ('검토 대기 · 검색에는 원문 사용' if review['action'] == 'pending' else 'DB에 저장된 선택')
                selected = review['action']
                if selected == 'accept' and review['value'] != value:
                    selected = 'edit'
                    state = '이전에 채택한 문장 유지 · 새 제안과 비교 가능'
                options = ''.join(f'<option value="{a}" {"selected" if a == selected else ""}>{label}</option>' for a, label in
                                  [('pending', '검토 대기'), ('accept', '수정문 채택'), ('original', '원문 유지'), ('edit', '직접 편집')])
                entry = {'source_key': row.get('_source_key', ''), 'id': record['id'], 'revision': record['revision'], 'column': key}
                attrs = f' data-review="{esc(json.dumps(entry, ensure_ascii=False))}" data-original="{esc(original)}" data-proposal="{esc(value)}"'
                draft = review['value'] if selected != 'pending' else value
                control = f'<label>{esc(key)} 선택 <select aria-label="{esc(record["id"] + " " + key + " 검토 선택")}">{options}</select></label><textarea aria-label="{esc(record["id"] + " " + key + " 직접 편집")}" {"" if selected == "edit" else "disabled"}>{esc(draft)}</textarea>'
                if review['stale']:
                    control += f'<details><summary>이전에 저장한 문장 (현재 검색에 미반영)</summary><pre>{esc(review["previous"])}</pre></details>'
            cells += f'<tr{attrs}><th>{esc(key)}<br><small>{esc(location)}</small></th><td>{left}</td><td>{right}</td><td><p>{esc(state)}</p>{control}</td></tr>'
        preserved_html = (f'<details><summary>보존 열 보기 · 식별자·날짜·부품·수치</summary><div class="scroll"><table><tr><th>열 이름</th><th>원문 값</th><th>셀 위치</th></tr>{preserved}</table></div></details>' if preserved else '')
        comparisons += f"<section><h2>{esc(record['id'])} · 셀별 원문·수정문 비교</h2><div class='scroll'><table class='comparison'><tr><th>열 / 원본 위치</th><th>원문 (삭제 표시)</th><th>수정 제안 (추가 표시)</th><th>검토 선택</th></tr>{cells}</table></div>{preserved_html}</section>"
    cases = '<p>아직 답변을 생성하지 않았습니다. 아래 정리문을 원문과 비교해 주세요.</p>' if not report['cases'] else ''
    for case in report['cases']:
        hits = ''.join(f"<details><summary>{esc(hit['id'])} · 검색 점수 {hit['score']:.4f}</summary><pre>{esc(hit['content'])}</pre></details>" for hit in case['hits'])
        quality = case.get('quality', {})
        label = ('DB 집계 결과' if case.get('aggregate') else
                 '근거 부족' if quality.get('answer_status') == 'insufficient_evidence' else '답변')
        sources = {s['id']: s for hit in case['hits'] for s in hit.get('evidence', [])}
        evidence_html = ''
        for claim in quality.get('claims', []) + quality.get('source_supplements', []):
            quotes = ''.join(f'<li>{esc(e["id"])} / {esc(e["column"])}<br>{esc(sources[e["id"]]["locations"].get(e["column"], ""))}<blockquote>{esc(e["quote"]) if e["quote"].strip() else "(원문 셀이 비어 있음)"}</blockquote></li>' for e in claim['evidence'])
            evidence_html += f'<details><summary>{esc(claim["text"])}</summary><ul>{quotes}</ul></details>'
        error_html = f'<p role="alert">{esc(case["error"])}</p>' if case.get('error') else ''
        cases += f"<section><h2>{esc(case['question'])}</h2><h3>{label}</h3><pre>{esc(case['answer'])}</pre>{error_html}<p>{esc(quality.get('note', ''))}</p>{evidence_html}<h3>검색 원문</h3>{hits}<p>{case['seconds']:.1f}초</p></section>"
        if 'review' in case:
            checks = case.get('answer_checks', {})
            cases += f"<section><h3>사람이 확인할 기준</h3><pre>{esc(case['review'])}</pre><p>기대 근거: {esc(case['expected_ids'])} · 검색 포함 검사: {esc(case['retrieval_pass'])}</p><p>지정 표현 검사: {esc(checks.get('passed', '기준 없음'))}<br>누락: {esc(checks.get('missing', []))}<br>금지 표현: {esc(checks.get('forbidden', []))}</p><p>답변 의미의 정확성은 직접 검토해야 합니다.</p></section>"
    theme = (ASSETS / 'theme.css').read_text(encoding='utf-8')
    review_script = (ASSETS / 'review.js').read_text(encoding='utf-8')
    flow = 'CS 원문 보관 → 접수·작업별 문장 검토 → 접수 문서 검색 → 원문 셀 근거를 붙여 답변' if is_cs else 'Excel → GPT-4 셀별 문장 다듬기 → 원문과 함께 SQLite 저장 → 키워드 검색 → API 답변'
    body = cases + comparisons if is_cs and report['cases'] else comparisons + cases
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Excel RAG 실행 결과</title>
<style>{theme}</style>
<main><header class="hero"><div class="eyebrow">EXCEL · RAG WORKSPACE</div><h1>명확한 문장으로,<br>근거 있는 답변까지.</h1><p class="subtitle">셀의 의미를 지키며 문장을 다듬고, 검토한 내용으로 검색합니다.</p></header><p>{flow}</p><p class="metadata">파일: {esc(report['source_file'])} / {esc(report['source_sheet'])}<br>읽기 방식: {input_label}<br>모델: {esc(report['model'])} · 요청한 추론 옵션: {esc(report['reasoning_effort'])}<br>서버: {esc(report['base_url'])}<br>실행 시각: {esc(report['created_at'])}</p><section><details><summary>Excel 원본 {len(report['rows'])}건 · {len(columns)}열 보기</summary><div class="scroll"><table style="min-width:{max(900, len(columns)*150)}px">{table}</table></div></details></section><section><div class="review-toolbar"><div><h2>검토하고 반영하기</h2><p>수정 제안은 채택 전까지 검색에 사용하지 않습니다. {esc(apply_help)} 파일을 다운로드하는 것만으로 DB가 바뀌지는 않습니다.</p></div><button id="download-reviews" type="button">검토 파일 다운로드</button></div><p id="review-notice" role="status" data-apply-help="{esc(apply_help)}">빨간 취소선은 삭제, 초록 밑줄은 추가된 부분입니다. 원문 보존 열과 빈 셀은 편집할 수 없습니다.</p></section>{body}<p>검색 점수는 정답 확률이 아닙니다. 이 데모는 임베딩 의미 검색을 사용하지 않습니다.</p></main><script>{review_script}</script></html>'''
    destination.write_text(page, encoding='utf-8')


def render_comparison(report, destination):
    """동일 질문의 검색 결과와 채택문 품질을 함께 검토하는 독립 보고서입니다."""
    esc = lambda value: html.escape(str(value))
    result_rows, edits = [], []
    for case in report['cases']:
        cells = []
        for variant in ('original', 'accepted'):
            hits = case[f'{variant}_hits']
            passed = case[f'{variant}_pass']
            status = '기준 없음' if passed is None else ('포함' if passed else '누락')
            details = ''.join(f'<details><summary>{esc(h["id"])}</summary><pre>{esc(h["content"])}</pre></details>' for h in hits) or '검색 결과 없음'
            cells.append(f'<td><p>필요한 근거: {status}</p>{details}</td>')
        result_rows.append(f'<tr><th>{esc(case["question"])}<p>{esc(case["review"])}</p></th>{"".join(cells)}</tr>')
    for index, edit in enumerate(report['edits']):
        before, after = highlight(edit['original'], edit['accepted'])
        edits.append(f'<tr data-index="{index}"><th>{esc(edit["id"])}<br>{esc(edit["column"])}</th><td>{before}</td><td>{after}</td><td><label>가독성 <select name="readability" aria-label="{esc(edit["id"] + edit["column"])} 가독성"><option>미평가</option><option>개선</option><option>같음</option><option>악화</option></select></label><label>의미 <select name="meaning" aria-label="{esc(edit["id"] + edit["column"])} 의미"><option>미평가</option><option>유지</option><option>변경됨</option><option>판단 어려움</option></select></label></td></tr>')
    status = f'원문과 다른 채택 셀 {len(report["edits"])}개' if report['edits'] else '변경된 채택문이 없어 두 검색군이 같습니다. 다듬기 효과를 평가할 수 없습니다.'
    totals = []
    for variant in ('original', 'accepted'):
        graded = [c[f'{variant}_pass'] for c in report['cases'] if c[f'{variant}_pass'] is not None]
        totals.append(f'{sum(graded)}/{len(graded)}' if graded else '기준 없음')
    metadata = json.dumps({k: report[k] for k in ('created_at', 'source_file', 'source_sheet', 'rows', 'edits')}, ensure_ascii=False).replace('<', '\\u003c')
    theme = (ASSETS / 'theme.css').read_text(encoding='utf-8')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>원문·채택문 검색 비교</title><style>{theme}</style><main>
<header class="hero"><div class="eyebrow">SEARCH COMPARISON</div><h1>다듬은 문장이,<br>검색에도 도움이 될까요?</h1><p>같은 질문·문서 단위·상위 3개 결과로 원문과 원문+채택문을 비교합니다. 모델을 호출하지 않습니다.</p></header>
<section><p>{esc(report['source_file'])} / {esc(report['source_sheet'])} · {report['rows']}건 · {esc(report['created_at'])}</p><p>{esc(report.get('provenance', ''))}</p><p>{esc(status)}</p><p>필요한 근거 포함: 원문 {totals[0]} · 원문+채택문 {totals[1]}</p><p>미채택 제안은 제외합니다. 기대 근거가 없는 질문은 점수를 매기지 않습니다. 검색 포함 검사는 답변 정확도가 아닙니다.</p></section>
<section><h2>같은 질문으로 검색 비교</h2><div class="scroll"><table><tr><th>질문과 검토 기준</th><th>원문</th><th>원문+채택문</th></tr>{''.join(result_rows)}</table></div></section>
<section><h2>채택문 가독성과 의미 검토</h2><p>원문의 숫자·단위·부정·예정·추정이 유지되는지 확인하세요. 아래 평가는 DB의 채택 선택을 바꾸지 않습니다.</p><div class="scroll"><table><tr><th>접수·열</th><th>원문</th><th>채택문</th><th>검토 결과</th></tr>{''.join(edits)}</table></div><button id="download-evaluation" type="button">품질 평가 다운로드</button><p id="evaluation-status" role="status">평가 선택은 이 화면에만 보관됩니다. 닫기 전에 다운로드하세요.</p></section></main>
<script id="evaluation-data" type="application/json">{metadata}</script><script>
document.getElementById('download-evaluation').addEventListener('click',()=>{{
 const data=JSON.parse(document.getElementById('evaluation-data').textContent);
 data.format='search-quality-review-v1';
 data.edits=data.edits.map((edit,index)=>{{const row=document.querySelector('[data-index="'+index+'"]');return {{...edit,readability:row.querySelector('[name="readability"]').value,meaning:row.querySelector('[name="meaning"]').value}};}});
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}));
 const link=document.createElement('a');link.href=url;link.download='search-quality-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 document.getElementById('evaluation-status').textContent='품질 평가 파일을 다운로드했습니다. DB의 검토 선택은 바뀌지 않았습니다.';
}});</script></html>'''
    destination.write_text(page, encoding='utf-8')
