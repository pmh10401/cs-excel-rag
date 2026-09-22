"""LM Studio 벡터 검색 결과 화면."""
import html
from .report import ASSETS
from .vectors import EMBED, CHAT, FIELDS
from .records import text_for

def render(report, destination):
    # 실행 결과를 HTML 파일로 만듭니다. 이 함수 자체는 모델이나 DB를 호출하지 않습니다.
    # '<', '&' 등을 HTML 기호로 해석하지 않고 글자로 표시하도록 변환합니다.
    # 원본과 모델 답변을 그대로 HTML에 끼워 넣지 않기 위한 처리입니다.
    esc = lambda value: html.escape(str(value))
    # th는 열 제목, td는 셀, tr은 행입니다. 입력한 순서대로 원본 표를 구성합니다.
    cols = ('id',) + FIELDS
    table = '<tr>' + ''.join(f'<th>{esc(c)}</th>' for c in cols) + '</tr>'
    table += ''.join('<tr>' + ''.join(f'<td>{esc(row[c])}</td>' for c in cols) + '</tr>' for row in report['rows'])
    # 질문마다 검색 후보와 실제 답변을 나란히 표시하는 영역을 만듭니다.
    # details/summary는 JavaScript 없이도 클릭해서 내용을 펼칠 수 있는 HTML 요소입니다.
    examples = ''
    for case in report['cases']:
        hits = ''.join(f"<details><summary>{esc(h['id'])} · 유사도 {h['score']:.4f}</summary><pre>{esc(h['content'])}</pre></details>" for h in case['hits'])
        examples += f"<article><h3>{esc(case['question'])}</h3><p class='muted'>실제 로컬 실행 · {case['seconds']:.1f}초 · 기대 근거: {esc(case['expected'] or '없음')}</p><div class='columns'><div><h4>③ 검색한 상위 3건</h4>{hits}<p class='muted'>유사도는 정답 확률이 아닙니다. 답이 없는 질문에도 검색 결과는 나옵니다.</p></div><div><h4>④ LM Studio 실제 답변</h4><pre class='answer'>{esc(case['answer'])}</pre></div></div><details><summary>모델에 전달한 근거 원문 보기</summary><pre>{esc(case['context'])}</pre></details></article>"
    # 공통 CSS를 HTML에 포함해 파일 하나로 열 수 있게 합니다.
    # 화면이 좁으면 질문·답변 두 열을 한 열로 배치합니다.
    theme = (ASSETS / 'theme.css').read_text(encoding='utf-8')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>로컬 RAG · 샘플 흐름</title>
<style>{theme}</style>
<main><span class="badge">LOCAL RAG / 실제 실행 결과</span><h1>열에 입력한 내용이<br>근거 있는 답변이 되기까지</h1><p>모든 장비와 점검 내용은 흐름 설명을 위해 만든 가상 데이터입니다. 실제 장비 작업 지침이 아닙니다.</p><div class="flow">① Excel 파일 읽기 → ② SQLite 저장·임베딩 → ③ 관련 기록 검색 → ④ LLM 답변</div>
<section><h2>① Excel에서 읽은 원본 {len(report["rows"])}건</h2><p>입력 파일: <strong>{esc(report["source_file"])}</strong> · 시트: <strong>{esc(report["source_sheet"])}</strong>. 실제 Excel의 첫 행을 열 이름으로 읽었습니다. 원본 값을 보존하고 열 이름을 붙여 검색용 본문으로 만듭니다. 이 단계에서는 LLM 요약을 사용하지 않았습니다.</p><div class="scroll"><table>{table}</table></div></section>
<section><h2>② SQLite와 임베딩</h2><p>파일: <code>demo.sqlite3</code> · 테이블: <code>records</code> · {report['dimensions']}차원 벡터 · 이번 실행에서 새로 임베딩한 행: {report['embedded_rows']}건</p><p>원본 JSON / 검색 본문 / 원본 ID / 본문 해시 / 모델 ID / 임베딩을 저장합니다. 짧은 한 행을 하나의 검색 단위로 사용합니다.</p><p class="muted">임베딩: {esc(EMBED)}<br>답변: {esc(CHAT)}<br>API: 127.0.0.1:1234 · 생성 시각: {esc(report['created_at'])}</p><details><summary>첫 번째 행이 검색용 본문으로 바뀐 모습</summary><pre>{esc(text_for(report['rows'][0]))}</pre></details></section>
<h2>③ → ④ 질문별 검색과 답변</h2>{examples}<p class="muted">이 화면은 저장된 실행 결과입니다. 다시 실행하면 결과가 갱신됩니다. 소수 샘플의 결과이며 한국어 검색 전반의 정확도를 보장하지 않습니다.</p></main></html>'''
    # 결과 화면을 덮어씁니다. 새 질문을 입력하는 웹앱이 아니라 저장된 실행 보고서입니다.
    destination.write_text(page, encoding='utf-8')
