// Node 표준 라이브러리만으로 실제 완료 처리 함수를 실행합니다. 서버 응답과 DOM 경계만 대체합니다.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../ragkit/assets/app.js'), 'utf8');
const functions = source.slice(source.indexOf('async function refreshJob()'), source.indexOf('function setView('));

(async () => {
  const cases = [
    ['bulk_review', {rows:2, applied_rows:1, saved:1, reviewed_cells:0, missing_rows:0,
      failures:[{id:'CS-FAILED', error:'숫자 검사 실패'}]}, 'batch-review-error', 'CS-FAILED: 숫자 검사 실패'],
    ['vectors', {rows:2, indexed:1, generated:1, failures:[{error:'임베딩 응답 오류'}]}, 'vectors-error', '임베딩 응답 오류'],
    ['english', {rows:2, generated:1, failures:[{id:'CS-EN', error:'번역 실패'}]}, 'english-error', 'CS-EN: 번역 실패'],
  ];
  for (const [kind, result, field, message] of cases) {
    const nodes = new Map(); let answerCount = 0;
    const element = id => { if (!nodes.has(id)) nodes.set(id, {textContent:'', writes:0}); return nodes.get(id); };
    const job = {kind:'ask', status:'done', result:{answer:'질문 답변'}, maintenance:{kind, status:'done', started:1, result}};
    const context = vm.createContext({
      ui:{view:'ask', refreshing:false},
      api:async url => url === '/api/job' ? job : {},
      $:element, setText:(node, text) => {node.textContent=text; node.writes++;},
      showAlert:(id,text) => {element(id).textContent=text; element(id).writes++;},
      renderAsk:() => answerCount++, renderJob:()=>{}, applyState:()=>{},
      startPoll:()=>{}, stopPoll:()=>{}, setBusy:()=>{}, englishSummary:()=>'',
    });
    vm.runInContext(functions, context);
    await context.refreshJob();
    assert.ok(element(field).textContent.includes(message), `${kind}: 준비 실패가 답변 뒤에 숨었습니다: ${element(field).textContent}`);
    assert.equal(answerCount, 1);
    const writes=element(field).writes;
    await context.refreshJob();
    assert.equal(element(field).writes, writes, '같은 결과를 매번 다시 처리합니다.');
    assert.equal(answerCount, 1);
    // 준비 실패로 복구 안내가 생겨도 내 질문 답변은 표시해야 합니다.
    context.ui.handled=''; job.recovery={id:5, active:false};
    await context.refreshJob();
    assert.equal(answerCount, 2, '자료 복구 안내가 내 답변을 가렸습니다.');
    // 질문이 계속 실행 중이어도 자료 실패 상세는 즉시 표시합니다.
    job.status='running'; job.maintenance.started=2; element(field).textContent='';
    await context.refreshJob();
    assert.ok(element(field).textContent.includes(message));
  }
  console.log('PASS: 질문 중·후의 자료 실패 상세 · 복구 안내 중 답변 · 결과 중복 처리 방지');
})().catch(error => { console.error(error); process.exitCode=1; });
