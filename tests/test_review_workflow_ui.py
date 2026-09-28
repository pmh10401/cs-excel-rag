"""Node 기본 모듈로 UI 회귀 검사. 서버·DB·브라우저·모델은 사용하지 않습니다."""
import json
import shutil
import subprocess
from pathlib import Path


def main(root):
    assets = root / 'ragkit' / 'assets'
    source = (assets / 'app.js').read_text(encoding='utf-8')
    html = (assets / 'app.html').read_text(encoding='utf-8')
    assert 'innerHTML' not in source and 'insertAdjacentHTML' not in source
    assert 'id="auto-english" checked' in html
    assert 'id="english-retry-btn"' in html
    assert 'max="1000" step="1" value="200"' in html
    assert '<option value="1000">' in html
    assert "$('english-btn').addEventListener('click', () => startEnglish())" in source
    assert "$('english-retry-btn').addEventListener('click', () => startEnglish(true))" in source
    # 초기 네트워크 요청·이벤트 등록 직전까지만 로드하고 실제 함수를 실행합니다.
    code = source[:source.index("\ndocument.querySelectorAll('.nav-btn')")]
    runner = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
// 이 화면에서 사용하는 선택자·이벤트만 지원하는 DOM 대역입니다.
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.listeners = {}; this._value = ''; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return (this.text || '') + this.children.map(c => c.textContent).join(''); }
  get childNodes() { return this.children; }
  get value() { return this.tagName === 'select' && !this._value ? (this.children.find(c => c.selected) || this.children[0] || {})._value || '' : this._value; }
  set value(value) { this._value = String(value); }
  append(...items) { for (let child of items) { if (typeof child === 'string') { const text = new Element('#text'); text.textContent = child; child = text; } child.parentNode = this; this.children.push(child); } }
  prepend(child) { child.parentNode = this; this.children.unshift(child); }
  replaceChildren(...items) { this.children = []; this.text = ''; this.append(...items); }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(event); }
  focus() { document.activeElement = this; }
  scrollIntoView() { this.scrolled = true; }
  reportValidity() { return true; }
  showModal() { this.open = true; }
  get classList() { return {toggle: () => {}}; }
  matches(selector) {
    if (selector.endsWith(':checked')) return this.checked && this.matches(selector.slice(0, -8));
    const attr = selector.match(/^(.*?)\[([^=]+)="([^"]*)"\]$/);
    if (attr) return this.matches(attr[1]) && (this.attrs[attr[2]] ?? this[attr[2]]) === attr[3];
    if (selector[0] === '#') return this.id === selector.slice(1);
    if (selector[0] === '.') return (this.className || '').split(' ').includes(selector.slice(1));
    return this.tagName === selector;
  }
  closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector); }
  querySelectorAll(selector) {
    const all = this.children.flatMap(c => [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return all;
    return all.filter(el => selector.split(',').some(part => {
      const steps = part.trim().split(/\s+/); let current = el;
      if (!current.matches(steps.pop())) return false;
      while (steps.length) { current = current.parentNode?.closest(steps.pop()); if (!current) return false; }
      return true;
    }));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const document = new Element('document');
document.createElement = tag => new Element(tag);
document.getElementById = id => document.querySelector('#' + id);
const meta = new Element('meta'); meta.attrs.name = 'csrf-token'; meta.content = 'test'; document.append(meta);
for (const match of HTML.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g)) {
  const el = new Element(match[1]); el.id = match[2]; document.append(el);
}
const reader = new Element('input'); reader.attrs.name = 'reader'; reader.value = 'openpyxl'; reader.checked = true; document.append(reader);
document.getElementById('ask-result').append(new Element('h3'));
const context = vm.createContext({document, assert, Event: class { constructor(type) { this.type = type; } },
  setInterval: () => 1, clearInterval: () => {}, confirm: () => true,
  fetch: () => { throw Error('Network is forbidden'); }});
vm.runInContext(SOURCE, context);
vm.runInContext(String.raw`(async () => {
  ui.view = 'source'; $('source-import').open = false;
  showAlert('source-error', '자료.xlsx / 시트 · B1: 빈 이름 <img>');
  assert.equal($('source-import').open, true);
  assert.equal(document.activeElement, $('source-error'));
  assert.equal($('source-error').scrolled, true);
  assert.match($('source-error').textContent, /B1.*<img>/);
  assert.equal($('source-error').querySelectorAll('img').length, 0);
  showAlert('source-error', '');
  assert.equal($('source-error').hidden, true);
  const initialApi = api;
  api = async () => { throw new Error('새 파일.xlsx / 시트!B1: 열 이름 오류'); };
  $('source-meta').hidden = false;
  await inspectSource();
  assert.equal($('source-meta').hidden, true); // 이전 파일 검사 결과를 새 오류 아래에 표시하지 않습니다.
  api = initialApi;
  // 다른 브라우저의 질문은 내 질문 접수를 막지 않고 공유 자료 변경만 잠급니다.
  renderJob({status:'idle', kind:null, server_busy:true, maintenance_running:false});
  assert.equal($('ask-btn').disabled, false);
  assert.equal($('normalize-btn').disabled, true);
  renderJob({status:'queued', kind:'ask', question:'내 질문', server_busy:true, maintenance_running:false, queue_position:2});
  assert.equal($('question').value, '내 질문');
  assert.equal($('ask-btn').disabled, true);
  assert.ok($('ask-progress').textContent.includes('2번째'));
  renderJob({status:'done', kind:'ask', server_busy:true, maintenance_running:false});
  assert.equal($('ask-btn').disabled, false);
  assert.equal($('normalize-btn').disabled, true);
  renderJob({status:'running', kind:'normalize', server_busy:true, maintenance_running:true});
  assert.equal($('ask-btn').disabled, true);
  renderJob({status:'idle', kind:null, server_busy:false, maintenance_running:false});
  const change = (original, proposal) => ({original, proposal});
  assert.deepEqual(reviewReasons(change('압력 10 bar', '압력 20 bar')), ['숫자']);
  assert.deepEqual(reviewReasons(change('10 mm', '10 cm')), ['단위']);
  assert.deepEqual(reviewReasons(change('10mm', '10cm')), ['단위']);
  assert.ok(reviewReasons(change('누수 없음', '누수 있음')).includes('부정'));
  assert.ok(reviewReasons(change('교환 안함', '교환함')).includes('부정'));
  assert.ok(reviewReasons(change('교체 예정 · 미실시', '교체 완료')).includes('예정·미수행'));
  assert.deepEqual(reviewReasons(change('점검 예정', '점검 예정입니다.')), []);
  assert.equal(hasTextChange(change('0', '')), true);
  assert.equal(hasTextChange(change('원문', null)), false);
  for (const [value, peer] of [['<img src=x onerror=alert(1)>', '원문'], ['😀 10 mm', '😀 20 mm'], ['', '삭제'], ['동일', '동일']]) {
    const box = document.createElement('div'); highlightChange(box, value, peer);
    assert.equal(box.textContent, value);
    assert.ok(box.children.every(child => ['mark', '#text'].includes(child.tagName)));
  }
  const markBox = document.createElement('div'); highlightChange(markBox, '압력 20 bar', '압력 10 bar');
  assert.equal(markBox.querySelector('mark').textContent, '2');
  const missing = document.createElement('div'); highlightChange(missing, null, '원문');
  assert.equal(missing.textContent, '제안 없음');

  const cell = (column, original, proposal) => ({column, original, proposal, editable: true, action: 'pending', value: original});
  const rec = {id: 'A', source_key: 'source-A', revision: 'rev-A', has_proposal: true, groups: [
    {name: '일반', cells: [cell('유지', '같음', '같음'), cell('첫째', '점검 예정', '점검 예정입니다.')]},
    {name: '중요', cells: [cell('압력', '10 bar', '20 bar'), cell('둘째', '점검 예정', '점검 예정입니다.')]}
  ]};
  const snapshot = JSON.stringify(rec);
  assert.equal(reviewGroups(rec)[0].name, '중요');
  ui.reviewGrouped = true; ui.reviewChanged = true;
  assert.deepEqual(reviewGroups(rec).map(g => g.cells.length), [1, 2]);
  assert.equal(JSON.stringify(rec), snapshot, '보기 정렬이 원본 레코드를 수정하면 안 됩니다.');
  ui.record = rec; ui.selected = rec.id;
  ui.dirty = {'첫째': {action: 'edit', value: '<script>편집</script>'}};
  renderRecord(rec);
  let articles = document.querySelectorAll('#record-detail .cell');
  assert.equal(articles.length, 3);
  assert.equal(articles[1].querySelector('textarea').value, '<script>편집</script>');
  $('review-changed').checked = false; $('review-changed').dispatchEvent(new Event('change'));
  assert.equal(ui.dirty['첫째'].value, '<script>편집</script>');
  assert.equal(document.querySelectorAll('#record-detail .cell').length, 4);
  ui.reviewChanged = true; renderRecord(rec);
  articles = document.querySelectorAll('#record-detail .cell');
  articles[1].querySelector('.review-pick').checked = true;
  applyBulkReview('accept');
  assert.equal(ui.dirty['첫째'].action, 'accept');
  assert.equal(ui.dirty['둘째'], undefined, '동일 쌍의 다른 셀을 자동 채택하면 안 됩니다.');
  let savedBody;
  api = async (path, options) => {
    if (path === '/api/reviews') { savedBody = options.body; return {saved: 1}; }
    if (path.startsWith('/api/record?')) return rec;
    if (path === '/api/state') return {};
    throw Error(path);
  };
  const realApplyState = applyState, realRefresh = refreshJob, realLoadRecords = loadRecords;
  applyState = () => {}; refreshJob = async () => {}; loadRecords = async () => {};
  await saveReviews();
  assert.equal(savedBody.entries.length, 1);
  assert.deepEqual(savedBody.entries[0], {source_key: 'source-A', id: 'A', revision: 'rev-A', column: '첫째', action: 'accept', value: '점검 예정입니다.'});
  assert.equal(hasDirty(), false);
  api = async path => path.includes('?id=')
    ? {request:'<script>prompt</script>',response:'<img src=x>',request_truncated:false,response_truncated:true}
    : {items:[{id:'trace-1',time:'2026-09-14T00:00:00Z',kind:'normalize',status:'done',model:'<img>'}]};
  await loadLLMMessages();
  const traceDetail = $('llm-messages-list').querySelector('details');
  assert.match(traceDetail.textContent, /문장 다듬기.*응답 수신/);
  traceDetail.open = true; traceDetail.dispatchEvent(new Event('toggle'));
  await Promise.resolve(); await Promise.resolve();
  assert.match(traceDetail.textContent, /<script>prompt<\/script>/);
  assert.match(traceDetail.textContent, /표시 길이 제한/);
  assert.equal(traceDetail.querySelectorAll('script').length, 0);
  assert.equal(traceDetail.querySelectorAll('img').length, 0);

  const fallbackCell = {...cell('원인(1)', '10 bar', '10 bar'), fallback: '<img>숫자 보정 실패'};
  const fallbackRecord = {...rec, groups: [{name: '작업1', cells: [fallbackCell]}]};
  ui.reviewChanged = false;
  renderRecord(fallbackRecord);
  assert.match($('record-detail').textContent, /원문 유지 · 자동 재보정/);
  assert.match($('record-detail').textContent, /원문 유지 \(자동 보정 실패\)/);
  assert.match($('record-detail').textContent, /<img>숫자 보정 실패/);
  assert.equal($('record-detail').querySelectorAll('img').length, 0);
  assert.equal(fallbackCell.action, 'pending');

  const pipeline = {review: {status: 'pending', pending_cells: 2}, english: {status: 'failed', error: '<img>失敗'}, vector: {status: 'disabled'}};
  const statuses = document.createElement('div'); appendPipeline(statuses, pipeline);
  assert.match(statuses.textContent, /검토 대기 2셀.*영어 실패.*벡터 꺼짐/);
  assert.match(statuses.textContent, /<img>失敗/);
  for (const [key, states] of Object.entries({review: ['complete', 'unavailable'], english: ['ready', 'pending', 'empty'], vector: ['ready', 'pending']})) {
    for (const status of states) { const el = document.createElement('div'); appendPipeline(el, {[key]: {status}}); assert.ok(el.textContent); }
  }
  const absent = document.createElement('div'); appendPipeline(absent); assert.equal(absent.children.length, 0);
  loadRecords = realLoadRecords;
  api = async () => ({total: 1, items: [{id: 'A', pipeline}]});
  await loadRecords(); assert.match($('record-list').textContent, /검토 대기 2셀/);
  renderRecord({...rec, pipeline}); assert.match($('record-detail').textContent, /영어 실패: <img>/);

  ui.last = {normalization_workers: 3}; openSettings();
  assert.equal($('auto-english').checked, true); assert.equal($('normalization-workers').value, '3');
  ui.last.settings = {auto_english: false, normalization_workers: 4}; openSettings();
  assert.equal($('auto-english').checked, false);
  let request;
  api = async (path, options) => { request = {path, body: options.body}; return options.body; };
  await saveSettings();
  assert.deepEqual(request, {path: '/api/settings', body: {normalization_workers: 4, auto_english: false}});
  openSettings(); assert.equal($('auto-english').checked, false);
  $('excel-path').value = '/source.xlsx'; $('english-limit').value = '20';
  await startEnglish(true);
  assert.deepEqual(request, {path: '/api/english', body: {limit: 0, failed_only: true}});
  ui.busy = false; await startEnglish();
  assert.equal(request.body.limit, 20); assert.equal(request.body.failed_only, undefined);
  ui.busy = false; ui.pending = false;
  await startOvernight();
  assert.deepEqual(request, {path:'/api/overnight',body:{}});
  renderJob({kind:'overnight',status:'running',stage:'재시도 대기',overnight:{phase:'normalize',status:'waiting',input:{rows:3},attempts:{normalize:2},retry_at:Date.now()/1000+60,failures:[{id:'A',error:'<script>실패'}]}});
  assert.equal($('overnight-btn').disabled,true);
  assert.equal($('overnight-stop-btn').disabled,false);
  assert.match($('overnight-status').textContent,/문장 다듬기 2회차/);
  assert.match($('overnight-error').textContent,/<script>실패/);
  assert.equal($('overnight-steps').children.length,6);
  assert.equal(Number($('overnight-progress').value),1);
  assert.equal($('overnight-badge').textContent,'재시도 대기');
  assert.match($('overnight-retry').textContent,/초 후 자동 재시도/);
  assert.equal($('overnight-ask-btn').hidden,true);
  await stopOvernight();
  assert.deepEqual(request,{path:'/api/overnight/stop',body:{}});
  renderJob({kind:'overnight',status:'running',overnight:{stop_requested:true,status:'waiting'}});
  assert.equal($('overnight-stop-btn').disabled,true);
  assert.equal($('overnight-badge').textContent,'중지 요청됨');
  renderJob({kind:'overnight',status:'done',result:{ready:false,stopped:true}});
  assert.match($('overnight-status').textContent,/중지됨/);
  assert.doesNotMatch($('overnight-status').textContent,/검색 준비 완료/);
  assert.equal($('overnight-stop-btn').disabled,true);
  renderJob({kind:'overnight',status:'done',result:{ready:true}});
  assert.match($('overnight-status').textContent,/검색 준비 완료/);
  assert.equal($('overnight-ask-btn').hidden,false);
  renderJob({kind:'import',status:'done',result:{rows:4}});
  assert.equal($('overnight-ask-btn').hidden,true); // 다른 자료 작업 뒤 이전 완료 표시를 남기지 않습니다.
  ui.busy = false; ui.pending = false;
  ui.last.db = {records: 3, english_failed: 1}; updateLocks(); assert.equal($('english-retry-btn').disabled, false);
  ui.last.db.english_failed = 0; updateLocks(); assert.equal($('english-retry-btn').disabled, true);

  const autoJob = {kind: 'auto_search', status: 'done', result: {
    english: {generated: 2, reused: 3, failures: [{id: 'A', error: '번역 실패'}]},
    english_warning: '영어 예외', embedding_warning: '벡터 예외'
  }};
  renderJob(autoJob);
  assert.match($('job-banner').textContent, /영어·벡터 자동 갱신/);
  assert.match($('job-banner').textContent, /영어 새 변환 2건/);
  assert.match($('job-banner').textContent, /영어 예외/);
  assert.match($('job-banner').textContent, /벡터 예외/);
  const updatedState = {config: {}, db: {records: 3, english_records: 2, english_failed: 1, vector_records: 1}, job: autoJob};
  api = async path => path === '/api/job' ? autoJob : updatedState;
  applyState = realApplyState;
  refreshJob = realRefresh; ui.view = 'source'; await refreshJob();
  assert.match($('english-count').textContent, /2\/3건 · 실패 1건/);
  assert.match($('vectors-count').textContent, /1\/3건/);
  applyState = () => {};
  assert.match($('english-status').textContent, /재사용 3건/);
  assert.match($('english-error').textContent, /A: 번역 실패/);
  assert.match($('english-error').textContent, /영어 예외/);
  assert.match($('vectors-error').textContent, /벡터 예외/);
  ui.view = 'review'; ui.handled = '';
  ui.dirty = {'첫째': {action: 'edit', value: '저장 전 편집'}}; renderRecord(rec);
  const detailBefore = $('record-detail').textContent;
  let listRefreshes = 0, stateRefreshes = 0;
  api = async path => {
    if (path === '/api/job') return autoJob;
    if (path === '/api/state') { stateRefreshes++; return {}; }
    if (path.startsWith('/api/records?')) { listRefreshes++; return {total: 1, items: [{id: 'A', pipeline}]}; }
    throw Error('미저장 검토가 있으면 상세를 다시 읽지 않아야 합니다: ' + path);
  };
  await refreshJob();
  assert.ok(stateRefreshes > 0); assert.equal(listRefreshes, 1);
  assert.equal(ui.dirty['첫째'].value, '저장 전 편집');
  assert.equal($('record-detail').textContent, detailBefore);
  renderJob({kind: 'auto_search', status: 'running'});
  assert.equal(ui.busy, true);
  assert.equal($('save-btn').disabled, true);
  assert.equal($('auto-english').disabled, true);
  assert.equal($('review-changed').disabled, true);
  assert.equal($('english-retry-btn').disabled, true);
  assert.ok(document.querySelectorAll('#record-detail select').every(el => el.disabled));
  setPending(true); setPending(false); assert.equal(ui.busy, true);
  assert.equal($('save-btn').disabled, true, '요청 종료가 실행 중인 자동 갱신 잠금을 풀면 안 됩니다.');
  // 완료 후 상세를 읽는 동안 시작한 편집도 덮어쓰지 않습니다.
  ui.dirty = {}; ui.handled = '';
  api = async path => {
    if (path === '/api/job') return autoJob;
    if (path === '/api/state') return {};
    if (path.startsWith('/api/records?')) return {total: 1, items: [{id: 'A', pipeline}]};
    if (path.startsWith('/api/record?')) { ui.dirty = {'첫째': {action: 'edit', value: '도중 편집'}}; return {...rec, id: '새 상세'}; }
    throw Error(path);
  };
  await refreshJob();
  assert.equal(ui.dirty['첫째'].value, '도중 편집'); assert.equal(ui.record.id, 'A');
  assert.equal($('record-detail').textContent, detailBefore);
  ui.dirty = {}; ui.view = 'source';
  realApplyState({config: {}, db: {records: 3, english_records: 2, english_failed: 1, vector_records: 1}});
  assert.match($('english-count').textContent, /2\/3건 · 실패 1건/);
  assert.match($('vectors-count').textContent, /1\/3건/);

  renderAsk({status: 'done', result: {status: 'ok', answer: '답변', validation_events: [
    {attempt: 1, elapsed_seconds: 0.2, finish_reason: 'stop', category: 'validation', error: '<script>검사 오류</script>', column: '조치', outcome: 'retry'},
    {error: '두 번째'}, '세 번째', {errors: ['네 번째']}
  ]}});
  assert.match($('ask-note').textContent, /검사 안내: <script>검사 오류<\/script>/);
  assert.match($('ask-note').textContent, /외 1건/);
  renderAsk({status: 'error', error: '검사 실패', result: {status: 'validation_failed', answer: '검사 실패', validation_events: [{error:'숫자 오류'}]}});
  assert.equal($('ask-result').hidden, false);
  assert.match($('ask-note').textContent, /숫자 오류/);
  renderAsk({status: 'done', result: {status: 'ok', answer: '답변'}});
  assert.match($('ask-note').textContent, /한국어 지시/);
  renderConversation([{question:'<script>지난 질문</script>', answer_claims:[{text:'배선 수정',receipts:['CS-1']}],answer:'',truncated:false}]);
  assert.equal($('conversation-history').hidden, false);
  assert.match($('conversation-turns').textContent, /<script>지난 질문<\/script>/);
  assert.equal($('conversation-turns').querySelectorAll('script').length, 0);
  ui.followUp = true; renderConversation(ui.conversation); updateLocks();
  assert.equal($('ask-btn').textContent, '이어서 질문하기');
  assert.equal($('compare-btn').disabled, true);
  renderAsk({status:'done',result:{status:'ok',answer:'배선 수정',follow_up:true,resolved_question:'CS-1의 조치는?',instruction_language:'en'}});
  assert.equal($('ask-context').hidden, false);
  assert.equal($('ask-resolved-question').textContent, 'CS-1의 조치는?');
  assert.match($('ask-note').textContent, /영어 지시/);
  renderAsk({status:'done', result:{status:'clarification', answer:'어떤 사례를 뜻하나요?', follow_up:true}});
  assert.equal($('ask-result-title').textContent, '질문 대상 확인');
  assert.equal($('ask-details').open, false);
  assert.equal($('ask-error').hidden, true);
  ui.busy = false; ui.pending = false; ui.searchInitialized = false;
  realApplyState({config: {}, db: {}, settings: {reference_limit: 0}});
  assert.equal($('reference-mode').value, 'auto');
  assert.equal($('reference-limit').disabled, true);
  assert.equal($('reference-manual-fields').hidden, true);
  $('reference-mode').value = 'manual'; $('reference-limit').value = '5'; updateLocks();
  assert.equal($('reference-limit').disabled, false);
  assert.equal($('reference-manual-fields').hidden, false);
  $('question').value = '압력 알람'; $('search-mode').value = 'hybrid';
  $('instruction-language').value = 'en';
  const askRequests = [];
  api = async (path, options) => { askRequests.push({path, body: options.body}); return options.body; };
  refreshJob = async () => {};
  await startAsk();
  assert.equal(askRequests[0].path, '/api/settings'); assert.equal(askRequests[0].body.reference_limit, 5);
  assert.equal(askRequests[1].path, '/api/ask'); assert.equal(askRequests[1].body.reference_limit, 5);
  assert.equal(askRequests[1].body.instruction_language, 'en'); assert.equal(askRequests[1].body.follow_up, false);
  assert.equal($('reference-mode').disabled, true);
  ui.busy = false; $('reference-mode').value = 'auto';
  renderConversation([{question:'지난 질문', answer_claims:[],answer:'이전 답변',truncated:false}]); ui.followUp = true;
  await startAsk(); assert.equal(askRequests.at(-1).body.reference_limit, 0);
  assert.equal(askRequests.at(-1).body.follow_up, true);
  ui.followUp = false; renderConversation([]);
  assert.equal($('conversation-history').hidden, true);
  ui.busy = false; $('reference-mode').value = 'manual'; $('reference-limit').value = '101';
  const callsBefore = askRequests.length;
  await startAsk(); assert.equal(askRequests.length, callsBefore);
  $('reference-limit').value = '1';
  api = async () => { throw Error('설정 저장 실패'); };
  await startAsk(); assert.equal(ui.busy, false);
  assert.match($('ask-error').textContent, /설정 저장 실패/);
  renderAsk({status: 'done', result: {status: 'ok', answer: '답변', references: {
    mode: 'auto', selected: 2, candidates: 100, context_source: 'server', context_window: 8192,
    input_reserve: 3900, output_reserve: 4096, tokenizer: 'o200k_harmony', limited_by: 'tokens'
  }}});
  assert.match($('ask-references').textContent, /참고 사례 2건 \/ 검색 후보 100건/);
  assert.match($('ask-references').textContent, /서버 실제 로드 문맥 8,192토큰/);
  renderAsk({status: 'done', result: {status: 'aggregate', references: {applied_to: 'aggregate'}}});
  assert.match($('ask-references').textContent, /전체 DB 집계/);
  renderAsk({status: 'done', result: {status: 'no_match'}});
  assert.equal($('ask-references').textContent, '');
  const e1 = {id:'CS-1/작업1',receipt:'CS-1',work:'작업1',column:'조치내용(1)',quote:'  교체 안함.\n11.8 L/min. <script>원문</script>',location:'CSList!A2'};
  const e2 = {...e1,id:'CS-1/작업2',work:'작업2',column:'조치내용(2)',quote:'교체 후 30분 재발 없음. 장기 미확인.'};
  const e3 = {...e1,id:'CS-2/접수',receipt:'CS-2',work:'접수',column:'MTTR',quote:''};
  const display = {status:'ok',answer:'모델 답변\n원문 보완 · 본문에는 보이면 안 됩니다.',elapsed_seconds:2,
    claims:[{text:'교체하지 않았습니다. <img>답변',evidence:[e1,e1]},
            {text:'MTTR은 미기재입니다.',evidence:[e3,e1]}],
    source_supplements:[{evidence:[e1,e2,e2]}]};
  renderAsk({status:'done',result:display});
  assert.equal($('ask-answer').querySelectorAll('.answer-claim').length,2);
  assert.equal($('ask-answer').querySelectorAll('.citation-link').length,3);
  assert.ok(!$('ask-answer').textContent.includes('원문 보완'));
  assert.ok(!$('ask-answer').textContent.includes('11.8 L/min'));
  assert.equal($('ask-answer').querySelectorAll('img').length,0);
  assert.equal($('ask-evidence').querySelectorAll('details').length,2);
  assert.equal($('ask-evidence').querySelectorAll('.quote').length,2);
  assert.equal($('ask-evidence').querySelector('.quote').textContent,e1.quote);
  assert.equal($('ask-evidence').querySelectorAll('script').length,0);
  assert.equal($('ask-supplements').querySelectorAll('.quote').length,1);
  assert.equal($('ask-supplements').querySelector('.quote').textContent,e2.quote);
  assert.equal($('ask-supplements-panel').open,false);
  assert.equal($('ask-details').open,false);
  $('ask-answer').querySelector('.citation-link').dispatchEvent(new Event('click'));
  assert.equal($('ask-evidence').querySelector('details').open,true);
  assert.equal(document.activeElement,$('ask-evidence').querySelector('summary'));
  assert.equal($('ask-evidence').querySelector('details').scrolled,true);
  const actionCell = {...e1,quote:'센서 교체 후에도 동일 증상 발생. 배선 수정 후 정상화.',action_outcomes:[
    {column:e1.column,order:1,action_quote:'센서 교체',outcome_quote:'동일 증상 발생.',status:'증상 지속'},
    {column:e1.column,order:2,action_quote:'배선 수정',outcome_quote:'정상화.',status:'정상화'}
  ]};
  const pendingCell = {...e2,quote:'센서 <img> 청소.',action_outcomes:[
    {column:e2.column,order:1,action_quote:'센서 <img> 청소.',outcome_quote:'',status:'결과 미확인'}
  ]};
  renderAsk({status:'done',result:{status:'ok',claims:[{text:'배선 수정 후 정상화되었습니다.',evidence:[actionCell,actionCell]}],
    source_supplements:[{evidence:[actionCell,pendingCell]}]}});
  assert.equal($('ask-evidence').querySelectorAll('.action-outcomes').length,1);
  assert.match($('ask-evidence').textContent,/센서 교체 → 증상 지속/);
  assert.match($('ask-evidence').textContent,/배선 수정 → 정상화/);
  assert.equal($('ask-evidence').querySelector('.quote').textContent,actionCell.quote);
  assert.ok(!$('ask-answer').textContent.includes('센서 교체'));
  assert.match($('ask-supplements').textContent,/결과 미확인/);
  assert.match($('ask-supplements').textContent,/결과가 없다는 뜻은 아닙니다/);
  assert.equal($('ask-supplements').querySelectorAll('img').length,0);
  assert.equal($('ask-supplements-panel').open,false);
  const translatedQuestion = 'The sensor <img src=x> is still faulty. What was done?';
  renderAsk({status:'done',result:{status:'ok',answer:'배선 수정 후 정상화되었습니다.',claims:[],
    search:{mode:'korean',translation_status:'translated',english_query:translatedQuestion,
      english_records:0,korean_ids:[],english_ids:[],warning:''}}});
  assert.ok($('ask-translation'),'번역 질문 확인 영역이 없습니다.');
  assert.equal($('ask-translation').hidden,false);
  assert.equal($('ask-translated-question').textContent,translatedQuestion);
  assert.equal($('ask-translated-question').querySelectorAll('img').length,0);
  assert.ok(!$('ask-answer').textContent.includes(translatedQuestion));
  renderAsk({status:'error',error:'근거 검사 실패',result:{status:'validation_failed',answer:'검사 실패',claims:display.claims}});
  assert.equal($('ask-translation').hidden,true);
  assert.equal($('ask-translated-question').textContent,'');
  assert.equal($('ask-answer').textContent,'검사 실패');
  assert.equal($('ask-details').open,true);
  assert.equal($('ask-evidence').hidden,true);
  assert.equal($('ask-supplements-panel').hidden,true);
  renderAsk({status:'done',result:{status:'aggregate',answer:'전체 100건'}});
  assert.equal($('ask-answer').textContent,'전체 100건');
  assert.equal($('ask-answer').querySelectorAll('.citation-link').length,0);
})()`, context).catch(error => { console.error(error); process.exitCode = 1; });
'''
    node = shutil.which('node')
    assert node, '이 정적 UI 검사는 Node.js가 필요합니다.'
    subprocess.run([node, '--check', str(assets / 'app.js')], check=True)
    subprocess.run([node], input='const SOURCE = ' + json.dumps(code) + ';\nconst HTML = ' + json.dumps(html) + ';\n' + runner,
                   text=True, check=True, timeout=30)
    print('PASS: 변경 강조·우선순위·묶기·편집 보존·개별 저장·설정·재시도·파이프라인·진단 (모델 호출 없음)')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1])
