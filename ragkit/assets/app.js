const TOKEN = document.querySelector('meta[name="csrf-token"]').content;
const $ = (id) => document.getElementById(id);
const CONFIG_LABELS = {
  model: '모델', base_url: '서버', reasoning_effort: '추론',
  timeout_seconds: '시간 제한(초)', max_context_chars: '근거 길이 제한',
  max_tokens: '출력 토큰 한도', context_window: '문맥 한도(설정)', temperature: 'temperature', top_p: 'top_p',
  answer_max_tokens: '답변 토큰', answer_temperature: '답변 temperature', answer_top_p: '답변 top_p'
};
const ASK_TITLE = {
  ok: '답변', aggregate: '집계 결과', no_match: '없는 기록',
  insufficient_evidence: '근거 부족',
  validation_failed: '근거 검사 실패', source_changed: '원문 변경', error: '답변을 만들지 못했습니다'
};

const ui = {
  view: 'source', page: 1, query: '', selected: null, record: null,
  dirty: {}, last: null, handled: '', poll: null, busy: false,
  pending: false, refreshing: false, pages: 1, total: 0, listRequest: 0,
  mappingBlocked: false, mappingPath: '', mappingDirty: false,
  aliasDraft: {}, unknownHeaders: [], missingColumns: [],
  schemaColumns: [], idColumn: '', reviewImportant: true, reviewChanged: false, reviewGrouped: false
};

async function api(path, opts) {
  opts = opts || {};
  const headers = {'X-CSRF-Token': TOKEN};
  if (opts.body && typeof opts.body !== 'string') {
    headers['Content-Type'] = 'application/json';
    opts = Object.assign({}, opts, {body: JSON.stringify(opts.body)});
  }
  Object.assign(headers, opts.headers || {});
  const res = await fetch(path, Object.assign({credentials: 'same-origin'}, opts, {headers: headers}));
  const data = await res.json().catch(function () { return {}; });
  if (!res.ok) {
    const err = new Error(data.error || '요청을 처리하지 못했습니다.');
    err.code = data.code;
    err.status = res.status;
    err.mapping = data.mapping;
    throw err;
  }
  return data;
}

function setText(el, value) { el.textContent = value == null ? '' : String(value); }
function showAlert(id, msg) {
  const el = $(id);
  el.hidden = !msg;
  setText(el, msg || '');
}
function sourceBody() {
  const reader = document.querySelector('input[name="reader"]:checked');
  return {path: $('excel-path').value, reader: reader ? reader.value : 'openpyxl'};
}
function hasDirty() { return Object.keys(ui.dirty).length > 0; }
function confirmLeave() {
  return !hasDirty() || confirm('저장하지 않은 검토 변경이 있습니다. 이동하면 사라집니다. 계속할까요?');
}
function markUnsaved() { $('unsaved').hidden = !hasDirty(); }
function discardReview() {
  ui.dirty = {};
  ui.selected = null;
  ui.record = null;
  markUnsaved();
  setText($('review-status'), '');
  const root = $('record-detail');
  root.replaceChildren();
  const p = document.createElement('p');
  p.className = 'muted';
  p.textContent = '왼쪽 목록에서 접수를 선택하세요.';
  root.append(p);
}
function fillMeta(dl, pairs) {
  dl.replaceChildren();
  dl.hidden = !pairs.length;
  pairs.forEach(function (pair) {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = pair[0];
    dd.textContent = pair[1];
    dl.append(dt, dd);
  });
}
function setBusy(running) {
  ui.busy = running;
  updateLocks();
}
function setPending(running) {
  ui.pending = running;
  updateLocks();
}
function updateLocks() {
  const jobLock = ui.busy || ui.serverBusy;
  const reqLock = ui.pending;
  $('reference-mode').disabled = ui.busy || reqLock;
  $('reference-manual-fields').hidden = $('reference-mode').value === 'auto';
  $('reference-limit').disabled = ui.busy || reqLock || $('reference-mode').value === 'auto';
  ['inspect-btn', 'import-btn', 'normalize-btn', 'header-mapping-save', 'save-settings', 'normalization-workers', 'auto-english', 'vectors-btn', 'vectors-limit', 'review-important', 'review-changed', 'review-grouped'].forEach(function (id) {
    if ($(id)) $(id).disabled = jobLock || reqLock;
  });
  ['ask-btn', 'compare-btn', 'search-mode'].forEach(id => { $(id).disabled = ui.busy || reqLock; });
  const hasRecords = !!(ui.last && ui.last.db && ui.last.db.records);
  if ($('normalize-btn')) $('normalize-btn').disabled = jobLock || reqLock || !hasRecords;
  if ($('import-btn')) $('import-btn').disabled = jobLock || reqLock || ui.mappingBlocked || ui.mappingDirty;
  $('convert-btn').disabled = jobLock || reqLock || ui.mappingBlocked || ui.mappingDirty || !/\.xlsx["']?$/i.test($('excel-path').value.trim());
  const path = $('excel-path');
  if (path) path.disabled = jobLock || reqLock;
  document.querySelectorAll('input[name="reader"]').forEach(function (el) {
    el.disabled = jobLock || reqLock;
  });
  document.querySelectorAll('#header-mapping select, #header-mapping button').forEach(function (el) {
    el.disabled = jobLock || reqLock;
  });
  document.querySelectorAll('.nav-btn').forEach(function (btn) {
    btn.disabled = reqLock;
  });
  if ($('search-q')) $('search-q').disabled = reqLock;
  const searchForm = $('search-form');
  const searchBtn = searchForm && searchForm.querySelector('button');
  if (searchBtn) searchBtn.disabled = reqLock;
  if ($('prev-page')) $('prev-page').disabled = reqLock || ui.page <= 1;
  if ($('next-page')) $('next-page').disabled = reqLock || ui.page >= (ui.pages || 1) || !ui.total;
  document.querySelectorAll('#record-list button').forEach(function (btn) {
    btn.disabled = reqLock;
  });
  document.querySelectorAll('#record-detail select').forEach(function (el) {
    el.disabled = reqLock || jobLock;
  });
  document.querySelectorAll('#record-detail textarea').forEach(function (el) {
    if (reqLock || jobLock) el.disabled = true;
    else {
      const select = el.parentNode.querySelector('select');
      el.disabled = !select || select.value !== 'edit';
    }
  });
  const save = $('save-btn');
  if (save) save.disabled = jobLock || reqLock || !hasDirty();
  $('accept-pending-btn').disabled = jobLock || reqLock || hasDirty() || !(ui.last && ui.last.db && ui.last.db.has_proposals);
  $('english-btn').disabled = jobLock || reqLock || hasDirty() || !hasRecords;
  $('english-retry-btn').disabled = $('english-btn').disabled || !(ui.last && ui.last.db && ui.last.db.english_failed);
  $('english-limit').disabled = jobLock || reqLock;
  $('bilingual').disabled = ui.busy || reqLock;
  $('vectors-btn').disabled = jobLock || reqLock || hasDirty() || !hasRecords;
  $('overnight-btn').disabled = jobLock || reqLock || hasDirty() || !hasRecords;
  $('overnight-stop-btn').disabled = reqLock || !ui.overnightRunning;
  updateBulkReview();
}
function startPoll() {
  if (!ui.poll) ui.poll = setInterval(refreshJob, 1000);
}
function stopPoll() {
  if (ui.poll) { clearInterval(ui.poll); ui.poll = null; }
}

function applyState(s) {
  ui.last = s;
  const path = $('excel-path');
  if (path && document.activeElement !== path && !path.disabled) {
    if (s.excel) path.value = s.excel;
  }
  if (path && ui.mappingPath && path.value !== ui.mappingPath) {
    ui.mappingBlocked = false;
    ui.unknownHeaders = [];
    ui.missingColumns = [];
    ui.mappingPath = '';
  }
  document.querySelectorAll('input[name="reader"]').forEach(function (el) {
    if (el.disabled || document.activeElement === el) return;
    el.checked = el.value === (s.reader || 'openpyxl');
  });
  setText($('output-path'), s.output_dir || '');
  const db = s.db || {};
  setText($('vectors-count'), '검색 가능한 벡터: ' + (db.vector_records || 0) + '/' + (db.records || 0) + '건 · 모델: ' + (s.config.embedding_model || '미설정'));
  const embedding = s.embedding_status || {};
  setText($('embedding-auto-status'), embedding.message || (s.config.embedding_auto_index ? '임베딩 자동 준비·갱신 사용' : '벡터 자동 갱신 꺼짐'));
  $('embedding-auto-status').className = embedding.status === 'error' ? 'error' : 'status';
  if (!ui.searchInitialized) {
    $('search-mode').value = s.config.embedding_auto_index ? 'hybrid' : 'keyword';
    let count = s.settings && s.settings.reference_limit != null ? s.settings.reference_limit : 3;
    try {
      const saved = localStorage.getItem('cs-rag-reference-limit');
      if (saved !== null && Number.isInteger(Number(saved)) && Number(saved) >= 0 && Number(saved) <= 100) count = Number(saved);
    } catch (_) { /* 브라우저 저장을 차단해도 현재 세션에서는 설정을 사용합니다. */ }
    $('reference-mode').value = count === 0 ? 'auto' : 'manual';
    $('reference-limit').value = String(count || 3);
    ui.searchInitialized = true;
  }
  setText($('english-count'), '검색 가능한 영어 자료: ' + (db.english_records || 0) + '/' + (db.records || 0) + '건 · 실패 ' + (db.english_failed || 0) + '건');
  setText($('batch-review-scope'), '대상: 현재 DB 전체 ' + (db.records || 0) + '건 · ' + (s.excel || 'Excel 미선택'));
  fillMeta($('source-meta'), [
    ['입력 경로', s.excel || '아직 없음'],
    ['시트', s.sheet || 'CSList'],
    ['양식', (s.schema_name || '') + ' · ' + (s.columns || 0) + '열'],
    ['현재 DB', db.exists ? db.records + '건' : '없음'],
    ['수정 제안', db.has_proposals ? '있음' : '없음'],
    ['읽기 방식', s.reader === 'xlwings' ? 'Windows Excel에서 열린 파일' : '일반 파일']
  ]);
  ui.schemaColumns = s.schema_columns || [];
  ui.idColumn = s.id_column || ui.idColumn || '';
  if (!ui.mappingDirty) ui.aliasDraft = Object.assign({}, s.header_aliases || {});
  renderHeaderMapping();
  renderJob(s.job || {status: 'idle'});
  if (s.job && (s.job.server_busy || ['running', 'queued'].includes(s.job.status))) startPoll();
}

function renderJob(job) {
  const banner = $('job-banner');
  if (job.kind === 'ask' && job.question && !$('question').value.trim()) $('question').value = job.question;
  const running = ['running', 'queued'].includes(job.status);
  ui.serverBusy = job.server_busy ?? running;
  ui.maintenance = job.maintenance_running ?? (running && job.kind !== 'ask');
  ui.overnightRunning = running && job.kind === 'overnight';
  setBusy(running || ui.maintenance);
  $('convert-progress-panel').hidden = job.kind !== 'convert' || job.status === 'idle';
  if (job.status === 'idle' || !job.kind) {
    if (!running) banner.hidden = true;
    setText($('ask-progress'), ui.serverBusy ? '다른 질문을 처리 중입니다. 질문을 제출하면 대기열에 등록됩니다.' : '');
    return;
  }
  banner.hidden = false;
  banner.className = 'job-banner' + (job.status === 'error' ? ' is-error' : job.status === 'done' ? ' is-done' : '');
  const kind = job.kind === 'import' ? 'DB 등록' : job.kind === 'normalize' ? '문장 다듬기' : job.kind === 'ask' ? '질문' : job.kind === 'bulk_review' ? '전체 제안 채택' : job.kind === 'english' ? '영어 변환' : job.kind === 'convert' ? '입력 파일 변환' : job.kind === 'vectors' ? '벡터 생성' : job.kind === 'auto_vectors' ? '임베딩 자동 준비·갱신' : job.kind === 'auto_search' ? '영어·벡터 자동 갱신' : job.kind;
  let msg = kind + ' · ' + (job.stage || job.status) + ' · ' + (job.elapsed_seconds || 0) + '초';
  if (job.status === 'queued') msg = '질문 대기 중 · 대기 순서 ' + job.queue_position + '번째';
  else if (job.status === 'running') msg = '진행 중 · ' + msg;
  else if (job.status === 'error') msg = (job.code === 'normalization_partial' ? '처리 종료 · ' : '실패 · ') + (job.error || msg);
  else if (job.status === 'done') msg = '완료 · ' + msg;
  if (job.result && job.result.embedding_warning) msg += ' · 임베딩 갱신 실패: ' + job.result.embedding_warning;
  if (job.result && job.result.english_warning) msg += ' · 영어 갱신 실패: ' + job.result.english_warning;
  if (job.result && job.result.english) msg += ' · ' + englishSummary(job.result.english);
  const progress = job.progress;
  if (job.kind === 'convert') {
    const bar = $('convert-progress');
    if (progress && typeof progress.percent === 'number') bar.value = progress.percent;
    else bar.removeAttribute('value');
    let detail = msg;
    if (progress) {
      const rows = (progress.processed_rows || 0).toLocaleString();
      const total = typeof progress.total_rows === 'number' ? '/' + progress.total_rows.toLocaleString() : '';
      detail += ' · 시트 ' + rows + total + '행 · 변환 ' + (progress.completed || 0).toLocaleString() + '건';
      if (typeof progress.percent === 'number') detail += ' · 현재 단계 ' + progress.percent + '%';
      if (job.status === 'error' && progress.stage) detail += ' · 중단 단계: ' + progress.stage;
    }
    setText($('convert-progress-detail'), detail);
    $('convert-excel-hint').hidden = !running || !progress || progress.phase !== 'reading';
  }
  if (progress) {
    if (typeof progress.workers === 'number') msg += ' · 동시 처리 설정 ' + progress.workers + '건';
    if (job.status === 'running' && typeof progress.active === 'number') msg += ' · 처리 중 ' + progress.active + '건';
    if (typeof progress.completed === 'number' && typeof progress.total === 'number') {
      msg += ' · 처리 ' + (progress.attempted ?? progress.completed) + '/' + progress.total + '건 · 성공 ' + progress.completed + '건';
      if (typeof progress.failed === 'number') msg += ' · 실패 ' + progress.failed + '건';
    }
    if (typeof progress.reused === 'number') msg += ' · 재사용 ' + progress.reused + '건';
    if (progress.failed_id) msg += ' · 실패 접수 ' + progress.failed_id;
    if (job.status === 'running' && progress.current_id) msg += ' · ' + progress.current_id;
  }
  setText(banner, msg);
  if (job.kind === 'overnight') {
    const night = job.overnight || job.result || {};
    const names = {normalize:'문장 다듬기',embedding:'벡터',english:'영어 번역'};
    const attempts = Object.entries(night.attempts || {}).map(([key,n]) => (names[key] || key) + ' ' + n + '회차').join(' · ');
    let detail = (night.ready ? '검색 준비 완료' : night.stopped ? '중지됨 · 성공 결과 보관' : msg) + (attempts ? ' · ' + attempts : '');
    const pending = Object.entries(night.pending || {}).map(([key,n]) => (names[key] || key) + ' 오류 ' + n + '건').join(' · ');
    if (pending) detail += ' · ' + pending;
    if (night.embedding && night.embedding.skipped) detail += ' · ' + night.embedding.reason;
    if (night.retry_at) detail += ' · 다음 재시도 ' + new Date(night.retry_at * 1000).toLocaleTimeString();
    setText($('overnight-status'), detail);
    showAlert('overnight-error', [job.error, night.error, ...(night.failures || []).map(item =>
      (item.id || item.phase || '') + ': ' + item.error)].filter(Boolean).join('\n'));
  }
  if (job.kind === 'ask') setText($('ask-progress'), msg);
  if (job.kind === 'bulk_review' && running) setText($('batch-review-status'), msg);
  if (['vectors', 'auto_vectors', 'auto_search'].includes(job.kind) && running) setText($('vectors-status'), msg);
  if (['english', 'auto_search'].includes(job.kind) && running) setText($('english-status'), msg);
}

function englishSummary(result) {
  return '영어 새 변환 ' + (result.generated || 0) + '건 · 동일 내용 재사용 ' + (result.reused || 0) +
    '건 · 실패 ' + (result.failures || []).length + '건' +
    (result.indexed != null ? ' · 검색 가능 ' + result.indexed + '/' + result.rows + '건' : '');
}

async function refreshJob() {
  if (ui.refreshing) return;
  ui.refreshing = true;
  try {
    const job = await api('/api/job');
    renderJob(job);
    if (['running', 'queued'].includes(job.status)) { startPoll(); return; }
    if (job.server_busy) startPoll(); else stopPoll();
    const key = job.status + ':' + job.kind + ':' + (job.error || '') + ':' + JSON.stringify(job.result || null);
    if (!job.kind || job.status === 'idle' || ui.handled === key) return;
    ui.handled = key;
    if (job.kind === 'overnight') applyState(await api('/api/state'));
    if (job.kind === 'convert') {
      if (job.status === 'done') {
        $('excel-path').value = job.result.path;
        document.querySelector('input[name="reader"][value="openpyxl"]').checked = true;
        applyState(await api('/api/state'));
        setText($('source-status'), job.result.rows.toLocaleString() + '건을 변환했습니다. 현재 입력은 변환 파일입니다. 입력 검사 또는 문장 다듬기를 실행하세요.');
        showAlert('source-error', '');
      } else {
        showAlert('source-error', job.error || '변환에 실패했습니다. Excel 입력과 열 대응을 확인하세요.');
      }
    }
    if (job.kind === 'normalize' || job.kind === 'import') {
      if (job.status === 'done') {
        const s = await api('/api/state');
        applyState(s);
        const rows = job.result && job.result.rows;
        const reused = job.result && job.result.reused;
        let text = job.kind === 'import' ? 'DB 등록을 마쳤습니다. Excel 없이 계속 사용할 수 있습니다. ' : '문장 다듬기를 마쳤습니다. ';
        if (rows) text += rows + '건을 저장했습니다. ';
        if (typeof reused === 'number') text += '재사용 ' + reused + '건. ';
        const imported = job.result && job.result.import;
        if (imported) text += '추가 ' + imported.added + '건 · 원문 갱신 ' + imported.updated + '건 · 원문 동일 ' + imported.unchanged + '건 · DB 누적 ' + imported.total + '건. ';
        const fallbackCells = job.result && job.result.fallback_cells || 0;
        if (fallbackCells) text += '숫자·단위 보정 실패로 원문 유지 ' + fallbackCells + '셀. 검토 화면에서 확인할 수 있습니다. ';
        if (s.config && s.config.embedding_auto_index) text += '검토를 반영하면 채택문으로 벡터를 갱신합니다. ';
        text += '출력: ' + s.output_dir;
        setText($('source-status'), text);
        showAlert('source-error', '');
      } else if (job.kind === 'import') {
        setText($('source-status'), 'DB 등록을 완료하지 못했습니다.');
        showAlert('source-error', job.error || '기존 자료는 유지됩니다. 입력을 확인하고 다시 등록하세요.');
      } else {
        setText($('source-status'), job.code === 'normalization_partial' ? '전체 행의 처리를 마쳤습니다. 아래 실패 목록을 확인하세요.' : '');
        const failures = job.progress && job.progress.failures || [];
        const retry = job.code === 'normalization_partial'
          ? ' 나머지 행도 모두 처리했습니다. 성공 결과는 DB에 보관했습니다. 같은 원본·설정으로 이어서 실행하면 실패한 행만 재시도합니다. 검색 자료는 전체 성공 후 갱신됩니다.'
          : ' 성공한 행은 따로 저장되어 같은 단추로 이어서 실행할 수 있습니다.';
        showAlert('source-error', (job.error || '문장 다듬기에 실패했습니다. 원문과 저장된 검토는 유지됩니다.') + retry +
          failures.map(item => '\n' + item.id + ': ' + item.error).join(''));
      }
    }
    if (job.kind === 'ask') renderAsk(job);
    const updatesVectors = ['vectors', 'auto_vectors', 'auto_search'].includes(job.kind) || (job.result && (job.result.embedding || job.result.embedding_warning));
    const updatesEnglish = job.kind === 'english' || job.kind === 'auto_search' || (job.result && (job.result.english || job.result.english_warning));
    if (updatesVectors || updatesEnglish) applyState(await api('/api/state'));
    if (updatesVectors) {
      const r = job.result && (job.result.embedding || (job.kind === 'vectors' || job.kind === 'auto_vectors' ? job.result : null));
      setText($('vectors-status'), r ? r.skipped ? r.reason : '검색 가능 ' + r.indexed + '/' + r.rows + '건 · 새 벡터 ' + r.generated + '개 · 실패 ' + (r.failures || []).length + '개' : '');
      showAlert('vectors-error', [r && (r.failures || []).map(item => item.error).join('\n'), job.result && job.result.embedding_warning,
        job.kind !== 'auto_search' && !r && job.error].filter(Boolean).join('\n'));
    }
    if (updatesEnglish) {
      const result = job.result && (job.result.english || (job.kind === 'english' ? job.result : null));
      const warning = job.result && job.result.english_warning;
      if (result) {
        setText($('english-status'), englishSummary(result));
        showAlert('english-error', [warning, ...(result.failures || []).map(item => item.id + ': ' + item.error)].filter(Boolean).join('\n'));
      } else {
        setText($('english-status'), '');
        showAlert('english-error', warning || (job.status === 'error' ? job.error || '영어 변환을 완료하지 못했습니다.' : ''));
      }
    }
    if (job.kind === 'bulk_review') {
      if (job.status === 'done') {
        const result = job.result;
        applyState(await api('/api/state'));
        setText($('batch-review-status'), '전체 ' + result.rows + '건 중 ' + result.applied_rows + '건의 ' + result.saved +
          '개 셀을 채택·저장하고 검색에 반영했습니다. 기존 검토 ' + result.reviewed_cells + '개 셀 유지 · 제안 없는 접수 ' + result.missing_rows + '건 제외.');
        const failures = result.failures || [];
        showAlert('batch-review-error', failures.length ? '검사에 실패한 ' + failures.length + '건은 제외했습니다.\n' +
          failures.map(item => item.id + ': ' + item.error).join('\n') : '');
      } else {
        setText($('batch-review-status'), '');
        showAlert('batch-review-error', job.error || '일괄 채택을 완료하지 못했습니다.');
      }
    }
    if (ui.view === 'review' && job.kind !== 'ask') {
      await loadRecords();
      if (ui.selected && !hasDirty()) {
        const selected = ui.selected;
        const refreshed = await api('/api/record?id=' + encodeURIComponent(selected));
        if (ui.view === 'review' && ui.selected === selected && !hasDirty()) {
          ui.record = refreshed;
          renderRecord(refreshed);
        }
      }
    }
  } catch (err) {
    stopPoll();
    setBusy(false);
    const banner = $('job-banner');
    banner.hidden = false;
    banner.className = 'job-banner is-error';
    const msg = '서버에 연결하지 못했습니다. ' + (err.message || '');
    setText(banner, msg);
    setText($('ask-progress'), '');
    setText($('source-status'), '');
    if (ui.view === 'ask') showAlert('ask-error', msg);
    else showAlert('source-error', msg);
  } finally {
    ui.refreshing = false;
  }
}

function setView(name) {
  if (ui.pending) return;
  if (ui.view === 'review' && name !== 'review' && !confirmLeave()) return;
  if (ui.view === 'review' && name !== 'review') discardReview();
  ui.view = name;
  document.querySelectorAll('[data-view]').forEach(function (el) { el.hidden = el.dataset.view !== name; });
  document.querySelectorAll('.nav-btn').forEach(function (btn) {
    btn.setAttribute('aria-current', btn.dataset.nav === name ? 'page' : 'false');
  });
  if (name === 'review') loadRecords();
}

function currentSourcePath() {
  return $('excel-path') ? $('excel-path').value : '';
}
function mappingRow(source, targets, isUnknown) {
  const row = document.createElement('div');
  row.className = 'mapping-row';
  const lab = document.createElement('label');
  const cap = document.createElement('span');
  cap.textContent = source;
  const select = document.createElement('select');
  select.setAttribute('aria-label', source + ' 양식 열');
  const empty = document.createElement('option');
  empty.value = '';
  empty.textContent = isUnknown ? '양식 열 선택' : '양식 열';
  select.append(empty);
  const chosen = Object.hasOwn(ui.aliasDraft, source) ? ui.aliasDraft[source] : '';
  const options = (targets || []).slice();
  if (chosen && options.indexOf(chosen) === -1) options.push(chosen);
  options.forEach(function (col) {
    const o = document.createElement('option');
    o.value = col;
    o.textContent = col;
    if (col === chosen) o.selected = true;
    select.append(o);
  });
  select.addEventListener('change', function () {
    ui.mappingDirty = true;
    if (select.value) ui.aliasDraft[source] = select.value;
    else delete ui.aliasDraft[source];
  });
  lab.append(cap, select);
  row.append(lab);
  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'quiet';
  del.textContent = '삭제';
  del.addEventListener('click', function () {
    ui.mappingDirty = true;
    delete ui.aliasDraft[source];
    if (isUnknown) select.value = '';
    else renderHeaderMapping();
  });
  row.append(del);
  return row;
}
function renderHeaderMapping() {
  const panel = $('header-mapping');
  if (!panel) return;
  const savedCount = Object.keys(ui.aliasDraft || {}).length;
  const showUnknown = ui.mappingBlocked && ui.mappingPath && ui.mappingPath === currentSourcePath();
  panel.hidden = !(showUnknown || savedCount > 0 || ui.mappingDirty);
  const fileEl = $('header-mapping-file');
  if (fileEl) setText(fileEl, showUnknown ? ('대상 파일: ' + ui.mappingPath) : '');
  const missEl = $('header-mapping-missing');
  if (missEl) setText(missEl, showUnknown && ui.missingColumns.length ? ('없는 양식 열: ' + ui.missingColumns.join(', ')) : '');
  const unknownBox = $('header-mapping-unknown');
  if (unknownBox) {
    unknownBox.replaceChildren();
    if (showUnknown) {
      ui.unknownHeaders.forEach(function (name) {
        unknownBox.append(mappingRow(name, ui.missingColumns, true));
      });
    }
  }
  const savedBox = $('header-mapping-saved');
  if (savedBox) {
    savedBox.replaceChildren();
    Object.keys(ui.aliasDraft || {}).forEach(function (name) {
      if (showUnknown && ui.unknownHeaders.indexOf(name) !== -1) return;
      savedBox.append(mappingRow(name, ui.schemaColumns, false));
    });
  }
  const savedTitle = $('header-mapping-saved-title');
  if (savedTitle && savedBox) savedTitle.hidden = savedBox.childNodes.length === 0;
  updateLocks();
}
async function saveHeaderMappings() {
  if (ui.busy || ui.pending) return;
  showAlert('source-error', '');
  setPending(true);
  try {
    const aliases = {};
    Object.keys(ui.aliasDraft || {}).forEach(function (key) {
      if (ui.aliasDraft[key]) aliases[key] = ui.aliasDraft[key];
    });
    const result = await api('/api/header-mappings', {method: 'POST', body: {aliases: aliases}});
    ui.aliasDraft = Object.assign({}, result.aliases || aliases);
    ui.mappingDirty = false;
    if (ui.last) ui.last.header_aliases = Object.assign({}, ui.aliasDraft);
  } catch (err) {
    showAlert('source-error', err.message);
    setPending(false);
    return;
  }
  setPending(false);
  await inspectSource();
}
async function inspectSource() {
  if (ui.busy) return;
  showAlert('source-error', '');
  setText($('source-status'), '입력 검사 중');
  setBusy(true);
  try {
    const result = await api('/api/inspect', {method: 'POST', body: sourceBody()});
    const db = result.db || {};
    ui.mappingBlocked = false;
    ui.mappingPath = result.path || '';
    ui.unknownHeaders = [];
    ui.missingColumns = [];
    fillMeta($('source-meta'), [
      ['입력 경로', result.path],
      ['시트', result.sheet],
      ['입력 검사', result.rows + '건 / ' + result.columns + '열'],
      ['식별 열', result.id_column],
      ['현재 DB', db.exists ? db.records + '건' : '없음'],
      ['저장된 원문', result.matches_stored == null ? '비교할 DB 없음' : (result.matches_stored ? '같음' : '다름')]
    ]);
    const incoming = result.import;
    setText($('source-status'), '입력 검사 완료. DB에 추가·갱신을 실행하세요.' + (incoming ?
      ' 추가 ' + incoming.added + '건 · 갱신 ' + incoming.updated + '건 · 동일 ' + incoming.unchanged + '건 · 충돌 ' + incoming.conflicts.length + '건.' : ''));
    showAlert('source-error', result.warning || '');
    const s = await api('/api/state');
    setText($('output-path'), s.output_dir || '');
    ui.last = s;
    ui.schemaColumns = s.schema_columns || ui.schemaColumns;
    ui.idColumn = s.id_column || ui.idColumn;
    if (!ui.mappingDirty) ui.aliasDraft = Object.assign({}, s.header_aliases || {});
    renderHeaderMapping();
  } catch (err) {
    setText($('source-status'), '');
    showAlert('source-error', err.message);
    if (err.code === 'header_mapping_required' && err.mapping) {
      ui.mappingBlocked = true;
      // 입력한 경로와 서버가 해석한 경로는 심볼릭 링크·따옴표 때문에 다를 수 있습니다.
      ui.mappingPath = currentSourcePath();
      ui.missingColumns = err.mapping.missing || [];
      ui.unknownHeaders = err.mapping.unknown || [];
      ui.schemaColumns = err.mapping.schema_columns || ui.schemaColumns;
      ui.idColumn = err.mapping.id_column || ui.idColumn;
      if (!ui.mappingDirty) ui.aliasDraft = Object.assign({}, err.mapping.aliases || {});
      setText($('source-status'), '열 이름을 양식에 맞춘 뒤 저장하고 다시 검사하세요.');
    } else {
      ui.mappingBlocked = false;
    }
    renderHeaderMapping();
  } finally {
    setBusy(false);
  }
}

async function startConvert() {
  if (ui.busy || ui.pending || ui.mappingBlocked || ui.mappingDirty) return;
  if (!confirmLeave()) return;
  discardReview();
  showAlert('source-error', '');
  setText($('source-status'), 'Excel 변환 준비 중');
  setPending(true);
  try {
    await api('/api/convert', {method: 'POST', body: sourceBody()});
    ui.handled = '';
    setBusy(true);
    startPoll();
    await refreshJob();
  } catch (err) {
    showAlert('source-error', err.message);
  } finally {
    setPending(false);
  }
}

async function startDataJob(kind) {
  if (ui.busy) return;
  if (kind === 'import' && ui.mappingDirty) {
    showAlert('source-error', '변경한 열 대응을 먼저 저장하고 입력 검사를 완료하세요.');
    return;
  }
  if (kind === 'import' && ui.mappingBlocked) {
    showAlert('source-error', '열 이름을 맞추고 입력 검사를 통과한 뒤에 문장 다듬기를 실행하세요.');
    return;
  }
  if (!confirmLeave()) return;
  discardReview();
  showAlert('source-error', '');
  ui.handled = '';
  setBusy(true);
  try {
    await api('/api/' + kind, {method: 'POST', body: kind === 'import' ? sourceBody() : {}});
    startPoll();
    await refreshJob();
  } catch (err) {
    setBusy(false);
    showAlert('source-error', err.message);
  }
}

async function loadRecords() {
  const request = ++ui.listRequest;
  showAlert('review-error', '');
  try {
    const data = await api('/api/records?q=' + encodeURIComponent(ui.query) + '&page=' + ui.page + '&page_size=20');
    if (request !== ui.listRequest) return;
    ui.total = data.total || 0;
    ui.pages = Math.max(1, Math.ceil(ui.total / (data.page_size || 20)));
    const pages = ui.pages;
    setText($('list-status'), data.warning || (ui.total ? ui.total + '건' : '저장된 자료가 없습니다.'));
    setText($('page-info'), ui.page + ' / ' + pages + ' 페이지');
    $('prev-page').disabled = ui.pending || ui.page <= 1;
    $('next-page').disabled = ui.pending || ui.page >= pages || !ui.total;
    const list = $('record-list');
    list.replaceChildren();
    (data.items || []).forEach(function (item) {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.disabled = ui.pending;
      btn.classList.toggle('is-selected', item.id === ui.selected);
      btn.textContent = item.id;
      const small = document.createElement('small');
      small.textContent = [item.title, item.customer, item.date].filter(Boolean).join(' · ');
      btn.append(small);
      appendPipeline(btn, item.pipeline);
      btn.addEventListener('click', function () { loadRecord(item.id); });
      li.append(btn);
      list.append(li);
    });
  } catch (err) {
    showAlert('review-error', err.message);
  }
}

function chosenValue(cell, action, editValue) {
  if (action === 'accept') return cell.proposal || '';
  if (action === 'edit') return editValue;
  return cell.original;
}

function appendPipeline(root, pipeline) {
  if (!pipeline) return;
  const labels = {
    review: {pending: '검토 대기', complete: '검토 완료', unavailable: '검토 제안 없음'},
    english: {ready: '영어 준비됨', pending: '영어 대기', failed: '영어 실패', empty: '영어 없음'},
    vector: {ready: '벡터 준비됨', pending: '벡터 대기', disabled: '벡터 꺼짐'}
  };
  const line = document.createElement('small');
  line.className = 'pipeline-status';
  line.textContent = Object.keys(labels).map(key => {
    const state = pipeline[key];
    if (!state || !labels[key][state.status]) return '';
    return labels[key][state.status] + (key === 'review' && state.status === 'pending' ? ' ' + (state.pending_cells || 0) + '셀' : '');
  }).filter(Boolean).join(' · ');
  root.append(line);
  if (pipeline.english && pipeline.english.status === 'failed' && pipeline.english.error) {
    const error = document.createElement('small');
    error.className = 'error';
    error.textContent = '영어 실패: ' + pipeline.english.error;
    root.append(error);
  }
}

function hasTextChange(cell) {
  return cell.proposal != null && String(cell.original ?? '') !== String(cell.proposal);
}

function reviewReasons(cell) {
  if (!hasTextChange(cell)) return [];
  // ponytail: 표현 토큰 비교는 의미 판정이 아닙니다. 누락 사례가 쌓이면 패턴을 보완합니다.
  return [
    ['숫자', /[+−-]?\d+(?:[.,]\d+)*/g],
    ['단위', /(?:[μµ°℃℉%]|(?<![a-z])(?:mm|cm|km|m|ml|l|kg|mg|g|mpa|kpa|pa|bar|psi|rpm|hz|khz|v|mv|a|ma|kw|w|ms|sec|min)(?![a-z])|밀리미터|센티미터|미터|리터|킬로그램|그램|초|분|시간|회|도)/gi],
    ['부정', /없|않|못|아니|불가|불량|금지|(?:^|\s)안(?=\s|[가-힣])|\b(?:no|not|never|without)\b/gi],
    ['예정·미수행', /예정|계획|미\s*(?:실시|수행|조치|완료|확인|점검)|보류|대기|추후|\b(?:planned|pending|unperformed)\b/gi]
  ].filter(([, pattern]) => JSON.stringify(String(cell.original ?? '').match(pattern) || []) !==
    JSON.stringify(String(cell.proposal).match(pattern) || [])).map(([label]) => label);
}

function highlightChange(box, value, other) {
  if (value == null) { box.textContent = '제안 없음'; return; }
  const text = Array.from(String(value));
  const peer = Array.from(String(other ?? ''));
  if (other == null || String(value) === String(other)) { box.textContent = String(value); return; }
  let start = 0, end = text.length, peerEnd = peer.length;
  while (start < end && start < peerEnd && text[start] === peer[start]) start++;
  while (end > start && peerEnd > start && text[end - 1] === peer[peerEnd - 1]) { end--; peerEnd--; }
  // ponytail: 여러 변경 사이의 공통 부분도 포함하는 선형 비교. 정밀 구간이 필요하면 토큰 diff로 교체합니다.
  box.append(text.slice(0, start).join(''));
  if (end > start) {
    const mark = document.createElement('mark');
    mark.textContent = text.slice(start, end).join('');
    box.append(mark);
  }
  box.append(text.slice(end).join(''));
}

function reviewGroups(rec) {
  let groups = rec.groups.map(group => ({...group, cells: group.cells.filter(cell => !ui.reviewChanged || hasTextChange(cell))}));
  if (ui.reviewGrouped) {
    const pairs = new Map();
    groups.flatMap(group => group.cells).forEach(cell => {
      const key = JSON.stringify([cell.original, cell.proposal]);
      if (!pairs.has(key)) pairs.set(key, {name: '원문·제안 쌍', cells: []});
      pairs.get(key).cells.push(cell);
    });
    groups = Array.from(pairs.values()).map(group => ({...group, name: '동일 원문·제안 · ' + group.cells.length + '개 셀'}));
  }
  if (ui.reviewImportant) {
    const priority = cell => Number(Boolean(cell.stale || cell.fallback)) || reviewReasons(cell).length;
    groups.forEach(group => group.cells.sort((a, b) => priority(b) - priority(a)));
    groups.sort((a, b) => Math.max(0, ...b.cells.map(priority)) - Math.max(0, ...a.cells.map(priority)));
  }
  return groups.filter(group => group.cells.length);
}

function updateBulkReview() {
  const all = $('review-select-all');
  if (!all) return;
  const boxes = Array.from(document.querySelectorAll('.review-pick'));
  const selected = boxes.filter(box => box.checked);
  const locked = ui.busy || ui.pending;
  boxes.forEach(box => { box.disabled = locked; });
  all.disabled = locked || !boxes.length;
  all.checked = boxes.length > 0 && selected.length === boxes.length;
  all.indeterminate = selected.length > 0 && selected.length < boxes.length;
  const missing = selected.some(box => box.closest('.cell').querySelector('option[value="accept"]').disabled);
  $('bulk-accept').disabled = locked || !selected.length || missing;
  $('bulk-original').disabled = locked || !selected.length;
  setText($('bulk-count'), '현재 표시된 검토 셀 중 ' + selected.length + '/' + boxes.length + '개 선택' +
    (missing ? ' · 제안 없는 셀을 해제하면 일괄 채택할 수 있습니다.' : ''));
}

function applyBulkReview(action) {
  if (ui.busy || ui.pending || !['accept', 'original'].includes(action)) return;
  const selects = Array.from(document.querySelectorAll('.review-pick:checked'))
    .map(box => box.closest('.cell').querySelector('select'));
  if (!selects.length || selects.some(select => select.querySelector('option[value="' + action + '"]').disabled)) return;
  selects.forEach(select => {
    select.value = action;
    select.dispatchEvent(new Event('change')); // 개별 검토와 동일한 변경·값 검사를 사용합니다.
  });
  setText($('review-status'), selects.length + '개 셀의 선택을 변경했습니다. 검토 저장을 누르면 변경한 셀 모두가 DB에 반영됩니다.');
}

function bindCell(cell, select, textarea) {
  function sync() {
    const action = select.value;
    textarea.disabled = action !== 'edit';
    if (action === 'accept') textarea.value = cell.proposal || '';
    if (action === 'original' || action === 'pending') textarea.value = cell.original;
    const value = chosenValue(cell, action, textarea.value);
    if (action === cell.action && value === cell.value) delete ui.dirty[cell.column];
    else ui.dirty[cell.column] = {action: action, value: value};
    markUnsaved();
    const save = $('save-btn');
    if (save) save.disabled = ui.busy || !hasDirty();
    updateLocks();
  }
  select.addEventListener('change', sync);
  textarea.addEventListener('input', sync);
}

function renderRecord(rec) {
  const root = $('record-detail');
  root.replaceChildren();
  const toolbar = document.createElement('div');
  toolbar.className = 'review-toolbar';
  const title = document.createElement('h2');
  title.textContent = rec.id;
  const save = document.createElement('button');
  save.type = 'button';
  save.id = 'save-btn';
  save.textContent = '검토 저장';
  save.addEventListener('click', saveReviews);
  toolbar.append(title, save);
  root.append(toolbar);
  const bulk = document.createElement('div');
  bulk.className = 'actions';
  const allLabel = document.createElement('label');
  allLabel.className = 'choice';
  const all = document.createElement('input');
  all.type = 'checkbox';
  all.id = 'review-select-all';
  allLabel.append(all, '현재 표시된 셀 전체 선택');
  all.addEventListener('change', () => {
    document.querySelectorAll('.review-pick').forEach(box => { box.checked = all.checked; });
    updateBulkReview();
  });
  bulk.append(allLabel);
  [['accept', '수정문 일괄 채택'], ['original', '원문 일괄 유지']].forEach(([action, text]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.id = 'bulk-' + action;
    button.className = 'quiet';
    button.textContent = text;
    button.addEventListener('click', () => applyBulkReview(action));
    bulk.append(button);
  });
  const count = document.createElement('p');
  count.id = 'bulk-count';
  count.className = 'muted';
  count.setAttribute('role', 'status');
  root.append(bulk, count);
  const head = document.createElement('p');
  head.className = 'muted';
  const h = rec.header || {};
  head.textContent = [h['제목'], h['고객사명'], h['설비코드'], h['접수일자']].filter(Boolean).join(' · ');
  root.append(head);
  appendPipeline(root, rec.pipeline);
  const options = document.createElement('div');
  options.className = 'actions';
  [['review-important', 'reviewImportant', '중요 변경 먼저'], ['review-changed', 'reviewChanged', '변경된 셀만'],
    ['review-grouped', 'reviewGrouped', '동일 원문·제안 묶기']].forEach(([id, key, text]) => {
    const label = document.createElement('label');
    label.className = 'choice';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.id = id;
    input.checked = ui[key];
    input.addEventListener('change', () => {
      ui[key] = input.checked;
      renderRecord(rec);
      $(id).focus();
    });
    label.append(input, text);
    options.append(label);
  });
  const hint = document.createElement('p');
  hint.className = 'muted';
  hint.textContent = '강조는 첫 변경부터 마지막 변경까지의 범위입니다. 중요 표시는 숫자·단위·부정·예정·미수행 표현의 차이를 찾는 보조 기능이며 의미 보존을 보증하지 않습니다. 정렬·필터·묶기는 현재 접수에만 적용되며, 묶인 셀도 개별 검토합니다. 보기 변경 시 일괄 선택은 해제되고 저장 전 편집은 유지됩니다.';
  root.append(options, hint);
  if (!rec.has_proposal) {
    const warn = document.createElement('p');
    warn.className = 'stale';
    warn.textContent = '수정 제안이 없습니다. 자료 화면에서 문장 다듬기를 실행하세요.';
    root.append(warn);
  }
  const groups = reviewGroups(rec);
  if (!groups.length) {
    const empty = document.createElement('p');
    empty.textContent = '현재 조건에 맞는 셀이 없습니다.';
    root.append(empty);
  }
  groups.forEach(function (group) {
    const cells = group.cells.filter(function (c) { return c.editable || !c.preserve; });
    const preserved = group.cells.filter(function (c) { return c.preserve && !c.editable; });
    if (!cells.length && !preserved.length) return;
    const sec = document.createElement('section');
    sec.className = 'group';
    const h3 = document.createElement('h3');
    h3.textContent = group.name;
    sec.append(h3);
    cells.forEach(function (cell) {
      const article = document.createElement('article');
      article.className = 'cell';
      const headEl = document.createElement('p');
      headEl.className = 'cell-head';
      headEl.textContent = cell.column + (cell.location ? ' · ' + cell.location : '');
      article.append(headEl);
      if (cell.fallback) {
        const note = document.createElement('p');
        note.className = 'stale';
        note.textContent = '원문 유지 · 자동 재보정 후에도 숫자·단위 검사를 통과하지 못했습니다. ' + cell.fallback;
        article.append(note);
      }
      const reasons = reviewReasons(cell);
      if (reasons.length) {
        const note = document.createElement('p');
        note.className = 'stale';
        note.textContent = '우선 확인: ' + reasons.join(' · ') + ' 표현 변경';
        article.append(note);
      }
      const compare = document.createElement('div');
      compare.className = 'cell-compare';
      ['원문', '수정 제안'].forEach(function (label, idx) {
        const wrap = document.createElement('div');
        const cap = document.createElement('label');
        cap.textContent = idx === 1 && cell.fallback ? '원문 유지 (자동 보정 실패)' : label;
        const box = document.createElement('div');
        box.className = 'box';
        const value = idx === 0 ? cell.original : cell.proposal;
        highlightChange(box, value, idx === 0 ? cell.proposal : cell.original);
        wrap.append(cap, box);
        compare.append(wrap);
      });
      article.append(compare);
      if (cell.stale) {
        const stale = document.createElement('p');
        stale.className = 'stale';
        stale.textContent = '원문이 바뀌어 이전 검토를 다시 확인하세요.' + (cell.previous ? ' 이전 값: ' + cell.previous : '');
        article.append(stale);
      }
      if (cell.editable) {
        const pickLabel = document.createElement('label');
        pickLabel.className = 'choice';
        const pick = document.createElement('input');
        pick.type = 'checkbox';
        pick.className = 'review-pick';
        pick.setAttribute('aria-label', rec.id + ' ' + cell.column + ' 일괄 검토 선택');
        pick.addEventListener('change', updateBulkReview);
        pickLabel.append(pick, '일괄 검토에 포함');
        article.prepend(pickLabel);
        let action = cell.action;
        if (action === 'accept' && cell.value !== cell.proposal) {
          action = 'edit';
          const keep = document.createElement('p');
          keep.className = 'stale';
          keep.textContent = '이전에 채택한 문장 유지 · 새 제안과 비교 가능';
          article.append(keep);
        }
        const lab = document.createElement('label');
        lab.textContent = cell.column + ' 검토';
        const select = document.createElement('select');
        select.setAttribute('aria-label', rec.id + ' ' + cell.column + ' 검토 선택');
        [['pending', '검토 대기'], ['accept', '수정문 채택'], ['original', '원문 유지'], ['edit', '직접 편집']].forEach(function (opt) {
          const o = document.createElement('option');
          o.value = opt[0];
          o.textContent = opt[1];
          if (opt[0] === 'accept' && cell.proposal == null) o.disabled = true;
          if (opt[0] === action) o.selected = true;
          select.append(o);
        });
        const textarea = document.createElement('textarea');
        textarea.setAttribute('aria-label', rec.id + ' ' + cell.column + ' 직접 편집');
        textarea.value = action === 'edit' ? cell.value : (action === 'accept' ? (cell.proposal || '') : cell.original);
        const draft = ui.dirty[cell.column];
        if (draft) {
          select.value = draft.action;
          textarea.value = draft.value;
        }
        textarea.disabled = select.value !== 'edit';
        lab.append(select);
        article.append(lab, textarea);
        bindCell(Object.assign({}, cell, {action: action}), select, textarea);
      }
      sec.append(article);
    });
    if (preserved.length) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = '보존 열 보기';
      details.append(summary);
      preserved.forEach(function (cell) {
        const item = document.createElement('p');
        item.textContent = cell.column + '\n' + (cell.original == null ? '' : String(cell.original)) + '\n' + (cell.location || '');
        details.append(item);
      });
      sec.append(details);
    }
    root.append(sec);
  });
  save.disabled = ui.busy || ui.pending || !hasDirty();
  markUnsaved();
  updateLocks();
}

async function loadRecord(id) {
  if (ui.pending) return;
  if (id === ui.selected && ui.record && ui.record.id === id) return;
  if (!confirmLeave()) return;
  setPending(true);
  showAlert('review-error', '');
  setText($('review-status'), '');
  try {
    const rec = await api('/api/record?id=' + encodeURIComponent(id));
    ui.dirty = {};
    markUnsaved();
    ui.selected = id;
    ui.record = rec;
    renderRecord(rec);
    document.querySelectorAll('#record-list button').forEach(function (btn) {
      const label = btn.childNodes[0] && btn.childNodes[0].textContent;
      btn.classList.toggle('is-selected', label === id);
    });
  } catch (err) {
    showAlert('review-error', err.message);
  } finally {
    setPending(false);
  }
}

async function saveReviews() {
  const rec = ui.record;
  if (!rec || !hasDirty() || ui.pending || ui.busy) return;
  const entries = Object.keys(ui.dirty).map(function (column) {
    const change = ui.dirty[column];
    return {
      source_key: rec.source_key, id: rec.id, revision: rec.revision,
      column: column, action: change.action, value: change.value
    };
  });
  showAlert('review-error', '');
  setPending(true);
  try {
    const result = await api('/api/reviews', {method: 'POST', body: {format: 'cell-reviews-v1', entries: entries}});
    const saved = await api('/api/record?id=' + encodeURIComponent(rec.id));
    ui.dirty = {};
    ui.record = saved;
    renderRecord(saved);
    applyState(await api('/api/state'));
    markUnsaved();
    setText($('review-status'), result.saved + '개 셀의 검토를 저장했습니다.');
    await loadRecords();
    await refreshJob();
  } catch (err) {
    showAlert('review-error', err.message);
  } finally {
    setPending(false);
  }
}

async function acceptPendingReviews() {
  if (ui.busy || ui.pending || hasDirty()) return;
  setPending(true);
  showAlert('batch-review-error', '');
  setText($('batch-review-status'), '현재 자료 전체의 미검토 제안을 확인합니다.');
  try {
    await api('/api/reviews/accept-pending', {method: 'POST', body: {}});
    ui.handled = '';
    setBusy(true);
    startPoll();
    await refreshJob();
  } catch (err) {
    setText($('batch-review-status'), '');
    showAlert('batch-review-error', err.message);
  } finally {
    setPending(false);
  }
}

async function startEnglish(failedOnly = false) {
  if (ui.busy || ui.pending || hasDirty()) return;
  setPending(true);
  showAlert('english-error', '');
  try {
    const scope = failedOnly ? {limit: 0, failed_only: true} : {limit: Number($('english-limit').value)};
    await api('/api/english', {method: 'POST', body: scope});
    ui.handled = '';
    setBusy(true);
    startPoll();
    await refreshJob();
  } catch (err) {
    showAlert('english-error', err.message);
  } finally {
    setPending(false);
  }
}

async function startOvernight() {
  if (ui.busy || ui.pending || hasDirty()) return;
  setPending(true);
  showAlert('overnight-error', '');
  try {
    await api('/api/overnight', {method:'POST', body:{}});
    ui.handled = '';
    setBusy(true);
    startPoll();
    await refreshJob();
  } catch (err) { showAlert('overnight-error', err.message); }
  finally { setPending(false); }
}

async function stopOvernight() {
  if (ui.pending || !ui.overnightRunning) return;
  setPending(true);
  try {
    await api('/api/overnight/stop', {method:'POST',body:{}});
    await refreshJob();
  } catch (err) { showAlert('overnight-error',err.message); }
  finally { setPending(false); }
}

async function startVectors() {
  if (ui.busy || ui.pending || hasDirty()) return;
  setPending(true);
  showAlert('vectors-error', '');
  try {
    await api('/api/vectors', {method: 'POST', body: {limit: Number($('vectors-limit').value)}});
    ui.handled = '';
    setBusy(true);
    startPoll();
    await refreshJob();
  } catch (err) { showAlert('vectors-error', err.message); }
  finally { setPending(false); }
}

function renderEvidence(claims, boxId = 'ask-evidence', seen = new Set()) {
  const box = $(boxId);
  box.replaceChildren();
  const groups = new Map();
  (claims || []).forEach(function (claim) {
    (claim.evidence || []).forEach(function (ev) {
      const key = JSON.stringify([ev.id, ev.column]);
      if (seen.has(key)) return;
      seen.add(key);
      const receipt = ev.receipt || ev.id.split('/')[0];
      if (!groups.has(receipt)) {
        const detail = document.createElement('details');
        detail.className = 'evidence';
        const summary = document.createElement('summary');
        detail.append(summary);
        box.append(detail);
        groups.set(receipt, {number: groups.size + 1, detail, summary, count: 0});
      }
      const group = groups.get(receipt);
      group.count += 1;
      group.summary.textContent = (boxId === 'ask-evidence' ? '[' + group.number + '] ' : '') +
        receipt + ' · 원문 ' + group.count + '개';
      const cell = document.createElement('section');
      cell.className = 'evidence-cell';
      const heading = document.createElement('h4');
      heading.textContent = [ev.work, ev.column].filter(Boolean).join(' · ');
      const loc = document.createElement('p');
      loc.className = 'muted';
      loc.textContent = ev.location || '셀 위치 없음';
      const quote = document.createElement('pre');
      quote.className = 'quote';
      quote.textContent = ev.quote ? ev.quote : '(원문 셀이 비어 있음)';
      cell.append(heading, loc);
      if ((ev.action_outcomes || []).length) {
        const outcomes = document.createElement('div');
        outcomes.className = 'action-outcomes';
        outcomes.setAttribute('aria-label', '원문에서 확인한 조치별 결과');
        ev.action_outcomes.forEach(function (step) {
          const line = document.createElement('p');
          // 원문과 같은 긴 미확인 구절은 아래 원문 영역에서 한 번만 보여 줍니다.
          line.textContent = step.action_quote === ev.quote && step.status === '결과 미확인'
            ? '결과 미확인 · 조치별 자동 연결 보류'
            : step.order + '. ' + step.action_quote + ' → ' + step.status;
          outcomes.append(line);
        });
        if (ev.action_outcomes.some(step => step.status === '결과 미확인')) {
          const note = document.createElement('p');
          note.className = 'muted';
          note.textContent = '자동 연결을 확정하지 못했습니다. 원문에 결과가 없다는 뜻은 아닙니다. 관찰 내용과 제한을 확인하세요.';
          outcomes.append(note);
        }
        cell.append(outcomes);
      }
      cell.append(quote);
      group.detail.append(cell);
    });
  });
  box.hidden = groups.size === 0;
  return groups;
}

function renderAnswer(result, showClaims) {
  const seen = new Set();
  const groups = renderEvidence(showClaims ? result.claims : [], 'ask-evidence', seen);
  const extras = renderEvidence(showClaims ? result.source_supplements : [], 'ask-supplements', seen);
  $('ask-evidence-title').hidden = !groups.size;
  $('ask-evidence-title').textContent = '인용 근거 · ' + groups.size + '건';
  $('ask-supplements-panel').hidden = !extras.size;
  $('ask-supplements-panel').open = false;
  const box = $('ask-answer');
  box.replaceChildren();
  if (!showClaims || !(result.claims || []).length) {
    box.textContent = result.answer || '';
    return;
  }
  result.claims.forEach(function (claim) {
    const paragraph = document.createElement('p');
    paragraph.className = 'answer-claim';
    paragraph.textContent = claim.text;
    const cited = new Set();
    (claim.evidence || []).forEach(function (ev) {
      const receipt = ev.receipt || ev.id.split('/')[0];
      const group = groups.get(receipt);
      if (!group || cited.has(receipt)) return;
      cited.add(receipt);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'citation-link';
      button.textContent = '[' + group.number + ']';
      button.setAttribute('aria-label', '인용 근거 ' + group.number + ', ' + receipt + ' 원문 보기');
      button.addEventListener('click', function () {
        group.detail.open = true;
        group.summary.focus();
        group.detail.scrollIntoView({block: 'nearest'});
      });
      paragraph.append(button);
    });
    box.append(paragraph);
  });
}

function renderAsk(job) {
  const result = job.result;
  const failed = job.status === 'error' || (result && result.status && !['ok', 'aggregate', 'no_match', 'insufficient_evidence'].includes(result.status));
  const code = (result && result.status) || job.code || (job.status === 'error' ? 'error' : 'ok');
  showAlert('ask-error', job.status === 'error' ? (job.error || '') : (code === 'source_changed' || code === 'validation_failed' ? (result && result.error || job.error || '') : ''));
  if (job.status === 'error' && !result) {
    $('ask-result').hidden = true;
    return;
  }
  if (!result) return;
  $('ask-result').hidden = false;
  $('ask-result').className = 'kind-' + code;
  setText($('ask-result-title'), result.comparison ? '검색 비교' : ASK_TITLE[code] || '답변');
  renderAnswer(result, code === 'ok' && !result.comparison);
  setText($('ask-status'), [code === 'ok' && (result.claims || []).length ? '인용·수치 검사 통과' : '',
    result.elapsed_seconds != null ? result.elapsed_seconds + '초' : ''].filter(Boolean).join(' · '));
  $('ask-details').open = failed;
  const refs = result.references;
  let referenceText = '';
  if (refs) {
    if (refs.applied_to === 'aggregate') referenceText = '전체 DB 집계 · 참고 사례 수 제한을 적용하지 않습니다.';
    else if (refs.applied_to === 'comparison') referenceText = '검색 비교 후보 ' + refs.selected + '건 · 답변 입력 분량 계산은 질문할 때 적용됩니다.';
    else {
      referenceText = (refs.mode === 'auto' ? '자동' : '직접 지정 최대 ' + refs.requested_limit + '건') +
        ' · 답변 참고 사례 ' + refs.selected + '건 / 검색 후보 ' + refs.candidates + '건';
      if (refs.mode === 'auto' && refs.context_source) {
        const source = {server: '서버 실제 로드 문맥', config: '설정 문맥 · 서버 미확인',
          config_and_server: '설정값과 서버 문맥 중 작은 한도', unknown: '문맥 미확인 · 문자 제한 사용'};
        referenceText += ' · ' + source[refs.context_source] + (refs.context_window ? ' ' + refs.context_window.toLocaleString() + '토큰' : '');
        referenceText += ' · 입력 예약 ' + refs.input_reserve.toLocaleString() + '토큰 (' +
          (refs.tokenizer === 'o200k_harmony' ? 'GPT-OSS 토큰 계산 + 여유' : 'UTF-8 바이트 보수적 추정') + ')';
        if (refs.output_reserve) referenceText += ' · 답변·추론 예약 ' + refs.output_reserve.toLocaleString() + '토큰';
        if (refs.limited_by === 'tokens') referenceText += ' · 문맥 한도에 맞춰 사례 수를 줄였습니다.';
        if (refs.limited_by === 'characters') referenceText += ' · 문자 제한에 맞춰 사례 수를 줄였습니다.';
      }
    }
  }
  setText($('ask-references'), referenceText);
  setText($('ask-note'), (result.elapsed_seconds != null ? result.elapsed_seconds + '초' : '') + (result.note ? ' · ' + result.note : ''));
  const diagnostics = (Array.isArray(result.validation_events) ? result.validation_events : []).flatMap(event => {
    if (typeof event === 'string') return [event];
    if (!event) return [];
    return event.errors || event.error || [];
  }).filter(item => typeof item === 'string' && item);
  if (diagnostics.length) setText($('ask-note'), [$('ask-note').textContent,
    '검사 안내: ' + diagnostics.slice(0, 3).map(error => error.slice(0, 180)).join(' / ') +
    (diagnostics.length > 3 ? ' 외 ' + (diagnostics.length - 3) + '건' : '')].filter(Boolean).join(' · '));
  const search = result.search;
  const translatedQuestion = search && search.translation_status === 'translated' ? search.english_query : '';
  $('ask-translation').hidden = !translatedQuestion;
  setText($('ask-translated-question'), translatedQuestion || '');
  $('ask-search').hidden = !search;
  if (search) setText($('ask-search-detail'), [
    search.mode === 'bilingual' ? '한국어 + 영어 검색' : '한국어 검색',
    '영어 검색 자료: ' + search.english_records + '건',
    '영어 질문: ' + (search.english_query || '사용하지 않음'),
    '한국어 후보: ' + search.korean_ids.join(', '),
    '영어 후보: ' + search.english_ids.join(', '), search.warning
  ].filter(Boolean).join('\n'));
  if (search && search.warning) setText($('ask-note'), $('ask-note').textContent + ' · ' + search.warning);
  const vector = result.vector_search;
  if (vector) {
    $('ask-search').hidden = false;
    const text = ['검색 방식: ' + vector.mode, '벡터 자료: ' + vector.vector_records + '건',
      '키워드 후보: ' + vector.keyword_ids.join(', '), '벡터 후보: ' + vector.vector_ids.join(', '),
      '결합 후보: ' + vector.hybrid_ids.join(', '), vector.warning].filter(Boolean).join('\n');
    setText($('ask-search-detail'), (search ? $('ask-search-detail').textContent + '\n\n' : '') + text);
    if (vector.warning) setText($('ask-note'), $('ask-note').textContent + ' · ' + vector.warning);
  }
  const timingLabels = {source_check: 'DB 원문 확인', index_prepare: '검색 자료 준비',
    keyword: '키워드 검색', bilingual: '한국어·영어 검색', index_load: '벡터 자료 준비',
    embedding: '질문 임베딩', vector_search: '벡터 검색', source_lookup: '근거 원문 조회',
    fusion: '검색 순위 결합', retrieval_total: '검색 전체', answer: '답변 생성·검사', total: '전체'};
  const times = Object.entries(result.timings || {}).filter(([key]) => timingLabels[key])
    .map(([key, value]) => timingLabels[key] + ': ' + Number(value).toFixed(3) + '초');
  if (times.length) {
    $('ask-search').hidden = false;
    const previous = search || vector ? $('ask-search-detail').textContent + '\n\n' : '';
    setText($('ask-search-detail'), previous + '처리 시간\n' + times.join('\n') +
      (result.source_db ? '\nDB에 등록된 원문 사용 · Excel 파일 불필요' : '') +
      (result.index_rebuilt ? '\n이전 버전의 검색 자료를 한 번 갱신했습니다.' : ''));
  }
  if (['no_match', 'aggregate', 'insufficient_evidence'].includes(code)) showAlert('ask-error', '');
  if (failed && result.error) showAlert('ask-error', result.error);
}

async function startAsk(comparison = false) {
  if (ui.busy || ui.pending) return;
  const question = $('question').value.trim();
  if (!question) { showAlert('ask-error', '질문을 입력하세요.'); return; }
  const referenceLimit = $('reference-mode').value === 'auto' ? 0 : Number($('reference-limit').value);
  if (!Number.isInteger(referenceLimit) || referenceLimit < ($('reference-mode').value === 'auto' ? 0 : 1) || referenceLimit > 100) {
    showAlert('ask-error', '참고 사례 수를 1~100 사이 정수로 입력하세요.'); return;
  }
  showAlert('ask-error', '');
  $('ask-result').hidden = true;
  ui.handled = '';
  setBusy(true);
  try {
    const settings = await api('/api/settings', {method: 'POST', body: {reference_limit: referenceLimit}});
    if (ui.last) ui.last.settings = Object.assign({}, ui.last.settings, settings);
    try { localStorage.setItem('cs-rag-reference-limit', String(referenceLimit)); } catch (_) {}
    await api(comparison ? '/api/compare-search' : '/api/ask', {method: 'POST', body: {question: question, bilingual: $('bilingual').checked, search_mode: $('search-mode').value, reference_limit: referenceLimit}});
    startPoll();
    await refreshJob();
  } catch (err) {
    setBusy(false);
    setText($('ask-progress'), '');
    showAlert('ask-error', err.message);
  }
}

async function loadLLMMessages() {
  const list = $('llm-messages-list');
  const status = $('llm-messages-status');
  setText(status, '기록을 불러오는 중입니다.');
  try {
    const data = await api('/api/llm-messages');
    list.replaceChildren();
    setText(status, data.warning || (data.items.length ? '항목을 펼쳐 확인하세요. 새로고침하면 진행 상태를 갱신합니다.' : '아직 기록된 요청이 없습니다.'));
    const kinds = {normalize: '문장 다듬기', english: '영어 번역', auto_search: '자동 번역', ask: '질문·답변'};
    for (const item of data.items) {
      const detail = document.createElement('details');
      const title = document.createElement('summary');
      title.textContent = new Date(item.time).toLocaleString() + ' · ' + (kinds[item.kind] || item.kind) + ' · ' +
        ({running: '응답 대기', done: '응답 수신', error: '요청 실패'}[item.status] || item.status) + ' · ' + item.model;
      detail.append(title);
      let loaded = false;
      detail.addEventListener('toggle', async function () {
        if (!detail.open || loaded) return;
        loaded = true;
        try {
          const record = await api('/api/llm-messages?id=' + encodeURIComponent(item.id));
          for (const [key, label] of [['request', '보낸 요청'], ['response', '받은 응답']]) {
            const heading = document.createElement('h3');
            heading.textContent = label + (record[key + '_truncated'] ? ' (표시 길이 제한으로 일부 생략)' : '');
            const body = document.createElement('pre');
            body.className = 'box';
            body.textContent = record[key] || '아직 응답이 없습니다.';
            detail.append(heading, body);
          }
        } catch (err) {
          const message = document.createElement('p'); message.textContent = err.message; detail.append(message);
        }
      });
      list.append(detail);
    }
  } catch (err) { setText(status, err.message); }
}

function openSettings() {
  const s = ui.last || {};
  $('normalization-workers').value = String((s.settings && s.settings.normalization_workers) || s.normalization_workers || 200);
  $('auto-english').checked = !s.settings || s.settings.auto_english !== false;
  setText($('settings-status'), '');
  showAlert('settings-error', '');
  updateLocks();
  const cfg = s.config || {};
  const pairs = [];
  Object.keys(CONFIG_LABELS).forEach(function (key) {
    if (cfg[key] != null) pairs.push([CONFIG_LABELS[key], String(cfg[key])]);
  });
  if (cfg.max_tokens == null) pairs.push(['출력 토큰 한도', '서버 기본값']);
  if (cfg.context_window != null) pairs.push(['요청별 출력 한도', '입력 분량과 여유 공간을 제외해 자동 조절']);
  pairs.push(['출력 폴더', s.output_dir || '']);
  pairs.push(['양식', s.schema_name || '']);
  fillMeta($('settings-meta'), pairs);
  $('settings').showModal();
}

async function saveSettings() {
  if (ui.busy || ui.pending) return;
  if (!$('normalization-workers').reportValidity()) return;
  setPending(true);
  showAlert('settings-error', '');
  setText($('settings-status'), '');
  try {
    const result = await api('/api/settings', {method: 'POST', body: {
      normalization_workers: Number($('normalization-workers').value),
      auto_english: $('auto-english').checked
    }});
    const settings = result.settings || result;
    if (ui.last) {
      ui.last.normalization_workers = settings.normalization_workers;
      ui.last.settings = Object.assign({}, ui.last.settings, settings);
    }
    setText($('settings-status'), '동시 처리 ' + settings.normalization_workers + '건 · 검토 저장 후 영어 자동 갱신 ' +
      (settings.auto_english === false ? '꺼짐' : '켜짐') + '으로 저장했습니다.');
  } catch (err) {
    showAlert('settings-error', err.message);
  } finally {
    setPending(false);
  }
}

function changePage(delta) {
  if (ui.pending) return;
  if (!confirmLeave()) return;
  discardReview();
  ui.page += delta;
  if (ui.page < 1) ui.page = 1;
  loadRecords();
}

document.querySelectorAll('.nav-btn').forEach(function (btn) {
  btn.addEventListener('click', function () { setView(btn.dataset.nav); });
});
$('open-settings').addEventListener('click', openSettings);
$('save-settings').addEventListener('click', saveSettings);
$('source-form').addEventListener('submit', function (ev) { ev.preventDefault(); inspectSource(); });
$('normalize-btn').addEventListener('click', () => startDataJob('normalize'));
$('import-btn').addEventListener('click', () => startDataJob('import'));
$('overnight-btn').addEventListener('click', startOvernight);
$('overnight-stop-btn').addEventListener('click', stopOvernight);
$('convert-btn').addEventListener('click', startConvert);
$('accept-pending-btn').addEventListener('click', acceptPendingReviews);
$('vectors-btn').addEventListener('click', startVectors);
$('compare-btn').addEventListener('click', () => startAsk(true));
$('search-mode').addEventListener('change', updateLocks);
$('reference-mode').addEventListener('change', updateLocks);
$('open-llm-messages').addEventListener('click', () => { $('llm-messages').showModal(); loadLLMMessages(); });
$('refresh-llm-messages').addEventListener('click', loadLLMMessages);
$('english-btn').addEventListener('click', () => startEnglish());
$('english-retry-btn').addEventListener('click', () => startEnglish(true));
if ($('header-mapping-save')) $('header-mapping-save').addEventListener('click', saveHeaderMappings);
if ($('excel-path')) $('excel-path').addEventListener('input', function () {
  updateLocks();
  if (ui.mappingPath && $('excel-path').value !== ui.mappingPath) {
    ui.mappingBlocked = false;
    ui.unknownHeaders = [];
    ui.missingColumns = [];
    ui.mappingPath = '';
    renderHeaderMapping();
    updateLocks();
  }
});
document.querySelectorAll('input[name="reader"]').forEach(function (el) {
  el.addEventListener('change', function () {
    ui.mappingBlocked = false;
    ui.mappingPath = '';
    ui.unknownHeaders = [];
    ui.missingColumns = [];
    renderHeaderMapping();
  });
});
$('search-form').addEventListener('submit', function (ev) {
  ev.preventDefault();
  if (ui.pending) return;
  if (!confirmLeave()) return;
  discardReview();
  ui.query = $('search-q').value.trim();
  ui.page = 1;
  loadRecords();
});
$('prev-page').addEventListener('click', function () { if (ui.page > 1) changePage(-1); });
$('next-page').addEventListener('click', function () { changePage(1); });
$('ask-form').addEventListener('submit', function (ev) { ev.preventDefault(); startAsk(); });
window.addEventListener('beforeunload', function (ev) {
  if (!hasDirty() && !ui.mappingDirty) return;
  ev.preventDefault();
  ev.returnValue = '';
});

api('/api/state').then(function (s) {
  applyState(s);
  if (s.job && s.job.kind) refreshJob();
}).catch(function (err) {
  showAlert('source-error', err.message);
});
