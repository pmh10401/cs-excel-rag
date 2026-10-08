// 선택 개발 검사: 설치된 Playwright로 실행하며 운영 DB·LLM을 수정하거나 호출하지 않습니다.
// node tests/test_relations_ui.cjs http://127.0.0.1:8766 /tmp/rag-relations-ui
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8766';
  const output = process.argv[3];
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    const expectedFailures = new Set();
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (['error', 'warning'].includes(message.type()) &&
          !(expectedFailures.has(message.location().url) && /status of 503/.test(message.text()))) errors.push(message.text());
    });
    const cell = (column, original) => ({column, original, location: '샘플!' + column});
    const records = {
      'detail-a': {id: 'detail-a', header: {'제목': '회전척 진동 점검'}, groups: [
        {name: '접수', cells: [cell('설비', '세정장비'), cell('기인 Unit(부품)', '회전척'), cell('불량발생일', '2026-10-07'), cell('현상', '회전 중 진동'), cell('원인', ''), cell('조치', '축 정렬 후 정상화. <img src=x onerror="window.injected=true">')]}]},
      'detail-b': {id: 'detail-b', header: {'제목': '같은 CS의 다른 부품'}, groups: [
        {name: '접수', cells: [cell('설비코드', '세정 장비'), cell('기인 Unit(부품)', '배수밸브'), cell('조치', '배수밸브 교체 예정')]}]},
      'split': {id: 'split', header: {}, groups: [
        {name: '접수', cells: [cell('UNIT명', '이송모듈'), cell('문제현상', '이송 중 진동')]},
        {name: '작업1', cells: [cell('Part명(1)', '흡착척'), cell('조치내용(1)', '다른 작업의 조치')]},
        {name: '작업2', cells: [cell('Part명(2)', '이송 암'), cell('조치내용(2)', '작업 2 점검 예정')]}]}
    };
    let fail = false, changed = false, slow = false;
    await page.route('**/api/record?*', async route => {
      const id = new URL(route.request().url()).searchParams.get('id');
      if (slow && id === 'detail-a') await new Promise(resolve => setTimeout(resolve, 300));
      const record = structuredClone(records[id]);
      if (changed) record.groups[0].cells[0].original = '변경된 설비';
      if (fail) expectedFailures.add(route.request().url());
      await route.fulfill({status: fail ? 503 : 200, contentType: 'application/json', body: JSON.stringify(fail ? {error: '모의 조회 실패'} : record)});
    });
    await page.goto(url);
    await page.waitForFunction(() => ui.last != null);
    assert.equal(new URL(page.url()).origin, new URL(url).origin);
    assert.match(await page.title(), /CS/);
    assert.ok(await page.locator('h1').first().textContent());
    await page.locator('.nav-btn[data-nav="ask"]').click();
    const kinds = {'설비':'equipment', '설비코드':'equipment', '기인 Unit(부품)':'part', 'UNIT명':'module', 'Part명(2)':'part'};
    const routePath = (id, work, columns) => ({record_id: id, receipt: 'CS-SAME', evidence_id: id + '/' + work,
      cells: columns.map(([column, name]) => ({column, name, kind:kinds[column], entity:name.normalize('NFC').toLowerCase().replace(/\s+/g,''), location: '샘플!' + column}))});
    const paths = [routePath('detail-b', '접수', [['설비코드', '세정 장비'], ['기인 Unit(부품)', '배수밸브']]),
      routePath('detail-a', '접수', [['설비', '세정장비'], ['기인 Unit(부품)', '회전척']])];
    const result = {status: 'ok', answer: '과거 사례의 조치와 결과를 확인하세요.',
      claims: [{text: '축 정렬 후 정상화된 기록이 있습니다.', evidence: [{id: 'detail-a/접수', receipt: 'CS-SAME', work: '접수', column: '조치', quote: '축 정렬 후 정상화'}]}],
      vector_search: {mode: 'graph', vector_records: 2, keyword_ids: [], vector_ids: [], hybrid_ids: [],
        relations: {status: 'used', reason: '같은 상세의 원문 이름을 연결했습니다.', matches: [], candidate_ids: [], paths}}};
    const render = value => page.evaluate(value => renderAsk({status: 'done', result: value}), value);
    await render(result);
    assert.equal(await page.locator('#ask-relations').evaluate(el => el.open), false);
    await page.locator('#ask-relations > summary').click();
    await page.waitForFunction(() => document.querySelectorAll('.relation-edge').length === 4);
    assert.equal(await page.locator('.relation-entity').count(), 3);
    assert.equal(await page.locator('.relation-entity').filter({hasText:'세정장비'}).count(), 1);
    assert.equal(await page.locator('#relation-entities').evaluate(el => new Set(Array.from(el.children, n => n.id)).size), 3);
    const buttons = page.locator('.relation-case');
    assert.equal(await buttons.count(), 2);
    assert.match(await buttons.first().textContent(), /상세 1.*답변에 인용됨/);
    assert.match(await buttons.nth(1).textContent(), /상세 2.*검색 후보/);
    await buttons.first().click();
    await page.getByText('2026-10-07', {exact: true}).waitFor();
    assert.equal(await page.locator('.relation-edge.is-active').count(), 2);
    assert.equal((await buttons.first().getAttribute('aria-describedby')).split(' ').length, 2);
    assert.match(await page.locator('#relation-source').textContent(), /원문 셀이 비어 있음/);
    assert.equal(await page.locator('#relation-source img').count(), 0);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    if (output) { fs.mkdirSync(output, {recursive: true}); await page.locator('#ask-result').screenshot({path: path.join(output, 'desktop.png')}); }
    slow = true;
    await buttons.first().click();
    await buttons.nth(1).click();
    await page.getByText('배수밸브 교체 예정', {exact: true}).waitFor();
    await page.waitForTimeout(400);
    assert.doesNotMatch(await page.locator('#relation-source').textContent(), /축 정렬/);
    slow = false;
    fail = true;
    await buttons.first().click();
    await page.getByText(/모의 조회 실패.*다시 선택/).waitFor();
    fail = false;
    changed = true;
    await buttons.first().click();
    await page.getByText(/관계 검색 후 원문이 변경/).waitFor();
    changed = false;
    await buttons.first().click();
    await page.getByText('2026-10-07', {exact: true}).waitFor();
    await page.setViewportSize({width: 390, height: 844});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    assert.ok(await page.locator('.relation-edge').evaluateAll(lines => lines.every(line => !/NaN/.test(line.getAttribute('d')) && line.getBBox().width > 0)));
    await buttons.nth(1).focus();
    await page.keyboard.press('Enter');
    await page.getByText('배수밸브 교체 예정', {exact: true}).waitFor();
    assert.equal(await buttons.nth(1).getAttribute('aria-pressed'), 'true');
    if (output) await page.locator('#ask-result').screenshot({path: path.join(output, 'mobile.png')});
    const split = routePath('split', '작업2', [['UNIT명', '이송모듈'], ['Part명(2)', '이송 암']]);
    await render({...result, claims: [], vector_search: {...result.vector_search, relations: {...result.vector_search.relations, paths: [split]}}});
    await page.locator('#ask-relations > summary').click();
    await page.locator('.relation-case').click();
    await page.getByText('작업 2 점검 예정', {exact: true}).waitFor();
    assert.doesNotMatch(await page.locator('#relation-source').textContent(), /다른 작업의 조치|흡착척/);
    const many = Array.from({length: 12}, (_, i) => ({...paths[1], record_id: 'many-' + i, evidence_id: 'many-' + i + '/접수'}));
    await render({...result, claims: [{text: '선택 사례', evidence: [{id: 'many-11/접수', receipt: 'CS-SAME', column: '현상', quote: '진동'}]}], vector_search: {...result.vector_search, relations: {...result.vector_search.relations, paths: many}}});
    assert.equal(await page.locator('.relation-case').count(), 10);
    assert.equal(await page.locator('.relation-entity').count(), 2);
    assert.match(await page.locator('.relation-case').first().textContent(), /답변에 인용됨/);
    await render({...result, status: 'validation_failed'});
    assert.equal(await page.locator('.relation-case[data-cited="true"]').count(), 0);
    const roles = [routePath('detail-a', '접수', [['설비', '회전장비'], ['UNIT명', '회전장비']])];
    await render({...result, claims: [], vector_search: {...result.vector_search, relations: {...result.vector_search.relations, paths: roles}}});
    assert.equal(await page.locator('.relation-entity').count(), 2); // 이름이 같아도 역할이 다르면 합치지 않습니다.
    const legacy = paths.map(p => ({...p, cells:p.cells.map(c => ({column:c.column, name:c.name, location:c.location}))}));
    await render({...result, vector_search: {...result.vector_search, relations: {...result.vector_search.relations, paths: legacy}}});
    assert.equal(await page.locator('.relation-case').count(), 2);
    await render({...result, vector_search: {...result.vector_search, relations: {...result.vector_search.relations, paths: [], reason: '조건을 확인할 수 없어 기존 검색을 유지했습니다.'}}});
    assert.equal(await page.locator('#relation-layout').isVisible(), false);
    assert.match(await page.locator('#relation-status').textContent(), /기존 검색/);
    await render({...result, vector_search: undefined});
    assert.equal(await page.locator('#ask-relations').isVisible(), false);
    // 일반 검색에서도 반환한 사례만 표시하고, 후보 보완의 경로로 화면 범위를 넓히지 않습니다.
    await render({...result, comparison: true, relation_view: {reason: '반환된 사례의 원문', paths: [paths[1]]}});
    assert.equal(await page.locator('.relation-case').count(), 1);
    assert.equal(await page.locator('.relation-case[data-cited="true"]').count(), 0);
    await render({...result, vector_search: undefined, relation_view: {reason: '코드 없이 반환한 원문', paths: [split]}});
    await page.locator('#ask-relations > summary').click();
    await page.locator('.relation-case').click();
    await page.getByText('작업 2 점검 예정', {exact: true}).waitFor();
    assert.equal(await page.locator('.relation-entity').count(), 2);
    assert.doesNotMatch(await page.locator('#relation-source').textContent(), /다른 작업의 조치|흡착척/);
    await render({...result, vector_search: undefined, relation_view: {reason: '이름 한 종류',
      paths: [{...paths[1], cells: paths[1].cells.slice(0, 1)}]}});
    await page.locator('#ask-relations > summary').click();
    await page.waitForFunction(() => document.querySelectorAll('.relation-edge').length === 1);
    assert.equal(await page.locator('.relation-entity').count(), 1);
    await render({...result, relation_view: {reason: '원문 이름 없음', paths: []}});
    assert.equal(await page.locator('#relation-layout').isVisible(), false);
    assert.match(await page.locator('#relation-status').textContent(), /원문 이름 없음/);
    await render({...result, relation_view: null});
    assert.equal(await page.locator('#ask-relations').isVisible(), false);
    // GPU 실험 상태는 사실 검사의 통과 여부와 구별해 보여 줍니다.
    await render({...result, vector_search: undefined, timings: {total: 20},
      clef_search: {model: 'clef:flash', seconds: 12.3, status: 'used',
      backend: 'mlx', device: 'gpu', reason: '후보 안의 상대 선택 확률입니다.',
      baseline_ids: ['detail-a', 'detail-b'], reranked_ids: ['detail-b', 'detail-a']}});
    assert.match(await page.locator('#ask-search-detail').textContent(), /used.*mlx\/gpu/);
    assert.match(await page.locator('#ask-search-detail').textContent(), /기존 순서: detail-a, detail-b.*\n재정렬: detail-b, detail-a/);
    assert.match(await page.locator('#ask-search-detail').textContent(), /처리 시간/);
    await render({...result, clef_search: {model: 'clef:flash', seconds: 1, status: 'error',
      reason: '기존 검색 유지 <img src=x onerror="window.injected=true">',
      baseline_ids: ['detail-a'], reranked_ids: ['detail-a']}});
    assert.match(await page.locator('#ask-search-detail').textContent(), /error/);
    assert.equal(await page.locator('#ask-note img').count(), 0);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    await render(result);
    assert.doesNotMatch(await page.locator('#ask-search-detail').textContent(), /Clef:/);
    assert.deepEqual(errors, []);
    console.log('PASS: 실제 브라우저 · 공유 노드/역할 구분/정규화/구형 응답 · 연결선/선택 강조/리사이즈 · 상세/작업 분리 · 인용 구분 · 날짜/빈 셀 · XSS · 조회 오류/재시도 · 원문 변경 · 응답 경합 · 10개 제한 · 키보드/390px · 예상 오류 외 콘솔 오류 0');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
