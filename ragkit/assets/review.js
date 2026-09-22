// 별도 웹 서버 없이 검토 파일을 다운로드합니다. DB 반영은 Python의 검토 적용 기능이 수행합니다.
const rows = Array.from(document.querySelectorAll('tr[data-review]'));
const notice = document.getElementById('review-notice');
for (const row of rows) {
  const select = row.querySelector('select');
  const text = row.querySelector('textarea');
  select.addEventListener('change', () => {
    text.disabled = select.value !== 'edit';
    if (select.value === 'accept') text.value = row.dataset.proposal;
    if (select.value === 'original') text.value = row.dataset.original;
    notice.textContent = '선택이 변경되었습니다. 검토 파일을 다운로드한 뒤 적용해야 DB에 저장됩니다.';
  });
  text.addEventListener('input', () => {
    notice.textContent = '편집 중입니다. 다운로드 전 숫자·단위·조건이 원문과 같은지 확인하세요.';
  });
}
document.getElementById('download-reviews').addEventListener('click', () => {
  const entries = rows.map(row => {
    const action = row.querySelector('select').value;
    const value = action === 'accept' ? row.dataset.proposal
      : action === 'edit' ? row.querySelector('textarea').value : row.dataset.original;
    return {...JSON.parse(row.dataset.review), action, value};
  });
  const blob = new Blob([JSON.stringify({format:'cell-reviews-v1', entries}, null, 2)], {type:'application/json;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'review_decisions.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  notice.textContent = notice.dataset.applyHelp;
});
