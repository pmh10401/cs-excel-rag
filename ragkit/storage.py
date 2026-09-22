"""LLM 제안과 검토 선택을 분리 저장하고, 채택문으로 검색 본문을 구성합니다."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import time
from .records import public_cells, row_revision, text_for, record_id
from .llm import editable, validate_normalized, normalize, normalization_prompt
from .input_formats import row_rules, row_schema

class NormalizationBatchError(ValueError):
    """모든 행을 시도한 뒤 실패 목록을 호출자에게 전달합니다."""
    def __init__(self, failures, total):
        self.failures = failures
        summary = f'전체 {total}건 처리 종료: 성공 {total - len(failures)}건, 실패 {len(failures)}건.'
        super().__init__(summary + '\n' + '\n'.join(f"{item['id']}: {item['error']}" for item in failures))


class ProcessingCancelled(Exception):
    """완료한 요청은 저장하고 새 요청을 예약하지 않습니다."""


def validate_workers(value=200):
    # bool과 소수·문자열은 동시 처리 수로 받지 않습니다.
    if type(value) is not int or not 1 <= value <= 1000:
        raise ValueError('문장 다듬기·영어 번역 동시 처리 수는 1~1000의 정수여야 합니다.')
    return value


def import_summary(db, rows):
    """접수번호로 비교합니다. 다른 파일의 다른 원문은 자동으로 덮어쓰지 않습니다."""
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='normalized_records'").fetchone()
    result = {'added': 0, 'updated': 0, 'unchanged': 0, 'conflicts': []}
    for row in rows:
        identity = record_id(row)
        old = db.execute('SELECT original_json FROM normalized_records WHERE id=?', (identity,)).fetchone() if exists else None
        if old is None:
            result['added'] += 1
        else:
            original = json.loads(old[0])
            if row_revision(row) == row_revision(original):
                result['unchanged'] += 1
            elif row.get('_source_key', '') != original.get('_source_key', ''):
                result['conflicts'].append(identity)
            else:
                result['updated'] += 1
    return result


def ingest(db, rows, rules, client=None, config=None, schema=None, progress=None, *, previous_errors=None, cancelled=None,
           append=False, import_result=None):
    workers = validate_workers((config or {}).get('normalization_workers', 200))
    # 수정 제안은 normalized_json에, 검색에 사용할 원문·채택문은 search_text에 저장합니다.
    # 원본과 LLM 정리문을 별도로 보관합니다. 기존 키워드 데모 테이블은 수정하지 않습니다.
    if not rows or len({record_id(row) for row in rows}) != len(rows):
        raise ValueError('데이터가 비어 있거나 ID가 중복되어 있습니다.')
    version = db.execute('PRAGMA data_version').fetchone() if append else None
    ensure_reviews(db)
    db.execute('CREATE TABLE IF NOT EXISTS normalized_records (id TEXT PRIMARY KEY, original_json TEXT NOT NULL, original_text TEXT NOT NULL, normalized_json TEXT NOT NULL, search_text TEXT NOT NULL, fingerprint TEXT NOT NULL)')
    summary = import_summary(db, rows) if append else None
    if summary and summary['conflicts']:
        raise ValueError('다른 파일에 같은 접수번호의 다른 원문이 있습니다. 기존 자료는 유지합니다. '
                         '접수번호를 확인하거나 기존 원본 파일에서 수정하세요: ' + ', '.join(summary['conflicts'][:20]))
    if client is not None:
        # ponytail: 행마다 최신 지문만 보관. 버전 이력이 필요해지면 지문 키를 추가합니다.
        db.execute('CREATE TABLE IF NOT EXISTS normalization_checkpoints (source_key TEXT, record_id TEXT, fingerprint TEXT NOT NULL, normalized_json TEXT NOT NULL, PRIMARY KEY(source_key, record_id))')
    # 원문뿐 아니라 서버·모델·프롬프트가 바뀌어도 다시 정리하도록 지문을 계산합니다. worker 수는 지문에 넣지 않습니다.
    settings = {key: config[key] for key in ('base_url', 'model', 'reasoning_effort', 'max_tokens', 'context_window', 'temperature', 'top_p')
                if key in config} if config else {}
    total, completed, reused_count, active, fallback_count = len(rows), 0, 0, 0, 0
    failures = []

    def report(current_id, stage, failed_id=None):
        if progress:
            payload = {'total': total, 'completed': completed, 'reused': reused_count,
                       'fallback_cells': fallback_count,
                       'current_id': current_id, 'stage': stage, 'failed_id': failed_id or (failures[0]['id'] if failures else None),
                       'attempted': completed + len(failures), 'failed': len(failures),
                       'failures': list(failures)}
            if client is not None:
                payload['workers'] = workers
                payload['active'] = active
            progress(payload)

    report('', '원문 검색 자료를 준비합니다' if client is None else '정규화를 시작합니다')
    prepared, updates, pending = [None] * total, [None] * total, []

    def store(index, row, identity, original, fingerprint, normalized, reused_row):
        nonlocal completed, reused_count, fallback_count
        record = prepared_record(db, row, normalized, rules)
        fallback_count += len(record['fallbacks'])
        search_text = search_content(record)
        prepared[index] = record
        updates[index] = (identity, json.dumps(row, ensure_ascii=False), original,
                          json.dumps(normalized, ensure_ascii=False), search_text, fingerprint if normalized else 'raw-only')
        completed += 1
        if reused_row:
            reused_count += 1
            report(identity, '이전 정리 결과를 재사용했습니다')
        elif client is None:
            report(identity, '원문 행을 준비했습니다')
        else:
            if record['fallbacks']:
                print(f"{identity}: 숫자·단위 검사 실패로 원문 유지: {', '.join(record['fallbacks'])}", flush=True)
            report(identity, '숫자·단위 검사 실패 셀은 원문으로 보존했습니다' if record['fallbacks'] else '행 정리를 저장했습니다')

    for index, row in enumerate(rows):
        if cancelled and cancelled():
            raise ProcessingCancelled()
        identity = record_id(row)
        old = db.execute('SELECT normalized_json,fingerprint,original_json FROM normalized_records WHERE id=?', (identity,)).fetchone()
        if append and old:
            original_row = json.loads(old[2])
            if row_revision(row) == row_revision(original_row):
                # 복사본을 다시 읽어도 기존 파일의 인용 위치와 검토 선택을 유지합니다.
                if row == original_row or row.get('_source_key', '') != original_row.get('_source_key', ''):
                    row = original_row
                if client is None:
                    store(index, row, identity, text_for(row), old[1], json.loads(old[0]), True)
                    continue
        source_key = row.get('_source_key', '')
        original = text_for(row)
        active_rules, active_schema = row_rules(row, rules), row_schema(row, schema)
        fingerprint = hashlib.sha256((row_revision(row) + normalization_prompt(row) + json.dumps([settings, active_rules, active_schema] if active_schema else [settings, active_rules], sort_keys=True)).encode()).hexdigest()
        try:
            report(identity, '원문 행을 준비하는 중입니다' if client is None else '행을 정리하는 중입니다')
            if client is None:
                # --search-only는 서버 없이 원본 검색만 확인합니다. 정리문을 생성한 척하지 않습니다.
                store(index, row, identity, original, fingerprint, None, False)
                continue
            cached = db.execute(
                'SELECT normalized_json FROM normalization_checkpoints WHERE source_key=? AND record_id=? AND fingerprint=?',
                (source_key, identity, fingerprint)).fetchone()
            if not cached:
                if old and old[1] == fingerprint and json.loads(old[2]).get('_source_key', '') == source_key:
                    cached = old
            if cached:
                normalized = json.loads(cached[0])
                validate_normalized(normalized, row, rules)
                store(index, row, identity, original, fingerprint, normalized, True)
            else:
                pending.append((index, row, identity, source_key, original, fingerprint))
        except Exception:
            report(identity, '원문 행 준비에 실패했습니다' if client is None else '행 정리에 실패했습니다', identity)
            raise

    if pending:
        def call_normalize(row):
            previous = (previous_errors or {}).get(record_id(row), '')
            if previous:
                return normalize(client, config, row, rules, schema=schema, previous_error=previous)
            return normalize(client, config, row, rules, schema=schema) if schema else normalize(client, config, row, rules)

        in_flight = {}
        work = iter(pending)

        def submit(pool):
            nonlocal active
            while len(in_flight) < workers:
                if cancelled and cancelled():
                    return
                try:
                    item = next(work)
                except StopIteration:
                    return
                _, row, identity, _, _, _ = item
                print(f"LLM 한국어 정리: {identity}", flush=True)
                in_flight[pool.submit(call_normalize, row)] = item
                active = len(in_flight)
                report(identity, '행을 정리하는 중입니다')

        # 작업 스레드는 normalize만 호출합니다. 동시에 workers개만 예약하고 검사·checkpoint·progress는 호출 스레드에서 처리합니다.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            submit(pool)
            while in_flight:
                finished, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
                for future in finished:
                    index, row, identity, source_key, original, fingerprint = in_flight.pop(future)
                    active = len(in_flight)
                    try:
                        normalized = future.result()
                        validate_normalized(normalized, row, rules)
                    except Exception as exc:
                        # 모델 요청·응답 검사의 실패는 해당 행에만 적용합니다.
                        # DB 저장 오류는 이 블록 밖에서 처리하여 저장 실패를 숨기지 않습니다.
                        failure = {'id': identity, 'error': str(exc)}
                        if getattr(exc, 'status_code', None) is not None:
                            failure['status_code'] = exc.status_code
                        failures.append(failure)
                        report(identity, '실패를 기록하고 나머지 행을 계속 처리합니다', identity)
                        continue
                    with db:
                        db.execute('INSERT OR REPLACE INTO normalization_checkpoints VALUES (?,?,?,?)',
                                   (source_key, identity, fingerprint, json.dumps(normalized, ensure_ascii=False)))
                        fallback_key = (source_key, identity, fallback_signature(row, normalized))
                        if getattr(normalized, 'fallbacks', {}):
                            db.execute('INSERT OR REPLACE INTO normalization_fallbacks VALUES (?,?,?,?)',
                                       (*fallback_key, json.dumps(normalized.fallbacks, ensure_ascii=False)))
                        else:
                            db.execute('DELETE FROM normalization_fallbacks WHERE source_key=? AND record_id=? AND signature=?', fallback_key)
                    store(index, row, identity, original, fingerprint, normalized, False)
                submit(pool)
        if cancelled and cancelled():
            raise ProcessingCancelled()
        if failures:
            report(failures[0]['id'], '전체 행 처리 종료 · 실패한 행만 재시도할 수 있습니다')
            raise NormalizationBatchError(failures, total)
    # 모든 입력 행이 성공한 뒤 확정합니다. 누적 등록에서는 입력에 없는 기존 접수를 지우지 않습니다.
    if append and schema:
        from .cslist import ensure_search_index
        ensure_search_index(db, rules, schema)
    with db:
        if append:
            if not db.in_transaction:
                db.execute('BEGIN IMMEDIATE')
            if db.execute('PRAGMA data_version').fetchone() != version:
                raise ValueError('처리 중 다른 연결에서 DB가 변경되었습니다. 기존 자료를 덮어쓰지 않았습니다. 다시 실행하세요.')
            changed = []
            for record, values in zip(prepared, updates):
                if db.execute('SELECT * FROM normalized_records WHERE id=?', (record['id'],)).fetchone() != values:
                    db.execute('INSERT INTO normalized_records VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                               'original_json=excluded.original_json, original_text=excluded.original_text, '
                               'normalized_json=excluded.normalized_json, search_text=excluded.search_text, fingerprint=excluded.fingerprint', values)
                    changed.append(record)
        else:
            db.execute('DELETE FROM normalized_records')
            db.executemany('INSERT INTO normalized_records VALUES (?,?,?,?,?,?)', updates)
            changed = prepared
        if schema:
            from .cslist import rebuild_search
            if changed:
                rebuild_search(db, changed, schema, rules=rules, incremental=append)
        if client is not None:
            db.executemany('DELETE FROM normalization_checkpoints WHERE source_key=? AND record_id=?',
                           [(record['original_cells'].get('_source_key', ''), record['id']) for record in prepared])
    if import_result is not None and summary is not None:
        import_result.update(summary, total=db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0])
    return prepared


def ensure_reviews(db):
    # 현재 조회용 자료와 별도로 저장하여 재실행하거나 다른 샘플을 열어도 최근 검토 결과를 보존합니다.
    db.execute('CREATE TABLE IF NOT EXISTS cell_reviews (source_key TEXT, record_id TEXT, column_name TEXT, revision TEXT NOT NULL, action TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(source_key, record_id, column_name))')
    db.execute('CREATE TABLE IF NOT EXISTS normalization_fallbacks (source_key TEXT, record_id TEXT, signature TEXT, fallbacks_json TEXT NOT NULL, PRIMARY KEY(source_key, record_id, signature))')


def fallback_signature(row, normalized):
    # 실패한 배치의 체크포인트와 기존 검색 DB가 공존할 수 있어 원문·제안이 모두 같을 때만 표시합니다.
    return hashlib.sha256(json.dumps([row_revision(row), normalized], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepared_record(db, row, normalized, rules):
    effective = public_cells(row)
    revision = row_revision(row)
    fallbacks = {}
    if normalized is not None and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='normalization_fallbacks'").fetchone():
        stored = db.execute('SELECT fallbacks_json FROM normalization_fallbacks WHERE source_key=? AND record_id=? AND signature=?',
                            (row.get('_source_key', ''), record_id(row), fallback_signature(row, normalized))).fetchone()
        if stored:
            fallbacks = json.loads(stored[0])
    saved = {column: (rev, action, value) for column, rev, action, value in db.execute(
        'SELECT column_name, revision, action, value FROM cell_reviews WHERE source_key=? AND record_id=?',
        (row.get('_source_key', ''), record_id(row)))}
    reviews = {}
    for column in effective:
        previous = saved.get(column)
        review = {'action': 'pending', 'value': effective[column], 'stale': False}
        if previous:
            old_rev, action, value = previous
            if old_rev == revision and editable(row, column, rules):
                candidate = public_cells(row) | {column: value}
                # 규칙 변경으로 허용되지 않는 편집이 되었으면 원문으로 돌아가고 재검토 표시합니다.
                try:
                    validate_normalized(candidate, row, rules)
                except ValueError:
                    review.update(stale=True, previous=value)
                else:
                    effective[column] = value
                    review.update(action=action, value=value)
            elif editable(row, column, rules):
                review.update(stale=True, previous=value)
        reviews[column] = review
    return {'id': record_id(row), 'original': text_for(row), 'original_cells': row,
            'normalized': normalized, 'effective': effective, 'reviews': reviews, 'revision': revision, 'fallbacks': fallbacks}


def search_content(record):
    # LLM 제안 자체는 검색·답변에 쓰지 않습니다. 사용자가 채택한 셀만 포함합니다.
    original = record['original']
    if record['effective'] == public_cells(record['original_cells']):
        return original
    return original + '\n\n검토 후 채택한 셀:\n' + text_for(record['effective'], record['original_cells'].get('_id_column', 'id'))


def accept_pending_reviews(db, rules, schema=None, progress=None, changed_ids=None):
    """현재 DB 전체의 미검토 제안을 채택합니다. 기존 선택과 재검토 대상은 보존합니다."""
    stored = db.execute('SELECT id, original_json, normalized_json FROM normalized_records ORDER BY id').fetchall()
    reviewed = set(db.execute("SELECT source_key,record_id,column_name FROM cell_reviews WHERE action != 'pending'"))
    result = {'rows': len(stored), 'applied_rows': 0, 'saved': 0,
              'reviewed_cells': 0, 'missing_rows': 0, 'failures': []}
    entries = []
    for index, (identity, raw, encoded) in enumerate(stored, 1):
        row = json.loads(raw)
        source = row.get('_source_key', '')
        columns = [key for key in public_cells(row) if editable(row, key, rules)]
        pending = [key for key in columns if (source, identity, key) not in reviewed]
        result['reviewed_cells'] += len(columns) - len(pending)
        if pending:
            try:
                proposal = json.loads(encoded)
                if proposal is None:
                    result['missing_rows'] += 1
                else:
                    # 저장 이후 규칙이 바뀌었거나 잘못된 제안이 있으면 그 접수만 제외합니다.
                    # 숫자·단위·빈칸 검사는 기존 검토 저장과 동일하게 유지합니다.
                    validate_normalized(proposal, row, rules)
                    base = {'source_key': source, 'id': identity, 'revision': row_revision(row), 'action': 'accept'}
                    entries.extend(base | {'column': key, 'value': proposal[key]} for key in pending)
                    result['applied_rows'] += 1
            except ValueError as error:
                result['failures'].append({'id': identity, 'error': str(error)})
        if progress and (index % 25 == 0 or index == len(stored)):
            progress(index, len(stored))
    if entries:
        # 브라우저에서 수천 번 저장하지 않고 기존 검토 저장을 한 번만 호출합니다.
        # 전체 검토와 검색 색인이 함께 확정되며 저장 오류는 예외로 전파해 롤백합니다.
        apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': entries}, rules, schema=schema)
    if changed_ids is not None:
        changed_ids.update(entry['id'] for entry in entries)
    result['saved'] = len(entries)
    return result


def apply_reviews(db, payload, rules, schema=None):
    # 다운로드한 JSON도 외부 입력입니다. 파일·행·버전·열을 검사한 뒤 한 번에 반영합니다.
    ensure_reviews(db)
    if not isinstance(payload, dict) or payload.get('format') != 'cell-reviews-v1' or not isinstance(payload.get('entries'), list):
        raise ValueError('올바른 검토 JSON 파일이 아닙니다.')
    rows = {record_id: (json.loads(raw), json.loads(proposal)) for record_id, raw, proposal in db.execute(
        'SELECT id, original_json, normalized_json FROM normalized_records')}
    seen, changes = set(), []
    for entry in payload['entries']:
        required = {'source_key', 'id', 'revision', 'column', 'action', 'value'}
        if not isinstance(entry, dict) or set(entry) != required or any(not isinstance(entry[k], str) for k in required):
            raise ValueError('검토 항목 형식이 올바르지 않습니다.')
        row, proposal = rows.get(entry['id'], (None, None))
        if row is None or entry['source_key'] != row.get('_source_key', '') or entry['revision'] != row_revision(row):
            raise ValueError(f"{entry['id']}: 입력 파일 또는 원문이 달라졌습니다. 새 비교 화면에서 다시 검토하세요.")
        column, action = entry['column'], entry['action']
        identity = (entry['source_key'], entry['id'], column)
        if identity in seen or column not in public_cells(row) or not editable(row, column, rules):
            raise ValueError(f"{entry['id']} / {column}: 중복되었거나 수정할 수 없는 열입니다.")
        seen.add(identity)
        if action not in ('pending', 'original', 'accept', 'edit'):
            raise ValueError('알 수 없는 검토 선택입니다.')
        value = row[column] if action in ('pending', 'original') else entry['value']
        if action == 'accept' and (proposal is None or value != proposal[column]):
            raise ValueError(f"{entry['id']} / {column}: 수정 제안이 달라졌습니다. 직접 편집 또는 새 제안을 선택하세요.")
        validate_normalized(public_cells(row) | {column: value}, row, rules)
        changes.append((*identity, entry['revision'], action, value, time.strftime('%Y-%m-%d %H:%M:%S')))
    with db:
        for change in changes:
            if change[4] == 'pending':
                db.execute('DELETE FROM cell_reviews WHERE source_key=? AND record_id=? AND column_name=?', change[:3])
            else:
                db.execute('INSERT OR REPLACE INTO cell_reviews VALUES (?,?,?,?,?,?,?)', change)
        prepared = [prepared_record(db, row, proposal, rules) for row, proposal in rows.values()]
        for record in prepared:
            db.execute('UPDATE normalized_records SET search_text=? WHERE id=?', (search_content(record), record['id']))
        if schema:
            from .cslist import rebuild_search
            rebuild_search(db, prepared, schema, rules=rules)
    return prepared
