"""CS 양식과 작업 구분. 원본 행을 유지하면서 모델 입력·검색 단위를 나눕니다."""
import json
import hashlib
import re
from .records import public_cells, record_id
from .search import word_tokens, word_grams, search_concepts
from .input_formats import row_schema, CASE_NUMBER_SQL

# 검색 본문·채택문 검증 규칙이 바뀌면 올립니다. 기존 DB는 첫 사용 때 한 번 갱신합니다.
SEARCH_VERSION = 3


def search_signature(schema, rules=None):
    if rules is None:
        rules = {key: 'rewrite' if key in schema['rewrite_columns'] else 'preserve' for key in schema['columns']}
    return hashlib.sha256(json.dumps([SEARCH_VERSION, schema, rules], sort_keys=True).encode()).hexdigest()


def ensure_document_ids(db):
    """FTS 내부 테이블을 수정하지 않고 공개 rowid와 접수번호를 연결합니다."""
    db.execute('CREATE TABLE IF NOT EXISTS rag_document_ids (record_id TEXT PRIMARY KEY, document_rowid INTEGER NOT NULL UNIQUE)')
    db.execute('DELETE FROM rag_document_ids')
    db.execute('INSERT INTO rag_document_ids SELECT record_id,rowid FROM rag_documents')


def ensure_search_index(db, rules, schema):
    """입수·검토 때 만든 색인을 재사용하고 구형 DB만 한 번 마이그레이션합니다."""
    tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {
        'rag_documents': ['document_id', 'record_id', 'content', 'evidence'],
        'rag_original_documents': ['document_id', 'record_id', 'content', 'evidence'],
    }
    if 'rag_search_meta' in tables and all(
            [c[1] for c in db.execute(f'PRAGMA table_info({table})')] ==
            prefix + ['keywords', 'symptom', 'alarm', 'equipment', 'action', 'grams']
            for table, prefix in expected.items()):
        if db.execute('SELECT signature FROM rag_search_meta WHERE id=1').fetchone() == (search_signature(schema, rules),):
            if 'rag_document_ids' not in tables or 'rag_chunks' in tables:
                db.execute('SAVEPOINT prune_legacy_search')
                try:
                    if 'rag_document_ids' not in tables:
                        ensure_document_ids(db)
                    # 접수 문서에 작업 근거가 이미 있습니다. 구형 중복 색인만 제거합니다.
                    db.execute('DROP TABLE IF EXISTS rag_chunks')
                except BaseException:
                    db.execute('ROLLBACK TO prune_legacy_search')
                    raise
                finally:
                    db.execute('RELEASE prune_legacy_search')
            return 'rag_chunks' in tables
    from .storage import prepared_record, stored_rows
    prepared = (prepared_record(db, json.loads(raw), json.loads(proposal), rules) for _, raw, proposal in stored_rows(db))
    # rebuild_search의 SAVEPOINT가 단독 실행과 외부 트랜잭션의 원자성을 함께 보장합니다.
    rebuild_search(db, prepared, schema, rules=rules)
    return True


def load_schema(path):
    schema = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(schema, dict):
        raise ValueError('CS 양식 설정은 JSON 객체여야 합니다.')
    for field in ('columns', 'rewrite_columns', 'context_columns'):
        values = schema.get(field)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v or v.startswith('_') for v in values) or len(set(values)) != len(values):
            raise ValueError(f'{field}: 중복 없는 열 이름 목록이 필요합니다.')
    columns = schema['columns']
    if schema.get('id_column') not in columns or len(columns) < 2:
        raise ValueError('양식에 식별 열과 내용 열이 필요합니다.')
    if not set(schema['rewrite_columns'] + schema['context_columns']).issubset(columns):
        raise ValueError('수정·문맥 열은 원본 양식에 있는 열이어야 합니다.')
    if schema['id_column'] in schema['rewrite_columns']:
        raise ValueError('CS 접수번호는 수정 대상이 될 수 없습니다.')
    return schema, {key: 'rewrite' if key in schema['rewrite_columns'] else 'preserve' for key in columns}


def column_group(key):
    # 숫자 접미사가 있는 열만 같은 작업으로 묶습니다. '(합산)'은 접수 정보에 남습니다.
    match = re.search(r'\((\d+)\)$', key)
    return '작업' + match[1] if match else '접수'


def normalization_groups(row, rules, schema):
    # 모델에는 수정 대상과 필요한 문맥만 보냅니다. 작업 1 요청에 작업 2를 넣지 않습니다.
    from .llm import editable
    schema = row_schema(row, schema)
    original = public_cells(row)
    groups = {}
    for key in original:
        if editable(row, key, rules):
            groups.setdefault(column_group(key), []).append(key)
    for group, targets in groups.items():
        allowed = set(targets) | {schema['id_column']}
        allowed.update(key for key in schema['context_columns'] if column_group(key) in ('접수', group))
        context = {key: value for key, value in original.items() if key in allowed and value.strip()}
        context['_id_column'] = schema['id_column']
        context['_typed_cells'] = {key: value for key, value in row.get('_typed_cells', {}).items() if key in context}
        context['_work_group'] = group
        if '_source_sheet' in row:
            context['_source_sheet'] = row['_source_sheet']
        yield context, {key: 'rewrite' if key in targets else 'preserve' for key in public_cells(context)}


def action_outcomes(cells):
    """명확한 '조치 후 결과'만 연결하고, 인용 구절은 원문의 부분 문자열로 남깁니다.

    결과 미확인은 자동 연결을 확정하지 못했다는 뜻입니다. 원문에 결과가 없다는
    뜻이 아니므로 복합 경과·관찰 시간·제한 사항은 원문 전체로 확인해야 합니다.
    """
    items = []
    endings = r'(?:됨|되었음|되었다|되었습니다|됐음|됐다|됐습니다|함|했음|했다|했습니다)?'
    patterns = (
        ('정상화', r'정상화' + endings),
        ('일부 개선', r'일부\s*개선' + endings),
        ('증상 지속', r'(?:(?:동일|같은)\s*증상(?:이|은)?\s*(?:발생|지속)|재발)' + endings),
    )
    for column, value in cells.items():
        if not re.fullmatch(r'조치내용\(\d+\)|조치', column) or not value.strip():
            continue
        steps = []
        # ponytail: 명확한 문장 경계와 후 연결만 지원합니다. 복합 문장은 원문 대조로 보류합니다.
        fragments = re.split(r'(?<=[.!?。！？])\s+|[\r\n]+', value.strip())
        for fragment in fragments:
            match = re.fullmatch(r'(.+?)\s+후(?:에도|에는|에)?\s*(.+)', fragment)
            action, outcome = (match[1], match[2]) if match else (fragment, '')
            status = '결과 미확인'
            if match and not re.search(r'예정|계획|미실시|미수행|보류|가정|의심|않|못', action):
                status = next((label for label, pattern in patterns
                               if re.fullmatch(pattern + r'[.。]?', outcome)), status)
            steps.append({'column': column, 'order': len(steps) + 1,
                          'action_quote': action, 'outcome_quote': outcome, 'status': status})
        if len(steps) > 1 and any(step['status'] == '결과 미확인' for step in steps):
            # 뒤의 '장기 재발 미확인' 같은 제한을 앞 문장의 성공에서 분리하지 않습니다.
            steps = [{'column': column, 'order': 1, 'action_quote': value,
                      'outcome_quote': '', 'status': '결과 미확인'}]
        items.extend(steps)
    return items


def record_sections(record, schema):
    """문서 안의 작업 경계를 유지합니다. 근거 검사용 셀에는 빈칸도 남깁니다."""
    row = record['original_cells']
    schema = row_schema(row, schema)
    original = public_cells(row)
    base = {key for key in schema['context_columns'] if column_group(key) == '접수'
            and key not in schema['rewrite_columns']}
    base.update(key for key in ('접수일시', '접수일자') if key in schema['columns'])
    groups = dict.fromkeys(column_group(key) for key, value in original.items() if value.strip())
    sections = []
    for group in groups:
        keys = {key for key in original if column_group(key) == group} | base | {schema['id_column']}
        cells = {key: value for key, value in original.items() if key in keys}
        sections.append({'id': f'{record_id(row)}/{group}', 'kind': group, 'cells': cells,
                         'action_outcomes': action_outcomes(cells),
                         'accepted': {key: record['effective'][key] for key in cells if record['effective'][key] != original[key]},
                         'locations': {key: row.get('_cell_locations', {}).get(key, '') for key in cells}})
    return sections


def search_fields(cells, accepted):
    """검색 전용 파생 값입니다. 열 이름·화면용 제목은 점수에 넣지 않습니다."""
    fields = [[] for _ in range(5)]  # 전체 값, 현상, 알람, 설비, 조치 순서입니다.
    for key, original in cells.items():
        values = [original] + ([accepted[key]] if key in accepted else [])
        if not any(value.strip() for value in values):
            continue
        # 셀 단위로 분석해 반복 문장 캐시를 재사용하고 서로 다른 셀의 표현을 합치지 않습니다.
        words = [word for value in values for word in (*word_tokens(value), *sorted(search_concepts(value)))]
        fields[0].extend(words)
        base = re.sub(r'\(.*?\)$', '', key)
        if key in ('제목', '문제현상', '현상', 'Defect 유형', 'Defect 상세'):
            fields[1].extend(words)
        elif key == '알람코드':
            fields[2].extend(words)
        elif base in ('설비코드', '제품군', '고객사호기', 'UNIT명', '설비군', 'Part명', 'Part Decs', '유닛/부품',
                      '설비', '설비군 간소화', '기인 모듈', '기인 Unit', '설비 타입'):
            fields[3].extend(words)
        elif base in ('조치내용', '조치') or key == '재발방지대책':
            fields[4].extend(words)
    grams = [gram for word in fields[0] for gram in word_grams(word)]
    return [' '.join(words) for words in [*fields, grams]]


def rebuild_search(db, prepared, schema, *, rules=None, incremental=False):
    # 원본 128열과 검토 이력은 그대로 두고 파생 검색 테이블만 갱신합니다.
    # SAVEPOINT는 독립 호출과 기존 검토 저장 트랜잭션 안의 호출 모두에서 원자성을 지킵니다.
    db.execute('SAVEPOINT rebuild_search')
    try:
        db.execute(f'CREATE INDEX IF NOT EXISTS records_case_number ON normalized_records (lower({CASE_NUMBER_SQL}))')
        db.execute('DROP TABLE IF EXISTS rag_chunks')
        for table, prefix in (
            ('rag_documents', 'document_id, record_id, content, evidence'),
            ('rag_original_documents', 'document_id, record_id, content, evidence'),
        ):
            columns = prefix.split(', ') + ['keywords', 'symptom', 'alarm', 'equipment', 'action', 'grams']
            existing = [column[1] for column in db.execute(f'PRAGMA table_info({table})')]
            if existing and existing != columns:
                db.execute(f'DROP TABLE {table}')
            definition = ', '.join(column + ' UNINDEXED' for column in columns[:4])
            db.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5({definition}, keywords, symptom, alarm, equipment, action, grams)')
            if not incremental:
                db.execute(f'DELETE FROM {table}')
        changed_ids = []
        for record in prepared:
            identity = record['id']
            rowid = None
            if incremental:
                old = db.execute('SELECT document_rowid FROM rag_document_ids WHERE record_id=?', (identity,)).fetchone()
                rowid = old[0] if old else None
                if old:
                    for table in ('rag_documents', 'rag_original_documents'):
                        db.execute(f'DELETE FROM {table} WHERE rowid=?', (rowid,))
            sections = record_sections(record, schema)
            originals, contents = [], []
            for section in sections:
                chunk_id, group = section['id'], section['kind']
                raw = f'근거 ID: {chunk_id}\n구분: {group}\n원문:\n'
                raw += '\n'.join(f'{key}: {value}' for key, value in section['cells'].items() if value.strip())
                content = raw
                if section['accepted']:
                    content += '\n검토 후 채택한 셀:\n' + '\n'.join(f'{key}: {value}' for key, value in section['accepted'].items())
                location = '\nExcel 위치: ' + record['original_cells'].get('_excel_location', '')
                originals.append(raw + location)
                contents.append(content + location)
            original = public_cells(record['original_cells'])
            accepted = {key: value for key, value in record['effective'].items() if value != original[key]}
            for table, parts in (('rag_documents', contents), ('rag_original_documents', originals)):
                is_original = table == 'rag_original_documents'
                evidence = [section | {'accepted': {}} for section in sections] if is_original else sections
                content = '\n\n'.join(parts)
                cursor = db.execute(f'INSERT INTO {table}(rowid,document_id,record_id,content,evidence,keywords,symptom,alarm,equipment,action,grams) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                    (rowid, f'{identity}/문서', identity, content, json.dumps(evidence, ensure_ascii=False),
                                     *search_fields(original, {} if is_original else accepted)))
                rowid = cursor.lastrowid
            if incremental:
                db.execute('INSERT OR REPLACE INTO rag_document_ids VALUES (?,?)', (identity, rowid))
                changed_ids.append(identity)
        # 전체 교체 뒤 남은 삭제 표시와 색인 조각을 합쳐 반복 갱신 시 검색 저하를 막습니다.
        if not incremental:
            for table in ('rag_documents', 'rag_original_documents'):
                db.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
            ensure_document_ids(db)
        # 검토·원문이 바뀐 모든 경로에서 오래된 영어 번역도 함께 검색에서 제외합니다.
        from .english import rebuild_english
        rebuild_english(db, schema, record_ids=changed_ids if incremental else None)
        from .semantic import sync_sources
        sync_sources(db, record_ids=changed_ids if incremental else None)
        db.execute('CREATE TABLE IF NOT EXISTS rag_search_meta (id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO rag_search_meta VALUES (1,?)', (search_signature(schema, rules),))
    except BaseException:
        db.execute('ROLLBACK TO rebuild_search')
        raise
    finally:
        db.execute('RELEASE rebuild_search')
