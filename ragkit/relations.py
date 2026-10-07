"""원문 셀에 함께 적힌 설비·모듈·부품을 상세별로 연결하는 선택 검색입니다."""
import json
import re
import unicodedata
from .search import QUERY_FILLER, search_concepts, word_tokens

COLUMNS = {'설비': 'equipment', '설비코드': 'equipment', '고객사호기': 'equipment',
           '기인 모듈': 'module', 'UNIT명': 'module', '기인 Unit(부품)': 'part',
           'Part명(1)': 'part', 'Part명(2)': 'part'}
GENERIC = {'펌프', '팬', 'pump', 'fan', '모듈', 'module', '부품', 'part', 'unit', '설비'}


def key(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', text).casefold())


def exists(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='rag_relations'").fetchone())


def sync(db, record_ids=None):
    # 기존 DB에서는 선택 기능을 처음 사용할 때만 생성합니다. 이후 모든 색인 갱신을 공유합니다.
    if not exists(db):
        return
    if record_ids is None:
        db.execute('DELETE FROM rag_relations')
        rows = db.execute('SELECT record_id,evidence FROM rag_documents')
    else:
        db.executemany('DELETE FROM rag_relations WHERE record_id=?', [(i,) for i in record_ids])
        rows = (row for identity in record_ids for row in db.execute(
            'SELECT record_id,evidence FROM rag_documents WHERE rowid='
            '(SELECT document_rowid FROM rag_document_ids WHERE record_id=?)', (identity,)))
    for identity, encoded in rows:
        for section in json.loads(encoded):
            for column, kind in COLUMNS.items():
                value = section['cells'].get(column, '').strip()
                entity = key(value)
                # ponytail: 한 셀의 짧은 이름만 연결합니다. 복수 부품·긴 설명은 별도 현장 검증 후 확장합니다.
                if (2 <= len(entity) <= 80 and entity not in GENERIC
                        and not re.search(r'[\r\n,;/|]|미상|미기재|불명|예정|의심', value)):
                    db.execute('INSERT INTO rag_relations VALUES (?,?,?,?,?,?,?)',
                               (identity, section['id'], column, kind, entity, value,
                                section.get('locations', {}).get(column, '')))


def ensure(db):
    if exists(db):
        return False
    db.execute('SAVEPOINT relations_prepare')
    try:
        db.execute('CREATE TABLE rag_relations (record_id TEXT, section_id TEXT, column_name TEXT, '
                   'kind TEXT, entity TEXT, source_value TEXT, location TEXT, '
                   'PRIMARY KEY(record_id,section_id,column_name))')
        db.execute('CREATE INDEX rag_relations_entity ON rag_relations(entity,kind,record_id,section_id)')
        sync(db)
    except BaseException:
        db.execute('ROLLBACK TO relations_prepare')
        raise
    finally:
        db.execute('RELEASE relations_prepare')
    return True


def candidates(db, question):
    info = {'status': 'skipped', 'reason': '', 'matches': [], 'candidate_ids': [], 'paths': []}
    if not exists(db):
        info['reason'] = '관계 자료가 없어 기존 결합 검색을 유지했습니다.'
        return [], info
    if re.search(r'\d|아니|않|없|못\s|\b안\s|안됨|정상|제외|미발생|미확인|미실시|미수행|미교체|'
                 r'예정|계획|보류|가정|의심|또는|혹은|\b(?:not|without|normal|or)\b', question, re.I):
        info['reason'] = '숫자·부정·정상·불확실·선택 조건은 기존 검색으로 확인합니다.'
        return [], info
    # 조사만 제거하고 원래 단어 경계와 연속 구절을 유지합니다. 저장 이름을 임의로 번역하지 않습니다.
    parts = [re.sub(r'(?:에서|으로|에서의|의|은|는|이|가|을|를)$', '', p)
             for p in re.findall(r'[\w-]+', question)]
    names = {key(word) for word in word_tokens(question)}
    names.update(key(''.join(parts[i:i+n])) for n in (1, 2, 3) for i in range(len(parts)-n+1))
    names = sorted(n for n in names if 2 <= len(n) <= 80 and n not in GENERIC)
    if not names or len(names) > 384:
        info['reason'] = '명확한 설비·모듈·부품 이름이 부족합니다.'
        return [], info
    marks = ','.join('?' for _ in names)
    matched = db.execute(f'SELECT DISTINCT kind,entity FROM rag_relations WHERE entity IN ({marks})', names).fetchall()
    matched = [(kind, entity) for kind, entity in matched
               if not any(kind == other_kind and entity != other and entity in other
                          for other_kind, other in matched)]
    kinds = {kind for kind, _ in matched}
    info['matches'] = [{'kind': kind, 'name': entity} for kind, entity in matched]
    if len(kinds) < 2 or len(matched) != len(kinds) or len(set(entity for _, entity in matched)) != len(matched):
        info['reason'] = '서로 다른 역할의 이름이 두 개 이상 명확할 때만 관계를 보완합니다.'
        return [], info
    names = [entity for _, entity in matched]
    marks = ','.join('?' for _ in names)
    pairs = db.execute(f'SELECT record_id,section_id FROM rag_relations WHERE entity IN ({marks}) '
                       'GROUP BY record_id,section_id HAVING COUNT(DISTINCT kind)=? '
                       'ORDER BY record_id,section_id LIMIT 101', (*names, len(kinds))).fetchall()
    if len(pairs) > 100:
        info['reason'] = '관계 후보가 100개를 넘어 기존 검색을 유지했습니다. 이름을 더 구체적으로 입력하세요.'
        return [], info
    required = {c for c in search_concepts(question) if c.startswith('symptom')}
    query_words = set(word_tokens(question)) - QUERY_FILLER
    found = []
    for identity, section_id in pairs:
        row = db.execute('SELECT document_id,record_id,content,evidence FROM rag_documents WHERE rowid='
                         '(SELECT document_rowid FROM rag_document_ids WHERE record_id=?)', (identity,)).fetchone()
        if not row:
            continue
        document_id, _, content, encoded = row
        sections = json.loads(encoded)
        section = next(s for s in sections if s['id'] == section_id)
        common = next((s['cells'] for s in sections if s['kind'] == '접수'), {})
        symptom = '\n'.join(v for k, v in (common | section['cells']).items()
                            if k in ('현상', '문제현상', 'Defect 상세', 'Defect 유형'))
        if not required.issubset(search_concepts(symptom)):
            continue
        paths = [dict(column=column, name=value, location=location) for column, value, location in db.execute(
            f'SELECT column_name,source_value,location FROM rag_relations WHERE record_id=? AND section_id=? '
            f'AND entity IN ({marks}) ORDER BY column_name', (identity, section_id, *names))]
        info['paths'].append({'record_id': identity, 'evidence_id': section_id, 'cells': paths})
        found.append({'id': document_id, 'record_id': identity, 'kind': '문서', 'content': content,
                      'evidence': sections, 'evidence_ids': [s['id'] for s in sections],
                      'score': len(query_words & set(word_tokens(symptom)))})
    found.sort(key=lambda h: (-h['score'], h['id']))
    found = list({h['id']: h for h in found}.values())
    info.update(status='used' if found else 'no_match', reason='원문 셀의 같은 상세·작업 경계에서 함께 기록된 이름을 조회했습니다.',
                candidate_ids=[h['id'] for h in found])
    return found, info
