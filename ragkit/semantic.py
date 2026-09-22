"""CS 벡터 색인과 키워드 결합 검색. 원문·검토는 SQLite에 그대로 보존합니다."""
import json
import re
import time
import uuid
from contextlib import closing
from urllib.parse import urlparse

import numpy as np
import faiss
from openai import OpenAI, APIError

from .english import document_cells, fingerprint
from .search import retrieve, exact_terms, focused_query, distinct_hits, word_tokens, QUERY_FILLER

FIELDS = ('제목', '문제현상', '현상', '제품군', 'UNIT명', '알람코드')
_index_cache = {}  # ponytail: 한 프로세스에 최근 색인 하나만 유지합니다. 다중 사용자라면 용량 제한 캐시로 확장합니다.


def settings(config):
    value = config.get('embedding') or {}
    if not isinstance(value, dict):
        raise ValueError('embedding 설정은 JSON 객체여야 합니다.')
    result = {'base_url': '', 'model': '', 'api_key': 'EMPTY', 'document_prefix': '',
              'query_prefix': '', 'revision': '1', 'batch_size': 16, 'min_score': 0.3,
              'timeout_seconds': 120, 'auto_start': True, 'auto_index': True, **value}
    for key in ('base_url', 'model', 'api_key', 'document_prefix', 'query_prefix', 'revision'):
        if not isinstance(result[key], str):
            raise ValueError(f'embedding.{key}는 문자열이어야 합니다.')
    address = urlparse(result['base_url'])
    if not result['model'].strip() or address.scheme not in ('http', 'https') or not address.netloc or address.username or address.password:
        raise ValueError('임베딩 서버 주소와 모델을 embedding 설정에 지정하세요. 답변 모델과 별도입니다.')
    if type(result['batch_size']) is not int or not 1 <= result['batch_size'] <= 128:
        raise ValueError('임베딩 batch_size는 1~128의 정수여야 합니다.')
    for key in ('auto_start', 'auto_index'):
        if type(result[key]) is not bool:
            raise ValueError(f'embedding.{key}는 참 또는 거짓이어야 합니다.')
    for key, low, high in [('min_score', -1, 1), ('timeout_seconds', 1, 3600)]:
        if type(result[key]) not in (int, float) or not low <= result[key] <= high:
            raise ValueError(f'embedding.{key}는 {low}~{high} 사이의 숫자여야 합니다.')
    return result


def profile(config):
    cfg = settings(config)
    return fingerprint(['cs-symptom-vector-v1', FIELDS,
        {key: cfg[key] for key in ('base_url', 'model', 'document_prefix', 'query_prefix', 'revision')}])


def client_for(config, *, progress=None):
    from .embedding_runtime import prepare
    cfg = settings(config)
    # 다른 서버에 답변용 인증키를 보내지 않습니다. 임베딩용 키를 따로 지정합니다.
    client = OpenAI(base_url=cfg['base_url'], api_key=cfg['api_key'], timeout=cfg['timeout_seconds'], max_retries=0)
    try:
        prepare(client, cfg, progress)
        return client
    except Exception:
        client.close()
        raise


def document_text(encoded):
    cells, accepted = document_cells(encoded)
    lines = []
    for key in FIELDS:
        # 미채택 제안은 포함하지 않습니다. 원문 표현과 채택한 표현은 둘 다 검색합니다.
        for value in dict.fromkeys([cells.get(key, ''), accepted.get(key, '')]):
            if value.strip():
                lines.append(f'{key}: {value}')
    return '\n'.join(lines)


def ensure_tables(db):
    db.execute('CREATE TABLE IF NOT EXISTS vector_sources (record_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL, content TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS vector_source_hash ON vector_sources(source_hash)')
    db.execute('''CREATE TABLE IF NOT EXISTS vector_embeddings (
        source_hash TEXT, profile TEXT, dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
        PRIMARY KEY(source_hash, profile))''')
    db.execute('CREATE TABLE IF NOT EXISTS vector_meta (id INTEGER PRIMARY KEY CHECK(id=1), revision TEXT NOT NULL)')
    db.execute('INSERT OR IGNORE INTO vector_meta VALUES (1,?)', (uuid.uuid4().hex,))


def touch(db):
    db.execute('UPDATE vector_meta SET revision=? WHERE id=1', (uuid.uuid4().hex,))


def sync_sources(db, record_ids=None):
    """검토·원문 갱신 트랜잭션에서 호출합니다. 오래된 벡터는 즉시 검색에서 제외됩니다."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='vector_sources'").fetchone():
        return
    before = db.total_changes
    from .english import translation_documents
    for identity, encoded in translation_documents(db, record_ids):
        content = document_text(encoded)
        db.execute('''INSERT INTO vector_sources VALUES (?,?,?) ON CONFLICT(record_id) DO UPDATE
            SET source_hash=excluded.source_hash, content=excluded.content
            WHERE source_hash != excluded.source_hash''', (identity, fingerprint(content), content))
    if record_ids is None:
        db.execute('DELETE FROM vector_sources WHERE record_id NOT IN (SELECT record_id FROM rag_documents)')
    if db.total_changes != before:
        touch(db)


def vector_count(db, config):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='vector_sources'").fetchone():
        return 0
    try:
        key = profile(config)
    except ValueError:
        return 0
    return db.execute('''SELECT COUNT(*) FROM vector_sources s JOIN vector_embeddings e
        ON s.source_hash=e.source_hash AND e.profile=? WHERE s.content!='' ''', (key,)).fetchone()[0]


def embed(client, texts, cfg, *, query=False):
    prefix = cfg['query_prefix'] if query else cfg['document_prefix']
    result = client.embeddings.create(model=cfg['model'], input=[prefix + text for text in texts], encoding_format='float')
    entries = sorted(result.data, key=lambda item: item.index)
    if [item.index for item in entries] != list(range(len(texts))):
        raise ValueError('임베딩 응답의 개수·순서가 입력과 다릅니다.')
    try:
        values = np.asarray([item.embedding for item in entries], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError('임베딩 응답의 숫자 배열이 올바르지 않습니다.') from error
    if values.ndim != 2 or not 1 <= values.shape[1] <= 65536 or not np.isfinite(values).all():
        raise ValueError('임베딩 차원 또는 숫자가 올바르지 않습니다.')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('0 벡터 또는 유효하지 않은 임베딩입니다.')
    return values / norms


def build_vectors(db, client, config, *, limit=0, progress=None, cancelled=None):
    cfg, key = settings(config), profile(config)
    if type(limit) is not int or limit < 0:
        raise ValueError('벡터 처리 건수는 0(전체) 또는 양의 정수여야 합니다.')
    if progress:
        progress('임베딩할 자료와 기존 벡터 확인 중')
    with db:
        ensure_tables(db)
        sync_sources(db)
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='rag_document_ids'").fetchone():
            from .cslist import ensure_document_ids
            ensure_document_ids(db)
    total = db.execute("SELECT COUNT(*) FROM vector_sources WHERE content!=''").fetchone()[0]
    before = vector_count(db, config)
    dimensions = {d for (d,) in db.execute('SELECT DISTINCT dimensions FROM vector_embeddings WHERE profile=?', (key,))}
    if len(dimensions) > 1:
        raise ValueError('같은 임베딩 설정의 벡터 차원이 다릅니다. revision을 바꾸고 다시 생성하세요.')
    dimension = next(iter(dimensions), None)
    pending = db.execute('''SELECT s.source_hash, MIN(s.content) FROM vector_sources s
        LEFT JOIN vector_embeddings e ON s.source_hash=e.source_hash AND e.profile=?
        WHERE e.source_hash IS NULL AND s.content!='' GROUP BY s.source_hash ORDER BY MIN(s.record_id) LIMIT ?''',
        (key, limit or -1)).fetchall()
    generated, failures = 0, []
    for start in range(0, len(pending), cfg['batch_size']):
        if cancelled and cancelled():
            from .storage import ProcessingCancelled
            raise ProcessingCancelled()
        batch = pending[start:start + cfg['batch_size']]
        try:
            vectors = embed(client, [content for _, content in batch], cfg)
            if dimension is not None and dimension != vectors.shape[1]:
                raise ValueError('임베딩 모델의 차원이 바뀌었습니다. revision을 변경해 재생성하세요.')
        except (APIError, ValueError) as error:
            failures.extend({'hash': digest, 'error': str(error)[:300],
                             **({'status_code': error.status_code} if getattr(error,'status_code',None) is not None else {})}
                            for digest, _ in batch)
            if progress:
                progress(f'{min(start + len(batch), len(pending))}/{len(pending)}개 문서 시도 · 실패 {len(failures)}개')
            continue
        dimension = vectors.shape[1]
        with db:
            db.executemany('INSERT OR REPLACE INTO vector_embeddings VALUES (?,?,?,?)',
                [(digest, key, dimension, vector.astype('<f4').tobytes()) for (digest, _), vector in zip(batch, vectors)])
            touch(db)
        generated += len(batch)
        if progress:
            progress(f'{start + len(batch)}/{len(pending)}개 문서 시도 · 새 벡터 {generated}개 · 실패 {len(failures)}개')
    return {'rows': total, 'indexed': vector_count(db, config), 'previously_indexed': before,
            'generated': generated, 'attempted': len(pending), 'failures': failures, 'dimensions': dimension,
            'model': cfg['model'], 'index': 'FAISS FlatIP'}


def load_index(db, config):
    key = profile(config)
    if not vector_count(db, config):
        raise ValueError('현재 모델로 생성한 벡터가 없습니다. 자료 화면에서 벡터 색인을 생성하세요.')
    revision = db.execute('SELECT revision FROM vector_meta WHERE id=1').fetchone()[0]
    path = db.execute('PRAGMA database_list').fetchone()[2]
    cache_key = (path, key, revision)
    if path and _index_cache.get('key') == cache_key:
        return _index_cache['value']
    identities, matrix, dimension = [], [], None
    for identity, size, blob in db.execute('''SELECT s.record_id,e.dimensions,e.vector FROM vector_sources s
        JOIN vector_embeddings e ON s.source_hash=e.source_hash AND e.profile=?
        WHERE s.content!='' ORDER BY s.record_id''', (key,)):
        vector = np.frombuffer(blob, dtype='<f4')
        if dimension is None:
            dimension = size
        if size != dimension or len(vector) != size or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError('저장된 벡터가 손상되었거나 차원이 다릅니다. embedding.revision을 변경하고 다시 생성하세요.')
        identities.append(identity)
        matrix.append(vector)
    # ponytail: 전체 비교 색인으로 10만 건까지 먼저 측정합니다. 검색 시간이 길어질 때 HNSW와 비교합니다.
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(matrix, dtype=np.float32))
    value = (identities, index)
    if path:
        _index_cache.clear()
        _index_cache.update(key=cache_key, value=value)
    return value


def vector_hits(db, question, config, *, limit=20, timings=None, embedding_client=None):
    timings = timings if timings is not None else {}
    question = focused_query(question)
    cfg = settings(config)
    started = time.monotonic()
    identities, index = load_index(db, config)
    timings['index_load'] = round(time.monotonic() - started, 4)
    started = time.monotonic()
    if embedding_client is None:
        with closing(client_for(config)) as client:
            query = embed(client, [question], cfg, query=True)
    else:
        # 평가 호출자는 이미 실행 중인 서버의 클라이언트를 소유합니다. 자동 준비를 하지 않습니다.
        query = embed(embedding_client, [question], cfg, query=True)
    timings['embedding'] = round(time.monotonic() - started, 4)
    if query.shape[1] != index.d:
        raise ValueError('질문과 문서의 벡터 차원이 다릅니다. 같은 모델로 다시 생성하세요.')
    timings.update(vector_search=0, source_lookup=0)
    fetch_limit = limit
    while True:
        started = time.monotonic()
        scores, positions = index.search(query, min(fetch_limit, index.ntotal))
        timings['vector_search'] += time.monotonic() - started
        candidates = [(identities[position], float(score)) for score, position in zip(scores[0], positions[0])
                      if position >= 0 and score >= cfg['min_score']]
        if not candidates:
            return []
        started = time.monotonic()
        placeholders = ','.join('?' for _ in candidates)
        # 공개 rowid를 기본키로 찾습니다. 원문과 근거는 기존 FTS 문서 그대로 반환합니다.
        indexed = db.execute("SELECT 1 FROM sqlite_master WHERE name='rag_document_ids'").fetchone()
        condition = (f'rowid IN (SELECT document_rowid FROM rag_document_ids WHERE record_id IN ({placeholders}))'
                     if indexed else f'record_id IN ({placeholders})')
        # ponytail: 구형 DB의 읽기 전용 호출은 전체 스캔으로 호환합니다. UI·벡터 생성 시 색인을 자동 이관합니다.
        rows = {row[1]: row for row in db.execute(
            f'SELECT document_id,record_id,content,evidence FROM rag_documents WHERE {condition}',
            [identity for identity, _ in candidates])}
        hits = []
        for parent, score in candidates:
            row = rows.get(parent)
            if row is None:
                continue
            identity, parent, content, encoded = row
            evidence = json.loads(encoded)
            hits.append({'id': identity, 'record_id': parent, 'kind': '문서', 'content': content,
                         'evidence': evidence, 'evidence_ids': [s['id'] for s in evidence], 'score': float(score)})
        timings['source_lookup'] += time.monotonic() - started
        unique = distinct_hits(hits)
        if len(unique) >= limit or len(candidates) < fetch_limit or fetch_limit >= 100:
            return unique[:limit]
        # 같은 질문 벡터로 한 번만 더 조회합니다. 원문·수치·조치가 다른 사례는 묶지 않습니다.
        fetch_limit = 100


def prefer_action_variants(hits, question):
    """같은 증상·조치 상태 안에서 반복 조치문을 뒤로 보냅니다. 원문이나 후보를 삭제하지 않습니다."""
    if re.search(r'\d', question) or len(hits) < 3:
        return hits  # 수치·모델 번호 조건은 기존 순서를 그대로 사용합니다.
    terms = set(word_tokens(focused_query(question))) - QUERY_FILLER
    keys, groups = [], {}
    for hit in hits:
        sections = hit.get('evidence', [])
        encoded = json.dumps(sections, ensure_ascii=False)
        symptom = document_text(encoded) if sections else ''
        actions = tuple(tuple(sorted((column, value.strip()) for section in sections
                                     for column, value in section.get(field, {}).items()
                                     if value.strip() and (column.startswith(('원인(', '조치내용('))
                                                           or column == '재발방지대책')))
                        for field in ('cells', 'accepted'))
        states = tuple((section.get('kind'), tuple((a['column'], a['status'])
                         for a in section.get('action_outcomes', []))) for section in sections)
        coverage = tuple(sorted(term for term in terms if term in hit.get('content', '').casefold()))
        key = (symptom, states, coverage) if symptom and any(actions) else hit['id']
        keys.append(key)
        groups.setdefault(key, []).append((hit, actions))
    for key, items in groups.items():
        seen, first, repeated = set(), [], []
        for hit, actions in items:
            (repeated if actions in seen else first).append(hit)
            seen.add(actions)
        groups[key] = iter(first + repeated)
    # ponytail: 같은 증상·상태·질문 단어가 일치하는 그룹 안에서만 재배치합니다. 의미가 비슷하다는 이유로 합치지 않습니다.
    return [next(groups[key]) for key in keys]


def search_cases(db, question, config, *, mode='hybrid', keyword_hits=None, details=None, embedding_client=None, limit=3):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('참고 사례 수는 1~100 사이 정수여야 합니다.')
    candidate_limit = max(20, limit)
    if mode not in ('keyword', 'vector', 'hybrid'):
        raise ValueError('검색 방식은 keyword, vector, hybrid 중 하나여야 합니다.')
    details = details if details is not None else {}
    timings = details.setdefault('timings', {})
    started = time.monotonic()
    # 보완어는 결합 검색의 후보에만 사용합니다. 키워드 단독 순위는 보존합니다.
    keyword = retrieve(db, question, limit=candidate_limit, expand_query=mode == 'hybrid') if keyword_hits is None else keyword_hits
    if keyword_hits is None:
        timings['keyword'] = round(time.monotonic() - started, 4)
    info = {'mode': mode, 'query': question, 'focused_query': focused_query(question),
            'keyword_ids': [h['id'] for h in keyword[:limit]], 'vector_ids': [], 'hybrid_ids': [],
            'vector_records': vector_count(db, config), 'warning': ''}
    total = db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0]
    if mode != 'keyword' and info['vector_records'] < total:
        info['warning'] = f"벡터 검색 범위는 전체 {total}건 중 {info['vector_records']}건입니다. 남은 벡터를 생성하세요."
    info['candidate_ids'] = {'keyword': [h['id'] for h in keyword], 'vector': []}
    details['vector_search'] = info
    if any(exact_terms(question)):
        info.update(mode='exact', warning='접수번호·코드·날짜 조건은 기존 정확 검색을 사용했습니다.')
        return keyword[:limit]
    if mode == 'keyword':
        return keyword[:limit]
    started = time.monotonic()
    try:
        options = {'embedding_client': embedding_client} if embedding_client is not None else {}
        vectors = vector_hits(db, question, config, limit=candidate_limit, timings=timings, **options)
    except (ValueError, RuntimeError, APIError) as error:
        if mode == 'vector':
            raise ValueError('벡터 검색을 완료하지 못했습니다. 임베딩 설정과 색인을 확인하세요.') from error
        info['warning'] = '벡터 검색을 사용할 수 없어 키워드 검색을 사용했습니다. 임베딩 설정과 색인을 확인하세요.'
        if keyword_hits is None:
            keyword = retrieve(db, question, limit=candidate_limit)
            info['keyword_ids'] = [h['id'] for h in keyword[:limit]]
            info['candidate_ids']['keyword'] = [h['id'] for h in keyword]
        return keyword[:limit]
    info.update(vector_ids=[h['id'] for h in vectors[:limit]], vector_seconds=round(time.monotonic() - started, 3))
    info['candidate_ids']['vector'] = [h['id'] for h in vectors]
    started = time.monotonic()
    scores, hits = {}, {}
    for found in (keyword, vectors):
        for rank, hit in enumerate(found, 1):
            hits.setdefault(hit['id'], hit)
            scores[hit['id']] = scores.get(hit['id'], 0) + 1 / (60 + rank)
    # RRF 점수는 유지합니다. 동점만 의미 검색 순위로 풀어 키워드 입력 순서의 편향을 없앱니다.
    vector_ranks = {hit['id']: rank for rank, hit in enumerate(vectors, 1)}
    ordered = sorted(hits, key=lambda key: (-scores[key], vector_ranks.get(key, len(vectors) + 1), key))
    combined = distinct_hits([hits[key] | {'score': scores[key]} for key in ordered])
    before = vectors if mode == 'vector' else combined
    selected = prefer_action_variants(before, question)[:limit]
    info['action_diversity'] = {'candidates': len(before),
                               'changed_positions': sum(a['id'] != b['id'] for a, b in zip(before, selected))}
    timings['fusion'] = round(time.monotonic() - started, 4)
    info['fusion'] = 'rrf_vector_tiebreak'
    info['hybrid_ids'] = [h['id'] for h in selected] if mode == 'hybrid' else [h['id'] for h in combined[:limit]]
    if mode == 'vector':
        info['vector_ids'] = [h['id'] for h in selected]
    return selected
