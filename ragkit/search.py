"""CS 자료는 한국어 단어를 우선 검색합니다. 원문과 채택한 셀만 사용합니다."""
from functools import lru_cache
import json
import re
import unicodedata
from .input_formats import CASE_NUMBER_SQL

# ponytail: 명시한 세 증상과 두 설비만 동의어로 연결합니다. 더 넓은 의미 검색은 별도 평가 후 확장합니다.
CONCEPTS = {
    'symptomnoise': r'소음|이상음|噪[音声聲]|异响|異響|(?<![a-z])noise(?![a-z])',
    'symptomleak': r'누수|누설|漏水|泄漏|洩漏|(?<![a-z])leak(?:age)?(?![a-z])',
    'symptomvibration': r'진동|振动|振動|(?<![a-z])vibration(?![a-z])',
    'equipmentpump': r'펌프|泵|(?<![a-z])pump(?![a-z])',
    'equipmentfan': r'팬|风扇|風扇|(?<![a-z])fan(?![a-z])',
}


def concepts(text):
    return {name for name, pattern in CONCEPTS.items() if re.search(pattern, text.casefold())}


def search_concepts(text):
    # 현장 표현은 검색 후보를 넓히는 데만 사용합니다. 원문 인용 검사에서 쓰는
    # concepts()는 바꾸지 않아, 표현 확장을 사실 확인으로 오해하지 않게 합니다.
    text = unicodedata.normalize('NFC', text)
    found = concepts(text)
    if re.search(r'(?:물|냉각수|오일|기름|액체)(?:이|가)?\s*(?:새(?:고|는|어|요|서|지|면|다)|샌(?:다|다는|다고)|샙니|샜|샘)', text):
        found.add('symptomleak')
    if re.search(r'웅웅|윙윙|덜그럭|달그락|이상한\s*소리|소리(?:가)?\s*(?:너무\s*)?(?:크|커|큼|큽)', text):
        found.add('symptomnoise')
    if re.search(r'떨림|떨리|떨려|덜덜', text):
        found.add('symptomvibration')
    return found


@lru_cache(maxsize=1)
def korean_analyzer():
    # 한 번 설치한 사전으로 로컬에서 분석합니다. LLM/API 호출이나 학습은 없습니다.
    from kiwipiepy import Kiwi
    analyzer = Kiwi(num_workers=1)
    for word in ('유량', '이상음'):
        analyzer.add_user_word(word, 'NNG')
    return analyzer


@lru_cache(maxsize=8192)
def word_tokens(text):
    # 반복되는 셀만 제한된 크기의 메모리에 재사용합니다. 원문 셀은 수정하지 않습니다.
    text = unicodedata.normalize('NFC', text).casefold()
    words = re.findall(r'[a-z0-9]+', text)  # E202 같은 영문+숫자 식별자를 보존합니다.
    if re.search('[가-힣]', text):
        words += [token.form for token in korean_analyzer().tokenize(text)
                  if token.tag.startswith(('NNG', 'NNP', 'VV', 'VA', 'XR'))
                  and re.fullmatch('[가-힣]+', token.form)]
    return tuple(words + [token for group in exact_terms(text) for token in group])


def word_grams(word):
    # 두 글자 조각은 의미 있는 한국어 단어에서만 만듭니다. '니다' 같은 어미는 제외됩니다.
    return [word[i:i+2] for i in range(len(word)-1)] if re.fullmatch('[가-힣]{3,}', word) else []


# 질문의 안내 표현이 모든 조치 기록을 검색 후보로 끌어들이지 않게 합니다.
# 원문과 색인에는 남기며, 질문에 적용할 때도 증상·코드·날짜는 유지합니다.
QUERY_FILLER = set('과거 이전 당시 사례 기록 관련 방법 조치 알려 알리 찾 찾아보 주 하 되 있 없 어떻 것 수 좀'.split())


def query_aliases(text):
    """현장 표현의 검색어만 보완합니다. 원문·인용 검사·모델에 보낼 질문은 바꾸지 않습니다."""
    text = unicodedata.normalize('NFC', text)
    words = set(word_tokens(text))
    extra = set()
    if '송풍기' in words:
        extra.add('fan')
    if '뜨겁' in words:
        extra.update(('온도', '고온'))
    if re.search(r'(?<!\w)대기압(?:으로)?\s*(?:복귀|(?:되)?돌아[오가갑와왔갔])', text):
        extra.update(('vent', '벤트'))
    # ponytail: 세 가지 확인된 표현만 확장합니다. 새 현장 용어는 회귀 질문과 함께 추가합니다.
    return extra


def exact_terms(text):
    # 하이픈·밑줄을 포함한 코드와 ISO 날짜는 글자 조각으로 비교하면 다른 기록과
    # 섞입니다. 구두점이 FTS에서 다시 분리되지 않도록 ASCII 16진수 토큰으로 저장합니다.
    codes = {m.casefold() for m in re.findall(r'(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+', text)
             if any(char.isdigit() for char in m)}
    dates = set(re.findall(r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)', text))
    return [['exactcode' + value.encode().hex() for value in sorted(codes)],
            ['exactdate' + value.replace('-', '') for value in sorted(dates)]]


def focused_query(question):
    """명시적인 'A가 아니라 B'의 B를 검색합니다. 답변용 원래 질문은 호출자가 보존합니다."""
    if any(exact_terms(question)):
        return question
    text = unicodedata.normalize('NFC', question)
    contrast = re.search(r'(?<=[이가게])\s+아니라\s+(\w[\s\S]*)', text)
    if contrast and text.count('아니라') == 1:
        head = text[:contrast.start()]
        if re.search(r'(?:뿐|만)(?:이)?$', head):
            return question  # '뿐만 아니라'는 제외가 아닌 추가 조건입니다.
        boundaries = list(re.finditer(r'[.!?;,]\s+|\n\s*', head))
        context = head[:boundaries[-1].end()] if boundaries else ''
        return context + contrast[1]
    # 전체 영어 번역은 답변에 그대로 사용하고, 이 명시적 대조만 검색에서 분리합니다.
    contrast = re.search(r'\bnot\s+(?!only\b|just\b|merely\b|simply\b)[^.!?;\n]+?\s+but\s+(\w[\s\S]*)', text, re.I)
    if contrast and len(re.findall(r'\bnot\b|\bbut\b', text, re.I)) == 2:
        return text[:contrast.start()] + contrast[1]
    # ponytail: 복잡한 부정/대조 문장은 원문 유지. 실제 회귀 질문을 확보한 표현만 추가합니다.
    return question


def distinct_hits(hits):
    """접수번호만 다르고 원문·채택 셀이 같은 결과를 한 번만 보여 줍니다. DB는 그대로 둡니다."""
    seen, result = set(), []
    for hit in hits:
        key = tuple(tuple(sorted({column: unicodedata.normalize('NFC', value).strip()
                                  for section in hit.get('evidence', [])
                                  for column, value in section.get(field, {}).items()
                                  if column not in ('CS 접수번호', 'CS NO') and value.strip()}.items()))
                    for field in ('cells', 'accepted'))
        if not key[0]:
            key = hit['id']  # 원문 근거가 없는 구형 자료는 임의로 합치지 않습니다.
        if key not in seen:
            result.append(hit)
            seen.add(key)
    return result


def tokens(text):
    # CS 양식이 없는 초기 데모와 아직 갱신하지 않은 구형 DB의 호환 경로입니다.
    # '냉각수를'를 '냉각', '각수', '수를'로 나누어 조사 차이를 일부 흡수합니다.
    # ponytail: 문자 겹침이라 동의어를 이해하지 못함; 의미 검색이 필요하면 임베딩 모델 추가.
    words = re.findall(r'[^\W_]+', text.casefold())
    return ([word[i:i+2] for word in words for i in range(max(1, len(word)-1))]
            + [token for group in exact_terms(text) for token in group])


def terms(text):
    return set(tokens(text))


def receipt_number(db, question):
    """명시적 CS 번호 또는 번호만 입력한 조회를 검색 방식 사이에서도 유지합니다."""
    explicit = re.search(r'(?:CS NO|CS 접수번호|접수번호)\s*[=:]?\s*"?([A-Za-z0-9_-]+)', question, re.I)
    if explicit:
        return explicit[1].casefold()
    candidate = question.strip(' "').casefold()
    if re.fullmatch(r'(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+', candidate) and db.execute(
            f'SELECT 1 FROM normalized_records WHERE lower({CASE_NUMBER_SQL})=? LIMIT 1', (candidate,)).fetchone():
        return candidate
    return ''


def retrieve(db, question, *, original_only=False, english_question=None, limit=3, expand_query=False):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('검색 후보 수는 1~100의 정수여야 합니다.')
    question = focused_query(question)
    if english_question is not None and not any(exact_terms(question)):
        english_question = focused_query(english_question)
    # 정리문과 원문을 함께 검색해, 정리 과정에서 빠진 표현도 찾을 기회를 남깁니다.
    query = terms(question)
    if not query:
        raise ValueError('질문에 검색할 문자를 입력하세요.')
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='rag_documents'").fetchone():
        target = 'rag_original_documents' if original_only else 'rag_documents'
        if english_question is not None:
            if original_only:
                raise ValueError('영어 검색은 현재 접수 문서 검색에서 사용하세요.')
            target = 'rag_english_documents'
        columns = {c[1] for c in db.execute(f'PRAGMA table_info({target})')}
        modern = 'grams' in columns
        named = search_concepts(question) if modern else concepts(question)
        query = (set(word_tokens(question)) - QUERY_FILLER if modern else query) | named
        if english_question is not None:
            filler = set('a an the is are was were be been being do does did has have had to of for in on at and or with from please find past case cases problem issue how what'.split())
            query = (set(word_tokens(english_question)) - filler) | named
        if not query:
            raise ValueError('검색할 현상이나 설비·접수번호를 함께 입력하세요.')
        # MATCH 문법을 사용자에게서 받지 않고, 분리한 검색어를 인용해 구성합니다.
        if len(query) > 128:
            raise ValueError('질문이 너무 깁니다. 한 가지 질문으로 나누어 입력하세요.')
        codes, dates = exact_terms(question)
        expanded = query
        if expand_query and modern and english_question is None and not codes and not dates:
            candidate = query | query_aliases(question)
            if len(candidate) <= 128:
                expanded = candidate  # 원래 단어와 보조 조각 검색의 조건은 보존합니다.
        expression = ' OR '.join('"' + token + '"' for token in sorted(expanded))
        if modern:
            expression = '{keywords symptom alarm equipment action} : (' + expression + ')'
        # 명시한 코드·날짜가 없다면 유사한 다른 번호로 대신 답하지 않습니다.
        # 여러 코드를 비교하는 질문은 코드 중 하나, 날짜가 있으면 해당 날짜도 맞아야 합니다.
        code_groups = {}
        for token in codes:
            value = bytes.fromhex(token.removeprefix('exactcode')).decode()
            family = re.split('[-_]', value)[0]
            code_groups.setdefault(family, []).append(token)
        # 접수번호와 설비코드를 OR로 묶으면 없는 접수번호 대신 다른 건을 찾습니다.
        # 같은 접두사의 복수 코드는 OR, 서로 다른 코드 종류와 날짜는 AND입니다.
        constraints = ['(' + ' OR '.join('"' + token + '"' for token in group) + ')'
                       for group in [*code_groups.values(), dates] if group]
        families = ({re.split('[-_]', identity.casefold())[0] for (identity,) in db.execute(f'SELECT lower({CASE_NUMBER_SQL}) FROM normalized_records')}
                    if code_groups else set())
        receipt_codes = [bytes.fromhex(token.removeprefix('exactcode')).decode()
                         for family, group in code_groups.items() if family in families for token in group]
        # 식별번호는 본문에 언급됐다는 이유로 일치시키지 않고 실제 부모 ID도 확인합니다.
        # 숫자만 있는 CS NO도 명시적 번호 조회로 지원합니다. 본문의 측정값과 혼동하지 않습니다.
        candidate = receipt_number(db, question) if not receipt_codes else ''
        if candidate:
            receipt_codes = [candidate]
        id_filter = (' AND record_id IN (SELECT id FROM normalized_records WHERE lower(' + CASE_NUMBER_SQL + ') IN ('
                     + ','.join('?' for _ in receipt_codes) + '))') if receipt_codes else ''
        focused = 'symptom' in columns
        rank_function = 'bm25()'
        if focused:
            rank_function = 'bm25(0, 0, 0, 0, 1, 8, 10, 4, 2' + (', 0.2)' if modern else ')')
            # 접수번호 조회는 그 접수의 전체 기록을 돌려줍니다. 증상 검색에서는
            # 명시한 증상 중 하나와 설비 중 하나가 모두 있어야 후보에 포함합니다.
            # 부정·미확인도 검색 대상이며, 일치 자체가 발생·해결을 뜻하지는 않습니다.
            if not receipt_codes:
                groups = [[name for name in sorted(named) if name.startswith(prefix)]
                          for prefix in ('symptom', 'equipment')]
                if groups[0]:
                    constraints += ['keywords : (' + ' OR '.join('"' + name + '"' for name in group) + ')'
                                    for group in groups if group]
        parameters = [' AND '.join(constraints + ['(' + expression + ')']), *receipt_codes]
        if not columns:
            raise ValueError('문서 색인이 없습니다. 같은 Excel을 다시 읽어 색인을 갱신하세요.')
        # 숨은 rank 열로 정렬해야 FTS가 상위 후보만 빠르게 가져옵니다.
        # bm25(...) AS rank는 이름만 같아서 전체 일치 문서를 정렬하게 됩니다.
        parameters.append(rank_function)
        sql = (f'SELECT document_id, record_id, content, evidence, rank FROM {target} '
               f'WHERE {target} MATCH ?{id_filter} AND rank MATCH ? ORDER BY rank LIMIT {limit}')
        found = list(db.execute(sql, parameters))
        if not found and modern and not codes and not dates and english_question is None:
            # ponytail: 단어 검색이 비었을 때만 보조 검색. 모든 질문 단어의 조각을
            # 함께 요구해 일부 두 글자만 우연히 겹치는 기록은 제외합니다. 오탈자 교정은 하지 않습니다.
            fallback = []
            for word in sorted(query):
                grams = word_grams(word)
                fallback.append('grams : "' + ' '.join(grams) + '"'
                                if grams else 'keywords : "' + word + '"')
            if any(word_grams(word) for word in query):
                parameters[0] = ' AND '.join(constraints + fallback)
                found = list(db.execute(sql, parameters))
        # ponytail: 동점 경계의 후보 선택은 FTS에 맡깁니다. 최신 접수 우선 등 업무 규칙이 생기면 별도 정렬합니다.
        found.sort(key=lambda row: (row[-1], row[0]))
        hits = []
        for identity, parent, content, evidence, rank in found:
            sections = json.loads(evidence)
            hits.append({'id': identity, 'record_id': parent, 'kind': '문서', 'content': content,
                         'evidence': sections, 'evidence_ids': [s['id'] for s in sections], 'score': -rank})
        if any(exact_terms(question)):
            return hits
        unique = distinct_hits(hits)
        if len(unique) < limit and len(hits) == limit and limit < 100:
            # ponytail: 중복으로 빈 자리만 최대 100위 안에서 보충합니다. 더 먼 후보는 별도 품질 평가 후 확대합니다.
            return retrieve(db, question, original_only=original_only, english_question=english_question,
                            limit=100, expand_query=expand_query)[:limit]
        return unique
    hits = []
    for record_id, original, normalized, content in db.execute('SELECT id, original_text, normalized_json, search_text FROM normalized_records ORDER BY id'):
        document = terms(content)
        score = len(query & document) / len(query | document)
        if score > 0:
            hits.append({'id': record_id, 'content': content, 'original': original,
                         'normalized': json.loads(normalized), 'score': score})
    return sorted(hits, key=lambda hit: -hit['score'])[:limit]


def evidence_ids(hits):
    return {identity for hit in hits for identity in hit.get('evidence_ids', [hit['id']])}


def retrieval_check(hits, case):
    if case.get('expect_no_match'):
        return not hits
    expected = set(case.get('expected_ids', []))
    return expected.issubset(evidence_ids(hits)) if expected else None


def compare_search(db, cases):
    """같은 DB 스냅샷에서 원문과 원문+채택문을 비교하며 검토 선택은 바꾸지 않습니다."""
    changed = sum(len(section['accepted']) for (encoded,) in db.execute('SELECT evidence FROM rag_documents')
                  for section in json.loads(encoded))
    result = []
    for case in cases:
        original = retrieve(db, case['question'], original_only=True)
        accepted = retrieve(db, case['question'])
        expected = set(case.get('expected_ids', []))
        result.append({'question': case['question'], 'review': case.get('review', ''),
                       'expected_ids': sorted(expected), 'changed_cells': changed,
                       'original_hits': original, 'accepted_hits': accepted,
                       'original_pass': retrieval_check(original, case),
                       'accepted_pass': retrieval_check(accepted, case)})
    return result


def count_records(db, question):
    """건수 질문만 처리합니다. 해석 못 한 조건을 버리고 전체 건수로 답하지 않습니다."""
    if not re.search(r'(몇\s*건|건수|총\s*건|총\s*개수)', question):
        return None
    # ponytail: 집계는 명시한 열="값"의 동등 조건만 지원; 자연어 기간·상태 조건은 별도 구현 전까지 거절합니다.
    columns = ('CS 접수번호', 'CS NO', '설비코드', '접수일자', '고객사명', '제품군', '유/무상')
    pattern = r'(' + '|'.join(map(re.escape, columns)) + r')\s*=\s*"([^"\n]*)"'
    filters = re.findall(pattern, question)
    remainder = re.sub(pattern, '', question)
    remainder = re.sub(r'전체|현재|저장된|저장|등록된|등록|데이터|기록|접수|총|몇|건수|건|개수|인가요|입니까|있나요|있습니까|알려주세요|알려줘|는|은|이|가|의|과|와|및|그리고|[\s?？.,]', '', remainder)
    repeated_columns = len({'CS 접수번호' if key == 'CS NO' else key for key, _ in filters}) != len(filters)
    if remainder or repeated_columns:
        return {'count': None, 'filters': filters, 'answer': '집계 조건을 확정할 수 없습니다. 전체 접수는 총 몇 건인가요? 또는 설비코드="DEMO-ETCH-01" 접수는 몇 건인가요?처럼 입력하세요. 기간·재발·완료 조건은 아직 지원하지 않습니다.'}
    clauses, params = [], []
    for key, value in filters:
        if key in ('CS 접수번호', 'CS NO'):
            expression = CASE_NUMBER_SQL
        else:
            expression = 'json_extract(original_json, ?)'
            params.append('$."' + key + '"')
        if key == '접수일자':
            expression = f'substr({expression}, 1, 10)'
        clauses.append(expression + ' = ?')
        params.append(value)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    count = db.execute(f'SELECT COUNT(DISTINCT lower({CASE_NUMBER_SQL})) FROM normalized_records' + where, params).fetchone()[0]
    scope = ', '.join(f'{k}="{v}"' for k, v in filters) or '전체'
    return {'count': count, 'filters': filters,
            'answer': f'현재 DB에 등록된 전체 자료에서 {scope} 접수는 {count}건입니다. 전체 원본의 CS 번호를 집계했으며, 다른 양식의 같은 번호는 한 건으로 셌습니다.'}
