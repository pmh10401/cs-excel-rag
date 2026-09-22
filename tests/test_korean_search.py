"""조사·어미가 같은 무관한 기록과 현장 표현을 실제 FTS5 검색으로 구별합니다."""
import sqlite3
from pathlib import Path
from ragkit import cslist, storage, search


CASES = [
    {'현상': '냉각수 누수 발생', '조치내용(1)': '호스를 교체했습니다.'},
    {'현상': '통신 단절 발생', '조치내용(1)': '케이블을 교체했습니다.'},
    {'현상': '온도 상승', '조치내용(1)': '히터를 점검했습니다.'},
    {'현상': '유량이 부족합니다.', '조치내용(1)': '스트레이너를 세척했습니다.'},
    {'현상': '펌프 이상음', '조치내용(1)': '베어링 점검 예정. 교체 안함.'},
    {'현상': '팬 진동', '조치내용(1)': '진동 수치 미기재. 원인 미확인.'},
    {'현상': '새로운 펌프 설치', '조치내용(1)': '시운전 예정.'},
    {'현상': '물이 새지 않음. 발생 여부 미확인.', '조치내용(1)': '점검만 했습니다.'},
    {'현상': '흡광광도계 감도 저하', '조치내용(1)': '측정 예정'},
    {'현상': '분광광도기', '조치내용(1)': '점검 예정'},
]

# 결과 전체가 허용한 접수에 속하는지 검사합니다. 정답 한 건과 오답 두 건을
# 함께 반환하는 검색을 통과시키지 않으며, 해당 사례가 없을 때도 검사합니다.
QUERIES = [
    ('물이 샙니다', {1, 8}), ('물이 샌다', {1, 8}), ('물이 새요', {1, 8}),
    ('냉각수가 새고 있어요', {1, 8}), ('물 새는 과거 사례', {1, 8}),
    ('펌프에서 웅웅거려요', {5}), ('펌프가 윙윙해요', {5}),
    ('펌프에서 달그락 소리가 납니다', {5}), ('pump noise', {5}),
    ('팬이 떨려요', {6}), ('팬 떨림', {6}), ('fan vibration', {6}),
    ('팬에서 물이 새요', set()), ('펌프가 떨려요', set()),
    ('냉각수를', {1}), ('통신이', {2}), ('온도가 상승합니다', {3}),
    ('유량이 부족해요', {4}), ('CS-KO-007 누수 여부', {7}),
    ('광도계', {9}), ('분광광도계', set()),
    ('누수', {1, 8}),
]


def main(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    rows = [{key: values.get(key, '') for key in schema['columns']} |
            {schema['id_column']: f'CS-KO-{index:03}', '_id_column': schema['id_column']}
            for index, values in enumerate(CASES, 1)]
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        for question, allowed in QUERIES:
            for original_only in (True, False):
                hits = search.retrieve(db, question, original_only=original_only)
                found = {int(h['record_id'].split('-')[-1]) for h in hits}
                assert found == allowed, ('관련 접수 누락·무관한 후보', question, original_only, found)
        assert '새지 않음' in search.retrieve(db, 'CS-KO-008')[0]['content']
        keywords = db.execute('SELECT keywords FROM rag_documents WHERE record_id=?', ('CS-KO-001',)).fetchone()[0].split()
        assert '냉각수' in keywords and not {'각수', '니다', '했습'} & set(keywords), keywords
        # 기본 단어가 없는 경우에만 모든 조각을 순서대로 찾습니다. 조각 하나만
        # 겹치는 '분광광도기'를 '분광광도계'로 대신 반환하면 안 됩니다.
        assert '광도계' not in db.execute('SELECT keywords FROM rag_documents WHERE record_id=?', ('CS-KO-009',)).fetchone()[0].split()
        for question in ('요', '습니다', '조치 방법 알려주세요'):
            try:
                assert not search.retrieve(db, question), question
            except ValueError as error:
                assert '검색' in str(error)
    # 검색용 표현 확장으로 답변의 원문 인용 검사까지 느슨해지면 안 됩니다.
    assert search.concepts('물이 샙니다') == set()
    print(f'PASS: 한국어 현장 질문 {len(QUERIES)}개 × 원문·채택문 문서 검색, 어미 오탐·부정 원문 보존')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
