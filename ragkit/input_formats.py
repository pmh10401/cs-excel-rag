"""지원하는 두 CS 양식. 원본 열을 바꾸지 않고 행마다 처리 규칙을 선택합니다."""

HISTORY_SHEET = '19~26년 누적_260925'
LEGACY_HISTORY_SHEET = '19~26년 누적_2600831'
HISTORY_COLUMNS = [
    'CS NO', '제목', '불량발생일', '제번', '설비', '설비군 간소화', '공정', '공정간소화',
    '반입일', '반입-불량 일수', '반입년', '반입년월', '기인 모듈', '기인 Unit(부품)',
    '작업처', 'Defect 유형', 'Defect 상세', '라인', '유형', '현상', '원인', '조치',
    '인원', 'SITE', '발생년', '불량발생년월', '분기', '설비 타입',
]
HISTORY_RENAMED_COLUMNS = {'Site': 'SITE', '불량발생월': '불량발생년월'}
LEGACY_HISTORY_COLUMNS = [next((old for old, new in HISTORY_RENAMED_COLUMNS.items() if new == key), key)
                          for key in HISTORY_COLUMNS]
HISTORY_DETAIL_COLUMNS = ['기인 모듈', '기인 Unit(부품)', '작업처', 'Defect 유형', 'Defect 상세', '라인', '유형']
HISTORY_REWRITE = ['제목', '설비', '설비군 간소화', '공정', '공정간소화', '기인 모듈',
                   '기인 Unit(부품)', 'Defect 유형', 'Defect 상세', '유형', '현상', '원인', '조치', '설비 타입']
HISTORY_SCHEMA = {'name': '누적 CS 28열', 'id_column': 'CS NO', 'columns': HISTORY_COLUMNS,
                  'rewrite_columns': HISTORY_REWRITE, 'context_columns': HISTORY_COLUMNS}
HISTORY_SCHEMAS = {
    HISTORY_SHEET: HISTORY_SCHEMA,
    LEGACY_HISTORY_SHEET: HISTORY_SCHEMA | {'columns': LEGACY_HISTORY_COLUMNS, 'context_columns': LEGACY_HISTORY_COLUMNS},
}
HISTORY_RULES = {key: 'rewrite' if key in HISTORY_REWRITE else 'preserve' for key in HISTORY_COLUMNS}
LEGACY_HISTORY_RULES = {key: 'rewrite' if key in HISTORY_REWRITE else 'preserve' for key in LEGACY_HISTORY_COLUMNS}


def select_sheet(names, requested):
    # 기존 CSList 요청은 새 양식만 있는 파일도 자동 인식합니다. 두 시트가 있으면
    # 명시된 시트를 우선하며, 알 수 없는 시트를 추측해서 읽지 않습니다.
    if requested == 'CSList' and requested not in names:
        for sheet in HISTORY_SCHEMAS:
            if sheet in names:
                return sheet
    if requested is None:
        if len(names) != 1:
            raise ValueError(f'시트가 여러 개입니다. 시트 이름을 지정하세요: {names}')
        return names[0]
    if requested not in names:
        raise ValueError(f'시트를 찾을 수 없습니다: {requested}')
    return requested


def input_layout(sheet, required_columns, id_column):
    if sheet in HISTORY_SCHEMAS and id_column in ('CS 접수번호', 'CS NO'):
        return HISTORY_SCHEMAS[sheet]['columns'], 'CS NO'
    return required_columns, id_column


def is_history(row):
    return row.get('_id_column') == 'CS NO' and row.get('_source_sheet') in HISTORY_SCHEMAS


def row_schema(row, schema):
    return HISTORY_SCHEMAS[row['_source_sheet']] if is_history(row) else schema


def row_rules(row, rules):
    # 작업 단위로 좁힌 입력은 이미 선택된 규칙을 사용합니다.
    if is_history(row) and '_work_group' not in row:
        return LEGACY_HISTORY_RULES if row['_source_sheet'] == LEGACY_HISTORY_SHEET else HISTORY_RULES
    return rules


# 두 양식의 문서는 별도 ID로 저장하지만 집계·접수번호 검색에서는 같은 사건입니다.
# 원본 JSON에 대한 식이므로 별도의 중복 매핑 테이블을 유지할 필요가 없습니다.
CASE_NUMBER_SQL = '''COALESCE(json_extract(original_json, '$."CS 접수번호"'),
                              json_extract(original_json, '$."CS NO"'), id)'''
