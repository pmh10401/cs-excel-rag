"""Excel 입력, 원본 보존, 검색용 문자열과 원문 버전 계산."""
import hashlib
import json
from datetime import date, time as daytime, timedelta
from pathlib import Path
from types import SimpleNamespace
from openpyxl.utils.cell import get_column_letter
from openpyxl import load_workbook
from .input_formats import select_sheet, input_layout, is_history

def cell_string(value):
    # 검색과 API용 문자열입니다. Excel의 자료형과 원래 값은 _typed_cells에 별도로 보존합니다.
    if value is None:
        return ''
    return value.isoformat() if isinstance(value, (date, daytime)) else str(value)


def public_cells(row):
    return {key: value for key, value in row.items() if not key.startswith('_')}


def record_id(row, id_column=None):
    # 내부 식별자는 원본 열에서 읽습니다. Excel에 id라는 가짜 열을 추가하지 않습니다.
    if not isinstance(row, dict):
        raise ValueError('기록은 열 이름과 값으로 구성되어야 합니다.')
    key = id_column or row.get('_id_column', 'id')
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'비어 있지 않은 문자열 {key}가 필요합니다.')
    # 같은 CS 번호의 다른 양식을 덮어쓰지 않습니다. 셀의 CS NO 자체는 그대로 둡니다.
    return '누적28:' + value if is_history(row) else value


def row_revision(row):
    # 행 번호가 바뀌어도 검토 결과를 유지하되 같은 행의 문맥·값이 바뀌면 재검토합니다.
    values = [public_cells(row), row.get('_typed_cells', {})]
    if row.get('_id_column', 'id') != 'id':
        values.append(row['_id_column'])
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_excel(path, sheet_name='점검기록', *, required_columns=None, allow_typed=True, id_column='id', reader='openpyxl', header_aliases=None, progress=None):
    # 변환 파일도 동일한 원문·자료형·셀 위치를 반환하여 후속 처리를 공유합니다.
    # 첫 행은 열 이름, 두 번째 행부터는 데이터입니다. 열 순서는 바꿔도 됩니다.
    # 숫자·날짜·참/거짓 셀은 값과 자료형 정보를 별도로 보존하며 LLM 수정 대상에서 제외합니다.
    path = Path(path)
    if path.name.lower().endswith('.rag.jsonl.gz'):
        from .source_cache import read_cache
        return read_cache(path, sheet_name, required_columns=required_columns,
                          allow_typed=allow_typed, id_column=id_column)
    if path.suffix.lower() != '.xlsx':
        raise ValueError('이 데모는 .xlsx 파일을 읽습니다.')
    if reader == 'xlwings':
        from .excel_app import read_open_book
        return read_open_book(path, sheet_name, required_columns, allow_typed, id_column, header_aliases, progress=progress)
    if reader != 'openpyxl':
        raise ValueError('읽기 방식은 openpyxl 또는 xlwings여야 합니다.')
    # data_only=False로 수식 자체를 확인합니다. 계산되지 않은 수식을 빈 값으로
    # 오해하지 않도록 수식 셀을 발견하면 아래에서 중단합니다. 파일을 수정하지 않습니다.
    book = load_workbook(path, data_only=False)
    try:
        sheet_name = select_sheet(book.sheetnames, sheet_name)
        sheet = book[sheet_name]
        if sheet.merged_cells.ranges:
            raise ValueError('입력 시트의 병합 셀을 해제하고 각 행에 값을 입력하세요.')
        return rows_from_cells(path, sheet_name, sheet.iter_rows(), required_columns, allow_typed, id_column, header_aliases)
    finally:
        # 성공하거나 오류가 발생해도 파일 핸들을 닫아 Excel 파일을 계속 편집할 수 있게 합니다.
        book.close()


class HeaderMappingError(ValueError):
    def __init__(self, message, *, missing, unknown, observed, id_column=None):
        super().__init__(message)
        self.missing = list(missing)
        self.unknown = list(unknown)
        self.observed = list(observed)
        self.id_column = id_column


# CSList 내보내기에서 확인된 표기만 내부 열 이름으로 맞춥니다. 다른 표기는 바꾸지 않습니다.
CSLIST_HEADER_ALIASES = {
    'Part Desc(1)': 'Part Decs(1)',
    'Part Desc(2)': 'Part Decs(2)',
    'UNIT(CIS)': 'UNIT(CS)',
    'Line': 'line',
    '공수1(1)': '공수(1)',
    '공수1(2)': '공수(2)',
    '도급여부 CASE': '도급여부CASE',
    '무상자재개수': '무상자개개수',
    '불출위치(1)': '분출위치(1)',
    '불출위치(2)': '분출위치(2)',
    '요청 납기일': '요청납기일',
    'CS조치결재상태(조치1/조치2)': '조치결재상태(조치1/조치2)',
    '협력시통합정산CS접수번호': '협력사통합전산CS접수번호',
    '협력사통합정산CS접수번호': '협력사통합전산CS접수번호',
}

MAX_HEADER_ALIASES = 400
MAX_HEADER_NAME = 200
ALIASES_FORMAT = 'column-aliases-v1'


def validate_header_aliases(aliases, required_columns, id_column):
    if not isinstance(aliases, dict):
        raise ValueError('열 이름 대응은 객체여야 합니다.')
    if required_columns is None:
        raise ValueError('양식 열이 있어야 열 이름을 대응할 수 있습니다.')
    required_set = set(required_columns)
    if id_column is not None and required_set and id_column not in required_set:
        raise ValueError('식별 열이 양식에 없습니다.')
    if len(aliases) > MAX_HEADER_ALIASES:
        raise ValueError('열 이름 대응이 너무 많습니다.')
    cleaned = {}
    for source, target in aliases.items():
        if not isinstance(source, str) or not source.strip() or source.strip().startswith('_'):
            raise ValueError('대응할 원본 열 이름이 올바르지 않습니다.')
        source = source.strip()
        if len(source) > MAX_HEADER_NAME:
            raise ValueError('대응할 원본 열 이름이 너무 깁니다.')
        if source in required_set:
            raise ValueError(f'양식 열 이름은 다른 열로 바꾸지 않습니다: {source}')
        if source in CSLIST_HEADER_ALIASES:
            raise ValueError(f'이미 정해진 열 이름 대응은 바꾸지 않습니다: {source}')
        if source in cleaned:
            raise ValueError(f'공백을 제외하면 중복된 원본 열입니다: {source}')
        if not isinstance(target, str) or not target.strip() or target.strip().startswith('_'):
            raise ValueError('맞출 양식 열 이름이 올바르지 않습니다.')
        target = target.strip()
        if len(target) > MAX_HEADER_NAME:
            raise ValueError('맞출 양식 열 이름이 너무 깁니다.')
        if target not in required_set:
            raise ValueError(f'맞출 양식 열이 없습니다: {target}')
        cleaned[source] = target
    return cleaned


def load_header_aliases(path, required_columns, id_column, schema_name=None):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except OSError as error:
        raise ValueError('열 이름 대응 파일을 읽지 못했습니다.') from error
    except ValueError as error:
        raise ValueError('열 이름 대응 파일이 올바른 JSON이 아닙니다.') from error
    if not isinstance(data, dict) or data.get('format') != ALIASES_FORMAT:
        raise ValueError('열 이름 대응 파일 형식을 확인하세요.')
    if data.get('schema_name') != (schema_name or 'CSList'):
        raise ValueError('저장된 열 이름 대응이 현재 양식과 다릅니다.')
    if data.get('id_column') != id_column or data.get('columns') != list(required_columns):
        raise ValueError('저장된 열 이름 대응이 현재 양식과 다릅니다.')
    if 'aliases' not in data:
        raise ValueError('열 이름 대응 파일 형식을 확인하세요.')
    return validate_header_aliases(data['aliases'], required_columns, id_column)


def save_header_aliases(path, aliases, required_columns, id_column, schema_name=None):
    validated = validate_header_aliases(aliases, required_columns, id_column)
    path = Path(path)
    payload = {
        'format': ALIASES_FORMAT,
        'schema_name': schema_name or 'CSList',
        'id_column': id_column,
        'columns': list(required_columns),
        'aliases': validated,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return validated


def _map_cslist_headers(headers, required_columns, id_column, header_aliases=None):
    if id_column != 'CS 접수번호' or required_columns is None:
        return headers, None
    required = set(required_columns)
    custom = header_aliases or {}
    mapped, source_columns = [], {}
    for key in headers:
        if key in required:
            mapped.append(key)
            continue
        canonical = CSLIST_HEADER_ALIASES.get(key)
        if canonical is None and key in custom:
            canonical = custom.get(key)
        if canonical is not None and canonical in required:
            mapped.append(canonical)
            source_columns[canonical] = key
        else:
            mapped.append(key)
    return mapped, source_columns or None


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_headers(headers, location):
    problems, positions = [], {}
    for column, name in enumerate(headers, 1):
        address = f'{get_column_letter(column)}1'
        if not isinstance(name, str) or not name:
            problems.append(f'{address}: 열 이름이 비어 있거나 문자가 아닙니다. 이 열의 제목을 입력하세요.')
        elif name.startswith('_'):
            problems.append(f'{address}: {name!r}의 _ 접두사는 예약되어 있습니다. 열 이름을 변경하세요.')
        else:
            positions.setdefault(name, []).append(address)
    problems.extend(f'{", ".join(cells)}: 중복 열 이름 {name!r}. 각 열의 제목을 구분하세요.'
                    for name, cells in positions.items() if len(cells) > 1)
    if not headers:
        problems.append('A1: 첫 행에 열 이름이 없습니다. 실제 제목을 1행에 입력하세요.')
    if problems:
        detail = '\n'.join(problems[:10])
        if len(problems) > 10:
            detail += f'\n그 외 {len(problems)-10}개 오류가 더 있습니다.'
        raise ValueError(f'{location} · 첫 행의 열 이름 오류\n{detail}')


def rows_from_cells(path, sheet_name, cells, required_columns, allow_typed, id_column, header_aliases=None):
    return list(iter_rows_from_cells(path, sheet_name, cells, required_columns, allow_typed, id_column, header_aliases))


def iter_rows_from_cells(path, sheet_name, cells, required_columns, allow_typed, id_column, header_aliases=None):
    # 파일 직접 읽기와 Excel 연결 읽기에 같은 검사를 적용합니다. 열·ID·빈칸 규칙이 달라지지 않습니다.
    required_columns, id_column = input_layout(sheet_name, required_columns, id_column)
    if id_column == 'CS NO':
        header_aliases = None  # CSList용 대응표를 다른 양식에 적용하지 않습니다.
    cells = iter(cells)
    header_cells = next(cells, ())
    headers = [cell.value.strip() if isinstance(cell.value, str) else cell.value
               for cell in header_cells]
    # 뒤쪽의 서식뿐인 빈 열은 제외합니다. 아래 행에 실제 값이 있으면 끝까지 검사해 거절합니다.
    while headers and _blank(headers[-1]):
        headers.pop()
    location = f'{Path(path).name} / {sheet_name}'
    _validate_headers(headers, location)
    if required_columns is None and headers.count(id_column) != 1:
        raise ValueError(f'{location} · 첫 행에 식별 열 {id_column!r}이 없습니다. 열 이름을 확인하세요.')
    observed = list(headers)
    aliases = None
    if header_aliases is not None:
        aliases = validate_header_aliases(header_aliases, required_columns, id_column)
    headers, source_columns = _map_cslist_headers(headers, required_columns, id_column, aliases)
    _validate_headers(headers, location)
    if required_columns is not None and set(headers) != set(required_columns):
        missing = sorted(set(required_columns) - set(headers))
        extra = sorted(set(headers) - set(required_columns))
        extra_locations = [f'{get_column_letter(index+1)}1: {name!r}'
                           for index, name in enumerate(headers) if name in extra]
        raise HeaderMappingError(
            f'{location} · 첫 행의 열이 양식과 다릅니다.\n누락: {missing}\n추가: {extra_locations}\n'
            '표시된 셀의 열 이름을 양식과 비교하세요.',
            missing=missing, unknown=extra, observed=observed, id_column=id_column)
    if headers.count(id_column) != 1:
        raise ValueError(f'{location} · 첫 행에 식별 열 {id_column!r}이 없습니다. 열 이름을 확인하세요.')
    seen = set()
    source_key = hashlib.sha256((str(path.resolve()) + '!' + sheet_name).encode()).hexdigest()
    for row_number, cell_row in enumerate(cells, start=2):
        # 읽기 전용 Excel의 빈 셀에도 원래 좌표를 붙입니다. 마지막 빈 열도 보존합니다.
        cell_row = list(cell_row)
        if len(cell_row) > len(headers):
            for column, cell in enumerate(cell_row[len(headers):], len(headers)+1):
                if not _blank(cell.value) or cell.data_type in ('f', 'e'):
                    letter = get_column_letter(column)
                    raise ValueError(f'{location}!{letter}{row_number}: 값이 있지만 {letter}1에 열 이름이 없습니다. '
                                     '제목을 입력하고 양식과 맞는지 확인하세요. 이 열의 값은 임의로 제외하지 않습니다.')
            cell_row = cell_row[:len(headers)]
        for index in range(len(headers)):
            if index >= len(cell_row):
                cell_row.append(SimpleNamespace(value=None, data_type='n', number_format='General'))
            if not hasattr(cell_row[index], 'coordinate'):
                cell = cell_row[index]
                cell_row[index] = SimpleNamespace(value=cell.value, data_type=cell.data_type,
                    number_format=cell.number_format, coordinate=f'{get_column_letter(index+1)}{row_number}')
        values = [cell.value for cell in cell_row]
        # 완전히 빈 행은 건너뛰되, 다른 값이 있는데 ID만 빠진 행은 오류로 처리합니다.
        for cell in cell_row:
            if cell.data_type == 'f':
                raise ValueError(f'{location}!{cell.coordinate}: 수식 셀입니다. 계산값을 읽으려면 Windows Excel에서 '
                                 '파일을 열고 계산을 마친 뒤 “Windows Excel에서 열린 파일”을 선택하세요.')
            if cell.data_type == 'e':
                raise ValueError(f'{location}!{cell.coordinate}: Excel 오류 값 {cell.value!r}입니다. 이 셀의 오류를 수정하세요.')
        if all(_blank(value) for value in values):
            continue

        if not allow_typed and any(v is not None and not isinstance(v, str) for v in values):
            raise ValueError('이 로컬 데모에서는 문자 값을 입력하세요.')
        # 빈 Excel 셀(None)은 빈 문자열로 바꿉니다. 실제 본문의 공백은 보존합니다.
        row = dict(zip(headers, (cell_string(value) for value in values)))
        row['_id_column'] = id_column
        row['_source_sheet'] = sheet_name
        if source_columns:
            row['_source_columns'] = source_columns
        row['_typed_cells'] = {key: {'kind': type(cell.value).__name__,
                              'value': cell_string(cell.value) if isinstance(cell.value, (date, daytime, timedelta)) else cell.value,
                              'number_format': cell.number_format}
                              for key, cell in zip(headers, cell_row) if cell.value is not None and not isinstance(cell.value, str)}
        row['_source_key'] = source_key
        try:
            text_for(row)
            identity = record_id(row)
        except ValueError as error:
            raise ValueError(f'{location} {row_number}행: {error}') from error
        if identity in seen:
            raise ValueError(f'{location}!{cell_row[headers.index(id_column)].coordinate}: 중복 ID ({id_column}) {identity}. '
                             '같은 접수번호의 행을 확인하세요.')
        seen.add(identity)
        # 사용자 작성 '출처'는 그대로 두고, 실제 파일·시트·행 위치를 별도로 기록합니다.
        row['_excel_location'] = f'{path.name} / {sheet_name}!{cell_row[0].coordinate}:{cell_row[-1].coordinate}'
        row['_cell_locations'] = {key: f'{path.name} / {sheet_name}!{cell.coordinate}'
                                  for key, cell in zip(headers, cell_row)}
        yield row
    if not seen:
        raise ValueError('Excel에 읽을 데이터 행이 없습니다.')


def text_for(row, id_column=None):
    # 원본 열 이름과 값을 그대로 보존합니다. 작성 내용의 정리는 normalize()에서 수행합니다.
    identity = record_id(row, id_column)
    id_column = id_column or row.get('_id_column', 'id')
    content = [(key, value) for key, value in row.items() if key != id_column and not key.startswith('_')]
    if not content or any(not isinstance(value, str) for _, value in content):
        raise ValueError('내용 열은 문자열이어야 합니다.')
    if not any(value.strip() for _, value in content):
        raise ValueError(f'{identity}: 내용이 비어 있습니다.')
    # CS의 빈 항목은 원본 JSON에 남깁니다. 검색 문서에 빈 열 이름 128개를 반복하지 않습니다.
    text = f"기록 ID: {identity}\n" + '\n'.join(f'{key}: {value}' for key, value in content if id_column == 'id' or value.strip())
    if row.get('_excel_location'):
        text += f"\nExcel 위치: {row['_excel_location']}"
    return text
