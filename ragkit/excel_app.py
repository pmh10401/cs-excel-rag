"""Windows Excel에서 이미 열린 통합문서를 xlwings로 읽습니다. 파일은 저장하거나 닫지 않습니다."""
import os
import sys
from types import SimpleNamespace
from openpyxl.styles.numbers import is_date_format, is_timedelta_format
from openpyxl.utils.cell import get_column_letter
from openpyxl.utils.datetime import from_excel, CALENDAR_MAC_1904, CALENDAR_WINDOWS_1900
from .records import rows_from_cells


def read_open_book(path, sheet_name, required_columns, allow_typed, id_column, header_aliases=None, *, progress=None):
    def report(stage, processed_rows=0, total_rows=None):
        if progress:
            progress({'phase': 'reading', 'stage': stage,
                      'processed_rows': processed_rows, 'total_rows': total_rows})
    if sys.platform != 'win32':
        raise RuntimeError('xlwings 입력은 Windows 데스크톱 Excel용입니다. 일반 파일은 openpyxl을 선택하세요.')
    try:
        report('열린 Excel에 연결 중')
        import xlwings as xw
    except ImportError as error:
        raise RuntimeError('xlwings가 필요합니다. setup.bat을 다시 실행해 패키지를 설치하세요.') from error
    try:
        # xw.Book(path)는 파일을 새로 열 수도 있으므로 쓰지 않습니다.
        # 사용자가 DRM 인증을 마치고 연 통합문서를 전체 경로로만 찾습니다.
        identity = os.path.normcase(os.path.abspath(path))
        matches = [book for app in xw.apps for book in app.books
                   if os.path.normcase(os.path.abspath(book.fullname)) == identity]
        if not matches:
            raise ValueError('해당 파일을 Windows Excel에서 먼저 열고 DRM 인증을 마친 뒤 다시 실행하세요. 파일 전체 경로도 확인하세요.')
        if len(matches) != 1:
            raise ValueError('같은 파일이 여러 Excel에 열려 있습니다. 사용할 통합문서 하나만 남겨 주세요.')
        book = matches[0]
        names = [sheet.name for sheet in book.sheets]
        if sheet_name is None:
            if len(names) != 1:
                raise ValueError(f'시트가 여러 개입니다. 시트 이름을 지정하세요: {names}')
            sheet_name = names[0]
        if sheet_name not in names:
            raise ValueError(f'시트를 찾을 수 없습니다: {sheet_name}')
        sheet = next(sheet for sheet in book.sheets if sheet.name == sheet_name)
        # expand()는 중간 빈 행에서 끊길 수 있습니다. 서식만 있는 먼 셀도 제외하도록
        # 실제 값/수식이 있는 마지막 행·열을 각각 찾습니다. 검색 조건은 매번 명시합니다.
        report('Excel 데이터 범위 확인 중')
        find = dict(What='*', LookIn=-4123, LookAt=2, SearchDirection=2,
                    MatchCase=False, MatchByte=False, SearchFormat=False)
        last_row = sheet.api.Cells.Find(SearchOrder=1, **find)
        last_column = sheet.api.Cells.Find(SearchOrder=2, **find)
        if last_row is None or last_column is None:
            raise ValueError('Excel에 읽을 데이터 행이 없습니다.')
        area = sheet.range((1, 1), (last_row.Row, last_column.Column))
        total_rows = max(0, last_row.Row - 1)
        report('Excel 병합 셀·수식 확인 중', total_rows=total_rows)
        # COM의 혼합 범위는 None(Null)이 될 수 있습니다. 명확한 False만 통과시킵니다.
        if area.api.MergeCells is not False:
            raise ValueError('입력 시트의 병합 셀을 해제하고 각 행에 값을 입력하세요.')
        if area.api.HasFormula is not False:
            raise ValueError('입력 시트에 수식이 있습니다. 확정된 값으로 준비하세요.')
        # 기본 옵션은 Excel 오류를 None으로 바꾸므로 오류 문자열을 반드시 남깁니다.
        report('Excel 값 읽는 중', total_rows=total_rows)
        values = area.options(ndim=2, err_to_str=True, chunksize=500).value
        epoch = CALENDAR_MAC_1904 if book.api.Date1904 else CALENDAR_WINDOWS_1900
        report('Excel 행·자료형 확인 중', total_rows=total_rows)

        def cells():
            for r, line in enumerate(values, 1):
                row = []
                for c, value in enumerate(line, 1):
                    coordinate = f'{get_column_letter(c)}{r}'
                    fmt = 'General'
                    if isinstance(value, str) and value.startswith('#'):
                        # 실제 오류와 '#N/A'라는 일반 문자를 구분합니다.
                        if not isinstance(sheet.range((r, c)).api.Value2, str):
                            raise ValueError(f'{sheet_name}!{coordinate}: Excel 오류 셀입니다.')
                    elif value is not None and not isinstance(value, str):
                        # ponytail: 숫자·날짜의 형식만 개별 조회; 대량 입력에서 느리면 열별 형식을 일괄 조회합니다.
                        cell = sheet.range((r, c))
                        fmt = cell.number_format
                        # Value2는 통화 형식에 따른 소수점 반올림을 피합니다.
                        value = cell.api.Value2
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if is_date_format(fmt):
                                value = from_excel(value, epoch, timedelta=is_timedelta_format(fmt))
                            elif isinstance(value, float) and value.is_integer():
                                value = int(value)
                    row.append(SimpleNamespace(value=value, number_format=fmt,
                                               coordinate=coordinate, data_type='s'))
                yield row
                report('Excel 행·자료형 확인 중', max(0, r - 1), total_rows)
        return rows_from_cells(path, sheet_name, cells(), required_columns, allow_typed, id_column, header_aliases)
    except ValueError:
        raise
    except Exception as error:
        # DRM이 자동화 읽기를 막으면 실패를 알립니다. 다른 읽기 방식으로 우회하지 않습니다.
        raise RuntimeError(f'Excel에서 읽지 못했습니다. Excel의 대화상자·접근 권한·DRM 자동화 허용 여부를 확인하세요. 상세: {error}') from error
