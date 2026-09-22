"""Windows에서 start.bat 하나로 CS 입력·검토·질문을 실행하는 작은 콘솔 메뉴."""
import json
from pathlib import Path
import webbrowser
from .records import read_excel, load_header_aliases
from .cslist import load_schema


def main(root, *, config_path=None, output_dir=None):
    from .cli import main as run
    settings_path = (output_dir or root) / 'cslist_settings.json'
    settings = {'excel': 'cslist_sample.xlsx', 'sheet': 'CSList', 'reader': 'openpyxl'}
    if settings_path.exists():
        saved = json.loads(settings_path.read_text(encoding='utf-8-sig'))
        if not isinstance(saved, dict) or not isinstance(saved.get('excel'), str) or saved.get('sheet') is not None and not isinstance(saved['sheet'], str):
            raise ValueError('cslist_settings.json의 excel과 sheet 값을 확인하세요.')
        settings.update(saved)
    settings['sheet'] = 'CSList'
    if settings['reader'] not in ('openpyxl', 'xlwings'):
        raise ValueError('cslist_settings.json의 reader는 openpyxl 또는 xlwings여야 합니다.')
    schema_path = root / 'cslist_schema.json'
    output = (output_dir or root / 'cslist-output') / 'demo.html'

    def input_path(message):
        # 탐색기에서 복사한 따옴표 포함 경로도 받습니다. 셸 명령으로 실행하지 않습니다.
        value = input(message).strip().strip('"\'')
        if not value:
            raise ValueError('파일 경로를 입력하세요.')
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    def check(excel, sheet, reader):
        schema, _ = load_schema(schema_path)
        aliases = load_header_aliases(
            output.parent / 'column_aliases.json', schema['columns'], schema['id_column'], schema.get('name'))
        rows = read_excel(excel, sheet, required_columns=schema['columns'], id_column=schema['id_column'],
                          reader=reader, header_aliases=aliases)
        print(f"입력 검사 완료: {len(rows)}건 / {len(schema['columns'])}열 / {rows[0]['_source_sheet']}")
        return rows[0]['_source_sheet']

    while True:
        try:
            print(f"\nCS RAG · 현재 입력: {settings['excel']} / 시트: CSList / 읽기: {settings['reader']}")
            print('1 입력 파일 선택·DB 등록  →  2 DB 문장 다듬기  →  3 검토 저장  →  4 질문하기')
            print('5 결과 화면 열기    6 입력 검사(API 없음)    7 원문·채택문 검색 비교    0 종료')
            choice = input('번호: ').strip()
            if choice == '0':
                return
            excel = Path(settings['excel'])
            if not excel.is_absolute():
                excel = root / excel
            args = ['--schema', str(schema_path), '--excel', str(excel), '--reader', settings['reader']]
            common = ['--config', str(config_path)] if config_path else []
            if output_dir:
                common += ['--output', str(output_dir)]
            args += common
            if settings['sheet']:
                args += ['--sheet', settings['sheet']]
            if choice == '1':
                path = input_path('Excel 경로 (파일 경로를 붙여 넣으세요): ')
                choice_reader = input('읽기 방식: 1 Excel에서 열린 파일(xlwings) / 2 일반 파일(openpyxl) [기본 1]: ').strip() or '1'
                if choice_reader not in ('1', '2'):
                    raise ValueError('읽기 방식은 1 또는 2를 입력하세요.')
                reader = 'xlwings' if choice_reader == '1' else 'openpyxl'
                sheet = check(path, 'CSList', reader)
                run(root, ['--schema', str(schema_path), '--excel', str(path), '--reader', reader, '--import-only'] + common)
                settings = {'excel': str(path.resolve()), 'sheet': sheet, 'reader': reader}
                settings_path.parent.mkdir(parents=True, exist_ok=True)
                settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
            elif choice == '2':
                run(root, args + ['--saved', '--normalize-only'])
            elif choice == '3':
                path = input_path('다운로드한 검토 JSON 경로: ')
                run(root, ['--schema', str(schema_path), '--apply-reviews', str(path)] + common)
            elif choice == '4':
                question = input('질문: ').strip()
                if not question:
                    raise ValueError('질문을 입력하세요.')
                run(root, args + ['--saved', '--question', question])
            elif choice == '5':
                if not output.exists():
                    raise ValueError('아직 결과 화면이 없습니다. 2번 메뉴부터 실행하세요.')
            elif choice == '6':
                check(excel, settings['sheet'], settings['reader'])
            elif choice == '7':
                path = input_path('평가 질문 JSON 경로 (기본 샘플은 cslist_cases.json): ')
                run(root, args + ['--compare-search', '--cases', str(path)])
                webbrowser.open((output.parent / 'search_comparison.html').resolve().as_uri())
            else:
                print('0부터 7까지의 번호를 선택하세요.')
            if choice in ('2', '3', '4', '5'):
                webbrowser.open(output.resolve().as_uri())
        except (EOFError, KeyboardInterrupt):
            print('\n메뉴를 종료합니다.')
            return
        except Exception as error:
            print(f'처리하지 못했습니다: {error}')
