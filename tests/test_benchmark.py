"""실제 모델이 두 근거를 한 대괄호에 묶었을 때 생긴 평가 오류를 재현합니다."""
from benchmark_cslist import check_citations


def main():
    # 쉼표로 두 실제 출처를 묶는 것은 허위 인용이 아닙니다.
    allowed = {'CS-2026-00020/접수','CS-2026-00020/작업1'}
    citations, valid = check_citations('금액은 0입니다 [CS-2026-00020/접수, CS-2026-00020/작업1].',allowed)
    assert valid and citations == ['CS-2026-00020/접수','CS-2026-00020/작업1']
    assert check_citations('금액은 0입니다 [CS-2026-00020/접수][CS-2026-00020/작업1].',allowed)[1]
    assert not check_citations('완료입니다 [CS-2026-99999/작업1].',allowed)[1]
    assert not check_citations('완료입니다 [CS-2026-00020/접수, 가짜근거].',allowed)[1]
    assert not check_citations('금액은 0입니다.',allowed)[1]
    assert check_citations('해당 접수 자료가 없어 확인할 수 없습니다.',set())[1]
    assert not check_citations('완료입니다 [CS-2026-00020/접수].',set())[1]
    print('PASS: 단일·복수 인용, 가짜·누락 인용, 근거 없는 질문의 인용 검사')


if __name__=='__main__':
    main()
