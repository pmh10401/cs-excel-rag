"""검색 재정렬 실험의 누락·원문 유출·동점·완료 검사를 모델 없이 확인합니다."""
from copy import deepcopy

from experiments.search_rerank import CAPACITY, checked_tokens, compare, sorted_indices, verify


def main():
    hits = [dict(id=str(i), score=99-i, expected='secret', evidence=[{'cells': {
        '현상': text, '조치': 'Do not send action', 'CS NO': 'Do not send CS', '작업자': 'Private'}}])
            for i, text in enumerate(('flow is 72 L/min', 'flow is 7.2 L/min', 'flow is 7.2 L/min'))]
    row = dict(case={'id': 'test', 'question': 'flow is 7.2 L/min', 'expected_ids': ['1']},
               hits=hits, exact=False, section='development')
    original = deepcopy(row)
    def predict(pairs):
        assert pairs == [(row['case']['question'], f'현상: {text}')
                         for text in ('flow is 72 L/min', 'flow is 7.2 L/min', 'flow is 7.2 L/min')]
        return [-2., 40., 40.], [10, 10, 10]
    result = compare(row, {}, predict)
    assert result['methods']['rerank']['ids'] == ['1', '2', '0']  # 확률로 반올림하지 않고 동점 순서를 보존
    assert row == original
    report = dict(development=[result], challenge=[])
    assert verify(report, [row], {}) == dict(processed=1, missing=0, errors=0, skipped=0)
    damaged = deepcopy(report)
    damaged['development'][0]['methods']['rerank']['ids'] = ['2', '1', '0']
    try:
        verify(damaged, [row], {})
        raise AssertionError('Corrupted ranking accepted')
    except ValueError:
        pass
    for values in ([1], [float('nan')]*3, [True]*3, [float('inf')]*3):
        failed = compare(row, {}, lambda _: (values, [10]*3))
        assert failed['pair']['status'] == 'fallback' and failed['methods']['baseline'] == failed['methods']['rerank']
    for ids in ([], [[]], [[1]*(CAPACITY+1)]):
        try:
            checked_tokens(ids)
            raise AssertionError('Invalid length accepted')
        except ValueError:
            pass
    for exact, selected in ((True, hits), (False, [])):
        bypass = row | {'exact': exact, 'hits': selected}
        skipped = compare(bypass, {}, lambda _: (_ for _ in ()).throw(AssertionError('Model called on bypass')))
        assert skipped['pair']['status'] == 'skipped'
    assert sorted_indices([40., 41.], 2) == [1, 0]
    print('search_rerank check passed')


if __name__ == '__main__':
    main()
