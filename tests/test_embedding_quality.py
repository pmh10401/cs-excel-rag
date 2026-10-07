"""숫자 조건·근거 없음 진단과 정답 없는 질문의 집계 분모를 확인합니다."""
from experiments.embedding_quality import MODELS, METHODS, condition_result, summarize
from experiments.gliner_decide import metrics


def main():
    assert condition_result([.6, .8], 1, .3)['correct'] is True
    assert condition_result([.1, .2], None, .3)['rejected'] is True
    assert condition_result([.8, .9], None, .3)['correct'] is None
    assert condition_result([.1, .2], 1, .3)['rejected'] is True  # 순위가 맞아도 잘못 거절할 수 있습니다.
    for values in ([.1], [float('nan'), .1], [float('inf'), .1]):
        try:
            condition_result(values, 0, .3)
            raise AssertionError('Invalid scores accepted')
        except ValueError:
            pass
    cases = [dict(id='found', question='flow', expected_ids=['correct']),
             dict(id='absent', question='flow 11.8', expected_ids=[])]
    hit = dict(id='correct', evidence=[{'cells': {'현상': 'flow 1.18'}}])
    rows = [dict(case=case, methods={method: dict(status='ok', seconds=1,
                metrics=metrics(case, [hit], {})) for method in METHODS}) for case in cases]
    conditions = [dict(language='ko', **{model: condition_result([.6, .8], expected, .3)
                                       for model in MODELS}) for expected in (1, None)]
    result = summarize(rows, conditions)
    for method in METHODS:
        assert result[method]['top1'] == result[method]['positive_count'] == 1
        assert result[method]['negative_count'] == 1 and result[method]['correct_rejections'] == 0
        assert result[method]['precision_at_5'] == .5
    for model in MODELS:
        assert result[model + '_conditions']['top1'] == 1
        assert result[model + '_conditions']['correct_rejections'] == 0
    print('embedding_quality check passed')


if __name__ == '__main__':
    main()
