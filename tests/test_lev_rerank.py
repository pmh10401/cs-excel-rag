"""LEV 확률의 의미와 실패 시 원문·후보 순서 보존을 모델 없이 검사합니다."""
from copy import deepcopy

from experiments.gliner_followup import case_text, relevance
from experiments.gliner_translation import translated_pair
from experiments.lev_rerank import relevance_answer


def main():
    for probability in (0., .1, .5, .9, 1.):
        answer = relevance_answer({'related': probability, 'unrelated': 1-probability})
        assert abs(relevance(answer)-probability) < 1e-12
    for invalid in ({}, {'related': .9}, {'related': .9, 'unrelated': .9},
                    {'related': float('nan'), 'unrelated': .1},
                    {'related': True, 'unrelated': 0.}, {'related': -1., 'unrelated': 2.}):
        try:
            relevance_answer(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid probability accepted: {invalid}')
    hits = [{'id': str(i), 'evidence': [{'cells': {'현상': value}}]}
            for i, value in enumerate(('소음 없음', '소음 발생'))]
    before = deepcopy(hits)
    translations = {'소음': 'Noise', case_text(hits[0]): 'No noise', case_text(hits[1]): 'Noise occurs'}
    def predict(texts):
        assert texts == ['Query: Noise\nCase: No noise', 'Query: Noise\nCase: Noise occurs']
        return [relevance_answer({'related': p, 'unrelated': 1-p}) for p in (.1, .9)]
    selected, info = translated_pair('소음', hits, predict, translations)
    assert selected == list(reversed(hits)) and info['status'] == 'ok' and hits == before
    def failed(texts):
        return [relevance_answer({'related': float('nan'), 'unrelated': .1})]
    selected, info = translated_pair('소음', hits, failed, translations)
    assert selected == before and info['status'] == 'fallback'
    assert translated_pair('소음', hits, failed, {}, exact=True)[1]['status'] == 'skipped'
    print('PASS: LEV label probability, invalid distributions, original evidence and fallback order')


if __name__ == '__main__':
    main()
