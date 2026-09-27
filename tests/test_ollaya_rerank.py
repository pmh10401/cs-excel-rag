"""Ollaya 응답 검사와 장애 시 원문·검색 순서 보존을 서버 없이 확인합니다."""
from copy import deepcopy

from experiments.gliner_followup import case_text, relevance
from experiments.gliner_translation import translated_pair
from experiments.ollaya_rerank import checked_answer


def main():
    response = dict(model='laya:en', state_truncated=False, done_reason='decide',
        usage={'output_tokens': 0}, answers={'relevance': dict(type='choice', choice='related',
        confidence=.6, probabilities={'related': .8, 'unrelated': .2})})
    # confidence는 선택지 간 차이일 수 있습니다. 관련 확률 자체를 사용해야 합니다.
    assert relevance(checked_answer(response, 'laya:en')) == .8
    rounded = deepcopy(response)
    rounded['answers']['relevance']['probabilities'] = {'related': .8, 'unrelated': .1999}
    assert abs(relevance(checked_answer(rounded, 'laya:en')) - .8/.9999) < 1e-12
    invalid = [response | {'model': 'another'}, response | {'state_truncated': True},
               response | {'done_reason': 'load'}, response | {'usage': {'output_tokens': 1}},
               response | {'usage': {'output_tokens': False}}, response | {'answers': {}}]
    for probabilities in ({'related': .8}, {'related': .8, 'unrelated': .8},
                          {'related': float('nan'), 'unrelated': .2},
                          {'related': True, 'unrelated': 0}, {'related': -.1, 'unrelated': 1.1}):
        broken = deepcopy(response)
        broken['answers']['relevance']['probabilities'] = probabilities
        invalid.append(broken)
    wrong_choice = deepcopy(response)
    wrong_choice['answers']['relevance']['choice'] = 'unrelated'
    invalid.append(wrong_choice)
    hits = [{'id': str(i), 'evidence': [{'cells': {'현상': text}}]}
            for i, text in enumerate(('소음 없음', '소음 발생'))]
    original = deepcopy(hits)
    translations = {'소음': 'Noise', case_text(hits[0]): 'No noise', case_text(hits[1]): 'Noise occurs'}
    for broken in invalid:
        def predict(texts):
            return [checked_answer(broken, 'laya:en') for _ in texts]
        selected, info = translated_pair('소음', hits, predict, translations)
        assert info['status'] == 'fallback' and info['error'] and selected == original and hits == original
    def timed_out(texts):
        raise TimeoutError('Ollaya timed out')
    selected, info = translated_pair('소음', hits, timed_out, translations)
    assert info['status'] == 'fallback' and selected == original and hits == original
    assert translated_pair('소음', hits, timed_out, {}, exact=True)[1]['status'] == 'skipped'
    print('PASS: Ollaya probabilities, truncation, malformed answers, timeout and original evidence')


if __name__ == '__main__':
    main()
