"""후보 집합 확률의 검증과 중복 문서·원문·실패 순서 보존을 실제 모델 없이 확인합니다."""
from copy import deepcopy

from experiments.clm_rerank import native_rerank
from experiments.gliner_followup import case_text


def main():
    hits = [dict(id=str(i), evidence=[{'cells': {'현상': '소음'}}]) for i in range(3)]
    before = deepcopy(hits)
    translations = {'질문': 'Noise?', case_text(hits[0]): 'Noise occurs'}
    def answer(query, questions):
        assert query == 'Noise?'
        assert questions['rank']['criteria'] == {str(i): 'Noise occurs' for i in range(3)}
        return {'answers': {'rank': {'probabilities': {'0': .2, '1': .4, '2': .4}}}}
    selected, info = native_rerank('질문', hits, answer, translations)
    assert [h['id'] for h in selected] == ['1', '2', '0'] and info['status'] == 'ok'
    assert 'predicted_no_match' not in info and hits == before
    for invalid in ({}, {'0': 1.}, {'0': .5, '1': .5, '2': .5},
                    {'0': float('nan'), '1': .5, '2': .5}, {'0': True, '1': 0., '2': 0.}):
        result = lambda *args: {'answers': {'rank': {'probabilities': invalid}}}
        selected, info = native_rerank('질문', hits, result, translations)
        assert selected == before and info['status'] == 'fallback'
    assert native_rerank('질문', hits, answer, {})[1]['status'] == 'fallback'
    assert native_rerank('질문', hits, answer, {}, exact=True)[1]['status'] == 'skipped'
    assert native_rerank('질문', [], answer, {})[1]['status'] == 'skipped'
    print('PASS: CLM candidate probabilities, duplicate documents, stable ties, evidence, fallback and exact bypass')


if __name__ == '__main__':
    main()
