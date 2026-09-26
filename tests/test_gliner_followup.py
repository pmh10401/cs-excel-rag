"""추가 GLiNER 실험의 점수·실패 복구·출처 보존 검사. 모델 호출 없음."""
from copy import deepcopy
from experiments.gliner_followup import relevance, pair_rerank, boost_tags


def main():
    def result(label, value):
        return {'relevance': {'label': label, 'confidence': value}}
    assert relevance(result('related', .9)) == .9
    assert abs(relevance(result('unrelated', .9)) - .1) < 1e-12
    for value in (True, float('nan'), -1, 1.1):
        try: relevance(result('related', value))
        except ValueError: pass
        else: raise AssertionError('비정상 신뢰도를 허용했습니다.')
    hits = [{'id': str(i), 'evidence': [{'cells': {'현상': text, '작업자': 'PRIVATE',
             '조치내용(1)': 'ANSWER_SECRET'}}]} for i, text in enumerate(('소음', '누수'))]
    before = deepcopy(hits)
    def predict(texts):
        assert all('Query:' in text and 'Case:' in text for text in texts)
        assert 'PRIVATE' not in str(texts) and 'ANSWER_SECRET' not in str(texts)
        return [result('unrelated', .9), result('related', .9)]
    ranked, info = pair_rerank('누수', hits, predict)
    assert ranked[0]['id'] == '1' and hits == before and info['status'] == 'ok'
    assert set(h['id'] for h in ranked) == {'0', '1'}
    def failure(_): raise RuntimeError('capacity')
    for predict_bad in (failure, lambda _: [], lambda _: [result('unknown', .9)] * 2):
        ranked, info = pair_rerank('누수', hits, predict_bad)
        assert ranked == hits and info['status'] == 'fallback'
    assert pair_rerank('ID-001', hits, failure, exact=True)[1]['status'] == 'skipped'
    assert pair_rerank('누수', [], failure)[1]['status'] == 'skipped'
    assert pair_rerank('누수', hits, lambda _: [result('related', .9)] * 2)[0] == hits
    assert boost_tags(hits, [['leak'], [], ['leak']], .02)[0]['id'] == '1'
    assert boost_tags(hits, [[], [], ['leak']], .02) == hits
    print('PASS: 관련성 점수, 부적합 응답 복구, 순위·후보·원문 보존, 비공개 필드 제외, 태그 재평가')


if __name__ == '__main__':
    main()
