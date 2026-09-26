"""GLiNER 실험의 순위 보존·안전한 실패·정답 라벨 비노출 검사. 모델 다운로드 없음."""
from copy import deepcopy
from experiments.gliner_decide import rerank, metrics


def main():
    hits = [{'id': str(i), 'evidence': [{'cells': {'제목': '사례', '현상': text,
             '작업자': 'PRIVATE', '조치내용(1)': 'LABEL_SECRET'}}]}
            for i, text in enumerate(('진동', '누수', '소음'))]
    original = deepcopy(hits)
    def result(label, confidence=.9):
        return {'symptom': [{'label': label, 'confidence': confidence}]}
    def predict(texts):
        assert len(texts) == 4 and texts[0] == '누수 사례'
        assert 'PRIVATE' not in str(texts) and 'LABEL_SECRET' not in str(texts)
        return [result(x) for x in ('leak', 'vibration', 'leak', 'noise')]
    ranked, info = rerank('누수 사례', hits, predict)
    assert ranked[0]['id'] == '1' and info['status'] == 'ok'
    assert sorted(h['id'] for h in ranked) == ['0', '1', '2'] and hits == original
    for bad in (float('nan'), -1, 1.1, True):
        ranked, info = rerank('누수 사례', hits, lambda _: [result('leak', bad)] * 4)
        assert ranked == hits and info['status'] == 'fallback'
    for predictions in ([result('unknown')] * 4, [result('leak')], [{'bad': []}] * 4):
        ranked, info = rerank('누수 사례', hits, lambda _: predictions)
        assert ranked == hits and info['status'] == 'fallback'
    def failure(_):
        raise ValueError('capacity')
    assert rerank('누수 사례', hits, failure)[1]['status'] == 'fallback'
    assert rerank('누수 사례', hits, lambda _: [result('leak', .2)] * 4)[0] == hits
    assert rerank('누수 사례', hits, lambda _: [result('leak')] * 4)[0] == hits
    for exact, candidates in ((True, hits), (False, [])):
        assert rerank('CS-001', candidates, failure, exact=exact)[1]['status'] == 'skipped'
    assert metrics({'expected_ids': ['1']}, ranked, {})['precision_at_5'] == 1/3
    assert metrics({'expected_ids': []}, [], {})['unexpected_hits'] is False
    print('PASS: GLiNER 실험 순위·후보·원문 보존, 비정상 응답/실패 복구, 정확 검색, 평가 지표')


if __name__ == '__main__':
    main()
