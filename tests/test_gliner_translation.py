"""번역 실험: 모델 입력만 교체하고 원문과 정답 채점 자료를 보존합니다."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
from experiments.gliner_translation import translated_pair, translate, validate_unit
from experiments.gliner_followup import case_text


def main():
    for original, invalid in [('광학 AF 오류', '광학 AF 오류'), ('유량 __CS_NUMBER_0__', 'Flow __CS_NUMBER_1__')]:
        try: validate_unit(original, invalid)
        except ValueError: pass
        else: raise AssertionError('번역되지 않은 문장이나 변경된 표시자를 허용했습니다.')
    hits = [{'id': str(i), 'evidence': [{'cells': {'현상': text, '작업자': 'PRIVATE',
             '조치내용(1)': 'ANSWER_SECRET'}}]} for i, text in enumerate(('소음', '누수'))]
    before = deepcopy(hits)
    translations = {'누수 질문': 'Leak question', case_text(hits[0]): 'Noise', case_text(hits[1]): 'Leak'}
    def predict(texts):
        assert texts == ['Query: Leak question\nCase: Noise', 'Query: Leak question\nCase: Leak']
        return [{'relevance': {'label': label, 'confidence': .9}} for label in ('unrelated', 'related')]
    ranked, info = translated_pair('누수 질문', hits, predict, translations)
    assert ranked[0]['id'] == '1' and info['status'] == 'ok' and hits == before
    def query_predict(texts):
        assert texts[0] == 'Query: Leak question\nCase: 현상: 소음'
        assert 'PRIVATE' not in str(texts) and 'ANSWER_SECRET' not in str(texts)
        return [{'relevance': {'label': 'related', 'confidence': .8}}] * 2
    assert translated_pair('누수 질문', hits, query_predict, translations, query_only=True)[0] == hits
    ranked, info = translated_pair('누수 질문', hits, predict, {})
    assert ranked == hits and info['status'] == 'fallback'
    assert translated_pair('누수 질문', hits, predict, {}, exact=True)[1]['status'] == 'skipped'
    assert translated_pair('누수 질문', [], predict, {})[1]['status'] == 'skipped'
    rows = [dict(exact=False, case={'question': '누수 질문'}, hits=hits)]
    with TemporaryDirectory() as folder:
        output = Path(folder)
        with patch('ragkit.english.request_json', side_effect=RuntimeError('offline')):
            try: translate(rows, output, 'http://127.0.0.1:1234/v1', 'test')
            except ValueError: pass
            else: raise AssertionError('번역 실패를 완료로 처리했습니다.')
        cache = json.loads((output / 'translations.json').read_text())
        assert not cache['complete'] and len(cache['errors']) == 3
        with patch('ragkit.english.request_json', return_value={'text_0': 'Leak question',
                   'text_1': 'Noise', 'text_2': 'Leak'}) as request:
            translate(rows, output, 'http://127.0.0.1:1234/v1', 'test')
            assert request.call_count == 1
            translate(rows, output, 'http://127.0.0.1:1234/v1', 'test')
            assert request.call_count == 1  # 저장된 성공 번역은 다시 요청하지 않습니다.
        assert json.loads((output / 'translations.json').read_text())['complete']
        try: translate(rows, output, 'http://127.0.0.1:1234/v1', 'changed-model')
        except ValueError: pass
        else: raise AssertionError('다른 모델의 캐시를 재사용했습니다.')
    with TemporaryDirectory() as folder:
        rows = [dict(exact=False, case={'question': '광학 AF 오류'},
                     hits=[{'id': '1', 'evidence': [{'cells': {'현상': '광학 AF 오류'}}]}])]
        with patch('ragkit.english.request_json', side_effect=[{'text_0': '광학 AF 오류'},
                                                             {'text_0': 'Optical AF error'}]) as request:
            translate(rows, Path(folder), 'http://127.0.0.1:1234/v1', 'test')
            assert request.call_count == 2 and 'Korean or Chinese text remains' in request.call_args.args[2]
    with TemporaryDirectory() as folder:
        rows = [dict(exact=False, case={'question': f'유량 {number} L/min'},
                     hits=[{'id': number, 'evidence': [{'cells': {'현상': f'유량 {number} L/min'}}]}])
                for number in ('11.8', '118')]
        with patch('ragkit.english.request_json', return_value={'text_0': 'Flow __CS_NUMBER_0__'}) as request:
            translate(rows, Path(folder), 'http://127.0.0.1:1234/v1', 'test')
            assert len(request.call_args.args[3]) == 1  # 숫자만 다른 문장의 번역을 공유합니다.
        cache = json.loads((Path(folder) / 'translations.json').read_text())
        for number in ('11.8', '118'):
            assert cache['entries'][f'유량 {number} L/min'] == f'Flow {number} L/min'
            assert cache['entries'][f'현상: 유량 {number} L/min'] == f'Symptom: Flow {number} L/min'
    print('PASS: 영어 입력 교체, 질문만 번역, 원문·후보 보존, 누락 번역 복구, 정확 검색 보존')


if __name__ == '__main__':
    main()
