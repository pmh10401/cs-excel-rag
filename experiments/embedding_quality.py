"""고정 합성 질문으로 기존 벡터를 읽기 전용 비교합니다. 운영 설정·DB를 쓰지 않습니다."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
import time

import numpy as np

from experiments.gliner_decide import digest, metrics
from ragkit.retrieval_evaluation import atomic_report, freeze
from ragkit import semantic
from ragkit.search import focused_query

MODELS = ('qwen', 'gemma2')
METHODS = ('keyword', 'qwen_vector', 'qwen_hybrid', 'gemma2_vector', 'gemma2_hybrid')


def condition_result(scores, expected, threshold):
    scores = [float(value) for value in scores]
    if len(scores) != 2 or not np.isfinite(scores).all():
        raise ValueError('조건 후보 두 개의 유한한 점수가 필요합니다.')
    best = int(np.argmax(scores))
    return dict(scores=scores, top1=best, correct=best == expected if expected is not None else None,
                rejected=max(scores) < threshold, margin=abs(scores[0] - scores[1]))


def summarize(rows, conditions):
    result = {}
    for method in METHODS:
        entries = [row['methods'][method] for row in rows]
        positives = [entry for entry in entries if entry['metrics']['top1'] is not None]
        semantic_only = [row['methods'][method] for row in rows
                         if row['case'].get('expected_templates')]
        negatives = [entry for entry in entries if entry['metrics']['unexpected_hits'] is not None]
        seconds = sorted(entry['seconds'] for entry in entries)
        result[method] = dict(positive_count=len(positives),
            top1=sum(entry['metrics']['top1'] for entry in positives),
            hit_at_3=sum(entry['metrics']['hit_at_3'] for entry in positives),
            hit_at_5=sum(entry['metrics']['hit_at_5'] for entry in positives),
            symptom_count=len(semantic_only), symptom_top1=sum(e['metrics']['top1'] for e in semantic_only),
            precision_at_5=statistics.mean(e['metrics']['precision_at_5'] for e in entries),
            negative_count=len(negatives), correct_rejections=sum(not e['metrics']['unexpected_hits'] for e in negatives),
            mean_seconds=statistics.mean(seconds), p95_seconds=seconds[int(np.ceil(.95 * len(seconds))) - 1],
            errors=sum(e['status'] != 'ok' for e in entries))
    for model in MODELS:
        values = [row[model] for row in conditions]
        positives = [v for v in values if v['correct'] is not None]
        negatives = [v for v in values if v['correct'] is None]
        result[model + '_conditions'] = dict(positive_count=len(positives),
            top1=sum(v['correct'] for v in positives), negative_count=len(negatives),
            correct_rejections=sum(v['rejected'] for v in negatives),
            false_rejections=sum(v['rejected'] for v in positives),
            languages={lang: sum(row[model]['correct'] is True for row in conditions if row['language'] == lang)
                       for lang in ('ko', 'en', 'zh')})
    return result


def run(db_path, dataset_path, config_paths, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise ValueError('기존 결과를 덮어쓰지 않습니다. 새 출력 폴더를 사용하세요.')
    frozen = json.loads(dataset_path.read_text())
    dataset, signature = freeze(frozen['dataset'])
    assert signature == frozen['freeze_hash']
    challenge_path = Path('experiments/gliner_conditions.json')
    challenge = json.loads(challenge_path.read_text())
    assert len(challenge['groups']) == 16
    configs = {name: {'embedding': json.loads(path.read_text())['embedding']}
               for name, path in zip(MODELS, config_paths, strict=True)}
    # 정답 고정과 파일 보존 해시를 저장한 뒤에만 모델을 준비·호출합니다.
    files = [db_path, dataset_path, challenge_path, Path(__file__), *config_paths,
             Path('ragkit/semantic.py'), Path('ragkit/search.py'), Path('ragkit/retrieval_evaluation.py')]
    protocol = dict(date='2026-10-07', freeze_hash=signature,
        hashes={str(path): digest(path) for path in files}, min_score=.3, limit=5, candidate_limit=20,
        models={name: {key: value for key, value in semantic.settings(cfg).items() if key != 'api_key'}
                | {'profile': semantic.profile(cfg)} for name, cfg in configs.items()},
        limits=['Known synthetic development questions, not independent field labels',
                '50 fresh full-corpus retrieval queries; separate 48 queries over two Korean candidates',
                'No translation, answer generation, reranker, training, threshold tuning or production promotion',
                'Raw vector mode deliberately bypasses exact CS lookup; hybrid retains production exact lookup',
                'One model block at a time, warm index, single pass; timings are not exclusive-resource benchmarks'])
    assert all(semantic.settings(c)['min_score'] == .3 for c in configs.values())
    atomic_report(output / 'protocol.json', protocol)
    atomic_report(output / 'dataset.json', frozen)
    atomic_report(output / 'challenge.json', challenge)
    rows = [dict(case=case, methods={}) for case in dataset['cases']]
    conditions = [dict(id=f"{group['id']}-{lang}", language=lang, question=question,
                       expected=group['expected'], cases=group['cases'])
                  for group in challenge['groups'] for lang, question in group['queries'].items()]
    report = dict(complete=False, protocol=protocol, development=rows, conditions=conditions, index_load={})
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        report['records'] = db.execute('SELECT COUNT(*) FROM normalized_records').fetchone()[0]
        assert report['records'] == 100000
        for row in rows:
            started = time.perf_counter()
            hits = semantic.search_cases(db, row['case']['question'], configs['qwen'], mode='keyword', limit=5)
            row['methods']['keyword'] = dict(hits=hits, seconds=time.perf_counter()-started, status='ok',
                metrics=metrics(row['case'], hits, dataset['template_titles']))
        for name, config in configs.items():
            with closing(semantic.client_for(config)) as client:
                started = time.perf_counter()
                identities, index = semantic.load_index(db, config)
                report['index_load'][name] = time.perf_counter() - started
                assert len(identities) == index.ntotal == report['records']
                assert index.d == (1024 if name == 'qwen' else 768)
                semantic.embed(client, ['임베딩 연결 상태 확인'], semantic.settings(config), query=True)
                for row in rows:
                    for mode in ('vector', 'hybrid'):
                        started, details = time.perf_counter(), {}
                        if mode == 'vector':
                            hits = semantic.vector_hits(db, row['case']['question'], config,
                                limit=20, timings=details, embedding_client=client)[:5]
                        else:
                            hits = semantic.search_cases(db, row['case']['question'], config,
                                limit=5, details=details, embedding_client=client)
                        info = details.get('vector_search', {}) if mode == 'hybrid' else {}
                        status = 'degraded' if info.get('warning') and info['mode'] != 'exact' else 'ok'
                        row['methods'][name + '_' + mode] = dict(hits=hits, details=details, status=status,
                            seconds=time.perf_counter()-started,
                            metrics=metrics(row['case'], hits, dataset['template_titles']))
                    atomic_report(output / 'results.json', report)
                # 별도 조건 시험은 10만 건 검색과 섞지 않습니다. 원문 셀의 현상 표기를 사용합니다.
                documents = [f'현상: {text}' for group in challenge['groups'] for text in group['cases']]
                queries = [focused_query(row['question']) for row in conditions]
                cfg = semantic.settings(config)
                vectors = []
                started = time.perf_counter()
                for texts, query in ((documents, False), (queries, True)):
                    vectors.append(np.vstack([semantic.embed(client, texts[i:i+16], cfg, query=query)
                                               for i in range(0, len(texts), 16)]))
                report[name + '_condition_embedding_seconds'] = time.perf_counter() - started
                docs, questions = vectors
                np.savez_compressed(output / (name + '-conditions.npz'), documents=docs, questions=questions)
                for i, row in enumerate(conditions):
                    row[name] = condition_result(docs[i//3*2:i//3*2+2] @ questions[i], row['expected'], cfg['min_score'])
                atomic_report(output / 'results.json', report)
                print(name, '50 retrieval questions and 48 condition questions finished', flush=True)
    report['preserved'] = {str(path): digest(path) == checksum for path, checksum in protocol['hashes'].items()}
    report['summary'] = summarize(rows, conditions)
    report['complete'] = (all(report['preserved'].values()) and
                          all(v['status'] == 'ok' for row in rows for v in row['methods'].values()))
    atomic_report(output / 'results.json', report)
    verify(output)


def verify(output):
    """모델 재호출 없이 저장 원문·정답과 실제 벡터의 점수를 대조합니다."""
    report = json.loads((output / 'results.json').read_text())
    frozen = json.loads((output / 'dataset.json').read_text())
    dataset, signature = freeze(frozen['dataset'])
    assert report['complete'] and signature == report['protocol']['freeze_hash'] == frozen['freeze_hash']
    assert all(report['preserved'].values()) and len(report['development']) == 50
    assert [row['case'] for row in report['development']] == dataset['cases']
    for row in report['development']:
        assert set(row['methods']) == set(METHODS)
        for entry in row['methods'].values():
            assert entry['status'] == 'ok' and 0 <= len(entry['hits']) <= 5
            assert len({hit['id'] for hit in entry['hits']}) == len(entry['hits'])
            assert metrics(row['case'], entry['hits'], dataset['template_titles']) == entry['metrics']
    challenge = json.loads((output / 'challenge.json').read_text())
    assert len(report['conditions']) == 48
    expected_rows = [(f"{g['id']}-{lang}", question, g['cases'], g['expected'])
                     for g in challenge['groups'] for lang, question in g['queries'].items()]
    assert [(r['id'], r['question'], r['cases'], r['expected']) for r in report['conditions']] == expected_rows
    for model in MODELS:
        with np.load(output / (model + '-conditions.npz')) as saved:
            docs, questions = saved['documents'], saved['questions']
        assert docs.shape == (32, 1024 if model == 'qwen' else 768)
        assert questions.shape == (48, docs.shape[1])
        assert np.isfinite(docs).all() and np.isfinite(questions).all()
        assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-4)
        assert np.allclose(np.linalg.norm(questions, axis=1), 1, atol=1e-4)
        for i, row in enumerate(report['conditions']):
            assert condition_result(docs[i//3*2:i//3*2+2] @ questions[i], row['expected'], .3) == row[model]
    assert summarize(report['development'], report['conditions']) == report['summary']
    atomic_report(output / 'verification.json', dict(complete=True, retrieval_results=250, condition_results=96,
        results_sha256=digest(output / 'results.json'), checked='Stored original evidence, labels, cosine vectors and all summary metrics'))
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('outputs/mac-cslist-ui/demo.sqlite3'))
    parser.add_argument('--dataset', type=Path, default=Path('outputs/embedding-quality-20261007/dataset.json'))
    parser.add_argument('--qwen-config', type=Path, default=Path('outputs/embeddinggemma2-20261007/mac_config.before.json'))
    parser.add_argument('--gemma-config', type=Path, default=Path('mac_config.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/embedding-quality-20261007'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        run(args.db, args.dataset, (args.qwen_config, args.gemma_config), args.output)


if __name__ == '__main__':
    main()
