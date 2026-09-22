"""Explicit, read-only search comparison; importing this module never calls a model.

Cases: {"provenance":"...", "cases":[{"id":"q1", "question":"...",
"expected_ids":["CS-..."], "required":["regex"], "forbidden":["regex"]}]}.
An empty expected_ids means no match is expected. Synthetic template labels are
also supported via expected_templates and template_titles (exact source titles).
Labels are scoring data only. Callbacks receive only question/method or question/hits.

Replay: {"freeze_hash":"<hash from --freeze-only>", "translation":{"status":
"available"}, "results":[{"id":"q1", "question":"...", "methods":{
"korean_hybrid":{"hits":[], "seconds":0.1}, "bilingual_hybrid":{"hits":[],
"seconds":0.2}}}]}. Full hit evidence is needed for action metrics. Answers and
search details can be copied from an earlier report. Missing data is never success.
"""
import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time

METHODS = ('korean_hybrid', 'bilingual_hybrid')


def freeze(dataset):
    """Validate and copy the entire question/label set before any search callback."""
    dataset = json.loads(json.dumps(dataset, ensure_ascii=False, allow_nan=False))
    if isinstance(dataset, list):
        dataset = {'provenance': 'User supplied; provenance unspecified', 'cases': dataset}
    if not isinstance(dataset, dict) or not isinstance(dataset.get('cases'), list) or not dataset['cases']:
        raise ValueError('At least one case is required.')
    titles = dataset.get('template_titles', {})
    if not isinstance(titles, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                                           for k, v in titles.items()) or len(set(titles.values())) != len(titles):
        raise ValueError('template_titles must map unique labels to unique source titles.')
    seen = set()
    for case in dataset['cases']:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k].strip()
                                             for k in ('id', 'question')) or case['id'] in seen:
            raise ValueError('Cases require unique nonempty id and question strings.')
        seen.add(case['id'])
        keys = [key for key in ('expected_ids', 'expected_templates') if key in case]
        if len(keys) != 1:
            raise ValueError('Use exactly one of expected_ids or expected_templates per case.')
        values = case[keys[0]]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError('Expected labels must be a list of nonempty strings (or []).')
        if keys[0] == 'expected_templates' and not set(values) <= titles.keys():
            raise ValueError('Every expected template needs a source title.')
        for key in ('required', 'forbidden'):
            patterns = case.get(key, [])
            if not isinstance(patterns, list) or any(not isinstance(p, str) or not p for p in patterns):
                raise ValueError('Answer rules must be lists of nonempty regex strings.')
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise ValueError(f'Invalid {key} regex: {error}') from error
    digest = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dataset, digest


def atomic_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            name = stream.name
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def source_cells(hit):
    return {key: value for section in hit.get('evidence', []) for key, value in section.get('cells', {}).items()}


def action_availability(hit):
    """Conservative surface evidence, not a semantic judgment or resolution label."""
    values = [value.strip() for key, value in source_cells(hit).items()
              if re.fullmatch(r'조치내용(?:\(\d+\))?|관찰결과|조치결과', key) and value.strip()]
    actual = False
    for value in values:
        for clause in re.split(r'[.;\n/]', value):
            # ponytail: explicit recorded verbs only; mixed/ambiguous clauses stay unknown.
            # Expand with human-adjudicated examples if this conservative rule misses useful logs.
            if re.search(r'예정|계획|필요|요청|권고|미실시|미확인|확인\s*불가|하지\s*않|안\s*함', clause):
                continue
            if re.search(r'(?:교체|조정|청소|세척|점검|측정|재체결|재부팅|확인|실시|관찰)(?:함|했|됨|되었|하였|\s*완료|\s*후)|'
                         r'재발(?:함|했|\s*없|\s*발생)|(?:누수|진동|압력|온도|유량).*(?:지속|상승|하락|감소|증가|관찰)', clause):
                actual = True
    return {'evidence_available': 'evidence' in hit, 'nonempty': bool(values), 'actual_or_observation': actual}


def score(case, hits, titles=None):
    """Top three, binary relevance; recall is over labeled IDs or template types."""
    hits = hits[:3]
    expected = set(case.get('expected_ids', case.get('expected_templates', [])))
    lookup = {title: label for label, title in (titles or {}).items()}
    labels = []
    for hit in hits:
        labels.append(({lookup.get(source_cells(hit).get('제목', '').strip())}
                       if 'expected_templates' in case else
                       {hit['id'], hit.get('record_id', hit['id'].removesuffix('/문서')),
                        *hit.get('evidence_ids', [])}) - {None})
    relevant = [bool(value & expected) for value in labels]
    found = set().union(*labels) if labels else set()
    actions = [action_availability(hit) for hit in hits]
    return {'top1': bool(relevant and relevant[0]) if expected else None,
            'hit_at_3': any(relevant) if expected else None,
            'recall_at_3': len(found & expected) / len(expected) if expected else None,
            'precision_at_3': sum(relevant) / len(hits) if hits else (0.0 if expected else None),
            'mrr_at_3': next((1 / (i + 1) for i, yes in enumerate(relevant) if yes), 0.0) if expected else None,
            'passed': any(relevant) if expected else not hits,
            'empty': not hits, 'missing_relevant': bool(expected) and not any(relevant),
            'missing_expected_ids': sorted(expected - found),
            'action_nonempty_at_3': any(a['nonempty'] for a in actions),
            'action_actual_at_3': any(a['actual_or_observation'] for a in actions),
            'relevant_action_at_3': any(yes and a['actual_or_observation'] for yes, a in zip(relevant, actions)),
            'top1_action_actual': bool(actions and actions[0]['actual_or_observation']),
            'top1_relevant_action': bool(relevant and relevant[0] and actions[0]['actual_or_observation']),
            'action_evidence_missing': any(not a['evidence_available'] for a in actions)}


def prefer_action(question, hits):
    """Evaluation only: stable permutation of existing top three; never drop a plan/failure."""
    # ponytail: top-three permutation cannot recover omitted actions or improve coverage.
    # Expand the candidate pool only after independent top1_relevant_action gains without regressions.
    from .search import exact_terms, search_concepts
    if any(exact_terms(question)):
        return list(hits)
    concepts = search_concepts(question)
    if not any(k.startswith('symptom') for k in concepts) or not any(k.startswith('equipment') for k in concepts):
        return list(hits)
    # These scores are RRF ranks, not semantic confidence. Also require all known
    # symptom/equipment concepts in source fields; abstain if that evidence is absent.
    eligible = []
    for i, hit in enumerate(hits[:3]):
        cells = source_cells(hit)
        text = ' '.join(cells.get(key, '') for key in ('제목', '문제현상', '현상', '제품군', 'UNIT명'))
        value = hit.get('score')
        if concepts <= search_concepts(text) and type(value) in (int, float) and math.isfinite(value) and value > 0:
            eligible.append(i)
    result = list(hits)
    if eligible:
        anchor = hits[eligible[0]]['score']
        positions = [i for i in eligible if abs(hits[i]['score'] - anchor) <= anchor * 0.02]
        ordered = sorted((hits[i] for i in positions), key=lambda h: not action_availability(h)['actual_or_observation'])
        for i, hit in zip(positions, ordered):
            result[i] = hit
    return result


def translation_availability(db):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    total = db.execute('SELECT COUNT(*) FROM rag_documents').fetchone()[0]
    indexed = db.execute('SELECT COUNT(*) FROM rag_english_documents').fetchone()[0] if 'rag_english_documents' in tables else 0
    stored = db.execute('SELECT COUNT(*) FROM english_translations').fetchone()[0] if 'english_translations' in tables else 0
    return {'status': 'unavailable' if not indexed else 'available' if indexed == total else 'partial',
            'source_records': total, 'indexed_records': indexed, 'stored_translations': stored,
            'freshness': 'not_revalidated', 'generated': 0,
            'note': 'Existing index counts only; no translation, rebuild, or semantic validation performed.'}


def summary(results, translation):
    metrics = ('top1', 'hit_at_3', 'recall_at_3', 'precision_at_3', 'mrr_at_3', 'passed',
               'action_nonempty_at_3', 'action_actual_at_3', 'relevant_action_at_3',
               'top1_action_actual', 'top1_relevant_action')
    result = {}
    for method in dict.fromkeys(name for row in results for name in row['methods']):
        entries = [row['methods'][method] for row in results if method in row['methods']]
        scored = [entry for entry in entries if 'score' in entry]
        result[method] = {'attempted': len(entries), 'scored': len(scored),
                          'errors': sum(e['status'] == 'error' for e in entries),
                          'missing': sum(e['status'] == 'missing' for e in entries),
                          'degraded': sum(e['status'] == 'degraded' for e in entries),
                          'answer_errors': sum(bool(e.get('answer_error')) for e in entries),
                          'answer_missing': sum(bool(e.get('answer_missing')) for e in entries),
                          'answer_checks_failed': sum(e.get('answer_checks', {}).get('passed') is False for e in entries)}
        for key in metrics:
            values = [entry['score'][key] for entry in scored if entry['score'][key] is not None]
            result[method][key] = sum(values) / len(values) if values else None
        seconds = [e['seconds'] for e in entries if e.get('seconds') is not None]
        result[method]['mean_seconds'] = sum(seconds) / len(seconds) if seconds else None
    paired = [row for row in results if all(row['methods'].get(m, {}).get('status') == 'ok' for m in METHODS)]
    available = translation.get('status') in ('available', 'partial')
    delta = {}
    if available and paired:
        for key in metrics:
            values = [float(row['methods'][METHODS[1]]['score'][key]) - float(row['methods'][METHODS[0]]['score'][key])
                      for row in paired if all(row['methods'][m]['score'][key] is not None for m in METHODS)]
            delta[key] = sum(values) / len(values) if values else None
    return {'methods': result, 'paired_cases': len(paired) if available else 0,
            'bilingual_minus_korean': delta or None,
            'comparison_note': 'Observed paired retrieval metrics only; no general benefit claim. '
            'Unknown/unavailable translations suppress comparison. Partial indexes have limited coverage.'}


def run_comparison(dataset, search_fn, output, *, translation=None, answer_fn=None, experiment=False, metadata=None,
                   methods=METHODS):
    dataset, digest = freeze(dataset)
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError('Select unique supported comparison methods.')
    translation = translation or {'status': 'unknown'}
    report = {'format': 'retrieval-comparison-v1', 'freeze_hash': digest, 'dataset': dataset,
              'translation': translation, 'metadata': metadata or {}, 'complete': False, 'results': [],
              'ranking_experiment': 'top-three stable 2% RRF/concept gate' if experiment else None,
              'limitations': 'Synthetic drafts are not field accuracy. Action rules are conservative surface signals, '
              'not proof of completed work or successful resolution. Missing evidence makes action metrics incomplete. '
              'The ranking experiment only permutes existing top three: action/relevance coverage cannot increase, '
              'and omitted candidates (including an FFU action outside top three) cannot be recovered. '
              'Production ranking requires independent measured top1_relevant_action gain without relevance regression.',
              'scoring_scope': 'Operational rehearsal on frozen cases; synthetic draft scores are not actual field accuracy.'}

    def save():
        report['summary'] = summary(report['results'], translation)
        atomic_report(output, report)

    save()
    for case in dataset['cases']:
        row = {'id': case['id'], 'question': case['question'], 'methods': {}}
        report['results'].append(row)
        for method in methods:
            started = time.monotonic()
            try:
                # Never pass the case object, expected IDs, regexes, or template labels to retrieval.
                entry = dict(search_fn(case['question'], method))
                entry.setdefault('seconds', time.monotonic() - started)
                if entry.get('error'):
                    entry['status'] = 'error'
                elif 'hits' not in entry:
                    entry['status'] = 'missing'
                    entry.setdefault('missing_reason', 'No saved search output.')
                else:
                    hits = entry['hits']
                    if not isinstance(hits, list) or any(not isinstance(h, dict) or not isinstance(h.get('id'), str) for h in hits):
                        raise ValueError('hits must be a list of objects with string id.')
                    detail = entry.get('details', {})
                    vector = detail.get('vector_search', {})
                    bilingual = detail.get('search', {})
                    entry['warnings'] = [d['warning'] for d in (vector, bilingual) if d.get('warning') and d.get('mode') != 'exact']
                    entry['status'] = 'degraded' if entry['warnings'] else 'ok'
                    if method == 'bilingual_hybrid' and translation.get('status') not in ('available', 'partial'):
                        entry['status'] = 'degraded'
                        entry['warnings'].append('Translation availability is unavailable or unknown; comparison excluded.')
                    entry['score'] = score(case, hits, dataset.get('template_titles'))
                    if answer_fn is not None:
                        answer_started = time.monotonic()
                        try:
                            # Whitelist source fields; replay annotations can never become model input.
                            sources = [{k: h[k] for k in ('id', 'record_id', 'kind', 'content', 'evidence', 'evidence_ids') if k in h}
                                       for h in hits[:3]]
                            entry['answer'] = answer_fn(case['question'], sources)
                        except Exception as error:
                            # SDK messages may echo credentials or response bodies; retain only the category.
                            entry['answer_error'] = f'{type(error).__name__}: answer failed; exception details omitted.'
                        entry['answer_seconds'] = time.monotonic() - answer_started
                    if 'answer' in entry and not entry.get('answer_error'):
                        from .evaluation import answer_checks
                        entry['answer_checks'] = answer_checks(entry['answer'], case)
                    entry['answer_missing'] = bool(case.get('required') or case.get('forbidden')) and 'answer_checks' not in entry
                if entry.get('seconds') is not None and (type(entry['seconds']) not in (int, float) or not math.isfinite(entry['seconds']) or entry['seconds'] < 0):
                    raise ValueError('seconds must be nonnegative and finite or null.')
                entry['latency_missing'] = entry.get('seconds') is None
            except Exception as error:
                entry = {'status': 'error', 'error': f'{type(error).__name__}: search evaluation failed; exception details omitted.',
                         'seconds': time.monotonic() - started}
            row['methods'][method] = entry
            if experiment and 'score' in entry:
                ranked = prefer_action(case['question'], entry['hits'])
                row['methods'][method + '_action'] = {
                    'status': entry['status'], 'hits': ranked, 'seconds': None,
                    'score': score(case, ranked, dataset.get('template_titles')),
                    'ranking_only': True, 'answer_missing': True,
                    'note': 'Same top three, so no action coverage gain is possible. No new search or answer. '
                    'Original failures/plans retained; inspect top1_relevant_action and relevance regressions.'}
            save()
    report['complete'] = True
    save()
    return report


def replay_search(replay, dataset):
    dataset, digest = freeze(dataset)
    if replay.get('freeze_hash') != digest:
        raise ValueError('Replay freeze_hash differs from the frozen questions/labels.')
    rows = replay.get('results', [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Replay results must be a list of case rows.')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate replay case ID.')
    by_question = {}
    for case in dataset['cases']:
        row = by_id.get(case['id'])
        if row and row.get('question') != case['question']:
            raise ValueError('Replay question differs from the frozen case.')
        if case['question'] in by_question:
            raise ValueError('Replay requires unique questions.')
        by_question[case['question']] = row or {}

    def search(question, method):
        entry = dict(by_question[question].get('methods', {}).get(method, {}))
        entry.setdefault('seconds', None)  # Replay processing time is not search latency.
        return entry
    return search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path; incrementally replaced atomically.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--replay', type=Path, help='Offline saved outputs; no model clients are created.')
    mode.add_argument('--live', action='store_true', help='Explicitly call already-running configured servers, sequentially.')
    mode.add_argument('--freeze-only', action='store_true', help='Validate and hash cases, without DB or model access.')
    parser.add_argument('--db', type=Path, help='Existing source DB, opened mode=ro; never migrated or rebuilt.')
    parser.add_argument('--config', type=Path, help='Existing model config (live only); settings are not changed.')
    parser.add_argument('--answers', action='store_true', help='Also generate answers live; replay checks saved answers automatically.')
    parser.add_argument('--prefer-action', action='store_true', help='Evaluation-only conservative permutation of top three.')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS,
                        help='Defaults to both; choose korean_hybrid for the untranslated main corpus.')
    parser.add_argument('--catalog', type=Path, help='Optional catalog.json titles for legacy expected_templates cases.')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('--output must be a new file; choose a separate report for each run.')
    if args.live and (not args.db or not args.config):
        parser.error('--live requires --db and --config.')
    if not args.live and (args.answers or args.db or args.config):
        parser.error('--answers, --db and --config require --live.')
    inputs = [p.resolve() for p in (args.cases, args.replay, args.db, args.config, args.catalog) if p]
    if args.output.resolve() in inputs or (args.db and str(args.output.resolve()) in [str(args.db.resolve()) + suffix for suffix in ('-wal', '-shm', '-journal')]):
        parser.error('Report must not overwrite an input or database sidecar.')
    try:
        dataset = json.loads(args.cases.read_text(encoding='utf-8-sig'))
        if args.catalog:
            if isinstance(dataset, list):
                dataset = {'cases': dataset}
            titles = json.loads(args.catalog.read_text(encoding='utf-8-sig'))['titles']
            if 'template_titles' in dataset and dataset['template_titles'] != titles:
                raise ValueError('Dataset template_titles conflicts with --catalog.')
            dataset['template_titles'] = titles
        dataset, digest = freeze(dataset)
        if args.freeze_only:
            atomic_report(args.output, {'freeze_hash': digest, 'dataset': dataset, 'complete': False, 'results': []})
            return 0
        if args.replay:
            replay = json.loads(args.replay.read_text(encoding='utf-8-sig'))
            report = run_comparison(dataset, replay_search(replay, dataset), args.output,
                                    translation=replay.get('translation'), experiment=args.prefer_action,
                                    metadata={'mode': 'replay', 'source': str(args.replay), 'live_calls': 0}, methods=args.methods)
        else:
            from openai import OpenAI
            from . import english, semantic, llm
            config = json.loads(args.config.read_text(encoding='utf-8-sig'))
            cfg = semantic.settings(config)
            with ExitStack() as stack:
                db = stack.enter_context(closing(sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)))
                db.execute('PRAGMA query_only=ON')
                db.execute('BEGIN')  # Both methods read one SQLite snapshot; no writes or migrations.
                translation = translation_availability(db)
                embedding = stack.enter_context(closing(OpenAI(base_url=cfg['base_url'], api_key=cfg['api_key'],
                                                               timeout=cfg['timeout_seconds'], max_retries=0)))
                client = None
                if (translation['indexed_records'] and 'bilingual_hybrid' in args.methods) or args.answers:
                    client = stack.enter_context(closing(OpenAI(base_url=config['base_url'], api_key=config.get('api_key', 'EMPTY'),
                                                                timeout=config.get('timeout_seconds', 120), max_retries=0)))

                def search(question, method):
                    if method == 'bilingual_hybrid' and not translation['indexed_records']:
                        return {'missing_reason': 'English index unavailable; bilingual search skipped.', 'seconds': None}
                    detail = {}
                    keywords = (english.bilingual_retrieve(db, question, client, config, details=detail, limit=20)
                                if method == 'bilingual_hybrid' else None)
                    hits = semantic.search_cases(db, question, config, keyword_hits=keywords,
                                                 details=detail, embedding_client=embedding)
                    return {'hits': hits, 'details': detail}

                answer_fn = (lambda question, hits: llm.answer(client, config, question, hits)[0]) if args.answers else None
                report = run_comparison(dataset, search, args.output, translation=translation, answer_fn=answer_fn,
                                        experiment=args.prefer_action, methods=args.methods,
                                        metadata={'mode': 'live', 'db': str(args.db.resolve()),
                                        'model': config.get('model'), 'embedding_model': cfg['model'],
                                        'reasoning_effort': config.get('reasoning_effort'),
                                        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                        'vector_profile': semantic.profile(config),
                                        'vector_records': semantic.vector_count(db, config),
                                        'scope_note': 'Results apply only to this source snapshot and frozen cases. '
                                        'Small isolated DB results do not establish full-corpus English benefit.',
                                        'timing_note': 'Sequential Korean then bilingual; cold caches included. No process preparation.'})
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    print(f"Saved {len(report['results'])} frozen cases: {args.output}")
    return int(any(entry.get('status') != 'ok' or entry.get('answer_error') or entry.get('answer_missing')
                   or entry.get('score', {}).get('passed') is False or entry.get('answer_checks', {}).get('passed') is False
                   for row in report['results'] for method, entry in row['methods'].items() if method in METHODS))


if __name__ == '__main__':
    raise SystemExit(main())
