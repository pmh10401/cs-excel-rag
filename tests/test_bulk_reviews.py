"""2,000건의 미검토 제안을 한 번에 채택하고 기존 검토·원문을 보존합니다."""
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
from ragkit import cslist, records, search, storage


def main(root):
    schema, rules = cslist.load_schema(root / 'cslist_schema.json')
    columns = schema['rewrite_columns']
    rows = [{key: '점검 예정' if key in columns else '' for key in schema['columns']} |
            {schema['id_column']: f'CS-BULK-{i:04}', '_id_column': schema['id_column'], '_source_key': 'bulk-source'}
            for i in range(1, 2001)]
    rows[-2][columns[0]] = ''
    rows[-2][columns[1]] = '0'
    rows[-2]['_typed_cells'] = {columns[1]: {'type': 'number', 'value': 0}}
    with sqlite3.connect(':memory:') as db:
        storage.ingest(db, rows, rules, schema=schema)
        # 실제 LLM 품질을 평가하는 검사가 아닙니다. 검사 가능한 제안을 DB에 준비합니다.
        proposals = [records.public_cells(row) | {key: '점검 예정입니다.' for key in columns if row[key] == '점검 예정'} for row in rows]
        proposals[-1][columns[0]] += ' qzxjvkwp'
        proposals[2] = None
        proposals[3][columns[0]] = '999회 점검했습니다.'  # 숫자를 추가한 잘못된 제안
        with db:
            db.executemany('UPDATE normalized_records SET normalized_json=? WHERE id=?',
                           [(json.dumps(p, ensure_ascii=False), records.record_id(row)) for row, p in zip(rows, proposals)])
        existing = [{'source_key': 'bulk-source', 'id': records.record_id(rows[0]),
                     'revision': records.row_revision(rows[0]), 'column': columns[0], 'action': 'edit', 'value': '점검은 예정되어 있습니다.'},
                    {'source_key': 'bulk-source', 'id': records.record_id(rows[1]),
                     'revision': records.row_revision(rows[1]), 'column': columns[0], 'action': 'original', 'value': '점검 예정'}]
        storage.apply_reviews(db, {'format': 'cell-reviews-v1', 'entries': existing}, rules, schema=schema)
        saved_reviews = db.execute('SELECT * FROM cell_reviews ORDER BY record_id').fetchall()
        raw = db.execute('SELECT id,original_json,normalized_json,fingerprint FROM normalized_records ORDER BY id').fetchall()
        assert not search.retrieve(db, 'qzxjvkwp')
        # 중간에 색인 저장이 실패하면 검토만 부분 채택되지 않아야 합니다.
        with patch.object(cslist, 'rebuild_search', side_effect=OSError('의도한 색인 저장 실패')):
            try:
                storage.accept_pending_reviews(db, rules, schema=schema)
            except OSError:
                pass
            else:
                raise AssertionError('색인 실패를 성공으로 처리했습니다.')
        assert db.execute('SELECT * FROM cell_reviews ORDER BY record_id').fetchall() == saved_reviews
        assert not search.retrieve(db, 'qzxjvkwp')
        result = storage.accept_pending_reviews(db, rules, schema=schema)
        assert result['rows'] == 2000 and result['applied_rows'] == 1998
        assert result['saved'] == 1998 * len(columns) - 4 > 4000, result
        assert result['reviewed_cells'] == 2 and result['missing_rows'] == 1
        assert len(result['failures']) == 1 and result['failures'][0]['id'] == 'CS-BULK-0004'
        assert db.execute('SELECT id,original_json,normalized_json,fingerprint FROM normalized_records ORDER BY id').fetchall() == raw
        assert db.execute('SELECT * FROM cell_reviews WHERE record_id IN (?,?) AND column_name=? ORDER BY record_id',
                          ('CS-BULK-0001', 'CS-BULK-0002', columns[0])).fetchall() == saved_reviews
        assert search.retrieve(db, 'qzxjvkwp')[0]['record_id'] == 'CS-BULK-2000'
        assert not search.retrieve(db, 'qzxjvkwp', original_only=True)
        assert not db.execute('SELECT 1 FROM cell_reviews WHERE record_id=? AND column_name IN (?,?)',
                              ('CS-BULK-1999', columns[0], columns[1])).fetchall(), '빈칸과 숫자 셀을 채택했습니다.'
        repeated = storage.accept_pending_reviews(db, rules, schema=schema)
        assert repeated['saved'] == repeated['applied_rows'] == 0
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
    print('PASS: 2,000건 일괄 채택·검색 반영, 4,000셀 초과, 기존 검토·원문 보존, 실패 제외·롤백·재실행')


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1] / 'windows-demo')
