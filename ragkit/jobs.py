"""자료 DB와 독립된 작업 저널. 실행 잠금은 프로세스 종료 시 OS가 해제합니다."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


class JobStorageError(RuntimeError):
    pass


class JobStore:
    def __init__(self, output):
        self.path = Path(output) / 'jobs.sqlite3'
        self.lock_path = Path(output) / 'jobs.lock'

    def acquire(self):
        handle = self.lock_path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.lock_path.stat().st_size == 0:
                    handle.write(b'0')
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            import errno
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return None
            raise
        return handle

    @staticmethod
    def release(handle):
        if handle is not None:
            handle.close()

    def latest(self, *, unfinished=False, locked=False):
        if not self.path.exists():
            return None
        handle = None
        try:
            if not locked:
                handle = self.acquire()
            # 종료 도중 남은 SQLite 저널은 쓰기 권한으로 열어야 자동 롤백됩니다.
            mode = 'rw' if locked or handle is not None else 'ro'
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=' + mode, uri=True, timeout=5)) as db:
                if db.execute('PRAGMA user_version').fetchone()[0] != 1:
                    raise ValueError('지원하지 않는 작업 DB 버전')
                row = db.execute('SELECT id, kind, request, snapshot FROM jobs '
                                 + ("WHERE status NOT IN ('done','dismissed') " if unfinished else '')
                                 + 'ORDER BY updated DESC, rowid DESC LIMIT 1').fetchone()
            if row is None:
                return None
            identity, kind, request, snapshot = row
            request, snapshot = json.loads(request), json.loads(snapshot)
            if kind not in ('import_many', 'overnight') or not isinstance(request, dict) or not isinstance(snapshot, dict):
                raise ValueError('작업 기록 형식 오류')
            return {'id': identity, 'kind': kind, 'request': request, 'snapshot': snapshot}
        except (sqlite3.Error, ValueError, OSError) as error:
            raise JobStorageError('작업 기록을 읽지 못했습니다. jobs.sqlite3를 보존하고 서버 저장 공간·권한을 확인하세요.') from error
        finally:
            self.release(handle)

    def save(self, kind, request, snapshot, identity=None):
        identity = identity or uuid.uuid4().hex
        try:
            with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
                version = db.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, 1):
                    raise ValueError('지원하지 않는 작업 DB 버전')
                db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, '
                           'request TEXT NOT NULL, snapshot TEXT NOT NULL, status TEXT NOT NULL, updated REAL NOT NULL)')
                db.execute('PRAGMA user_version=1')
                db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                           'snapshot=excluded.snapshot, status=excluded.status, updated=excluded.updated',
                           (identity, kind, json.dumps(request, ensure_ascii=False), json.dumps(snapshot, ensure_ascii=False),
                            snapshot['status'], time.time()))
                db.execute("DELETE FROM jobs WHERE status IN ('done','dismissed') AND id NOT IN "
                           "(SELECT id FROM jobs WHERE status IN ('done','dismissed') ORDER BY updated DESC, rowid DESC LIMIT 20)")
            return identity
        except (sqlite3.Error, ValueError, OSError) as error:
            raise JobStorageError('작업 상태 저장에 실패했습니다. 성공 자료는 보존하지만 자동 복구 상태는 확인이 필요합니다. 서버 저장 공간·권한을 확인하세요.') from error

    def interrupted(self):
        handle = self.acquire()
        if handle is None:
            return None
        try:
            return self.latest(unfinished=True, locked=True)
        finally:
            self.release(handle)
