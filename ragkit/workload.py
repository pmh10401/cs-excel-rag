"""한 서버의 모델 요청을 조율합니다. 실행 중인 요청은 강제로 중단하지 않습니다."""
from contextlib import contextmanager
import threading


class ModelTurns:
    def __init__(self):
        self.condition = threading.Condition()
        self.active = 0
        self.exclusive = False
        self.questions = 0
        self.background_waiting = 0
        self.background_turn = False

    def enqueue_question(self):
        with self.condition:
            self.questions += 1

    @contextmanager
    def question(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.active and
                                    not (self.background_turn and self.background_waiting))
            self.questions -= 1
            self.active, self.exclusive = 1, True
        try:
            yield
        finally:
            with self.condition:
                self.active, self.exclusive = 0, False
                self.background_turn = True
                self.condition.notify_all()

    @contextmanager
    def background(self, *, exclusive=False):
        with self.condition:
            self.background_waiting += 1
            try:
                self.condition.wait_for(lambda: not self.exclusive and not (exclusive and self.active)
                                        and (not self.questions or self.background_turn))
                self.active += 1
                self.exclusive = exclusive
                self.background_turn = False
            finally:
                self.background_waiting -= 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                if exclusive:
                    self.exclusive = False
                self.condition.notify_all()
