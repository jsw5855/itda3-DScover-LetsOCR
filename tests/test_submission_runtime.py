import multiprocessing as mp
import os
import threading
import time

import pytest

from submission_runtime import THREAD_VARIABLES, run_workers


def initialize():
    assert all(os.environ[key] == '2' for key in THREAD_VARIABLES)


def initialize_error():
    raise ValueError('initializer failure')


def initialize_slow_exit():
    threading.Thread(target=time.sleep, args=(60,), daemon=False).start()


def work(item):
    if item == 'crash':
        os._exit(17)
    if item == 'error':
        raise ValueError('prediction failure')
    if item == 'stall':
        time.sleep(60)
    return item


@pytest.fixture(autouse=True)
def no_leaked_workers():
    before = {p.pid for p in mp.active_children()}
    yield
    assert {p.pid for p in mp.active_children()} == before


def test_results_and_environment_restoration(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.delenv('NUMEXPR_NUM_THREADS', raising=False)
    assert sorted(run_workers(list(range(10)), initialize, work)) == list(range(10))
    assert os.environ['OMP_NUM_THREADS'] == '7'
    assert 'NUMEXPR_NUM_THREADS' not in os.environ


@pytest.mark.parametrize('initializer,jobs,message', [
    (initialize_error, [1], 'initializer failure'),
    (initialize, ['error'], 'prediction failure'),
    (initialize, ['crash'], 'Worker'),
])
def test_failure_propagates_and_reaps(initializer, jobs, message):
    with pytest.raises(RuntimeError, match=message):
        run_workers(jobs, initializer, work, task_timeout=10)


def test_stalled_prediction_is_bounded():
    with pytest.raises(TimeoutError, match='stalled'):
        run_workers(['stall'], initialize, work, task_timeout=3)


def test_shutdown_with_non_daemon_thread_is_bounded():
    started = time.monotonic()
    assert sorted(run_workers([1, 2], initialize_slow_exit, work, shutdown_timeout=0.2)) == [1, 2]
    assert time.monotonic() - started < 15


def test_parent_interrupt_reaps_workers():
    def interrupt(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_workers([1, 'stall'], initialize, work, progress=interrupt)


def test_total_deadline_is_bounded():
    with pytest.raises(TimeoutError, match='budget exceeded'):
        run_workers(['stall'], initialize, work, total_timeout=2)
