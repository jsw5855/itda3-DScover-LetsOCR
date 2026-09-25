"""Bounded spawn-worker lifecycle; no Queue feeder or Pool replacement loop."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _worker(connection, initializer, function):
    try:
        initializer()
        connection.send(('ready', None))
        while True:
            item = connection.recv()
            if item is None:
                return
            connection.send(('result', function(item)))
    except EOFError:
        return
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def _stop_workers(workers, graceful, grace_seconds):
    # One shared deadline per phase, not one full timeout per worker.
    deadline = time.monotonic() + (grace_seconds if graceful else 0)
    for process, _ in workers:
        process.join(max(0, deadline - time.monotonic()))
    forced = []
    for action in ('terminate', 'kill'):
        live = [p for p, _ in workers if p.is_alive()]
        for process in live:
            forced.append((process.pid, action))
            getattr(process, action)()
        deadline = time.monotonic() + 5
        for process in live:
            process.join(max(0, deadline - time.monotonic()))
    survivors = [p.pid for p, _ in workers if p.is_alive()]
    status = [(p.pid, p.exitcode) for p, _ in workers]
    for process, connection in workers:
        connection.close()
        if not process.is_alive():
            process.close()
    if survivors:
        raise RuntimeError(f'Workers did not exit after kill: {survivors}')
    print(f'[runtime] worker exits={status}; forced={forced}; survivors=[]', flush=True)


def run_workers(jobs, initializer, function, *, processes=2, threads=2,
                progress=None, task_timeout=180, total_timeout=7200,
                shutdown_timeout=10):
    """Return all results or fail with worker/job identity, always reaping workers.

    A timeout fails the run; it never substitutes predictions or writes partial CSV.
    The caller must use a __main__ guard when invoked from a Python script.
    Notebook callers import this module's worker target, which is spawn-safe.
    """
    context = mp.get_context('spawn')
    workers = []
    pending = {}
    remaining = iter(jobs)
    results = []
    started = time.monotonic()
    graceful = False
    previous = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    try:
        # Inherit limits BEFORE spawn imports the caller module / NumPy.
        for key in THREAD_VARIABLES:
            os.environ[key] = str(threads)
        for _ in range(processes):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, initializer, function), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            workers.append((process, parent))
            pending[parent] = ('initialization', time.monotonic())
        while pending:
            if time.monotonic() - started > total_timeout:
                raise TimeoutError(f'Submission worker time budget exceeded; pending={list(pending.values())}')
            ready = wait(list(pending), timeout=0.2)
            for process, connection in workers:
                if connection not in pending:
                    continue
                job, since = pending[connection]
                if connection in ready:
                    try:
                        kind, payload = connection.recv()
                    except (EOFError, OSError) as error:
                        raise RuntimeError(f'Worker {process.pid} disconnected during {job!r}') from error
                    if kind == 'error':
                        raise RuntimeError(f'Worker {process.pid} failed during {job!r}:\n{payload}')
                    if kind == 'result':
                        results.append(payload)
                        if progress is not None:
                            progress(len(results))
                    item = next(remaining, None)
                    connection.send(item)
                    if item is None:
                        del pending[connection]
                    else:
                        pending[connection] = (item, time.monotonic())
                elif process.exitcode is not None:
                    raise RuntimeError(f'Worker {process.pid} exited ({process.exitcode}) during {job!r}')
                elif time.monotonic() - since > task_timeout:
                    raise TimeoutError(f'Worker {process.pid} stalled during {job!r} for {task_timeout}s')
        graceful = True
    finally:
        try:
            _stop_workers(workers, graceful, shutdown_timeout)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    print(f'[runtime] workers stopped; {len(results)} results; {time.monotonic()-started:.1f}s', flush=True)
    return results
