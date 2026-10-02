"""Linux FTZ/DAZ evidence for the submission runtime. Diagnostic only.

Usage (repository root, inside the submission venv):
  python scripts/linux_ftz_probe.py IMAGE_DIR [--count 40] [--cpus 4] [--out DIR]

1. Which OpenMP runtime oneDNN's GOMP_parallel binds to, from glibc's own
   LD_DEBUG=bindings trace of a production engine start.
2. The MXCSR of the OCR main thread and of every OpenMP team thread, read inside
   a spawned production worker after one real OCR call.
3. The same images through the real 2-worker runtime with the helper on and
   with it disabled. Predictions must be identical; times are reported.

predict.ipynb never imports this file. Images 900301-900400 are refused.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOLDOUT = range(900301, 900401)
FTZ_DAZ = 0x8040
_ENGINE = None
_FLUSHED = []


def mxcsr():
    libm = ctypes.CDLL("libm.so.6")
    mode = (ctypes.c_uint32 * 2)()
    libm.fegetmode(mode)
    return mode[1]


def initialize_worker():
    """Production initialize_engine; records what flush_denormals returned."""
    global _ENGINE
    import ocr_pipeline
    original = ocr_pipeline.flush_denormals
    if os.environ.get("ITDA_PROBE_FTZ") == "off":
        ocr_pipeline.flush_denormals = lambda: 0
    else:
        ocr_pipeline.flush_denormals = lambda: _FLUSHED.append(original()) or _FLUSHED[-1]
    _ENGINE = ocr_pipeline.initialize_engine(enable_mkldnn=True, cpu_threads=2, recognition_batch_size=6)


def predict_one(item):
    import ocr_pipeline
    index, path = item
    started = time.perf_counter()
    prediction, method, attempts = ocr_pipeline.predict_image_detailed(_ENGINE, Path(path))
    return dict(index=index, image=Path(path).name, prediction=prediction, method=method,
                attempts=attempts, elapsed_sec=time.perf_counter() - started,
                pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)))


def probe_threads(item):
    """One real OCR call, then the MXCSR of the main thread and each OpenMP team."""
    import ocr_pipeline
    _, path = item
    _ENGINE.predict(ocr_pipeline.load_image(Path(path), 512))
    runtimes = sorted({line.split()[-1] for line in open("/proc/self/maps")
                       if re.search(r"/(libiomp5\.so|libgomp\.so\.1)$", line.strip())})
    teams = {}
    for library in runtimes:
        seen = []
        record = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            lambda _: seen.append([threading.get_native_id(), mxcsr()]))
        runtime = ctypes.CDLL(library)
        runtime.GOMP_parallel.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)
        runtime.GOMP_parallel(ctypes.cast(record, ctypes.c_void_p), None, 0, 0)
        teams[library] = sorted(seen)
    return dict(pid=os.getpid(), flushed=list(_FLUSHED),
                main=[threading.get_native_id(), mxcsr()], teams=teams,
                os_threads=len(os.listdir("/proc/self/task")))


def runtime_bindings(work):
    """glibc's record of where each library's OpenMP entry points resolved."""
    prefix = work / "ld"
    code = ("import ocr_pipeline; "
            "ocr_pipeline.initialize_engine(enable_mkldnn=True, cpu_threads=2, recognition_batch_size=6)")
    env = dict(os.environ, LD_DEBUG="bindings", LD_DEBUG_OUTPUT=str(prefix))
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pattern = re.compile(r"binding file (\S+) \[\d+\] to (\S+) \[\d+\]: normal symbol "
                         r"`(GOMP_parallel|omp_set_num_threads)'")
    found = {}
    for log in work.glob("ld.*"):
        with log.open(errors="replace") as stream:
            for line in stream:
                match = pattern.search(line)
                if match:
                    source, target, symbol = match.groups()
                    found.setdefault(f"{Path(source).name}:{symbol}", set()).add(Path(target).name)
        log.unlink()
    return {key: sorted(value) for key, value in sorted(found.items())}


def run_mode(mode, jobs, function, processes):
    from submission_runtime import run_workers
    os.environ["ITDA_PROBE_FTZ"] = mode
    started = time.perf_counter()
    results = run_workers(jobs, initialize_worker, function, processes=processes, threads=2)
    return time.perf_counter() - started, sorted(results, key=lambda r: r.get("index", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("Linux only")
    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if any(p.stem.isdigit() and int(p.stem) in HOLDOUT for p in paths):
        raise SystemExit("Cosmetic holdout images must never be probed")
    paths = paths[:args.count]
    if not paths:
        raise SystemExit("No images")
    allowed = sorted(os.sched_getaffinity(0))
    if args.cpus:
        os.sched_setaffinity(0, allowed[:args.cpus])
    out = Path(args.out or tempfile.mkdtemp(prefix="itda_ftz_probe_"))
    out.mkdir(parents=True, exist_ok=True)

    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                if line.startswith("model name")), "unknown")
    report = dict(platform=platform.platform(), machine=platform.machine(), libc=platform.libc_ver(),
                  python=sys.version.split()[0], cpu=cpu, cpus_allowed=allowed,
                  cpus_used=sorted(os.sched_getaffinity(0)), images=len(paths))
    print(f"[probe] {report['platform']} | {cpu} | cpus {report['cpus_used']}", flush=True)

    report["bindings"] = runtime_bindings(out)
    print("[probe] bindings", json.dumps(report["bindings"]), flush=True)

    _, threads = run_mode("on", [(0, str(paths[0]))], probe_threads, processes=1)
    report["threads"] = threads[0]
    print("[probe] threads", json.dumps(threads[0]), flush=True)

    jobs = [(i, str(p)) for i, p in enumerate(paths)]
    timing = {}
    rows = {}
    for mode in ("on", "off"):
        wall, results = run_mode(mode, jobs, predict_one, processes=2)
        rows[mode] = results
        timing[mode] = dict(wall_sec=wall, sec_per_image=wall / len(jobs),
                            affinity=sorted({tuple(r["affinity"]) for r in results}))
        print(f"[probe] ftz {mode}: {wall:.1f}s for {len(jobs)} images", flush=True)
    report["timing"] = timing
    report["prediction_differences"] = [
        a["image"] for a, b in zip(rows["on"], rows["off"])
        if (a["prediction"], a["attempts"]) != (b["prediction"], b["attempts"])]

    dnnl = report["bindings"].get("libdnnl.so.3:GOMP_parallel", [])
    team = next((v for k, v in report["threads"]["teams"].items() if dnnl and Path(k).name == dnnl[0]), [])
    report["checks"] = checks = dict(
        dnnl_binds_one_runtime=len(dnnl) == 1,
        helper_switched_main_thread=report["threads"]["flushed"] == [1],
        main_thread_ftz_daz=report["threads"]["main"][1] & FTZ_DAZ == FTZ_DAZ,
        dnnl_team_ftz_daz=bool(team) and all(m & FTZ_DAZ == FTZ_DAZ for _, m in team),
        predictions_identical=not report["prediction_differences"],
    )
    report["speedup_off_over_on"] = timing["off"]["wall_sec"] / timing["on"]["wall_sec"]
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for mode, results in rows.items():
            for r in results:
                stream.write(json.dumps(dict(mode=mode, **r), ensure_ascii=False) + "\n")
    for name, ok in checks.items():
        print(f"[probe] {'PASS' if ok else 'FAIL'} {name}")
    print(f"[probe] no-FTZ / FTZ wall ratio: {report['speedup_off_over_on']:.2f}x")
    print(f"[probe] report: {out / 'report.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
