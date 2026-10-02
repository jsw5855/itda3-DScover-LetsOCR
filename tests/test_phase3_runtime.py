"""Phase-3 early stop: strongest-evidence readings skip the 1024px re-read.

No image IDs and no ground-truth lookups; evidence is built from literal text.
"""
import pytest

import ocr_pipeline as p


def box(text, q, top=0):
    return {'text': text, 'confidence': q,
            'bbox': [[0, top], [250, top], [250, top + 25], [0, top + 25]]}


def evidence(boxes):
    _, found = p.stage_result(boxes)
    return found


def test_strong_anchored_complete_date_is_conclusive():
    found = evidence([box('EXP 2028.04.17', .99)])
    assert p.conclusive(found) is True
    assert p.retry_triggered(found) is False


def test_conclusive_overrides_multiple_distinct_candidates():
    """M alone no longer forces a re-read when the selected box labels itself
    as the expiry and was recognized confidently."""
    found = evidence([box('EXP 2028.04.17', .99), box('2026.01.02', .60, 40)])
    assert found['M'] is True
    assert p.conclusive(found) is True
    assert p.retry_triggered(found) is False


@pytest.mark.parametrize('boxes, reason', [
    ([box('EXP 2028.04.17', .85)], 'recognition confidence below the threshold'),
    ([box('EXP 2028.04.99', .99)], 'the day is damaged'),
    ([box('EXP 2028.04', .99)], 'the date is incomplete'),
    ([box('2028.04.17', .99)], 'the box does not label itself as the expiry'),
])
def test_not_conclusive(boxes, reason):
    assert p.conclusive(evidence(boxes)) is False, reason


def test_threshold_is_inclusive_and_not_fitted():
    assert p.conclusive(evidence([box('EXP 2028.04.17', p.RETRY_Q_THRESHOLD)])) is True
    assert p.conclusive(evidence([box('EXP 2028.04.17', p.RETRY_Q_THRESHOLD - .01)])) is False


def test_missing_confidence_is_never_conclusive():
    found = dict(q=None, M=False, short=False, damaged_day=False, self_anchor=True,
                 self_exclude=False, selected_date='2028-04-17', candidate_dates=['2028-04-17'])
    assert p.conclusive(found) is False
    assert p.retry_triggered(found) is True


def test_self_excluded_reading_is_never_conclusive():
    found = dict(q=.99, M=False, short=False, damaged_day=False, self_anchor=False,
                 self_exclude=True, selected_date='2028-04-17', candidate_dates=['2028-04-17'])
    assert p.conclusive(found) is False


def cascade(stage_boxes):
    requested = []

    def run_stage(name):
        requested.append(name)
        return p.stage_result(stage_boxes[name])

    prediction, method, attempts = p.run_cascade(run_stage)
    assert attempts == requested
    return prediction['final_date'], method, requested


def test_conclusive_original_requests_no_further_stage():
    date, method, requested = cascade({'original_512': [box('EXP 2028.04.17', .99)]})
    assert (date, method, requested) == ('2028-04-17', 'original_512', ['original_512'])


def test_conclusive_original_skips_highres_despite_second_date():
    """The saved-evidence case this change exists for: a confident self-labelled
    expiry alongside another printed date no longer pays for a 1024px re-read."""
    _, _, requested = cascade({'original_512': [box('EXP 2028.04.17', .99),
                                                box('2026.01.02', .60, 40)]})
    assert requested == ['original_512']


def test_uncertain_anchored_original_still_retries_at_highres():
    date, method, requested = cascade({
        'original_512': [box('EXP 2028.04.17', .85)],
        'highres_1024': [box('EXP 2028.04.17', .97)],
    })
    assert requested == ['original_512', 'highres_1024']
    assert date == '2028-04-17'
    assert method in ('highres_1024_retry', 'original_512_retry_kept')


def test_damaged_day_still_retries_at_highres():
    _, _, requested = cascade({
        'original_512': [box('EXP 2028.04.99', .99)],
        'highres_1024': [box('EXP 2028.04.17', .99)],
    })
    assert requested == ['original_512', 'highres_1024']


@pytest.mark.parametrize('platform', ['darwin', 'freebsd13', 'cygwin'])
def test_flush_denormals_is_a_no_op_on_other_platforms(monkeypatch, platform):
    monkeypatch.setattr(p.sys, 'platform', platform)
    monkeypatch.setattr(p.ctypes, 'CDLL', pytest.fail)
    assert p.flush_denormals() == 0


class FakeLibm:
    """glibc femode_t on x86_64: [control word | reserved, __mxcsr]."""

    def __init__(self, mxcsr=0x1F80, get=0, set_=0, keep=True):
        self.mxcsr, self.get, self.set_, self.keep = mxcsr, get, set_, keep

    def fegetmode(self, mode):
        mode[0], mode[1] = 0x037F, self.mxcsr
        return self.get

    def fesetmode(self, mode):
        if self.keep:
            self.mxcsr = mode[1]
        return self.set_


@pytest.fixture
def linux(monkeypatch, tmp_path):
    cpuinfo = tmp_path / 'cpuinfo'
    cpuinfo.write_text('processor\t: 0\nflags\t\t: fpu sse sse2 avx avx2\n')
    monkeypatch.setattr(p.sys, 'platform', 'linux')
    monkeypatch.setattr('platform.machine', lambda: 'x86_64')
    monkeypatch.setattr(p, '_CPUINFO', str(cpuinfo))
    libm = FakeLibm()
    monkeypatch.setattr(p.ctypes, 'CDLL', lambda name: libm if name == 'libm.so.6' else pytest.fail(name))
    return libm, cpuinfo


def test_linux_sets_ftz_and_daz_on_the_calling_thread(linux):
    libm, _ = linux
    assert p.flush_denormals() == 1
    assert libm.mxcsr == 0x1F80 | 0x8040


def test_linux_keeps_the_other_mxcsr_bits(linux):
    libm, _ = linux
    libm.mxcsr = 0x3F80  # rounding toward -inf, all exceptions masked
    assert p.flush_denormals() == 1
    assert libm.mxcsr == 0x3F80 | 0x8040


@pytest.mark.parametrize('machine', ['aarch64', 'arm64', 'i686', 'ppc64le', 'riscv64'])
def test_linux_other_architectures_are_untouched(linux, monkeypatch, machine):
    monkeypatch.setattr('platform.machine', lambda: machine)
    monkeypatch.setattr(p.ctypes, 'CDLL', pytest.fail)
    assert p.flush_denormals() == 0


@pytest.mark.parametrize('flags', ['flags\t\t: fpu sse sse2 sse4_2\n', 'processor\t: 0\n', ''])
def test_linux_without_avx_is_untouched(linux, monkeypatch, flags):
    _, cpuinfo = linux
    cpuinfo.write_text(flags)
    monkeypatch.setattr(p.ctypes, 'CDLL', pytest.fail)
    assert p.flush_denormals() == 0


def test_linux_unreadable_cpuinfo_is_untouched(linux, monkeypatch, tmp_path):
    monkeypatch.setattr(p, '_CPUINFO', str(tmp_path / 'missing'))
    assert p.flush_denormals() == 0


def test_linux_without_glibc_libm_is_untouched(linux, monkeypatch):
    def missing(name):
        raise OSError(f'{name}: cannot open shared object file')
    monkeypatch.setattr(p.ctypes, 'CDLL', missing)
    assert p.flush_denormals() == 0


def test_linux_without_femode_functions_is_untouched(linux, monkeypatch):
    monkeypatch.setattr(p.ctypes, 'CDLL', lambda name: object())
    assert p.flush_denormals() == 0


@pytest.mark.parametrize('libm', [
    FakeLibm(get=-1),              # fegetmode failed
    FakeLibm(mxcsr=0x00011F80),    # reserved bits set: not an MXCSR, wrong layout
])
def test_linux_unexpected_mode_is_never_written(linux, monkeypatch, libm):
    before = libm.mxcsr
    monkeypatch.setattr(p.ctypes, 'CDLL', lambda name: libm)
    assert p.flush_denormals() == 0
    assert libm.mxcsr == before


@pytest.mark.parametrize('libm', [FakeLibm(set_=-1), FakeLibm(keep=False)])
def test_linux_unconfirmed_switch_reports_zero(linux, monkeypatch, libm):
    monkeypatch.setattr(p.ctypes, 'CDLL', lambda name: libm)
    assert p.flush_denormals() == 0


def fake_engine_modules(monkeypatch, tmp_path, calls):
    import types
    for name in ('PP-OCRv6_medium_det', 'korean_PP-OCRv5_mobile_rec'):
        for filename in ('inference.json', 'inference.pdiparams', 'inference.yml'):
            (tmp_path / name).mkdir(exist_ok=True)
            (tmp_path / name / filename).write_bytes(b'')
    cv2 = types.ModuleType('cv2')
    cv2.setNumThreads = lambda n: calls.append('cv2')
    paddleocr = types.ModuleType('paddleocr')
    paddleocr.PaddleOCR = lambda **options: calls.append('engine') or 'engine'
    monkeypatch.setitem(p.sys.modules, 'cv2', cv2)
    monkeypatch.setitem(p.sys.modules, 'paddleocr', paddleocr)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'PADDLE_PDX_CACHE_HOME'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize('platform, order', [
    ('linux', ['flush', 'cv2', 'engine']),
    ('win32', ['cv2', 'engine', 'flush']),
])
def test_engine_flushes_before_threads_on_linux_after_init_on_windows(monkeypatch, tmp_path,
                                                                      platform, order):
    calls = []
    fake_engine_modules(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(p.sys, 'platform', platform)
    monkeypatch.setattr(p, 'flush_denormals', lambda: calls.append('flush') or 0)
    assert p.initialize_engine(weights_dir=tmp_path, cpu_threads=2) == 'engine'
    assert calls == order


def test_engine_starts_when_ftz_is_unavailable_on_linux(monkeypatch, tmp_path):
    calls = []
    fake_engine_modules(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(p.sys, 'platform', 'linux')
    monkeypatch.setattr('platform.machine', lambda: 'aarch64')
    assert p.initialize_engine(weights_dir=tmp_path, cpu_threads=2) == 'engine'
    assert calls == ['cv2', 'engine']


@pytest.mark.skipif(p.sys.platform != 'win32', reason='MSVC CRT control word')
def test_flush_denormals_switches_the_calling_thread():
    import ctypes
    crt = ctypes.CDLL('ucrtbase')
    crt._controlfp.argtypes = (ctypes.c_uint, ctypes.c_uint)
    crt._controlfp.restype = ctypes.c_uint
    previous = crt._controlfp(0, 0) & p._MCW_DN
    try:
        assert p.flush_denormals() >= 1
        assert crt._controlfp(0, 0) & p._MCW_DN == p._DN_FLUSH
    finally:
        crt._controlfp(previous, p._MCW_DN)
