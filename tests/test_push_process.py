import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object lifecycle')
@pytest.mark.parametrize('exit_mode', ['normal', 'error', 'interrupt', 'terminate'])
def test_push_job_reaps_descendants(tmp_path, exit_mode):
    import win32api
    import win32event
    child = tmp_path / 'child.py'
    ready = tmp_path / 'ready'
    release = tmp_path / 'release'
    child.write_text("import os,time\nfrom pathlib import Path\n"
                     "assert os.environ.get('FOR_DISABLE_CONSOLE_CTRL_HANDLER') == '1'\n"
                     f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                     "time.sleep(90)\n")
    owner = tmp_path / 'owner.py'
    owner.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
                     f"p=subprocess.Popen([sys.executable,{str(child)!r}])\n"
                     f"while not Path({str(release)!r}).exists(): time.sleep(.02)\n"
                     f"sys.exit({7 if exit_mode == 'error' else 0})\n")
    launcher = tmp_path / 'launcher.py'
    project = Path(__file__).resolve().parents[1]
    launcher.write_text("import sys,signal\n"
                        f"sys.path.insert(0,{str(project)!r})\n"
                        "signal.signal(signal.SIGBREAK,signal.default_int_handler)\n"
                        "from tcd_prg.scripts.push_process import run_push_process\n"
                        f"run_push_process([sys.executable,{str(owner)!r}],cwd={str(tmp_path)!r})\n")
    proc = subprocess.Popen([sys.executable, str(launcher)], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handle = None
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists()
        handle = win32api.OpenProcess(0x100000, False, int(ready.read_text()))
        if exit_mode == 'terminate':
            proc.terminate()
        elif exit_mode == 'interrupt':
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        else:
            release.touch()
        proc.wait(timeout=12)
        assert win32event.WaitForSingleObject(handle, 5000) == win32event.WAIT_OBJECT_0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if handle is not None:
            handle.Close()


def test_loader_iterators_shutdown_on_exception():
    from tcd_prg.trainers.push_workers import managed_push_workers, PushDataLoader
    with pytest.raises(RuntimeError, match='injected'):
        with managed_push_workers():
            loader = PushDataLoader(list(range(8)), batch_size=2, num_workers=1, persistent_workers=True)
            iterator = iter(loader)
            assert next(iterator).tolist() == [0, 1]
            workers = list(iterator._workers)
            raise RuntimeError('injected')
    assert all(not worker.is_alive() for worker in workers)


def test_loader_defers_interrupt_until_iterator_is_owned(monkeypatch):
    from torch.utils.data import DataLoader
    from tcd_prg.trainers.push_workers import managed_push_workers, PushDataLoader

    class Iterator:
        stopped = False

        def _shutdown_workers(self):
            self.stopped = True

    iterator = Iterator()

    def interrupt_during_construction(unused_loader):
        signal.raise_signal(signal.SIGINT)
        return iterator

    monkeypatch.setattr(DataLoader, '__iter__', interrupt_during_construction)
    with pytest.raises(KeyboardInterrupt):
        with managed_push_workers():
            iter(PushDataLoader([1], batch_size=1))
    assert iterator.stopped
