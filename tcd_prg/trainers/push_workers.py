"""Explicit ownership of C DataLoader iterators on normal/error/interrupt exit."""
from contextlib import contextmanager
from contextvars import ContextVar
import signal
import threading
import weakref
from torch.utils.data import DataLoader

_iterators = ContextVar('push_loader_iterators', default=None)


def ignore_worker_interrupts(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


class PushDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('worker_init_fn', ignore_worker_interrupts)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # PyTorch can leave _MultiProcessingDataLoaderIter only partly
        # initialized if KeyboardInterrupt is raised inside its constructor;
        # its destructor then crashes while trying to clean missing fields.
        # Defer the exception until construction finishes and the iterator is
        # registered with our owner, so the normal finally path can shut it down.
        deferred = []
        previous = {}
        can_manage_signals = threading.current_thread() is threading.main_thread()
        if can_manage_signals:
            def remember(signum, unused_frame):
                deferred.append(signum)

            for sig in (signal.SIGINT, getattr(signal, 'SIGBREAK', None)):
                if sig is not None:
                    previous[sig] = signal.signal(sig, remember)
        try:
            iterator = super().__iter__()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        owned = _iterators.get()
        if owned is not None:
            owned.add(iterator)
        if deferred:
            raise KeyboardInterrupt
        return iterator


@contextmanager
def managed_push_workers():
    owned = weakref.WeakSet()
    token = _iterators.set(owned)
    previous = {}
    if hasattr(signal, 'SIGBREAK'):
        previous[signal.SIGBREAK] = signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        yield
    finally:
        previous[signal.SIGINT] = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        try:
            for iterator in list(owned):
                shutdown = getattr(iterator, '_shutdown_workers', None)
                if shutdown is not None:
                    shutdown()
        finally:
            _iterators.reset(token)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
