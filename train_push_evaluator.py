"""Project-root launcher for independent PUSH evaluator training."""

import os
import sys
import signal
from pathlib import Path

if __name__ == '__mp_main__':
    # Spawned workers execute this before unpickling/importing torch. Let the
    # owning trainer shut them down instead of interrupting DLL initialization.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)

if __name__ == "__main__":
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        if os.name == 'nt' and os.environ.get('TCD_PUSH_SUPERVISED') != '1':
            from tcd_prg.scripts.push_process import run_push_process
            run_push_process([sys.executable, *sys.argv], cwd=Path(__file__).resolve().parent)
        else:
            from tcd_prg.scripts.train_push_evaluator import main
            main()
    except KeyboardInterrupt:
        print('PUSH training interrupted; worker cleanup completed.', flush=True)
        raise SystemExit(130)
