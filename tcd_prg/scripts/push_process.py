"""Windows process-tree ownership for the PUSH launcher only."""
import os
import signal
import subprocess

SUPERVISED = 'TCD_PUSH_SUPERVISED'


def run_push_process(command, *, cwd):
    if os.name != 'nt':
        return subprocess.run(command, cwd=cwd, check=True)
    import win32api
    import win32con
    import win32event
    import win32job
    import win32process

    job = win32job.CreateJobObject(None, '')
    limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    limits['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
    process = thread = None
    previous = {signal.SIGBREAK: signal.signal(signal.SIGBREAK, signal.default_int_handler)}
    try:
        startup = win32process.STARTUPINFO()
        startup.dwFlags |= win32con.STARTF_USESTDHANDLES
        startup.hStdInput = win32api.GetStdHandle(-10)
        startup.hStdOutput = win32api.GetStdHandle(-11)
        startup.hStdError = win32api.GetStdHandle(-12)
        # Intel Fortran runtime otherwise intercepts console events and aborts
        # before Python finally blocks (forrtl error 200). Scope this to C.
        env = dict(os.environ, **{SUPERVISED: '1', 'FOR_DISABLE_CONSOLE_CTRL_HANDLER': '1'})
        process, thread, pid, _ = win32process.CreateProcess(
            None, subprocess.list2cmdline(command), None, None, True,
            win32con.CREATE_SUSPENDED | win32con.CREATE_NEW_PROCESS_GROUP,
            env, str(cwd), startup)
        # Assign before the child can spawn workers: no unowned startup window.
        win32job.AssignProcessToJobObject(job, process)
        win32process.ResumeThread(thread)
        thread.Close()
        thread = None
        try:
            while win32event.WaitForSingleObject(process, 200) == win32event.WAIT_TIMEOUT:
                pass
        except KeyboardInterrupt:
            # Repeated Ctrl+C must not interrupt teardown halfway through.
            for sig in (signal.SIGINT, signal.SIGBREAK):
                previous.setdefault(sig, signal.getsignal(sig))
                signal.signal(sig, signal.SIG_IGN)
            try:
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            except OSError:
                pass
            win32event.WaitForSingleObject(process, 5000)
            raise
        code = win32process.GetExitCodeProcess(process)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    finally:
        # Also covers child exceptions and launcher termination. The job handle
        # is non-inheritable; workers cannot keep the ownership handle alive.
        job.Close()
        if process is not None:
            if thread is not None:  # Assignment failed while child was suspended.
                win32process.TerminateProcess(process, 1)
            win32event.WaitForSingleObject(process, 5000)
            process.Close()
        if thread is not None:
            thread.Close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
