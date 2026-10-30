"""Work queue for long Claude runs on the subscription, resumed by cron after usage-limit pauses.

  python -m qaemb.queue run       # run queued commands in order until one pauses or fails (cron calls this)
  python -m qaemb.queue status    # progress, pause/failure state, last log lines

outputs/qaemb/queue/queue.txt holds one command per line: the arguments after `python -m`, e.g.
  qaemb.run stage=embed +exp=[D0_smoke10,Q0_all,ANS_claude_opus_cli,M_all,G1,R1] tag=claude-cli
Lines starting with # are skipped. A command that exits 0 is recorded in done.txt and never rerun.
Exit code 75 (usage limit, see answerers/claude_cli.py) pauses the queue until the reset time in paused_until.
Any other non-zero exit stops the queue and writes FAILED; nothing runs again until that file is removed.
Only one runner works at a time (a lock file), so overlapping cron ticks are harmless. Log: queue.log.
"""
import fcntl
import shlex
import subprocess
import sys
import time
from datetime import datetime

from .answerers.claude_cli import PAUSE_FILE, PAUSED
from .config import ROOT

Q = ROOT / "outputs" / "qaemb" / "queue"
QUEUE, DONE, FAILED, LOG = Q / "queue.txt", Q / "done.txt", Q / "FAILED", Q / "queue.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d-%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pending() -> list[str]:
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    lines = [s.strip() for s in QUEUE.read_text().splitlines()] if QUEUE.exists() else []
    return [s for s in lines if s and not s.startswith("#") and s not in done]


def run() -> None:
    Q.mkdir(parents=True, exist_ok=True)
    lock = open(Q / "queue.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # another runner is working
    if FAILED.exists():
        return log(f"stopped: {FAILED.read_text().strip()} (remove {FAILED} to continue)")
    if PAUSE_FILE.exists():
        until = int(PAUSE_FILE.read_text())
        if time.time() < until + 60:
            return  # quiet: cron ticks while paused are expected
        PAUSE_FILE.unlink()
        log("usage window reset; resuming")
    for cmd in pending():
        log(f"start: {cmd}")
        with open(LOG, "a") as f:
            rc = subprocess.run([sys.executable, "-m", *shlex.split(cmd)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            with open(DONE, "a") as f:
                f.write(cmd + "\n")
            log(f"done: {cmd}")
        elif rc == PAUSED:
            until = int(PAUSE_FILE.read_text()) if PAUSE_FILE.exists() else None
            return log(f"paused until {datetime.fromtimestamp(until):%Y-%m-%d-%H:%M:%S}" if until else "paused")
        else:
            FAILED.write_text(f"exit code {rc}: {cmd}\n")
            return log(f"FAILED (exit {rc}): {cmd}; queue stopped, see the log above")
    log("queue empty")


def status() -> None:
    left = pending()
    done = DONE.read_text().splitlines() if DONE.exists() else []
    print(f"done {len(done)}, pending {len(left)}")
    if left:
        print(f"next: {left[0]}")
    if PAUSE_FILE.exists():
        print(f"paused until {datetime.fromtimestamp(int(PAUSE_FILE.read_text())):%Y-%m-%d-%H:%M:%S}")
    if FAILED.exists():
        print(f"FAILED: {FAILED.read_text().strip()}")
    if LOG.exists():
        print("--- last log lines ---\n" + "\n".join(LOG.read_text().splitlines()[-15:]))


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
