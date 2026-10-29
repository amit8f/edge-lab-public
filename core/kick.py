"""Morning starter for the all-day jobs. GitHub delays scheduled runs by hours on busy mornings (on Mon
2026-10-26 the 07:50 UTC listen cron ran at 14:35 and the 13:20 UTC watch cron at 19:21), but a job queued
with `gh workflow run` starts within seconds. So one small job stays up around the clock: it checks once a
minute, starts each workflow below once per trading day at its start time, and queues its own next job before
its 6-hour limit. The workflows' own crons stay as a backup; their concurrency groups keep any duplicate
waiting, and a job started after its session simply exits. Starting jobs only, nothing else.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HOLIDAYS = Path(__file__).resolve().parent.parent / "deploy" / "nyse_holidays.txt"
# workflow -> (first ET time to start it, last ET time it is still worth starting)
JOBS = {"listen.yml": (dtime(3, 55), dtime(19, 30)),
        "nx3-live.yml": (dtime(9, 15), dtime(15, 30)),
        "watch.yml": (dtime(9, 20), dtime(15, 30)),
        "scouts.yml": (dtime(9, 20), dtime(15, 30))}
# workflow -> [(first ET time, last ET time, minutes between starts)]: short jobs started again and again.
# The dashboard's own */10 cron never fired on its first day (2026-10-29), so the kick starts it instead: in
# market hours until the NX3 settle has landed, and early each morning (weekends too) for the nightly logs.
EVERY = {"dashboard.yml": [(dtime(9, 25), dtime(16, 45), 10), (dtime(5, 0), dtime(5, 30), 60)]}


def holidays(path=HOLIDAYS):
    out = set()
    for ln in Path(path).read_text().splitlines():
        head = ln.split("|")[0].strip()
        if head and not head.startswith("#"):
            out.add(date.fromisoformat(head))
    return out


def due(now, runs, closed):
    """Workflows to start now. now: an aware datetime; runs: {workflow: [created_at as aware datetimes]};
    closed: NYSE full-day closures. A workflow is due inside its window on a trading day when none of its runs
    was created on that ET day at or after its start time (an earlier one belongs to the night before). An EVERY
    workflow is due inside one of its windows when none of its runs was created in the last `gap` minutes."""
    et = now.astimezone(ET)
    out = []
    for wf, (lo, hi) in JOBS.items():
        if et.weekday() >= 5 or et.date() in closed or not lo <= et.time() < hi:
            continue
        start = datetime.combine(et.date(), lo, ET)
        if not any(c.astimezone(ET) >= start for c in runs.get(wf, [])):
            out.append(wf)
    for wf, windows in EVERY.items():
        for lo, hi, gap in windows:
            market = lo.hour >= 9          # market-hours windows skip weekends and closures; the early one doesn't
            if market and (et.weekday() >= 5 or et.date() in closed) or not lo <= et.time() < hi:
                continue
            if not any(now - c < timedelta(minutes=gap) for c in runs.get(wf, [])):
                out.append(wf)
    return out


def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def created(repo, wf):
    rows = json.loads(gh("run", "list", "-R", repo, "--workflow", wf, "--limit", "20", "--json", "createdAt"))
    return [datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--minutes", type=float, required=True)
    a = ap.parse_args(argv)
    closed = holidays()
    stop = time.monotonic() + a.minutes * 60
    while time.monotonic() < stop:
        try:
            for wf in due(datetime.now(ET), {wf: created(a.repo, wf) for wf in (*JOBS, *EVERY)}, closed):
                gh("workflow", "run", wf, "-R", a.repo, "--ref", "main")
                print(f"{datetime.now(ET):%Y-%m-%d %H:%M} ET started {wf}")
        except subprocess.CalledProcessError as e:  # a failed call is retried next minute
            print(f"::warning::gh failed ({e.returncode})", file=sys.stderr)
        time.sleep(60)


if __name__ == "__main__":
    main()
