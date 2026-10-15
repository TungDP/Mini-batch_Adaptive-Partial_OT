"""Run an experiment of scripts/experiments.py on the local GPUs.

    python scripts/launch.py E1 --status            # what is done / running / pending
    python scripts/launch.py E1 --dry               # print the commands
    nohup python scripts/launch.py E1 --gpus 0,1,2 > results/_launcher/E1.out 2>&1 &
    python scripts/launch.py E2 --only oh: --skip :curv: --seeds 1

Relaunching is safe: each trainer skips a run whose done.json exists, and a lock file keeps
two processes off the same run.  Run names come from src/apps.py (the trainers' own code), so
the launcher never rebuilds a name by hand.  The job list is fixed when the launcher starts,
so editing experiments.py does not affect a launcher that is already running.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import apps  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402

LOG_DIR = os.path.join(apps.RESULTS, "_launcher")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet under `| head`


def state(job):
    d = apps.run_dir(job.app, job.argv)
    if os.path.exists(os.path.join(d, "done.json")):
        return "done", d
    lock = d + ".lock"
    if os.path.exists(lock):
        try:
            os.kill(int(open(lock).read().strip()), 0)
            return "running", d
        except (OSError, ValueError):
            pass
    return "pending", d


def pid_alive(path):
    try:
        os.kill(int(open(path).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=list(EXPERIMENTS))
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--per_gpu", type=int, default=1, help="concurrent GPU jobs per card (1 for ResNet-50 at m=65)")
    p.add_argument("--cpu_slots", type=int, default=8, help="concurrent CPU-only jobs (flow, color)")
    p.add_argument("--seeds", default=None, help="comma list, overrides the experiment default")
    p.add_argument("--with", dest="with_", default="", help="optional extras: aot, celeba")
    p.add_argument("--only", default="", help="comma list of substrings; keep jobs whose label has any")
    p.add_argument("--skip", default="", help="comma list of substrings; drop jobs whose label has any")
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()

    kw = {"with_": tuple(x for x in a.with_.split(",") if x)}
    if a.seeds:
        kw["seeds"] = tuple(int(s) for s in a.seeds.split(","))
    jobs = EXPERIMENTS[a.exp](**kw)
    if a.only:
        jobs = [j for j in jobs if any(s in j.label for s in a.only.split(","))]
    if a.skip:
        jobs = [j for j in jobs if not any(s in j.label for s in a.skip.split(","))]
    states = [state(j) for j in jobs]
    n = {s: sum(st == s for st, _ in states) for s in ("done", "running", "pending")}
    left = sum(j.hours for j, (st, _) in zip(jobs, states) if st == "pending")
    gpu_slots = len(a.gpus.split(",")) * a.per_gpu
    print(f"{a.exp}: {len(jobs)} jobs  done {n['done']}  running {n['running']}  pending {n['pending']}  "
          f"~{left:.0f} GPU-h pending (~{left / max(1, gpu_slots):.1f} h on {gpu_slots} slots)")
    if a.status or a.dry:
        for j, (st, d) in zip(jobs, states):
            if a.dry and st == "pending":
                print(f"python {os.path.relpath(apps.script(j.app), apps.REPO)} {' '.join(j.argv)}")
            elif a.status:
                print(f"  {st:8s} {j.label:45s} {os.path.relpath(d, apps.RESULTS)}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    pidfile = os.path.join(LOG_DIR, f"{a.exp}.pid")
    if os.path.exists(pidfile) and pid_alive(pidfile):
        raise SystemExit(f"another launcher for {a.exp} is running (pid {open(pidfile).read().strip()})")
    with open(pidfile, "w") as fh:
        fh.write(str(os.getpid()))
    ledger = os.path.join(LOG_DIR, f"{a.exp}.jobs.csv")

    queue = [j for j, (st, _) in zip(jobs, states) if st == "pending"]
    free_gpu = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    free_cpu = a.cpu_slots
    running = []  # (proc, job, slot, t0, logfile)
    try:
        while queue or running:
            for item in list(running):
                proc, job, slot, t0, logf = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                logf.close()
                if slot is None:
                    free_cpu += 1
                else:
                    free_gpu.append(slot)
                new = not os.path.exists(ledger)
                with open(ledger, "a", newline="") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(["label", "returncode", "hours", "run_dir"])
                    w.writerow([job.label, proc.returncode, round((time.time() - t0) / 3600, 3),
                                apps.run_dir(job.app, job.argv)])
                print(f"[{time.strftime('%m-%d %H:%M')}] {'ok  ' if proc.returncode == 0 else 'FAIL'} {job.label}",
                      flush=True)
            for job in list(queue):
                if job.gpu and not free_gpu:
                    continue
                if not job.gpu and free_cpu <= 0:
                    continue
                queue.remove(job)
                slot = free_gpu.pop(0) if job.gpu else None
                if slot is None:
                    free_cpu -= 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot if slot is not None else "",
                           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
                name = job.label.replace(":", "_").replace("/", "-")
                logf = open(os.path.join(LOG_DIR, f"{a.exp}_{name}.log"), "w")
                proc = subprocess.Popen([sys.executable, apps.script(job.app)] + job.argv, cwd=apps.REPO, env=env,
                                        stdout=logf, stderr=subprocess.STDOUT)
                running.append((proc, job, slot, time.time(), logf))
                print(f"[{time.strftime('%m-%d %H:%M')}] start {job.label}  gpu={slot}  pid={proc.pid}", flush=True)
            time.sleep(10)
    finally:
        if os.path.exists(pidfile) and open(pidfile).read().strip() == str(os.getpid()):
            os.remove(pidfile)


if __name__ == "__main__":
    main()
