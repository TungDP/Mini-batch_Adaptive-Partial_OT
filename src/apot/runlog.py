"""Run directories, names and logs.

Safety rules (lessons from the m-KOT runners, which deleted finished runs whenever a launcher
rebuilt a run name slightly differently):
  * the trainer, not the launcher, decides whether a run is done: ``done.json`` exists;
  * a finished run is never touched unless --overwrite is passed;
  * only an unfinished run directory is wiped on (re)start;
  * a lock file with the owner PID keeps two processes off the same run;
  * numbers are formatted by one function, so 1, 1.0 and "1" give the same name.
"""
import atexit
import csv
import json
import os
import shutil
import time

import numpy as np


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.6g" % float(v)
    return str(v)


def run_name(fields):
    return "_".join(f"{k}{fmt(v)}" for k, v in fields if v is not None and v != "")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


class Run:
    def __init__(self, root, name):
        self.name = name
        self.dir = os.path.join(root, name)
        self.lock = self.dir + ".lock"
        self._log = None
        self._t0 = time.time()

    @property
    def done(self):
        return os.path.exists(os.path.join(self.dir, "done.json"))

    def claim(self):
        """Take the run's lock; False if another live process holds it."""
        os.makedirs(os.path.dirname(self.lock), exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                atexit.register(self.release)
                return True
            except FileExistsError:
                try:
                    pid = int(open(self.lock).read().strip() or -1)
                except (OSError, ValueError):
                    pid = -1
                if _pid_alive(pid):
                    return False
                os.remove(self.lock)  # stale lock of a dead process
        return False

    def release(self):
        try:
            if int(open(self.lock).read().strip()) == os.getpid():
                os.remove(self.lock)
        except (OSError, ValueError):
            pass

    def start(self, config, overwrite=False):
        if self.done and not overwrite:
            raise RuntimeError(f"refusing to restart finished run {self.dir}")
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1, default=str)
        self._log = open(os.path.join(self.dir, "log.txt"), "a")
        self._t0 = time.time()

    def log(self, msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self._log:
            self._log.write(line + "\n")
            self._log.flush()

    def row(self, fname, d):
        path = os.path.join(self.dir, fname)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            if new:
                w.writeheader()
            w.writerow(d)

    def finish(self, summary):
        summary = dict(summary, wall_seconds=round(time.time() - self._t0, 1))
        tmp = os.path.join(self.dir, "done.json.tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        os.replace(tmp, os.path.join(self.dir, "done.json"))
        self.log(f"done: {json.dumps(summary, default=float)}")


def open_run(root, fields, config, overwrite=False, print_name=False):
    """Common trainer entry: returns a started Run, or None if the run should not proceed
    (already done, held by another process, or --print_name)."""
    run = Run(root, run_name(fields))
    if print_name:
        print(run.dir)
        return None
    if run.done and not overwrite:
        print(f"[skip] done: {run.dir}", flush=True)
        return None
    if not run.claim():
        print(f"[skip] running elsewhere: {run.dir}", flush=True)
        return None
    run.start(config, overwrite=overwrite)
    return run


class MassLog:
    """Per-batch transport diagnostics, written to batches.csv and summarised in done.json."""

    def __init__(self, run, every=1):
        self.run, self.every = run, every
        self.s, self.ov, self.cross, self.ms = [], [], [], []

    def add(self, step, P, info, C, ms, ys=None, yt=None, k=0):
        s = float(P.sum())
        row = {"step": step, "k": k, "s": round(s, 6), "s_req": info.get("s_req", ""),
               "lam": info.get("lam", ""), "cost": float((P * C).sum()), "ms": round(ms, 3)}
        self.s.append(s)
        self.ms.append(ms)
        if ys is not None and yt is not None:
            from .overlap import batch_overlap
            ncls = int(max(ys.max(), yt.max())) + 1
            ov = batch_overlap(ys, yt, ncls)
            cross = float(P[ys[:, None] != yt[None, :]].sum()) / max(s, 1e-12)
            row.update(ov=round(ov, 6), cross=round(cross, 6))
            self.ov.append(ov)
            self.cross.append(cross)
        if step % self.every == 0:
            self.run.row("batches.csv", row)

    def summary(self):
        out = {"mean_s": float(np.mean(self.s)) if self.s else None,
               "mean_s_late": float(np.mean(self.s[len(self.s) // 2:])) if self.s else None,
               "sd_s": float(np.std(self.s)) if self.s else None,
               "mean_plan_ms": float(np.mean(self.ms)) if self.ms else None}
        if self.ov:
            from scipy.stats import spearmanr
            s, ov = np.asarray(self.s), np.asarray(self.ov)
            out["mean_ov"] = float(ov.mean())
            out["mean_cross"] = float(np.mean(self.cross))
            out["rho_s_ov"] = float(spearmanr(s, ov)[0]) if s.std() > 1e-9 and ov.std() > 1e-9 else None
        return out
