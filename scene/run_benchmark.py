# Sequential benchmark runner (detached): 4 stacks x {s1s2, s5} + report.
import subprocess, sys, datetime

PY = r"D:\mujoco\.venv\Scripts\python.exe"
TS = r"D:\mujoco\DOG4.6_description\tests\compare_stacks.py"
LOG = r"D:\mujoco\DOG4.6_description\results\benchmark_run.log"

with open(LOG, "w", buffering=1, encoding="utf-8") as log:
    log.write("START %s\n" % datetime.datetime.now().isoformat())
    for k in ("mit", "eth", "cvxwbc", "slqjt"):
        for scen in ("s1s2", "s5"):
            log.write("\n>>> %s %s\n" % (k, scen))
            r = subprocess.run([PY, "-u", TS, "run", k, scen],
                               stdout=log, stderr=subprocess.STDOUT)
            log.write("<<< exit %d\n" % r.returncode)
    log.write("\n>>> report\n")
    subprocess.run([PY, "-u", TS, "report"], stdout=log, stderr=subprocess.STDOUT)
    log.write("ALL BENCHMARK RUNS DONE %s\n" % datetime.datetime.now().isoformat())
