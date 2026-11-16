"""Apply IAM-only/20-epoch request and document completed epochs automatically."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

run = Path(__file__).resolve().parent
project = run.parents[3]
sys.path.insert(0, str(project))
from src.training.epoch_reporting import generate_report
request_path = run / "dataset_selection_request.json"
request = json.loads(request_path.read_text(encoding="utf-8"))
active = run / "iam"
last_report_token = None

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)

def update(state, **details):
    request.update(state=state, **details)
    write(request_path, request)

def refresh_report(force=False):
    global last_report_token
    history = active / "history.csv"
    if history.exists():
        token = history.stat().st_mtime_ns
        if force or token != last_report_token:
            path = generate_report(active, planned_epochs=request["epochs"])
            last_report_token = token
            print("Epoch report updated:", path, flush=True)

try:
    update("waiting_for_current_epoch")
    print("IAM-only, maximum 20 epochs scheduled. Current epoch will save first.", flush=True)
    refresh_report(force=True)
    while True:
        status = read(active / "status.json")
        if status["status"] != "training":
            raise RuntimeError("Training stopped independently; no automatic restart: " + str(status))
        history = active / "history.csv"
        if history.exists():
            with history.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows and int(rows[-1]["epoch"]) >= request["restart_after_completed_epoch"]:
                refresh_report()
                break
        time.sleep(1)
    marker = "dataset-selection-controller: IAM only, maximum 20 epochs"
    stop = active / "STOP"
    with stop.open("x", encoding="utf-8") as stream:
        stream.write(marker)
    update("waiting_for_clean_pause")
    while True:
        status = read(active / "status.json")
        if status["status"] == "interrupted":
            if status.get("reason") != "STOP file requested pause":
                raise RuntimeError("Training interrupted for another reason: " + str(status))
            break
        if status["status"] != "training":
            raise RuntimeError("Unexpected state during pause: " + str(status))
        time.sleep(1)
    old_request = run / "epoch_limit_request.json"
    deadline = time.monotonic() + 30
    while read(old_request)["state"] != "failed":
        if time.monotonic() >= deadline:
            raise RuntimeError("Old two-dataset supervisor did not acknowledge its paused subprocess")
        time.sleep(1)
    previous = read(old_request)
    previous.update(state="superseded_by_iam_only", superseded_by=request_path.name,
                    reason="Intentional clean pause: IAM only, maximum 20 epochs")
    write(old_request, previous)
    if stop.read_text(encoding="utf-8") != marker:
        raise RuntimeError("STOP marker changed; leaving it in place")
    stop.unlink()
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "src.training.ctc_training", "--datasets", "iam",
               "--output", str(run), "--epochs", str(request["epochs"]),
               "--patience", "3", "--threads", "4", "--resume"]
    child = subprocess.Popen(command, cwd=project, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    update("running_iam_only", training_pid=child.pid)
    print("Resuming IAM only, maximum 20 epochs. Per-epoch reporting enabled.", flush=True)
    while child.poll() is None:
        refresh_report()
        time.sleep(2)
    refresh_report(force=True)
    if child.returncode:
        raise RuntimeError(f"Training exited with code {child.returncode}; see IAM status.json")
    update("finished_iam_only", final_status=read(active / "status.json"))
except BaseException as error:
    update("failed", reason=str(error))
    raise
