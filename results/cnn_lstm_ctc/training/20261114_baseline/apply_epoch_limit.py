"""Apply this run's requested four-epoch limit without losing its active epoch."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

run = Path(__file__).resolve().parent
project = run.parents[3]
request_path = run / "epoch_limit_request.json"
request = json.loads(request_path.read_text(encoding="utf-8"))
active = run / request["active_dataset"]

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def update(state, **details):
    request.update(state=state, **details)
    temporary = request_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(request, indent=2), encoding="utf-8")
    temporary.replace(request_path)

try:
    update("waiting_for_current_epoch")
    print("Four-epoch limit scheduled; waiting for the current epoch checkpoint.", flush=True)
    while True:
        status = read_json(active / "status.json")
        if status["status"] in ("failed", "blocked_invalid_inputs", "interrupted"):
            raise RuntimeError("Training stopped independently; no automatic restart: " + str(status))
        if status["status"] in ("complete", "early_stopped"):
            break
        history = active / "history.csv"
        if history.exists():
            import csv
            with history.open(encoding="utf-8", newline="") as stream:
                completed = list(csv.DictReader(stream))
            if completed and int(completed[-1]["epoch"]) >= request["restart_after_completed_epoch"]:
                stop = active / "STOP"
                marker = "epoch-limit-controller: maximum 4 epochs"
                with stop.open("x", encoding="utf-8") as stream:
                    stream.write(marker)
                update("waiting_for_clean_pause")
                while True:
                    status = read_json(active / "status.json")
                    if status["status"] == "interrupted":
                        if status.get("reason") != "STOP file requested pause":
                            raise RuntimeError("Training stopped for another reason: " + str(status))
                        break
                    if status["status"] in ("complete", "early_stopped"):
                        break
                    if status["status"] == "failed":
                        raise RuntimeError("Training failed while pausing: " + str(status))
                    time.sleep(1)
                if stop.read_text(encoding="utf-8") != marker:
                    raise RuntimeError("STOP marker changed; leaving it in place")
                stop.unlink()
                break
        time.sleep(2)
    update("running_with_four_epoch_limit")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "src.training.ctc_training", "--output", str(run),
               "--epochs", "4", "--patience", "3", "--threads", "4", "--resume"]
    subprocess.run(command, cwd=project, env=env, check=True)
    update("finished")
except BaseException as error:
    update("failed", reason=str(error))
    raise
