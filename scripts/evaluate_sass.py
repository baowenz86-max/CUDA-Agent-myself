"""Rebuild original and recovered CUDA in isolated directories; no model calls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def run(command, directory, env, timeout, name):
    started = time.monotonic()
    with (directory / f"{name}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"returncode": None, "timeout": True, "seconds": time.monotonic()-started}
    return {"returncode": code, "timeout": False, "seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for each run")
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    python = str(args.python.absolute())
    if not Path(python).is_file():
        parser.error("Verification Python does not exist")
    config = json.loads((args.predictions / "generation_config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for position, task_id in enumerate(config["ids"], 1):
        if not task_id.startswith("task_") or not task_id[5:].isdigit():
            raise ValueError(f"Invalid task ID {task_id}")
        print(f"[{position}/{len(config['ids'])}] {task_id} original replay", flush=True)
        archive = args.results / task_id
        seed = 2026 + int(task_id[5:]) * 100
        row = {"id": task_id, "seed": seed}

        def prepare(kind, cuda):
            directory = (args.output / task_id / kind).absolute()
            directory.mkdir(parents=True)
            shutil.copytree(repo / "agent_workdir/utils", directory / "utils",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("binding.cpp", "binding_registry.h"):
                shutil.copy2(repo / "agent_workdir" / name, directory / name)
            for name in ("model.py", "model_new.py"):
                shutil.copy2(archive / name, directory / name)
            (directory / "kernels").mkdir()
            shutil.copy2(archive / "kernels/custom_ops_binding.cpp",
                         directory / "kernels/custom_ops_binding.cpp")
            shutil.copy2(cuda, directory / "kernels/custom_ops.cu")
            return directory

        def evaluate(directory):
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["CUDA_AGENT_RUN_DIR"] = str(directory / ".correctness")
            env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
            result = {"compile": run([python, "-m", "utils.compile"], directory,
                                      env, args.timeout, "compile")}
            if result["compile"]["returncode"] != 0:
                result["status"] = "compile_failed"
                return result
            result["verification"] = run([python, "-m", "utils.verification", "--seed", str(seed)],
                                         directory, env, args.timeout, "verification")
            path = directory / ".correctness/verification.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            result["verification_report"] = report
            passed = result["verification"]["returncode"] == 0 and report.get("status") == "passed"
            result["status"] = "passed" if passed else "verification_failed"
            return result

        try:
            original = prepare("original", archive / "kernels/custom_ops.cu")
            row["original"] = evaluate(original)
            if row["original"]["status"] != "passed":
                row["status"] = "original_replay_failed"
            else:
                prediction = args.predictions / task_id
                generation = json.loads((prediction / "result.json").read_text())
                row["generation"] = generation
                if generation.get("truncated") or not generation.get("format_ok"):
                    row["status"] = "generation_failed"
                else:
                    print(f"  {task_id} recovered compile/verify", flush=True)
                    candidate = prepare("recovered", prediction / "custom_ops.cu")
                    (candidate / ".correctness").mkdir()
                    shutil.copy2(original / ".correctness/reference_fixture.pt",
                                 candidate / ".correctness/reference_fixture.pt")
                    row["recovered"] = evaluate(candidate)
                    row["status"] = row["recovered"]["status"]
        except Exception as exc:
            row.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        (args.output / task_id).mkdir(parents=True, exist_ok=True)
        (args.output / task_id / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"  {task_id}: {row['status']}", flush=True)
    evaluable = [r for r in rows if r.get("original", {}).get("status") == "passed"
                 and r["status"] != "infrastructure_error"]
    summary = {"tasks": len(rows), "evaluable": len(evaluable),
               "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in rows),
               "original_replay_failed": sum(r["status"] == "original_replay_failed" for r in rows),
               "compiled": sum(r.get("recovered", {}).get("compile", {}).get("returncode") == 0 for r in evaluable),
               "functionally_correct": sum(r["status"] == "passed" for r in evaluable),
               "status_by_task": {r["id"]: r["status"] for r in rows}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
