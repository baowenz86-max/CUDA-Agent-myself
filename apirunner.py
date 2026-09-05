import subprocess
import os
import sys
#保存当前工作目录
import shutil
from pathlib import Path
from datetime import datetime
#数据
import argparse
from llm_agent import run_agent

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover - optional dependency for local environments
    load_dataset = None

DATASET_NAME = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"

WORKDIR = "./agent_workdir"
MAX_RETRY = 5

ENV = os.environ.copy()

# 固定 CUDA 12.6 和 GCC/G++ 13，确保 runner 启动的所有子进程
# （编译、验证、profiling 和产物生成）使用同一套工具链。
CUDA_HOME = "/public/app/cuda/12.6"
HOST_CC = "/usr/bin/gcc-13"
HOST_CXX = "/usr/bin/g++-13"

ENV["CUDA_HOME"] = CUDA_HOME
ENV["CUDACXX"] = f"{CUDA_HOME}/bin/nvcc"
ENV["CC"] = HOST_CC
ENV["CXX"] = HOST_CXX
ENV["CUDAHOSTCXX"] = HOST_CXX
ENV["PATH"] = f"{CUDA_HOME}/bin{os.pathsep}{ENV.get('PATH', '')}"
ENV["LD_LIBRARY_PATH"] = (
    f"{CUDA_HOME}/lib64{os.pathsep}{ENV.get('LD_LIBRARY_PATH', '')}"
)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="dataset task id"
    )

    return parser.parse_args()


def load_task(task_id):
    print(f"Loading dataset task {task_id}")

    arrow_file = Path(
        "datasets/BytedTsinghua-SIA___cuda-agent-ops-6_k/"
        "default/0.0.0/44a734c78c947bfcba5189cbfd13f57a6d29a698/"
        "cuda-agent-ops-6_k-train.arrow"
    )

    if arrow_file.exists():
        print(f"Using local Arrow dataset: {arrow_file}")
        dataset = Dataset.from_file(str(arrow_file))
        sample = dataset[task_id]
    else:
        if load_dataset is None:
            raise RuntimeError("datasets package is not installed.")

        print("Local Arrow dataset not found; trying Hugging Face Hub.")
        remote_dataset = load_dataset(DATASET_NAME)
        sample = remote_dataset["train"][task_id]

    print("operators:")
    print(sample["ops"])

    return sample


def prepare_model(sample):

    model_path = os.path.join(
        WORKDIR,
        "model.py"
    )


    with open(
        model_path,
        "w"
    ) as f:

        f.write(
            sample["code"]
        )


    print(
        "[OK] model.py generated"
    )


def reset_generated_workspace():
    """开始新任务前，清理上一任务生成的 CUDA 源码和编译产物。"""
    kernel_dir = Path(WORKDIR) / "kernels"
    generated_suffixes = {".cu", ".cpp", ".ptx", ".cubin", ".sass"}

    if kernel_dir.exists():
        for path in kernel_dir.iterdir():
            if path.is_file() and path.suffix in generated_suffixes:
                path.unlink()
                print(f"[CLEAN] removed: {path}")

    for filename in ("model_new.py", "cuda_extension.so"):
        path = Path(WORKDIR) / filename
        if path.exists():
            path.unlink()
            print(f"[CLEAN] removed: {path}")


def run(cmd):

    print("\n==========")
    print(cmd)
    print("==========")


    result = subprocess.run(
        cmd,
        cwd=WORKDIR,
        env=ENV,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )


    print(result.stdout)


    return (
        result.returncode == 0,
        result.stdout
    )


def send_feedback(error_log):

    prompt = f"""
当前任务：

model.py:
{open(
    os.path.join(WORKDIR,"model.py")
).read()}

错误：

{error_log}

请修改CUDA实现。

你正在优化 CUDA kernel。

刚刚生成的代码运行失败。

错误日志如下：

----------------
{error_log}
----------------


请：

1. 分析错误原因
2. 修改 model_new.py 或 kernels/
3. 不修改 utils/
4. 不修改 binding.cpp
5. 修改后重新运行：

bash utils/compile.sh

直到编译成功。
"""


    return run_agent(prompt, WORKDIR)

def generate_cuda():

    prompt = """
你现在是CUDA kernel优化agent。

请执行：

1. 阅读 SKILL.md
2. 阅读 model.py
3. 生成 CUDA 优化版本

要求：

- 修改 model_new.py
- 在 kernels/ 中生成CUDA kernel

禁止：

- 修改 utils
- 修改 binding.cpp
- 修改 binding_registry.h

完成后：

运行:
bash utils/compile.sh

如果失败继续修复。
"""


    return run_agent(prompt, WORKDIR)


def compile_with_feedback(max_retry=3):

    for i in range(max_retry):

        success, log = run(
            "bash utils/compile.sh"
        )

        if success:
            print("Compile success")
            return True


        print(log)

        send_feedback(log)


    return False


def verify_with_feedback(max_retry=3):

    for i in range(max_retry):

        success, log = run(
            "python3 -m utils.verification"
        )


        if success:

            print("Verification success")
            return True


        print("Verification failed")

        send_feedback(
            f"""
Verification failed:

{log}

请检查CUDA kernel计算逻辑。
重点检查：

1. tensor shape
2. index
3. dtype
4. reduction

重新修改。
"""
        )


def profile():

    return run(
        "python3 -m utils.profiling"
    )


def generate_kernel_artifacts():
    """
    调用工具脚本生成 .ptx 和 .sass，并保存在 agent_workdir/kernels 中。
    """

    repo_root = Path(__file__).resolve().parent

    for script_name in [
        "tools/cu_to_ptx.py",
        "tools/cu_to_sass.py",
    ]:

        script_path = repo_root / script_name

        print(f"\n[RUN] {script_name}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"[WARN] {script_name} failed")


def save_result(task_id=0, logs=None):
    """
    保存一次成功生成的CUDA程序
    """

    save_dir = Path(
        f"./results/task_{task_id:04d}"
    )

    # 每次保存都生成当前轮次的完整快照。如果目标目录已存在，
    # 先删除旧快照，避免不同名的旧 kernel/编译产物残留。
    if save_dir.exists():
        shutil.rmtree(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=False
    )


    # 保存 model.py
    files = [
        "model.py",
        "model_new.py",
    ]

    for file in files:

        src = Path(WORKDIR) / file

        if src.exists():

            shutil.copy(
                src,
                save_dir / file
            )


    # 保存 kernels
    kernel_src = Path(WORKDIR) / "kernels"

    kernel_dst = save_dir / "kernels"


    if kernel_src.exists():

        shutil.copytree(
            kernel_src,
            kernel_dst
        )


    # 保存日志
    if logs:

        for name, content in logs.items():

            with open(
                save_dir / name,
                "w"
            ) as f:

                f.write(content)



    # 保存时间信息

    with open(
        save_dir / "meta.txt",
        "w"
    ) as f:

        f.write(
            f"""
task_id: {task_id}
time: {datetime.now()}

status: success
"""
        )


    print(
        f"[SAVE] result saved to {save_dir}"
    )


def main():

    args = parse_args()


    print(
        "Start CUDA Agent"
    )


    # 1.读取任务
    sample = load_task(
        args.task_id
    )


    # 2. 清理上一任务的生成文件，并写入当前任务的 model.py
    reset_generated_workspace()

    prepare_model(
        sample
    )


    # 3.CUDA Agent流程

    print("Start CUDA Agent")

    for retry in range(MAX_RETRY):

        generate_cuda()

        if compile_with_feedback():
            if verify_with_feedback():

                success, profile_log = run(
                    "python3 -m utils.profiling"
                )

                generate_kernel_artifacts()

                save_result(
                    task_id=args.task_id,
                    logs={
                        "profile.log": profile_log
                    }
                )

                return
    print("Task failed")


if __name__=="__main__":
    main()
