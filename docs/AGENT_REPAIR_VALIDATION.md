# 三阶段生成与修复改进：平台验证

本地测试验证工作流和响应处理，不代表 CUDA 结果或成功率已经提升。真实重跑应在原学校平台、原数据集和工具链下进行。

## 修改内容

- Python 阶段生成只读 `custom_ops_contract.json` 元数据，冻结算子名称、参数顺序、标量/Tensor 类型、dtype/device/shape/layout 和返回结构。CUDA、绑定阶段按该契约实现；修复若确需变更接口，可同时提交新的 `operator_contract` 与所有受影响源码。
- Conv/ConvTranspose 使用 cuDNN，由 C++ 阶段实现。按参考实际调用决定是否使用 bias；筛选必须保序，转置不能假定连续，CPU 指针不能传给 CUDA kernel，整数指针统一类型。
- 源码回滚同时恢复该候选的验证报告、诊断、失败输入及算子契约。修复上下文提供基准候选 ID、源码哈希和历史修改。
- 按 sources/compile/interface/runtime/structure/numerical/memory 分类反馈。结构进展可以保留，已通过的样例和已匹配检查点不得退化；首个失败输入变化时，额外重放旧输入比较检查点。
- 响应格式、文件范围、契约校验失败可在原调用阶段纠正两次；总共最多三次请求。传输/I/O 错误不走格式纠正。所有变更完整校验后才应用，截断响应拒绝应用。
- 正常 forward 的五个固定样例仍是通过标准。新增诊断仅提供证据，不放宽容限，也不允许通过修改 debug_forward 获得成功。

## 在学校平台执行

先将源码改动同步到平台，保留原有 `logs/task_0011` 至 `logs/task_0030`，不要将新运行写回原基准目录。

```bash
python -m unittest discover -s tests -v
CUDA_AGENT_GPU_TESTS=1 python -m unittest tests.test_tensor_diagnostics -v
CUDA_AGENT_LOG_DIR=logs/agent-repair-v1 python main.py --task-start 11 --task-end 30
python -m scripts.compare_runs --before logs --after logs/agent-repair-v1 --output logs/agent-repair-v1/comparison.json
```

以上 `python` 应使用平台现有虚拟环境；模型、thinking、seed、GPU、并发数和 CUDA 数值设置保持与基准相同。命令不会在本地发起平台任务。

## 对比指标

`comparison.json` 按任务选取最新已完成报告，并列出选中的运行路径和缺失任务。缺失任务不进入成功率分母，因此必须检查 `missing_tasks` 为空，再比较完整批次。

- `success_rate`：完整工作流成功率；原 task11–30 基准为 12/20。
- `first_candidate_pass_rate`：首次候选即通过的比例；首次生成失败也计为未通过。
- `candidate_failures`：分类失败次数，包含中间候选；一个任务可能贡献多次失败。
- `api_calls` / `api_failed_calls` / `format_correction_calls`：包括格式纠正请求的调用成本。
- `effective_repair_rate`：有明确 API 阶段记录的已评估修复中，被保留为最佳候选的比例。旧日志缺少归属信息时显示 null，不推测。
- `summed_task_seconds`：任务耗时之和，不是并行批次墙钟耗时。

同时审查 task13/14/20/24 的首个错误检查点、task19/21/27 的绑定签名、task29 的返回类型和 task30 的设备路径。cuDNN 与精度设置仍需平台实测，不能仅凭提示词认定数值一致。

旧日志读取兼容；新生成必须带算子契约。动态参数和缺少文档的绑定不做静态猜测，仍依靠真实调用和验证。
