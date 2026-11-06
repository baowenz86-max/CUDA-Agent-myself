# CUDA Agent：保留可信进展，减少无效修复

本次完成代码实施、离线回归测试及历史证据回放。没有调用生成模型，没有安装依赖，没有运行真实 CUDA 编译、GPU correctness 或 benchmark。测试中的编译器和 API 均为替身。历史目录未被改写。

## 修改前架构与已复核基线

原调用链：

```text
reference preflight → fixture/task_contract
→ Python + operator_contract → CUDA → C++ binding
→ source checks → compile → compiled interface preflight
→ 五轮 normal-forward verification
→ 失败输入诊断／旧输入 replay
→ candidate snapshot + comparison
→ restore best → classified repair → 重新评估
→ correctness success → profiling → artifacts/report
```

修改前测试为 76 项：64 通过，12 跳过。按 task 81–140 各任务最新已完成报告复核：首候选通过 18/60（30%），最终通过 37/60（61.67%）。这两个数字仍然是历史基线，本次没有产生新的真实成功率。

旧比较器会要求诊断整体状态为 failed/matched，因此 unavailable 中的部分匹配不能被采纳；在部分分支中，最终 mismatch 上升还会否决 checkpoint 推进。评估后的失败没有统一排除 environment/resource。编译模块会覆盖已配置工具链。旧报告主要依赖 Markdown，正确性成功但 profiling 失败未形成独立机器状态。

## 文件改动与原因

| 文件 | 改动及目的 |
|---|---|
| `cuda_agent/candidates.py` | 结构化 progress、带理由的 compare、独立推进审计；保存修复基准和效果；恢复源码、fixture、contract、诊断及 precompile/dependency 证据。 |
| `cuda_agent/repair.py` | 终止类及 binding/contract/shape 路由；规范化根因 fingerprint；记录 hashes、文件增删改、修复结果与未落实标志。 |
| `cuda_agent/api.py` | 校验可选 target_files；无源码修改在应用前拒绝并记录效果；每个修复 API 事件记录 base candidate；上下文加入 contract 驱动 binding 片段。 |
| `agent_workdir/utils/checkpoint_trust.py`（新增） | 不依赖 PyTorch 的 AST 路径对应检查、可安全展开的单 return helper、原生 wrapper/launcher/kernel 调用关系及操作证据检查、实时 metadata 比较。 |
| `agent_workdir/utils/verification.py` | 诊断分开保存整体状态、静态映射和实际 checkpoint 观测；异常保留可信前缀；输入迁移仅在共享存储时 clone。 |
| `agent_workdir/utils/operator_contract.py` | required/optional/default 一致性；C++ 类型及 py::arg helper；可解析的 native binding 静态检查；编译后双向 nullable；模型签名和 state_dict metadata 检查。 |
| `agent_workdir/utils/preflight.py` | 增加 interface-only 扩展 stub，不运行 forward；参考 Conv/ConvTranspose 依赖检查；结构化终止分类；记录编译签名待检查项。 |
| `agent_workdir/utils/dependencies.py`（新增） | 分离 cuDNN header 发现和主库链接探测；支持显式路径、系统、CUDA、NVIDIA Python 包及 PyTorch 目录；生成依赖 fingerprint。 |
| `agent_workdir/utils/compile.py`、`compile.sh` | 尊重工具链覆盖，编译使用工作流 Python；应用依赖路径，扩展缓存 identity；缺开发依赖时删除暴露的旧二进制并终止。 |
| `cuda_agent/tools/toolchain.py` | 保留显式 CUDACXX 和 CUDAHOSTCXX。 |
| `cuda_agent/workflow.py` | 编译前接口检查；终止类不 repair、不消耗算法修复预算；诊断不可信时加入明确的证据限制；区分 correctness/profiling。 |
| `cuda_agent/validation.py` | kernel/launcher 局部 coverage warning；可解析 binding 检查及 pending 记录。 |
| `cuda_agent/metrics.py`（新增）、`report.py` | 统一机器指标、独立可信推进审计，新增 workflow_report.json，保留旧 Markdown。 |
| `scripts/compare_runs.py` | 新旧日志读取；任务数和候选事件分开；正确性、profiling、recovery、修复效果和固定分母。 |
| `scripts/replay_progress.py`（新增） | 校验快照 hashes，重新检查源码映射，回放七个案例的比较及分类；不导入生成模型，不反序列化 tensor。 |
| `tests/test_progress_evidence.py`（新增） | 48 项离线回归，包括八项要求中的核心场景及接口、诊断生命周期、指标和终止路由。 |
| 原有三个测试文件 | 将 checkpoint 测试迁移到显式 trust/fixture/sequence；补缓存失效和 CPU alias/inplace 测试，更新有意变更的分类期望。 |

## Comparison 与 rollback

`progress()` 从 tuple 改为结构化记录，包含完整 correctness、pipeline、通过样本数和可信匹配前缀。`better()` 保留布尔接口，新增 `compare()` 返回决策及 reason。

比较首先排除 reference/environment/resource，再保护固定 verification seed、样本 seed、原有正常观测及已经通过的样本。正常验证数据不完整不能获得完整成功。排序依次为完整 correctness、pipeline 推进、同输入可信 checkpoint 前缀、同一可信失败位置的局部 mismatch，最后才是正常输出证据；平局保留旧版本。

checkpoint 比较要求 fixture_id、input_seed 和 reference_sequence 一致。失败输入变化时沿用旧输入 replay。缺失 trust 不从旧 matched 数补出；诊断缺失不能构成 checkpoint 推进或回退，明确观测到的可信 regression 则优先拒绝。检查点推进不再被末端误差上升否决。

恢复最佳候选时同步恢复源码、operator/task contract、固定 fixture、失败输入、source/interface/precompile/dependency 与诊断报告。修复效果日志不回滚。构建继续依靠源码和环境 fingerprint 验证缓存，不删除全部原生增量缓存。

独立的 `trusted_advancement()` 不调用比较器，供审计“有可信推进却被拒绝”的事件。在七例快照回放中，旧记录有 3 次这种拒绝，新选择规则为 0。该计数不是实际任务成功数。

## Debug trust 的范围

每次诊断记录 checkpoint ID、参考 source/语义、候选表达式、native 调用证据、当前输入的参考和候选容器/shape/dtype、matched、mismatch、trustworthy 及原因。shape 不固定为首轮 preflight 的 shape。

直线型 forward/debug 计算及 yield 位置必须对应；仅支持能安全代入的本地单 return helper。原生端沿可见调用关系寻找操作证据，声明、标签或同 shape 不足以建立可信度。融合 sqrt/round、提前 yield 输入、漏步骤、路径不同及不能确认的映射会标为不可靠。后续不支持的 helper 不抹去前面独立确认的前缀。

unavailable 是诊断执行状态，不是对全部 checkpoint 的否定。实际执行在后续步骤异常时，已确认前缀仍保留。extra/missing checkpoint 和 live shape/dtype 不一致单独记录不可信。正常五轮 forward 始终决定 correctness。

静态检查不能证明任意 CUDA 数学、控制流或内存安全。当前操作识别有意保守，不能识别的实现退回正常输出、runtime、shape 和 source evidence；显式 native 操作证据也不替代数值比较。特殊函数提示将事实、数学推断和假设分开，不能由巨大 digamma 输出推断输入一定巨大。

## Contract、precheck 与路由

required 与 optional 必须互为反义；required 参数不能携带 default。nullable 与“参数可省略”独立。required nullable 生成 optional C++ 类型，但 py::arg 不带默认值；只有 optional 参数生成 default。

contract helper 进入生成上下文。常见直接 m.def 注册可在编译前检查类型、nullable、参数数量、py::arg 名称、顺序与默认值；动态/不可解析形式记录 pending。编译后仍检查真实导出签名，无法获取的签名明确记录 pending。模型 constructor、forward 签名、state_dict 键、shape/dtype 在编译前检查；扩展 stub 的任何实际调用都会失败，检查不会运行生成 forward。

cuDNN 探测分别检查 header 文件与主库是否可由 host C++ linker 链接。显式目录/库路径、CUDA/system 路径、Python NVIDIA cuDNN 包和 PyTorch 安装目录均作为候选；不以 PyTorch runtime 能用 cuDNN 代替开发依赖。参考 Conv/ConvTranspose 在生成前检查，生成源码声明的 cuDNN 在编译前检查。发现的 include/library/runtime 路径及依赖文件 identity 纳入编译缓存。真实系统可链接性本次未验证，离线测试使用 mock。

reference/environment/resource 保存证据后终止。初次 preflight 不满足条件时不生成；候选评估遇到终止类不继续 repair，已应用修复若评估因终止类失败，不计入算法修复预算。compile、binding/contract、runtime、shape、numerical、unknown 分别使用对应反馈；diagnostic_unreliable 限制根因断言，不抹掉其他可靠失败证据。

输入 `.to(device)` 返回原对象或共享存储时仍 clone；已建立独立存储时省掉第二次分配。CPU fixture、reference/candidate 输入独立，顺序执行及参考模型设备行为不变。

coverage 仅产生 warning：分别分析每个 launcher 对应 kernel，识别该 kernel 的 grid-stride 增量。只有能把 kernel 的实际边界参数关联到确定的 extent、grid、block 时，才报告明确覆盖不足；其余为 possible risk。没有自动改 kernel，也没有新增通过条件。

## Repair effectiveness 与指标口径

每个已应用修复关联实际恢复的 base candidate，记录前后 hashes、changed/added/deleted files、progress、fingerprint 和 hypothesis。fingerprint 清除路径前缀、内存地址和源码行号，保留异常、语义数字及 expected/actual 结构；编译取首个根因错误，避免构建进度产生假差异。

结果为 fixed、progressed、same_failure、regressed、different_failure；repair_not_applied 为独立布尔值。指纹相同但可靠进展改善仍为 progressed。target_files 中声称修改的文件 hash 未变会被记录；仅 `.cu` 未变不会否定 binding 修复。全无修改在原子应用前拒绝，记录效果事件；单次调用最多三次格式/校验纠正，连续 API 失败上限仍为三次。context 展示未落实修改、重复失败、前一假设，并在连续停滞时要求更换可验证根因。

新增机器指标包括首轮通过、最终 correctness、workflow success、correctness 后 profiling 失败、任务最终分类、经历各分类的去重任务数、候选事件数、推进接受、same failure、未落实修复、不可信 checkpoint 和可信推进误拒审计。binding/contract 在汇总指标中归入 interface，具体失败路由仍保留。

- recovery = 经 repair 通过的初始失败任务 / 实际进入 repair 的初始失败任务。
- successful_repair_rate = 以完整 correctness 结束的 code repairs / 已评估 code repairs。
- progression_effective_rate 单独统计 fixed/progressed，不与成功修复率混用。
- average_repair_attempts 使用已应用并评估的 code repair 次数；格式/API 重试和未应用效果另报。
- 新报告中的已知零事件为 0；旧记录缺失的新证据为 null（unknown），不反推 trust 或 repair_not_applied。

历史总分母固定为 60：18/60 first-pass、37/60 final。固定原基线可评估集合为 55，排除 task 101、105、107、115、116；对应 37/55（67.27%）。按最终失败证据新分类后，另有 5 个 environment 失败，新的可评估集合为 50，对应 37/50（74%）。这只是分母重分类，不能解释为提升。文件同时保存具体 task IDs；中间候选发生 environment 的任务另计，不混入最终分类分母。

历史进入 repair 的初始失败任务 36，恢复 19，recovery 为 19/36（52.78%）；已评估 code repairs 共 80，平均每任务 1.333 次。旧记录的新 trust/effect 指标仍为 unknown。新逻辑未跑新的 60 项实验。

## 测试与七个历史案例

本次运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`：125 项，112 通过、13 跳过。相比原基线新增 49 项，其中 48 项离线通过；新增真实 CPU alias/inplace 测试因无 PyTorch 跳过。原有 Torch/CUDA 运行测试继续跳过。

验证覆盖 checkpoint 推进优先、编译回退保护、unavailable 部分进展、不可信 shape、required nullable、cuDNN 环境分类、same failure 和 repair_not_applied 八项；补充种子/样本保护、fixture/序列、局部误差、恢复证据、接口顺序与 state、缓存失效、所有终止类不 repair、有界重试、指纹、coverage 局部性与新旧指标。Python AST 语法检查、bash -n 和 git diff --check 通过。

完整测试输出：[progress_offline_tests.txt](../analysis_reports/progress_offline_tests.txt)。回放结果：[progress_replay_81_140.json](../analysis_reports/progress_replay_81_140.json)。

| Task | 离线证据与结论 |
|---|---|
| 84 | cuDNN 缺开发头文件的编译证据归入 environment，不建议搬源码绕过。 |
| 87 | 重查 group_norm/permute 的 normal/debug/native 对应后，保留 unavailable 中的可信 0→2；LSTM 后续步骤错位不污染前两步。 |
| 99 | fused sqrt/round 不能对应单独 sqrt checkpoint，标 diagnostic_unreliable。 |
| 114 | 保留 digamma 误差样本事实；输出幅度不能确认输入幅度，near-pole 等只是待输入侧证据验证的假设。 |
| 122 | **原验收前提不成立**：三次修复的 binding hashes 均改变；候选 2 实际增加 torch/torch.h，候选 3 实际改变 topk 返回类型。未变的是 .cu，声明没有声称修改它。因此历史 repair_not_applied 无法成立，不做误报；对应功能由显式 target_files 的独立测试验证。 |
| 125 | debug 在实际卷积前 yield 输入，映射不可信，不能据其 shape 断言卷积内部错误。 |
| 136 | sigmoid/pooling/flatten 的对应关系重查后，保留可信 1→3 的候选，末端 mismatch 轻微上升不否决。 |

回放的 `all_passed` 指依据实际证据建立的七项选择/分类检查通过；`original_expectations_fully_supported=false` 和 `original_expectation_discrepancies=[122]` 明确记录原计划中一项历史预期不受快照支持。没有宣称七个任务已修复。

回放会核对每个源码文件与 manifest 的 SHA-256，并重新检查映射；历史 matched 观测仅在该映射重新确认后用于选择演示。没有重新执行历史 tensor，无法独立复现数值匹配，也不会把历史无 trust 的 matched 数直接接入在线比较器。

## 风险、未验证项与下一实验

未验证：真实 PyTorch CPU alias/inplace、真实 CUDA 编译与 pybind 签名、实际 cuDNN header/link/runtime、CUDA 五轮 correctness、OOM 生命周期、GPU benchmark、profiling 性能及真实修复收益。不会由离线测试推断这些项目已通过。

主要风险是保守 AST/native 对应检查可能拒绝可用的动态、复杂 helper 或合法融合诊断；C++ 静态注册解析覆盖有限；依赖链接成功不能证明所有运行时分库均可加载。诊断证据和 pending 状态应在真实七例实验中重点审阅。接口-only preflight 有额外模型初始化开销，本次没有测量其成本。

下一步在原 CUDA 平台、相同 provider/model、种子、atol/rtol、repair budget 和 benchmark 设置下，先运行 84、87、99、114、122、125、136，输出到新日志目录，再运行 81–140。默认 repair budget 仍为 3；verification seed 仍按原规则由 2026 加 task_id × 100 派生。

```bash
# 以下命令仅作为后续平台实验说明，本次没有执行。
for task in 84 87 99 114 122 125 136; do
  CUDA_AGENT_LOG_DIR=logs/progress-v1-seven python3 main.py --task-start "$task" --task-end "$task"
done
CUDA_AGENT_LOG_DIR=logs/progress-v1-81-140 python3 main.py --task-start 81 --task-end 140
python3 -m scripts.compare_runs --before logs --after logs/progress-v1-81-140 \
  --task-start 81 --task-end 140 --output analysis_reports/progress_real_comparison.json
```

优先观察总 60 和固定 55 集合的最终 correctness、trusted progression 保留率、useful_candidate_rollback_count（目标 0）、same_failure/repair_not_applied、平均 repair、successful repair 与 progression effective 两种比率、environment 终止是否正确，以及 correctness 后 profiling 失败。缺任务或新分母变化必须同时展示，不单看重分类后的成功率。
