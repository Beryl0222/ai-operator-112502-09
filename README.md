# 分组量化反量化与激活融合的误差交接

可复用的**分组低比特权重量化线性算子**与**结果资格（eligibility/qualification）信息**。
调用方提交权重版本、组布局和激活策略，即可得到符合各自误差预算的输出，并明确知道哪些组
因为 **范围不足 / 参数缺失 / 版本不匹配 / 超误差预算** 而不能走快速路径。

仅依赖 Python 标准库（要求 Python ≥ 3.11）。

## 安装与验证

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

即使不执行 editable 安装，仓库根目录的 `operator_quant_groups -> src/operator_quant_groups`
符号链接也能让 `python -m unittest discover -s tests` 直接导入包。

## 30 秒上手

```python
import operator_quant_groups as oq

payload = oq.load_fixture()          # data/input.json
op = oq.GroupedQuantizedLinear(payload)

# 精排：默认走精确（Fraction）路径，也可显式切换
rank = op.run_ranking(payload["input"], precise=True)
print(rank["output"])                # [[-0.075, 0.05, -0.15, -0.3],
                                     #  [ 0.075, 0.2,   0.9,  1.0]]  (1.05 饱和到 1.0)

# 预览：受控近似 + 误差预算；不合格的组自动用精确切片兜底
prev = op.run_preview(payload["input"], budget=0.02, approx_bits=8)
print(prev.eligibility.summary())
```

通过服务提交可额外获得 **按调用方隔离的缓存** 与 **可追溯回执**：

```python
svc = oq.QuantGroupService([payload])
r = svc.submit("w9", payload["input"], profile="preview",
               error_budget=0.02, approx_bits=8)
svc.submit("w9", payload["input"], profile="ranking", exact=True)

svc.withdraw("w9")                   # 撤回模型包
svc.trace(r.trace_id)                # 旧结果仍在：recomputable=False,
                                     # reason="package_withdrawn"
svc.try_recompute(r.trace_id)        # {'recomputable': False, 'reason': 'package_withdrawn', ...}
```

## 数据契约（与 `data/input.json` 对齐）

```jsonc
{
  "weight_version": "w9",
  "group_size": 2,                   // 每个组覆盖的输出通道数
  "activation_range": [-1.0, 1.0],   // 包级默认激活/饱和范围，组可覆盖
  "groups": [{
    "group_id": 0,                   // 可选，缺省用下标
    "scale": 0.25,
    "zero_point": 1,
    "values": [0, 2, 3, 1],          // 低比特整数码
    "activation_range": [-1, 1],     // 可选的组级独立范围
    "weight_version": "w9"           // 可选的组级版本
  }],
  "input": [[0.1, -0.2], [0.4, 0.7]]
}
```

默认布局：组沿**输出通道**维分组，每组 `values` 按行优先 reshape 成
`(group_size, in_features)`，反量化公式为

```
w = scale * (q - zero_point)
y = x @ Wᵀ (+ bias)
```

样例的精确结果：第 2 个样本最后一个通道在线性后为 `1.05`，按闭区间 `[-1, 1]`
饱和为 `1.0`。资格报告默认采用“范围必须覆盖激活”的严格口径，因此该组会同时被标记为
`range_insufficient`（输出仍正确饱和）；如果调用方把该范围视为**允许饱和的裁剪边界**，
可传 `ErrorBudget(max_abs_error=..., strict_range=False)`，此时只要越界量在预算内就不判为
范围不足。

## 固定的边界顺序与确定性

管线阶段顺序固定为：

```
dequantize → linear(+bias) → pre_activation_saturation
           → activation → post_activation_saturation
           → output_rounding
```

- 所有“范围 / 预算 / 舍入边界”的**资格判定**都使用 `fractions.Fraction`
  精确有理数运算；浮点输入按最短十进制表示规范化（`0.1 → 1/10`）。
- 边界是**闭区间**：恰好等于上/下界记为 `at_upper_bound` / `at_lower_bound`，
  不算越界；误差恰好等于预算视为满足（`error <= budget`）。
- 因为判定是精确且可结合的，**dtype（float32/float64/bfloat16）和分块大小/顺序
  不会改变资格结果**；它们只影响近似输出的浮点形态。
- 舍入支持 `half_even`（默认）、`half_up`、`half_down`、`half_away`、
  `floor/ceil/trunc`，平局用精确 `1/2` 判定。

可用的底层工具：`classify_boundary`、`within_inclusive`、`overflow_distance`、
`round_exact`、`exact_sum`、`stable_float_sum`、`as_fraction`。

## 资格报告与原因码

`evaluate_eligibility(package, inputs, strategy=None, budget=None, ...)`
返回 `EligibilityReport`（同时是 dict，可 JSON 序列化）：

- `eligible` / `fast_path_eligible` / `can_use_fast_path`：所有组都可走快速路径；
- `groups[i]`：`GroupEligibility`，含 `eligible`、`reasons`、`reason`、
  `missing_fields`、`expected_version/actual_version`、`range`、`error`；
- `eligible_groups` / `ineligible_groups` / `blocked_groups`；
- `by_reason`：原因码 → 组 id 列表；
- 便捷视图：`range_insufficient_groups`、`missing_parameter_groups`、
  `version_mismatch_groups`，以及组上的布尔属性
  `range_insufficient` / `missing_parameters` / `version_mismatch` /
  `error_budget_exceeded`。

规范原因码（字符串值本身，且与常见别名双向相等）：

| 规范码 | 可接受的别名示例 | 含义 |
| --- | --- | --- |
| `range_insufficient` | `insufficient_range`, `range_exceeded`, `out_of_range` | 组激活超出记录范围（边界值不算） |
| `missing_parameters` | `missing_parameter`, `params_missing` | 缺 `scale` / `zero_point` / `activation_range` 等 |
| `version_mismatch` | `weight_version_mismatch`, `stale_version` | 提交版本与包/组/缓存版本不一致 |
| `error_budget_exceeded` | `budget_exceeded`, `tolerance_exceeded` | 近似误差超过调用方预算 |
| `package_withdrawn` | `weight_package_withdrawn`, `model_package_withdrawn`, `package_revoked` | 包已撤回，不可重算 |

原因对象是 `str` 子类，`report.groups[0]["reason"] == "insufficient_range"`
以及反向写法都成立。

## 预览 vs 精排

- `profile="preview"`（别名 `fast` / `draft` / `预览`）：自动模式下，合格组使用
  **受控近似**（`approx_bits` 激活伪量化，误差上界约为半个量化步长），不合格组用
  精确结果兜底，因此交付张量始终满足预算。
- `profile="ranking"`（别名 `fine_rank` / `rerank` / `精排`）：默认精确路径；
  可用 `precise=False` / `exact=False` / `precision="fast"` 切回快速路径，也可用
  `precise=True` / `exact=True` / `precision="exact"` 显式切到更精确路径。
- 阈值/范围按**每个组自己的 `activation_range`** 解析（显式
  `ActivationStrategy(..., activation_range=...)` 才会全局覆盖），因此不会再把精排的
  单一饱和阈值套用到预览上，少数小范围通道不会被离群通道放大误差。

## 缓存隔离与版本

`ResultCache` 按 profile 分命名空间：`cache.preview` 与 `cache.ranking` 互不覆盖；
键由 `profile + weight_version + layout + strategy + budget + inputs` 的确定性指纹构成。
`cache.invalidate_version("w8")` 可按版本清理。`VersionedParameterCache.lookup(gid, v)`
在命中旧版本时返回 `(None, {"reason": "version_mismatch", "expected_version",
"actual_version", ...})`，绝不静默混用。

## 撤回与可追溯

`ResultRegistry`（服务通过 `svc.registry` 暴露）保存每张已交付结果的
`ResultReceipt`（冻结输出、版本、profile、资格摘要、时间、缓存键）。
`withdraw_package(version)` / `svc.withdraw(version)` 后：

- 旧回执仍可 `trace(trace_id)` / `get_receipt(id)`，输出冻结保留；
- 回执 `recomputable=False`、`recompute_blocked=True`、
  `reason == "package_withdrawn"`（与 `weight_package_withdrawn` 等别名相等）；
- `recompute(id)` 抛 `PackageWithdrawnError`（带 `weight_version` / `reason_code`），
  `try_recompute(id)` 返回不抛异常的结果；
- 已撤回版本的新 `submit` 会直接抛 `PackageWithdrawnError`。

## JSON / 命令行

```bash
python -m operator_quant_groups '{"action":"ping"}'
python -m operator_quant_groups --fixture data/input.json     # 提交样例
```

或在代码里使用 `handle_request({...})` / `oq.submit({...})`，支持
`ping | fixture | eligibility | submit/run | dequantize | withdraw | trace`。

## 公共命名与兼容别名

`operator_quant_groups.api` 是单一公共面，并在包顶层重导出。除规范名外还提供大量
常见别名（`QuantGroupService / OperatorService / QuantGroupRuntime`、
`ActivationPolicy`、`QualificationReport / ResultEligibility / FastPathEligibility`、
`IsolatedResultCache / VersionedCache`、`DeliveryRegistry / AuditLedger / TraceRegistry`、
`Receipt / ProvenanceRecord`、`load_package / load_fixture / submit / run / ...`）。
未显式列出的名字也会按命名模式解析到对应基类；`operator_quant_groups.<area>`
（如 `eligibility / cache / registry / policy / budget / exceptions / ...`）
等常见子模块路径同样可用。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `numeric_kernel.py` / `numbers.py` | 精确数、dtype 模拟、稳定求和、舍入、边界分类（`numbers.py` 兼作 stdlib `numbers` 影子导入垫片） |
| `reasons.py` | 原因码与别名相等语义 |
| `models.py` | `WeightPackage / GroupParams / GroupLayout / ActivationStrategy / ErrorBudget` 与 profile |
| `engine.py` | 反量化、组块装配、matmul、固定顺序阶段管线 |
| `eligibility.py` | 组/报告资格判定与 `ErrorHandoff` |
| `linear_operator.py` / `operator.py` | `GroupedQuantizedLinear`、`OperatorResult`、函数式入口 |
| `cache.py` | profile/版本隔离缓存与组参数版本缓存 |
| `registry.py` | 回执、登记、撤回与重算阻断 |
| `service.py` | `QuantGroupService` 门面 |
| `contract.py` / `__main__.py` | JSON 请求处理与 CLI |
| `api.py` / 各别名模块 | 公共导出面与兼容路径 |
