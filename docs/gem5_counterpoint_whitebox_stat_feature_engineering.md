# 高维 gem5 stat 到真机 PMU 双计数观测的异构属性图预实现设计

## 结论

本阶段主线收敛为：

```text
raw gem5 stat observations
raw PMU event observations
FDL/source/sidecar/audit evidence
workload/window observations
  -> heterogeneous attributed graph
  -> projection relation learning
  -> backend comparison
  -> pre-implementation acceptance gates
```

核心判断：

- `canonical metric` 不再作为 gem5 stat 与真机 PMU event 之间的统一指标中间层。
- `canonical_*`、`pmu_group`、`value_kind`、`denominator_metric` 等只作为旧实现事实和迁移对象存在。
- 新模型主线直接建模 `gem5_stat_node` 与 `pmu_event_node` 两层节点，以及 evidence / workload-window 辅助节点。
- 图模型负责生成并学习节点特征、边特征、观测 mask 和 projection relation，不由人工语义相关性或固定公式提前定义。
- 具体工具后端待评估；NetworkX、pandas/pyarrow、scipy.optimize.nnls、PyG、DGL、igraph 都只是候选实现。

一句话：

```text
先定义数学图对象和训练任务，再选择后端；
不先定义 canonical 中间层、手写拟合公式或固定图库。
```

## 当前事实与迁移约束

本文是预实现设计，不是当前代码说明。当前实现事实仍需保留为迁移约束。

`cpu_microarchitecture` 当前事实：

- `configs/schema/canonical.yaml` 仍包含旧 `target_metric_catalog.metrics`、`canonical_*`、`pmu_group`、`value_kind`、`denominator_metric`、`source_scope` 和 `max_residual_sources`。
- 当前 calibration 仍是 `semantic_residual_calibration`。
- 当前训练器仍按 target metric 拟合 `target_models`。
- 当前预测表仍包含 `raw_*`、`self_*`、`residual_*`、`pred_*` 多阶段列。
- `src/alignment/shared/selection.py`、`same_group_projection_sources()`、`fit_self_scale()`、`fit_evidence_residual()` 等仍属于旧主线。
- `src/alignment/dataset.py::build_gem5_raw_stat_long()` 已能物化 raw gem5 stats。
- PMU 采集和 confidence audit 已存在，但 confidence audit 是旁路审计，不是模型锚点。

已观察到的数据事实：

- `output/suites/microbench/pmu_resource_pressure_n2_v1/calibration/calibration_dataset.parquet` 是旧 canonical 包装后的训练表，有 425 行、160 列、36 个 workload、392 行 `source_quality_status == "ok"`。
- 该表含 22 个旧 canonical 包装下的 gem5 count 列和 22 个旧 canonical 包装下的 real count 列。
- 旧 canonical 同组字段存在高相关和重复解释风险，例如部分 branch / mem / L2 组 source 相关性接近或等于 1.0。
- `output/suites/microbench_gem5_smoke/gem5_smoke_n2_v1/roi_windows/tables/gem5_raw_stat_long.parquet` 已观察到 4001 个 raw stat，但只有 3 个 workload；raw stat pool 有价值，但不能直接自由回归。

`counterpoint-gem5` 当前事实：

- `counterpoint/gem5/sidecar.py` 是 thin sidecar，只应承载 `model_id/module_id/fdl_path/gem5_revision/source_refs/counter_bindings`。
- sidecar 明确拒绝 `pmu/g_group/metric/canonical/flow` 等业务字段。
- `counterpoint/gem5/raw_stats.py` 的目标 raw stat contract 是 `workload_id/parent_window_id/window_id/gem5_stat/gem5_value`。
- `counterpoint/gem5/audit.py`、FDL、source refs、sidecar binding 和 exact audit 只提供证据，不定义 PMU/G-group/graph/model schema。
- `countpointer` H1 接口只消费内部计数关系配置和外部 staged `raw_stat_candidates.parquet` / 映射文件，不动态扫描 `stats.txt` 或其他运行目录文件。

## 已撤回的旧路线

以下路线不再作为目标设计：

- 通过 `canonical metric` 人工统一 gem5 stat 与真机 PMU event。
- 使用 `canonical gem5_count -> canonical real_count` 作为主任务。
- 按旧观测组生成固定全连接人工标签块。
- 将 `value_kind`、`denominator_metric`、ratio/progress/sparse_count 手写公式作为拟合定义。
- 将 relation graph 与 projection graph 设计成两个耦合子系统。
- 将投影边简化为 source / target / weight / active 四元组。
- 将上一版 edge projection 模型文件或按 metric 命名的预测列作为最终主 schema。
- 预先指定 NetworkX 作为图结构主依赖。
- 预先把线性非负 baseline 升级为核心拟合器。
- 预先排除 PyG/DGL/igraph 等后端。

可保留但必须降级的旧内容：

- `canonical_*`：旧字段事实和迁移对象。
- `pmu_group`：旧采集/观测上下文，可迁移为 observation context 或 mask feature。
- `value_kind`、`denominator_metric`：旧人工先验，可迁移为 node / edge feature candidate 或 audit label。
- `scipy.optimize.nnls`：线性 baseline 候选。
- `NetworkX`：debug / 可视化 / 小图遍历候选。
- `pandas/pyarrow`：数据交换和落盘候选。

## 统一数学对象

预实现阶段的统一数学对象是 heterogeneous attributed graph：

```text
G = (V, E, X_V, X_E, Y, M)

V = V_gem5_stat
  ∪ V_pmu_event
  ∪ V_evidence
  ∪ V_workload_window

E = E_stat_relation
  ∪ E_projection_candidate
  ∪ E_evidence
  ∪ E_observation

X_V: node features
X_E: edge features
Y: observed gem5 / PMU counts
M: observation masks, split masks, confidence masks
```

关键点：

- `gem5_stat_node` 和 `pmu_event_node` 是独立语义层。
- PMU event 是观测标签节点，不是白盒机制节点。
- gem5 stat 是 white-box / simulator observation node，不是 PMU 语义同名替身。
- evidence node 只提供证据，不强行锚定投影关系。
- workload-window node 显式表达观测上下文和缺失 mask。

## 节点、边、特征与观测

### 节点

`gem5_stat_node`：

- stat path。
- SimObject / stat prefix。
- source location / source refs。
- unit / count-like status。
- observed distribution。
- cross-workload variation。
- structural neighborhood。

`pmu_event_node`：

- event name / event code。
- collector observation context。
- observed distribution。
- confidence / noise profile。
- co-observation mask。
- architecture / platform context。

`evidence_node`：

- FDL counter binding。
- source declaration / increment evidence。
- sidecar binding。
- exact audit status。
- confidence audit status。

`workload_window_node`：

- workload id / family。
- input shape。
- ROI / parent window / window id。
- split role。
- observation availability。

### 边

`stat_relation_edge`：

- gem5 stat-stat relation。
- hierarchy / parent-child。
- source-derived。
- total-part。
- unit transform candidate。

`projection_candidate_edge`：

- candidate relation from `gem5_stat_node` to `pmu_event_node`。
- edge score / confidence 由模型学习。
- 不由 canonical 名称或同组人工规则直接定义。

`evidence_edge`：

- evidence node to stat / relation / projection candidate。
- 表示 source、FDL、sidecar 或 audit 支持。

`observation_edge`：

- workload-window node to measured stat / PMU event。
- 承载 observed count、availability、mask 和 split 信息。

### 特征原则

- 人工先验只作为 feature candidate 或 audit label，不作为模型定义。
- ratio、denominator、progress、total/part、miss/access 等不直接写成目标公式。
- 特征必须可追溯到 raw observation、source evidence、sidecar binding 或明确配置。
- 对缺失观测必须显式进入 `M`，不能用默认值掩盖。

## 训练任务与模型目标

模型先行阶段优先定义任务，而不是先选库。

主任务：

- PMU event count prediction。
- projection relation confidence learning。
- projection edge stability across workload/window split。
- failure attribution。

辅助目标：

- edge confidence calibration。
- proxy dominance detection。
- evidence-supported explanation。
- relation consistency audit。
- missing observation robustness。

训练输入：

```text
G = (V, E, X_V, X_E, Y, M)
```

训练输出：

- PMU event prediction。
- projection candidate score / confidence。
- supporting evidence summary。
- failure attribution report。
- backend comparison metrics。

不在预实现阶段固定：

- 最终图学习算法。
- 最终图库。
- 最终存储格式。
- 最终模型 payload schema。

## 后端候选与 baseline

后端只作为候选实现，不定义主线。

候选后端：

- `PyG/DGL`：异构图学习原型候选。需要评估样本规模、mask 支持、解释性、训练稳定性和工程复杂度。
- `scipy.optimize.nnls`：线性 baseline 候选。用于确认非负线性映射能达到的下界表现，不定义核心拟合路线。
- `sklearn`：split、metrics、简单 baseline 和评估工具候选。
- `NetworkX`：debug、可视化、小图邻域检查候选，不定义训练图。
- `pandas/pyarrow`：数据交换和落盘候选，不定义模型 schema。
- `igraph`：高性能静态图分析候选，只有在图规模成为瓶颈时评估。

backend comparison report 至少比较：

- prediction error。
- edge confidence stability。
- mask handling。
- explanation traceability。
- training / inference cost。
- dependency complexity。
- reproducibility。

## CounterPoint 证据边界

`counterpoint-gem5` 不承担 PMU label、G-group、projection fitting 或 graph model schema。

它只输出：

- raw gem5 stat observation。
- FDL binding。
- source refs。
- sidecar binding。
- exact audit / confidence audit evidence。

它不输出：

- PMU event schema。
- G-group schema。
- canonical metric schema。
- projection edge schema。
- graph construction schema。
- model payload schema。

这些边界保证 CounterPoint 是证据提供者，而不是 PMU/gem5 对齐模型的业务层。

## 预实现输入输出 artifact

预实现阶段需要先定义 artifact，而不是直接改训练器。

输入 artifact：

- raw gem5 stat observation table。
- raw PMU event observation table。
- workload/window table。
- evidence table。
- source/sidecar/audit table。

中间 artifact：

- graph node table。
- graph edge table。
- node feature table。
- edge feature table。
- observation mask table。
- split mask table。

输出 artifact：

- backend comparison report。
- graph projection report。
- prediction report。
- projection relation confidence report。
- failure attribution report。

这些 artifact 的字段应最小充分，不重复保存可由路径、node id、edge id 或 source artifact 推导的信息。

## 验收检查

进入实现前必须满足：

- 文档不再把 `canonical metric` 作为模型节点或统一语义层。
- 文档不再把 G-group 同组全连接作为主拟合定义。
- 文档不再把 NNLS 写成主拟合器。
- 文档不再把 NetworkX 写成主图结构依赖。
- 文档明确 `G = (V, E, X_V, X_E, Y, M)`。
- 文档明确 `gem5_stat_node` 和 `pmu_event_node` 是独立语义层。
- 文档明确 backend candidate 与 baseline 的角色。
- 文档明确 CounterPoint 只提供证据。

静态检查应覆盖两类结果：旧强结论短语不再出现在目标设计语境中；异构属性图、节点特征、边特征、观测 mask、projection relation learning 和 backend candidate 等新主线短语存在。

## 最终收束

本文件收束后的主线是：

```text
不再通过 canonical metric 人工统一 gem5 和 PMU；
不再通过固定公式先定义投影；
不再先绑定具体图库；
先定义异构属性图、节点/边特征、观测 mask 和学习任务；
再比较后端并进入实现。
```
