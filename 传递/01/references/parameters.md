# RiskRules 参数参考

## 决策树参数 (DEFAULT_TREE_PARAMS)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_depth | int | 10 | 决策树最大深度，控制规则复杂度 |
| min_samples_leaf | int | 50 | 叶子节点最小样本数，防止过拟合 |
| criterion | str | "gini" | 分裂准则，可选 "gini" / "entropy" |
| class_weight | str/None | None | 类别权重，可设为 "balanced" 处理不平衡数据 |
| random_state | int | 42 | 随机种子，保证结果可复现 |

## 策略筛选阈值 (DEFAULT_STRATEGY_FILTER)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| min_bad_rate | float | 0.05 | 最小坏账率，过滤低风险策略 |
| max_bad_rate | float | 0.80 | 最大坏账率，排除异常高值 |
| min_samples | int | 30 | 最小样本量，保证统计显著性 |
| min_lift | float | 1.5 | 最小提升度（相对整体bad_rate的倍数） |

## 可视化参数 (DEFAULT_VIZ_PARAMS)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| figsize | tuple | (14, 10) | 图表尺寸 |
| dpi | int | 150 | 图表分辨率 |
| cmap | str | "RdYlGn_r" | 颜色映射 |
| max_rules_display | int | 30 | 单张图表中最多展示的规则数 |

## 特征采样参数 (DEFAULT_SAMPLE_PARAMS)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| n_features | int | 10 | 随机采样每次选取的特征数 |
| n_iterations | int | 5 | 随机采样迭代次数 |
| top_k_importance | int | 10 | 重要性采样选取的top-k特征数 |

## API 模块说明

| 模块 | 类 | 功能 |
|------|-----|------|
| data_loader | DataLoader | CSV数据加载与预处理 |
| tree_builder | TreeBuilder | 决策树构建与训练 |
| rule_extractor | RuleExtractor | XiaoWuGe算法规则提取 |
| strategy_analyzer | StrategyAnalyzer | 策略筛选、评估、覆盖率分析 |
| visualizer | StrategyVisualizer | matplotlib + plotly 双引擎可视化 |
| feature_sampler | FeatureSampler | 随机/重要性/业务分组采样 |
| exporter | StrategyExporter | CSV/Excel/HTML报告导出 |
| premium_dashboard | PremiumDashboard | Three.js + D3.js 高级交互式看板 |
