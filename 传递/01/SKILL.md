---
name: risk_strategy_mining
description: 基于决策树算法的自动化风控策略挖掘工具。输入基础数据，返回策略明细及多种可视化图表，支持规则提取、策略推荐、多维分析。
version: 2.2.0
author: XiaoWuGe
tags: [风控, 策略挖掘, 决策树, 金融, risk-control]
---

# Risk Strategy Mining

基于决策树（XiaoWuGe 算法）的自动化风控策略挖掘工具。输入 CSV 数据，输出策略明细及多种可视化图表。

## 核心能力

| 能力 | 说明 |
|------|------|
| 数据加载 | 自动读取CSV，识别特征列与标签列 |
| 决策树训练 | 可配置深度、叶子样本数、分裂准则等参数 |
| 规则提取 | XiaoWuGe算法，毫秒级提取全部叶子规则到 DataFrame |
| 策略分析 | bad_rate、lift、覆盖率等多维度评估与筛选 |
| 可视化 | 决策树图 + 策略分布图 + 特征重要性 + 热力图 + 交互式图表 |
| 特征采样 | 随机采样 / 重要性采样 / 业务分组采样，扩展规则多样性 |
| 多格式导出 | CSV / Excel / HTML报告 / Premium Dashboard |

## 使用方式

### CLI

```bash
# 基础使用
python scripts/main.py --data_path ./train.csv

# 自定义参数
python scripts/main.py --data_path ./train.csv --max_depth 8 --min_samples_leaf 100

# 完整参数
python scripts/main.py --data_path ./train.csv \
    --max_depth 10 --min_samples_leaf 50 \
    --min_bad_rate 0.05 --min_lift 1.5 \
    --sample_mode random --n_iterations 10 \
    --export_format all --output_dir ./output
```

### 代码调用

```python
from riskrules import (
    DataLoader, TreeBuilder, RuleExtractor,
    StrategyAnalyzer, StrategyVisualizer, StrategyExporter
)

# 1. 加载数据
X, y, feature_names = DataLoader.load("train.csv")

# 2. 训练决策树
builder = TreeBuilder({"max_depth": 10, "min_samples_leaf": 50})
builder.fit(X, y)

# 3. 提取规则
extractor = RuleExtractor(builder.model, feature_names)
rules_df = extractor.extract_to_dataframe()

# 4. 分析筛选
analyzer = StrategyAnalyzer(rules_df)
filtered = analyzer.filter(min_bad_rate=0.05, min_lift=1.5)
print(analyzer.summary_stats())

# 5. 可视化
viz = StrategyVisualizer("./output")
viz.plot_bad_rate_distribution(rules_df)
viz.plot_top_strategies(rules_df)

# 6. 导出
exporter = StrategyExporter("./output")
exporter.to_excel(filtered, importance_df=builder.feature_importance)
```

### 特征采样

```python
from riskrules import FeatureSampler

sampler = FeatureSampler()

# 模式1: 随机采样
rules_df = sampler.random_sample(X, y, n_features=10, n_iterations=5)

# 模式2: 重要性采样
importance_df, rules_df = sampler.importance_sample(X, y, top_k=10)
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_depth | 10 | 决策树最大深度 |
| min_samples_leaf | 50 | 叶子节点最小样本数 |
| min_bad_rate | 0.05 | 最小坏账率阈值 |
| min_lift | 1.5 | 最小提升度阈值 |
| min_samples | 30 | 最小样本量阈值 |
| criterion | gini | 分裂准则 (gini / entropy) |
| sample_mode | - | 采样模式 (random / importance / business) |

完整参数列表见 `references/parameters.md`。

## 输出结构

```
output/
├── strategies_*.csv / .xlsx            # 策略明细
├── feature_importance_*.csv            # 特征重要性
├── tree_sklearn.png                    # sklearn决策树图
├── tree_graphviz.png                   # graphviz决策树图
├── tree_dtreeviz.svg                   # dtreeviz决策树图
├── bad_rate_distribution.png           # 坏账率分布图
├── top_strategies.png                  # Top策略柱状图
├── coverage_vs_badrate.png             # 覆盖率vs坏账率气泡图
├── feature_importance.png              # 特征重要性图
├── rule_depth_analysis.png             # 规则深度分析图
├── rules_heatmap.png                   # 特征使用热力图
├── interactive_*.html                  # 交互式图表
├── strategy_report_*.html              # HTML综合报告
└── premium_dashboard_*.html            # Premium高级看板
```

## 依赖

```bash
pip install -r requirements.txt
# graphviz 需要系统包: brew install graphviz (macOS) / sudo apt install graphviz (Ubuntu)
```
