#!/usr/bin/env python
"""
RiskRules - 风控策略挖掘主入口

基于决策树的自动化风控策略挖掘工具。
输入基础数据，输出策略明细 + 多维可视化图表。

用法:
    python main.py --data_path /path/to/train.csv

    python main.py --data_path /path/to/train.csv \\
                   --max_depth 10 --min_samples_leaf 50 \\
                   --output_dir ./output

    python main.py --data_path /path/to/train.csv \\
                   --sample_mode random --n_iterations 10
"""

import argparse
import os
import sys
from pathlib import Path

# 将项目根目录加入路径（main.py 在 scripts/ 子目录下）
sys.path.insert(0, str(Path(__file__).parent.parent))

from riskrules import (
    DataLoader,
    TreeBuilder,
    RuleExtractor,
    StrategyAnalyzer,
    StrategyVisualizer,
    FeatureSampler,
    StrategyExporter,
)
from riskrules.config import (
    DEFAULT_TREE_PARAMS,
    DEFAULT_STRATEGY_FILTER,
    DEFAULT_VIZ_PARAMS,
    DEFAULT_SAMPLE_PARAMS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RiskRules - 基于决策树的自动化风控策略发现与推荐",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础使用
    python main.py --data_path ./train.csv

    # 自定义参数
    python main.py --data_path ./train.csv --max_depth 8 --min_samples_leaf 100

    # 启用特征采样
    python main.py --data_path ./train.csv --sample_mode random --n_iterations 10

    # 导出Excel报告
    python main.py --data_path ./train.csv --export_format excel
        """,
    )

    # 数据参数
    parser.add_argument("--data_path", required=True, help="训练数据CSV文件路径")
    parser.add_argument("--label_col", default="label", help="标签列名 (默认: label)")
    parser.add_argument("--id_col", default=None, help="主键列名")
    parser.add_argument("--feature_start", default=None, help="特征起始列名（包含）")
    parser.add_argument("--feature_end", default=None, help="特征结束列名（包含）")

    # 决策树参数
    parser.add_argument("--max_depth", type=int, default=10, help="决策树最大深度 (默认: 10)")
    parser.add_argument("--min_samples_leaf", type=int, default=50, help="叶子节点最小样本数 (默认: 50)")
    parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"], help="分裂准则")
    parser.add_argument("--class_weight", default=None, help="类别权重 (如: balanced)")

    # 策略筛选参数
    parser.add_argument("--min_bad_rate", type=float, default=0.05, help="最小坏账率 (默认: 0.05)")
    parser.add_argument("--max_bad_rate", type=float, default=0.80, help="最大坏账率 (默认: 0.80)")
    parser.add_argument("--min_samples", type=int, default=30, help="最小样本量 (默认: 30)")
    parser.add_argument("--min_lift", type=float, default=1.5, help="最小提升度 (默认: 1.5)")

    # 采样参数
    parser.add_argument("--sample_mode", default=None, choices=["random", "importance", "business"], help="特征采样模式")
    parser.add_argument("--n_features", type=int, default=10, help="随机采样特征数")
    parser.add_argument("--n_iterations", type=int, default=5, help="采样迭代次数")
    parser.add_argument("--top_k", type=int, default=10, help="重要性采样的top-k特征数")

    # 输出参数
    parser.add_argument("--output_dir", default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument("--export_format", default="csv", choices=["csv", "excel", "html", "all"], help="导出格式")

    # 可视化参数
    parser.add_argument("--skip_viz", action="store_true", help="跳过可视化")
    parser.add_argument("--max_rules_display", type=int, default=30, help="图表中最多展示的规则数")

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  RiskRules - 风控策略智能挖掘")
    print("  基于决策树的自动化策略发现与推荐")
    print("=" * 60)

    # ========== 1. 数据加载 ==========
    print("\n[1/6] 加载数据...")
    X, y, feature_names = DataLoader.load(
        path=args.data_path,
        label_col=args.label_col,
        id_col=args.id_col,
        feature_start=args.feature_start,
        feature_end=args.feature_end,
    )
    info = DataLoader.summary(X, y)
    print(f"  样本数: {info['n_samples']:,} | 特征数: {info['n_features']}")
    print(f"  坏账率: {info['bad_rate']:.2%} (bad={info['n_bad']:,}, good={info['n_good']:,})")

    # ========== 2. 决策树训练 ==========
    print("\n[2/6] 训练决策树模型...")
    tree_params = {
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "criterion": args.criterion,
        "class_weight": args.class_weight,
    }

    if args.sample_mode:
        # 特征采样模式
        sampler = FeatureSampler()
        if args.sample_mode == "random":
            print(f"  采样模式: 随机采样 (n_features={args.n_features}, iterations={args.n_iterations})")
            rules_df = sampler.random_sample(
                X, y,
                n_features=args.n_features,
                n_iterations=args.n_iterations,
                tree_params=tree_params,
            )
            builder = TreeBuilder(tree_params)
            builder.fit(X, y)
        elif args.sample_mode == "importance":
            print(f"  采样模式: 重要性采样 (top_k={args.top_k})")
            importance_df, rules_df = sampler.importance_sample(
                X, y, top_k=args.top_k, tree_params=tree_params
            )
            builder = TreeBuilder(tree_params)
            builder.fit(X, y)
        else:
            print("  采样模式: 业务分组采样 (需要配置feature_groups)")
            builder = TreeBuilder(tree_params)
            builder.fit(X, y)
            extractor = RuleExtractor(builder.model, feature_names)
            rules_df = extractor.extract_to_dataframe()
    else:
        # 标准模式：单一决策树
        builder = TreeBuilder(tree_params)
        builder.fit(X, y)
        tree_info = builder.get_tree_info()
        print(f"  节点数: {tree_info['node_count']} | 叶子数: {tree_info['n_leaves']} | 深度: {tree_info['max_depth']}")

        extractor = RuleExtractor(builder.model, feature_names)
        rules_df = extractor.extract_to_dataframe()

    print(f"  提取规则数: {len(rules_df)}")

    # ========== 3. 策略分析 ==========
    print("\n[3/6] 策略分析与筛选...")
    analyzer = StrategyAnalyzer(rules_df, filter_params={
        "min_bad_rate": args.min_bad_rate,
        "max_bad_rate": args.max_bad_rate,
        "min_samples": args.min_samples,
        "min_lift": args.min_lift,
    })

    filtered = analyzer.filter()
    summary = analyzer.summary_stats()

    print(f"  总规则数: {summary['total_rules']}")
    print(f"  筛选后: {len(filtered)} 条")
    print(f"  平均坏账率: {summary['avg_bad_rate']:.2%}")
    print(f"  最高坏账率: {summary['max_bad_rate']:.2%}")
    print(f"  平均提升度: {summary['avg_lift']:.2f}x")
    print(f"  最高提升度: {summary['max_lift']:.2f}x")
    print(f"  平均深度: {summary['avg_depth']:.1f}")

    # 展示Top-10
    print("\n  Top-10 策略:")
    top10 = analyzer.top_strategies(10)
    for i, (_, row) in enumerate(top10.iterrows()):
        rule_short = row["rule"] if len(row["rule"]) <= 80 else row["rule"][:77] + "..."
        print(f"  {i+1:2d}. {rule_short}")
        print(f"      Bad Rate: {row['bad_rate']:.2%} | Lift: {row['lift']:.1f}x | N={int(row['total'])} | Depth={int(row['depth'])}")

    # ========== 4. 可视化 ==========
    image_paths = []
    html_charts = []
    if not args.skip_viz:
        print("\n[4/6] 生成可视化图表...")
        viz = StrategyVisualizer(output_dir=args.output_dir)

        # 决策树静态可视化
        print("  -> sklearn plot_tree...")
        path1 = viz.plot_tree_sklearn(builder.model, feature_names)
        image_paths.append(path1)

        print("  -> graphviz...")
        path2 = viz.plot_tree_graphviz(builder.model, feature_names)
        if path2:
            image_paths.append(path2)

        print("  -> dtreeviz...")
        path3 = viz.plot_tree_dtreeviz(builder.model, X, y)
        if path3:
            image_paths.append(path3)

        # matplotlib 静态图表
        print("  -> 坏账率分布图...")
        path4 = viz.plot_bad_rate_distribution(rules_df)
        image_paths.append(path4)

        print("  -> Top策略图...")
        path5 = viz.plot_top_strategies(rules_df, n=min(args.max_rules_display, len(rules_df)))
        image_paths.append(path5)

        print("  -> 覆盖率vs坏账率...")
        path6 = viz.plot_coverage_vs_badrate(rules_df, total_samples=len(X))
        image_paths.append(path6)

        print("  -> 特征重要性...")
        path7 = viz.plot_feature_importance(builder.feature_importance)
        image_paths.append(path7)

        print("  -> 规则深度分析...")
        path8 = viz.plot_rule_depth_analysis(rules_df)
        image_paths.append(path8)

        # Plotly 交互式图表
        print("  -> [交互式] Top策略图...")
        chart1 = viz.plot_interactive_top_strategies(filtered, n=min(args.max_rules_display, len(filtered)))
        if chart1:
            with open(chart1, "r") as f:
                html_charts.append({"id": "top_strategies", "title": "Top 高风险策略", "html": f.read()})

        print("  -> [交互式] 覆盖率vs坏账率气泡图...")
        chart2 = viz.plot_interactive_coverage_badrate(filtered, total_samples=len(X))
        if chart2:
            with open(chart2, "r") as f:
                html_charts.append({"id": "coverage_badrate", "title": "策略覆盖率 vs 坏账率", "html": f.read()})

        print("  -> [交互式] 特征重要性...")
        chart3 = viz.plot_interactive_feature_importance(builder.feature_importance)
        if chart3:
            with open(chart3, "r") as f:
                html_charts.append({"id": "feature_importance", "title": "特征重要性排序", "html": f.read(), "full_width": True})

        print("  -> [交互式] 策略叠加Lift曲线...")
        chart4 = viz.plot_interactive_lift_curve(filtered)
        if chart4:
            with open(chart4, "r") as f:
                html_charts.append({"id": "lift_curve", "title": "策略叠加效果曲线", "html": f.read(), "full_width": True})

        print("  -> [交互式] 特征-策略桑基图...")
        chart5 = viz.plot_interactive_strategy_sankey(filtered, n=min(15, len(filtered)))
        if chart5:
            with open(chart5, "r") as f:
                html_charts.append({"id": "sankey", "title": "特征→策略流向图", "html": f.read(), "full_width": True})

        print(f"  共生成 {len(image_paths)} 个静态图 + {len(html_charts)} 个交互式图表")

    # ========== 5. 特征重要性 ==========
    print("\n[5/6] 特征重要性 Top-10:")
    for i, (_, row) in enumerate(builder.feature_importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:20s}  {row['importance']:.4f}")

    # ========== 6. 导出结果 ==========
    print(f"\n[6/6] 导出结果 (format={args.export_format})...")
    exporter = StrategyExporter(output_dir=args.output_dir)

    if args.export_format in ("csv", "all"):
        exporter.to_csv(filtered, include_importance=builder.feature_importance)
    if args.export_format in ("excel", "all"):
        exporter.to_excel(
            filtered,
            importance_df=builder.feature_importance,
            summary_stats=summary,
        )
    if args.export_format in ("html", "all"):
        # 组装静态图片信息
        static_images = []
        for p in image_paths:
            fname = os.path.basename(p)
            name = os.path.splitext(fname)[0]
            title_map = {
                "tree_sklearn": "决策树 (sklearn)",
                "tree_graphviz": "决策树 (graphviz)",
                "tree_dtreeviz": "决策树 (dtreeviz)",
                "bad_rate_distribution": "坏账率 & 提升度分布",
                "top_strategies": "Top 高风险策略",
                "coverage_vs_badrate": "覆盖率 vs 坏账率",
                "feature_importance": "特征重要性排序",
                "rule_depth_analysis": "规则深度分析",
            }
            static_images.append({
                "title": title_map.get(name, name),
                "path": p,
            })

        exporter.to_html_report(
            filtered,
            summary,
            importance_df=builder.feature_importance,
            html_charts=html_charts,
            static_images=static_images,
            tree_params=tree_params,
            data_info=info,
        )

    # Premium Dashboard（始终生成）
    print("  -> Premium Dashboard...")
    from riskrules.premium_dashboard import PremiumDashboard
    dashboard = PremiumDashboard(output_dir=args.output_dir)
    dashboard.generate(
        filtered,
        summary,
        importance_df=builder.feature_importance,
        model=builder.model,
        feature_names=list(feature_names),
        data_info=info,
        tree_params=tree_params,
    )

    print("\n" + "=" * 60)
    print(f"  挖掘完成！输出目录: {os.path.abspath(args.output_dir)}")
    print(f"  策略总数: {len(rules_df)} | 筛选后: {len(filtered)}")
    print("=" * 60)

    return rules_df, filtered, builder


if __name__ == "__main__":
    main()
