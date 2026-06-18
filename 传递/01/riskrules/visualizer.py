"""
可视化模块 — 决策树可视化 + 策略分析图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互式后端

from typing import Optional, List
from sklearn.tree import DecisionTreeClassifier
from .config import DEFAULT_VIZ_PARAMS


def _setup_chinese_font():
    """自动检测并配置中文字体，解决 matplotlib 中文乱码/缺失问题"""
    import matplotlib.font_manager as fm

    # macOS / Windows / Linux 常见中文字体候选列表
    candidate_fonts = [
        "PingFang HK",
        "Heiti TC",
        "STHeiti",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    chosen = None
    for font in candidate_fonts:
        if font in available:
            chosen = font
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fm._load_fontmanager(try_read_cache=False)
    else:
        cjk = [f.name for f in fm.fontManager.ttflist if any(
            ord(c) > 0x2000 for c in f.name
        )]
        if cjk:
            print(f"未找到推荐中文字体，可用CJK字体: {cjk[:10]}")

    return chosen


# 模块加载时自动配置
_FONT_CHOSEN = _setup_chinese_font()


class StrategyVisualizer:
    """策略可视化器：决策树可视化 + 策略分析图表"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        import os

        os.makedirs(output_dir, exist_ok=True)

    # ========== 决策树可视化 ==========

    def plot_tree_sklearn(
        self,
        model: DecisionTreeClassifier,
        feature_names: List[str],
        class_names: List[str] = None,
        figsize: tuple = (20, 12),
        dpi: int = 150,
        save_path: Optional[str] = None,
    ):
        """sklearn内置plot_tree可视化"""
        from sklearn import tree

        if class_names is None:
            class_names = ["good", "bad"]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        tree.plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
        )
        ax.set_title("Decision Tree (sklearn plot_tree)", fontsize=14, fontweight="bold")

        path = save_path or f"{self.output_dir}/tree_sklearn.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_tree_graphviz(
        self,
        model: DecisionTreeClassifier,
        feature_names: List[str],
        class_names: List[str] = None,
        save_path: Optional[str] = None,
    ):
        """graphviz可视化（需要安装graphviz）"""
        from sklearn import tree

        if class_names is None:
            class_names = ["good", "bad"]

        dot_data = tree.export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )

        path = save_path or f"{self.output_dir}/tree_graphviz"
        try:
            import graphviz

            graph = graphviz.Source(dot_data)
            graph.render(path, format="png", cleanup=True)
            return f"{path}.png"
        except Exception:
            with open(f"{path}.dot", "w") as f:
                f.write(dot_data)
            return f"{path}.dot"

    def plot_tree_dtreeviz(
        self,
        model: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        class_names: dict = None,
        orientation: str = "TD",
        fancy: bool = True,
        save_path: Optional[str] = None,
    ):
        """dtreeviz可视化（最佳效果，需要安装dtreeviz）"""
        if class_names is None:
            class_names = {0: "good", 1: "bad"}

        try:
            import dtreeviz

            viz_model = dtreeviz.model(
                model,
                X_train=X,
                y_train=y,
                target_name="label",
                feature_names=X.columns.tolist(),
                class_names=class_names,
            )

            v = viz_model.view(orientation=orientation, fancy=fancy)

            path = save_path or f"{self.output_dir}/tree_dtreeviz.svg"
            v.save(path)
            return path
        except ImportError:
            print("dtreeviz 未安装，跳过此可视化。安装: pip install dtreeviz")
            return None

    def plot_tree_dtreeviz_sample_path(
        self,
        model: DecisionTreeClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        sample_idx: int = 0,
        class_names: dict = None,
        save_path: Optional[str] = None,
    ):
        """dtreeviz可视化某个样本在树上的预测路径"""
        if class_names is None:
            class_names = {0: "good", 1: "bad"}

        try:
            import dtreeviz

            viz_model = dtreeviz.model(
                model,
                X_train=X,
                y_train=y,
                target_name="label",
                feature_names=X.columns.tolist(),
                class_names=class_names,
            )

            test_x = X.iloc[sample_idx, :]
            v = viz_model.view(x=test_x, show_just_path=True)

            path = save_path or f"{self.output_dir}/tree_path_sample{sample_idx}.svg"
            v.save(path)
            return path
        except ImportError:
            print("dtreeviz 未安装，跳过此可视化。")
            return None

    # ========== 策略分析图表 ==========

    def plot_bad_rate_distribution(
        self,
        rules_df: pd.DataFrame,
        bins: int = 30,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """策略坏账率分布直方图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 左图：坏账率分布
        axes[0].hist(rules_df["bad_rate"], bins=bins, color="#E74C3C", edgecolor="white", alpha=0.8)
        axes[0].axvline(
            rules_df["bad_rate"].mean(), color="blue", linestyle="--", linewidth=2, label="均值"
        )
        axes[0].axvline(
            rules_df["bad_rate"].median(), color="green", linestyle="--", linewidth=2, label="中位数"
        )
        axes[0].set_xlabel("Bad Rate")
        axes[0].set_ylabel("规则数量")
        axes[0].set_title("策略坏账率分布", fontsize=12, fontweight="bold")
        axes[0].legend()

        # 右图：提升度分布
        axes[1].hist(rules_df["lift"], bins=bins, color="#3498DB", edgecolor="white", alpha=0.8)
        axes[1].axvline(rules_df["lift"].mean(), color="red", linestyle="--", linewidth=2, label="均值")
        axes[1].axvline(1.0, color="black", linestyle="-", linewidth=1.5, label="基准线(lift=1)")
        axes[1].set_xlabel("Lift")
        axes[1].set_ylabel("规则数量")
        axes[1].set_title("策略提升度分布", fontsize=12, fontweight="bold")
        axes[1].legend()

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/bad_rate_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_top_strategies(
        self,
        rules_df: pd.DataFrame,
        n: int = 20,
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
    ):
        """Top-N策略的bad_rate水平柱状图"""
        top = rules_df.nlargest(n, "bad_rate")

        fig, ax = plt.subplots(figsize=figsize)

        labels = [self._shorten_rule(r, max_len=60) for r in top["rule"]]

        colors = plt.cm.RdYlGn_r(top["bad_rate"] / top["bad_rate"].max())

        bars = ax.barh(range(len(top)), top["bad_rate"], color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Bad Rate")
        ax.set_title(f"Top {n} 高风险策略 (按坏账率排序)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(
                row["bad_rate"] + 0.01,
                i,
                f"BR:{row['bad_rate']:.1%} | n={int(row['total'])} | lift:{row['lift']:.1f}",
                va="center",
                fontsize=6,
            )

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/top_strategies.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_coverage_vs_badrate(
        self,
        rules_df: pd.DataFrame,
        total_samples: int,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
    ):
        """策略命中率 vs 坏账率散点图（气泡图）"""
        rules_df = rules_df.copy()
        rules_df["coverage"] = rules_df["total"] / total_samples

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            rules_df["coverage"],
            rules_df["bad_rate"],
            s=rules_df["total"] * 3,
            c=rules_df["lift"],
            cmap="RdYlGn_r",
            alpha=0.6,
            edgecolors="gray",
            linewidth=0.3,
        )

        top5 = rules_df.nlargest(5, "bad_rate")
        for _, row in top5.iterrows():
            ax.annotate(
                self._shorten_rule(row["rule"], 30),
                (row["coverage"], row["bad_rate"]),
                fontsize=6,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
            )

        ax.set_xlabel("覆盖率 (命中样本/总样本)")
        ax.set_ylabel("坏账率")
        ax.set_title("策略覆盖率 vs 坏账率", fontsize=13, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="Lift")

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/coverage_vs_badrate.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None,
    ):
        """特征重要性排序图"""
        imp = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp)))
        ax.barh(range(len(imp)), imp["importance"], color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp["feature"], fontsize=10)
        ax.set_xlabel("Importance")
        ax.set_title(f"特征重要性 Top {top_n}", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_rule_depth_analysis(
        self,
        rules_df: pd.DataFrame,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """规则深度与bad_rate的关系分析"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        depth_counts = rules_df["depth"].value_counts().sort_index()
        axes[0].bar(depth_counts.index, depth_counts.values, color="#9B59B6", edgecolor="white")
        axes[0].set_xlabel("规则深度（条件数）")
        axes[0].set_ylabel("规则数量")
        axes[0].set_title("规则深度分布", fontsize=12, fontweight="bold")

        depth_data = [
            rules_df[rules_df["depth"] == d]["bad_rate"].values
            for d in sorted(rules_df["depth"].unique())
        ]
        bp = axes[1].boxplot(depth_data, labels=sorted(rules_df["depth"].unique()))
        axes[1].set_xlabel("规则深度")
        axes[1].set_ylabel("Bad Rate")
        axes[1].set_title("不同深度策略的Bad Rate分布", fontsize=12, fontweight="bold")

        plt.tight_layout()
        path = save_path or f"{self.output_dir}/rule_depth_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_rules_heatmap(
        self,
        rules_df: pd.DataFrame,
        n: int = 20,
        figsize: tuple = (16, 10),
        save_path: Optional[str] = None,
    ):
        """策略特征使用热力图 — 展示top策略中各特征的使用频率"""
        top_rules = rules_df.nlargest(n, "bad_rate")

        feature_usage = {}
        for rule in top_rules["rule"]:
            conditions = rule.split(" & ")
            for cond in conditions:
                feat = cond.split("<=")[0].split(">")[0].strip()
                feature_usage[feat] = feature_usage.get(feat, 0) + 1

        features = sorted(feature_usage.keys(), key=lambda x: feature_usage[x], reverse=True)
        if len(features) == 0:
            return None

        matrix = []
        for _, row in top_rules.iterrows():
            row_data = []
            for feat in features:
                row_data.append(1 if feat in row["rule"] else 0)
            matrix.append(row_data)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [self._shorten_rule(r, 50) for r in top_rules["rule"]], fontsize=6
        )
        ax.set_title(f"Top {n} 策略特征使用热力图", fontsize=13, fontweight="bold")
        ax.set_xlabel("特征")
        ax.set_ylabel("策略规则")

        plt.colorbar(im, ax=ax, label="使用该特征")
        plt.tight_layout()
        path = save_path or f"{self.output_dir}/rules_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _shorten_rule(rule: str, max_len: int = 50) -> str:
        """缩短规则字符串用于显示"""
        if len(rule) <= max_len:
            return rule
        return rule[: max_len - 3] + "..."

    # ========== Plotly 交互式图表 ==========

    def plot_interactive_top_strategies(
        self,
        rules_df: pd.DataFrame,
        n: int = 30,
        save_path: Optional[str] = None,
    ) -> str:
        """交互式 Top-N 策略柱状图"""
        import plotly.graph_objects as go
        import plotly.express as px

        top = rules_df.nlargest(n, "bad_rate").iloc[::-1]

        labels = [self._shorten_rule(r, 80) for r in top["rule"]]
        customdata = list(zip(
            top["rule"],
            top["bad_rate"],
            top["lift"],
            top["total"].astype(int),
            top["depth"].astype(int),
        ))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top["bad_rate"],
            y=labels,
            orientation="h",
            marker=dict(
                color=top["bad_rate"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="坏账率"),
                line=dict(color="rgba(180,40,40,0.3)", width=0.5),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>规则:</b> %{customdata[0]}<br>"
                "<b>坏账率:</b> %{customdata[1]:.2%}<br>"
                "<b>提升度:</b> %{customdata[2]:.1f}x<br>"
                "<b>样本量:</b> %{customdata[3]}<br>"
                "<b>深度:</b> %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            text=[f"BR:{br:.1%} | Lift:{lf:.1f}x | N:{int(n)}" for br, lf, n in zip(top["bad_rate"], top["lift"], top["total"])],
            textposition="outside",
            textfont=dict(size=10, color="#333"),
            insidetextanchor="start",
        ))

        fig.update_layout(
            title=dict(text=f"Top {n} 高风险策略", font=dict(size=20, family="Arial, sans-serif"), x=0.5),
            xaxis=dict(title="坏账率", tickformat=".0%", gridcolor="#f0f0f0"),
            yaxis=dict(title="", tickfont=dict(size=9)),
            height=max(600, n * 28),
            margin=dict(l=20, r=40, t=60, b=20),
            paper_bgcolor="white",
            plot_bgcolor="#fafafa",
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        overall_br = rules_df["bad"].sum() / rules_df["total"].sum()
        fig.add_vline(x=overall_br, line_dash="dash", line_color="#3498db",
                      annotation_text=f"整体坏账率 {overall_br:.1%}",
                      annotation_position="top right")

        path = save_path or f"{self.output_dir}/interactive_top_strategies.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        fig.write_html(path.replace(".html", ".json"), include_plotlyjs=False, full_html=False)
        return path

    def plot_interactive_coverage_badrate(
        self,
        rules_df: pd.DataFrame,
        total_samples: int,
        save_path: Optional[str] = None,
    ) -> str:
        """交互式覆盖率 vs 坏账率气泡图"""
        import plotly.graph_objects as go

        df = rules_df.copy()
        df["coverage"] = df["total"] / total_samples

        conditions = [
            df["bad_rate"] >= 0.25,
            (df["bad_rate"] >= 0.10) & (df["bad_rate"] < 0.25),
            df["bad_rate"] < 0.10,
        ]
        colors = ["#e74c3c", "#f39c12", "#3498db"]
        labels_risk = ["高风险 (≥25%)", "中风险 (10%-25%)", "低风险 (<10%)"]

        fig = go.Figure()
        for cond, color, label in zip(conditions, colors, labels_risk):
            subset = df[cond]
            if len(subset) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=subset["coverage"],
                y=subset["bad_rate"],
                mode="markers+text",
                name=label,
                marker=dict(
                    size=subset["total"] / total_samples * 800,
                    sizemode="area",
                    sizeref=0.01,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=[self._shorten_rule(r, 30) for r in subset["rule"]],
                textposition="top center",
                textfont=dict(size=7),
                customdata=list(zip(
                    subset["rule"],
                    subset["bad_rate"],
                    subset["lift"],
                    subset["total"].astype(int),
                    subset["coverage"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "坏账率: %{customdata[1]:.2%}<br>"
                    "提升度: %{customdata[2]:.1f}x<br>"
                    "样本量: %{customdata[3]}<br>"
                    "覆盖率: %{customdata[4]:.2%}<br>"
                    "<extra></extra>"
                ),
            ))

        overall_br = rules_df["bad"].sum() / rules_df["total"].sum()
        fig.add_hline(y=overall_br, line_dash="dash", line_color="gray",
                      annotation_text=f"整体坏账率 {overall_br:.1%}")

        fig.update_layout(
            title=dict(text="策略覆盖率 vs 坏账率", font=dict(size=20), x=0.5),
            xaxis=dict(title="覆盖率 (命中样本/总样本)", tickformat=".1%", gridcolor="#f0f0f0"),
            yaxis=dict(title="坏账率", tickformat=".0%", gridcolor="#f0f0f0"),
            height=650,
            paper_bgcolor="white",
            plot_bgcolor="#fafafa",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        path = save_path or f"{self.output_dir}/interactive_coverage_badrate.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> str:
        """交互式特征重要性图"""
        import plotly.graph_objects as go

        imp = importance_df.head(top_n).iloc[::-1]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=imp["importance"],
            y=imp["feature"],
            orientation="h",
            marker=dict(
                color=imp["importance"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="重要性"),
                line=dict(width=0),
            ),
            text=[f"{v:.4f}" for v in imp["importance"]],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"特征重要性 Top {top_n}", font=dict(size=20), x=0.5),
            xaxis=dict(title="重要性", gridcolor="#f0f0f0"),
            yaxis=dict(title="", tickfont=dict(size=11)),
            height=max(450, top_n * 25),
            paper_bgcolor="white",
            plot_bgcolor="#fafafa",
        )

        path = save_path or f"{self.output_dir}/interactive_feature_importance.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_lift_curve(
        self,
        rules_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> str:
        """交互式 Lift-覆盖率 双轴图"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df = rules_df.sort_values("bad_rate", ascending=False).reset_index(drop=True)
        df["cum_bad"] = df["bad"].cumsum()
        df["cum_total"] = df["total"].cumsum()
        total_bad = df["bad"].sum()
        total_all = df["total"].sum()
        df["cum_bad_rate"] = df["cum_bad"] / df["cum_total"]
        df["cum_coverage"] = df["cum_total"] / total_all
        df["cum_recall"] = df["cum_bad"] / total_bad

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df.index + 1, y=df["cum_bad_rate"],
                mode="lines+markers", name="累计坏账率",
                line=dict(width=2, color="#e74c3c"),
                marker=dict(size=5),
                hovertemplate="策略#%{x}: 累计坏账率=%{y:.2%}<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index + 1, y=df["cum_recall"],
                mode="lines+markers", name="累计召回率",
                line=dict(width=2, color="#3498db", dash="dash"),
                marker=dict(size=5),
                hovertemplate="策略#%{x}: 召回率=%{y:.2%}<extra></extra>",
            ),
            secondary_y=True,
        )

        overall_br = total_bad / total_all
        fig.add_hline(y=overall_br, line_dash="dot", line_color="gray",
                      annotation_text=f"整体坏账率 {overall_br:.1%}")

        fig.update_layout(
            title=dict(text="策略叠加效果曲线", font=dict(size=20), x=0.5),
            xaxis=dict(title="策略数量 (按坏账率降序累加)", gridcolor="#f0f0f0"),
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="#fafafa",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.12),
        )
        fig.update_yaxes(title_text="累计坏账率", tickformat=".0%", secondary_y=False, gridcolor="#f0f0f0")
        fig.update_yaxes(title_text="累计召回率", tickformat=".0%", secondary_y=True)

        path = save_path or f"{self.output_dir}/interactive_lift_curve.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_strategy_sankey(
        self,
        rules_df: pd.DataFrame,
        n: int = 10,
        save_path: Optional[str] = None,
    ) -> str:
        """交互式桑基图 — 展示Top策略的特征→规则→风险流向"""
        import plotly.graph_objects as go

        top = rules_df.nlargest(n, "bad_rate")

        feature_to_idx = {}
        rule_to_idx = {}
        sources, targets, values = [], [], []
        labels_list = []

        for _, row in top.iterrows():
            rule_label = self._shorten_rule(row["rule"], 60)
            if rule_label not in rule_to_idx:
                rule_to_idx[rule_label] = len(labels_list)
                labels_list.append(rule_label)

            conditions = row["rule"].split(" & ")
            for cond in conditions:
                feat = cond.split("<=")[0].split(">")[0].strip()
                if feat not in feature_to_idx:
                    feature_to_idx[feat] = len(labels_list)
                    labels_list.append(feat)
                sources.append(feature_to_idx[feat])
                targets.append(rule_to_idx[rule_label])
                values.append(1)

        if not sources:
            return None

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color="black", width=0.5),
                label=labels_list,
                color=["#3498db"] * len(feature_to_idx) + ["#e74c3c"] * len(rule_to_idx),
            ),
            link=dict(
                source=sources, target=targets, value=values,
                color=["rgba(180,180,180,0.3)"] * len(sources),
            ),
        )])

        fig.update_layout(
            title=dict(text=f"Top {n} 策略特征流向图", font=dict(size=20), x=0.5),
            height=550,
            paper_bgcolor="white",
        )

        path = save_path or f"{self.output_dir}/interactive_sankey.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_3d_scatter(
        self,
        rules_df: pd.DataFrame,
        total_samples: int,
        save_path: Optional[str] = None,
    ) -> str:
        """3D策略空间散点图 — Bad Rate × Lift × Coverage 三维探索"""
        import plotly.graph_objects as go

        df = rules_df.copy()
        df["coverage"] = df["total"] / total_samples
        df["risk_level"] = df["bad_rate"].apply(
            lambda x: "高风险" if x >= 0.25 else ("中风险" if x >= 0.1 else "低风险")
        )
        color_map = {"高风险": "#e74c3c", "中风险": "#f39c12", "低风险": "#3498db"}

        fig = go.Figure()
        for level, color in color_map.items():
            subset = df[df["risk_level"] == level]
            if len(subset) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=subset["bad_rate"],
                y=subset["lift"],
                z=subset["coverage"],
                mode="markers+text",
                name=level,
                marker=dict(
                    size=subset["total"] / total_samples * 400,
                    sizemode="area",
                    sizeref=0.005,
                    color=color,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=[self._shorten_rule(r, 25) for r in subset["rule"]],
                textposition="top center",
                textfont=dict(size=7),
                customdata=list(zip(
                    subset["rule"], subset["bad_rate"], subset["lift"],
                    subset["total"].astype(int), subset["coverage"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "坏账率: %{customdata[1]:.2%}<br>"
                    "提升度: %{customdata[2]:.1f}x<br>"
                    "样本量: %{customdata[3]}<br>"
                    "覆盖率: %{customdata[4]:.2%}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=dict(text="3D 策略空间探索", font=dict(size=20), x=0.5),
            scene=dict(
                xaxis=dict(title="坏账率", tickformat=".0%", gridcolor="#f0f0f0"),
                yaxis=dict(title="提升度 (Lift)", gridcolor="#f0f0f0"),
                zaxis=dict(title="覆盖率", tickformat=".1%", gridcolor="#f0f0f0"),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                bgcolor="#fafafa",
            ),
            height=700,
            paper_bgcolor="white",
            legend=dict(orientation="h", y=1.02),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        path = save_path or f"{self.output_dir}/interactive_3d_scatter.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_parallel_coordinates(
        self,
        rules_df: pd.DataFrame,
        total_samples: int,
        save_path: Optional[str] = None,
    ) -> str:
        """平行坐标图 — 多维度策略对比分析"""
        import plotly.express as px

        df = rules_df.copy()
        df["coverage"] = df["total"] / total_samples
        df["risk_score"] = df["bad_rate"] * df["lift"]
        df["risk_label"] = df["bad_rate"].apply(
            lambda x: "高风险" if x >= 0.25 else ("中风险" if x >= 0.1 else "低风险")
        )

        dimensions = [
            dict(label="坏账率", values=df["bad_rate"], tickformat=".0%",
                 range=[df["bad_rate"].min(), df["bad_rate"].max()]),
            dict(label="提升度", values=df["lift"],
                 range=[df["lift"].min(), df["lift"].max()]),
            dict(label="覆盖率", values=df["coverage"], tickformat=".1%",
                 range=[df["coverage"].min(), df["coverage"].max()]),
            dict(label="样本量", values=df["total"],
                 range=[df["total"].min(), df["total"].max()]),
            dict(label="深度", values=df["depth"],
                 range=[df["depth"].min(), df["depth"].max()]),
            dict(label="风险评分", values=df["risk_score"],
                 range=[df["risk_score"].min(), df["risk_score"].max()]),
        ]

        fig = px.parallel_coordinates(
            df,
            dimensions=dimensions,
            color="bad_rate",
            color_continuous_scale=px.colors.sequential.Reds,
            labels={},
        )

        fig.update_layout(
            title=dict(text="多维度策略平行坐标图", font=dict(size=20), x=0.5),
            height=550,
            paper_bgcolor="white",
            margin=dict(t=60, b=20, l=60, r=60),
        )

        path = save_path or f"{self.output_dir}/interactive_parallel_coords.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_treemap(
        self,
        rules_df: pd.DataFrame,
        importance_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Treemap矩形树图 — 策略层级结构可视化"""
        import plotly.express as px

        df = rules_df.nlargest(30, "bad_rate").copy()
        df["risk_level"] = df["bad_rate"].apply(
            lambda x: "高风险 ≥25%" if x >= 0.25
            else ("中风险 10-25%" if x >= 0.1 else "低风险 <10%")
        )

        records = []
        for _, row in df.iterrows():
            conditions = row["rule"].split(" & ")
            primary_feat = conditions[0].split("<=")[0].split(">")[0].strip()
            records.append({
                "risk_level": row["risk_level"],
                "primary_feature": primary_feat,
                "rule_short": self._shorten_rule(row["rule"], 50),
                "bad_rate": row["bad_rate"],
                "lift": row["lift"],
                "total": row["total"],
            })

        treemap_df = pd.DataFrame(records)

        fig = px.treemap(
            treemap_df,
            path=["risk_level", "primary_feature", "rule_short"],
            values="total",
            color="bad_rate",
            color_continuous_scale="Reds",
            hover_data={
                "bad_rate": ":.2%", "lift": ":.1f", "total": True,
            },
        )

        fig.update_traces(
            textinfo="label+value",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "坏账率: %{customdata[0]:.2%}<br>"
                "提升度: %{customdata[1]:.1f}x<br>"
                "样本量: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            textfont=dict(size=11),
        )

        fig.update_layout(
            title=dict(text="策略层级Treemap", font=dict(size=20), x=0.5),
            height=600,
            paper_bgcolor="white",
            margin=dict(t=60, b=10, l=10, r=10),
        )

        path = save_path or f"{self.output_dir}/interactive_treemap.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path

    def plot_interactive_radar(
        self,
        rules_df: pd.DataFrame,
        n: int = 6,
        save_path: Optional[str] = None,
    ) -> str:
        """雷达图 — Top策略多维指标对比"""
        import plotly.graph_objects as go

        top = rules_df.nlargest(n, "bad_rate")

        metrics = {
            "坏账率": top["bad_rate"],
            "提升度": (top["lift"] / top["lift"].max()),
            "样本覆盖率": (top["total"] / top["total"].max()),
            "规则简洁度": (1 - (top["depth"] / top["depth"].max())),
            "坏账数": (top["bad"] / top["bad"].max()),
        }

        fig = go.Figure()
        colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]

        for i, (_, row) in enumerate(top.iterrows()):
            values = [
                metrics["坏账率"].iloc[i],
                metrics["提升度"].iloc[i],
                metrics["样本覆盖率"].iloc[i],
                metrics["规则简洁度"].iloc[i],
                metrics["坏账数"].iloc[i],
            ]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=list(metrics.keys()) + [list(metrics.keys())[0]],
                name=self._shorten_rule(row["rule"], 35),
                fill="toself",
                opacity=0.35,
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra>%{fullData.name}</extra>",
            ))

        fig.update_layout(
            title=dict(text="Top策略多维雷达图", font=dict(size=20), x=0.5),
            polar=dict(
                radialaxis=dict(range=[0, 1.05], showline=False, gridcolor="#f0f0f0"),
                angularaxis=dict(gridcolor="#f0f0f0"),
                bgcolor="#fafafa",
            ),
            height=600,
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.1, font=dict(size=9)),
        )

        path = save_path or f"{self.output_dir}/interactive_radar.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=False)
        return path
