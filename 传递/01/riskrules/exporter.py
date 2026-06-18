"""
策略导出模块 — 支持 CSV / Excel / 精美HTML报告
"""

import pandas as pd
import os
import json
import base64
from typing import Optional, List
from datetime import datetime


class StrategyExporter:
    """策略导出器：将挖掘结果导出为多种格式"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def to_csv(
        self,
        rules_df: pd.DataFrame,
        filename: Optional[str] = None,
        include_importance: Optional[pd.DataFrame] = None,
    ) -> str:
        """导出策略到CSV文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"strategies_{timestamp}.csv"
        path = os.path.join(self.output_dir, filename)

        df = rules_df.sort_values("bad_rate", ascending=False)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"策略已导出: {path} ({len(df)} 条规则)")

        if include_importance is not None:
            imp_path = os.path.join(self.output_dir, f"feature_importance_{timestamp}.csv")
            include_importance.to_csv(imp_path, index=False, encoding="utf-8-sig")

        return path

    def to_excel(
        self,
        rules_df: pd.DataFrame,
        filename: Optional[str] = None,
        importance_df: Optional[pd.DataFrame] = None,
        summary_stats: Optional[dict] = None,
    ) -> str:
        """导出策略到Excel文件（多sheet）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"strategies_{timestamp}.xlsx"
        path = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            rules_df.sort_values("bad_rate", ascending=False).to_excel(
                writer, sheet_name="策略明细", index=False
            )
            if importance_df is not None:
                importance_df.to_excel(writer, sheet_name="特征重要性", index=False)
            if summary_stats:
                stats_df = pd.DataFrame({
                    "指标": list(summary_stats.keys()),
                    "值": list(summary_stats.values()),
                })
                stats_df.to_excel(writer, sheet_name="汇总统计", index=False)

        print(f"策略已导出: {path}")
        return path

    def to_html_report(
        self,
        rules_df: pd.DataFrame,
        summary_stats: dict,
        importance_df: Optional[pd.DataFrame] = None,
        html_charts: Optional[List[dict]] = None,
        static_images: Optional[List[dict]] = None,
        tree_params: Optional[dict] = None,
        data_info: Optional[dict] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        导出自包含的交互式HTML综合报告。

        Parameters
        ----------
        rules_df : pd.DataFrame - 策略明细
        summary_stats : dict - 汇总统计
        importance_df : pd.DataFrame, optional - 特征重要性
        html_charts : list of dict, optional - 每个元素 {"id": str, "title": str, "html": str}
        static_images : list of dict, optional - 每个元素 {"title": str, "path": str}
        tree_params : dict, optional - 决策树参数
        data_info : dict, optional - 数据概览
        filename : str, optional - 输出文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"strategy_report_{timestamp}.html"
        path = os.path.join(self.output_dir, filename)

        top30 = rules_df.nlargest(30, "bad_rate")

        report = self._build_report_html(
            rules_df=rules_df,
            top30=top30,
            summary_stats=summary_stats,
            importance_df=importance_df,
            html_charts=html_charts or [],
            static_images=static_images or [],
            tree_params=tree_params or {},
            data_info=data_info or {},
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"HTML报告已导出: {path}")
        return path

    # ==================== HTML 构建 ====================

    def _build_report_html(self, rules_df, top30, summary_stats,
                           importance_df, html_charts, static_images, tree_params,
                           data_info, timestamp) -> str:
        """构建完整的HTML报告"""

        overall_br = rules_df["bad"].sum() / rules_df["total"].sum()

        high_risk = len(rules_df[rules_df["bad_rate"] >= 0.25])
        mid_risk = len(rules_df[(rules_df["bad_rate"] >= 0.10) & (rules_df["bad_rate"] < 0.25)])
        low_risk = len(rules_df[rules_df["bad_rate"] < 0.10])

        charts_section = self._build_charts_section(html_charts)
        static_section = self._build_static_images_section(static_images)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>风控策略挖掘报告 - RiskRules</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        :root {{
            --bg: #f0f2f5;
            --card-bg: #ffffff;
            --primary: #1a73e8;
            --danger: #e74c3c;
            --warning: #f39c12;
            --success: #27ae60;
            --text: #2c3e50;
            --text-secondary: #7f8c8d;
            --border: #e8ecf1;
            --shadow: 0 2px 12px rgba(0,0,0,0.06);
            --radius: 12px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .navbar {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 28px 40px;
            position: sticky; top: 0; z-index: 100;
        }}
        .navbar h1 {{ font-size: 26px; font-weight: 700; letter-spacing: 1px; }}
        .navbar .subtitle {{ font-size: 14px; opacity: 0.75; margin-top: 6px; }}
        .navbar .meta {{ display: flex; gap: 30px; margin-top: 12px; font-size: 13px; opacity: 0.8; }}
        .navbar .meta span {{ display: flex; align-items: center; gap: 6px; }}

        .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border-radius: var(--radius);
            padding: 20px 24px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--primary);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }}
        .kpi-card.danger {{ border-left-color: var(--danger); }}
        .kpi-card.warning {{ border-left-color: var(--warning); }}
        .kpi-card.success {{ border-left-color: var(--success); }}
        .kpi-card .kpi-value {{
            font-size: 32px; font-weight: 700; line-height: 1.1;
            background: linear-gradient(135deg, #2c3e50, #34495e);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .kpi-card.danger .kpi-value {{ background: linear-gradient(135deg, #e74c3c, #c0392b); -webkit-background-clip: text; background-clip: text; }}
        .kpi-card.warning .kpi-value {{ background: linear-gradient(135deg, #f39c12, #e67e22); -webkit-background-clip: text; background-clip: text; }}
        .kpi-card.success .kpi-value {{ background: linear-gradient(135deg, #27ae60, #2ecc71); -webkit-background-clip: text; background-clip: text; }}
        .kpi-card .kpi-label {{ font-size: 13px; color: var(--text-secondary); margin-top: 4px; }}
        .kpi-card .kpi-sub {{ font-size: 11px; color: var(--text-secondary); margin-top: 2px; }}

        .section {{
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 28px 32px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 18px; font-weight: 700;
            padding-bottom: 16px; margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
            display: flex; align-items: center; gap: 10px;
        }}
        .section-title .icon {{ font-size: 22px; }}

        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .chart-grid .chart-full {{ grid-column: 1 / -1; }}
        .chart-box {{
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 16px;
            min-height: 400px;
        }}
        .chart-box.full {{ grid-column: 1 / -1; }}
        @media (max-width: 1000px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}

        .table-wrapper {{ overflow-x: auto; }}
        table.strategy-table {{
            width: 100%; border-collapse: collapse;
            font-size: 13px;
        }}
        table.strategy-table thead th {{
            background: linear-gradient(135deg, #1a1a2e, #0f3460);
            color: white; padding: 12px 14px; text-align: left;
            font-weight: 600; font-size: 12px; text-transform: uppercase;
            letter-spacing: 0.5px;
            position: sticky; top: 0;
        }}
        table.strategy-table tbody td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        table.strategy-table tbody tr {{ transition: background 0.15s; }}
        table.strategy-table tbody tr:hover {{ background: #f0f4ff; }}
        table.strategy-table .rule-cell {{
            max-width: 450px; font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
            font-size: 11px; line-height: 1.5; word-break: break-all;
        }}
        table.strategy-table .rule-cell .cond {{
            display: inline-block;
            background: #f0f4ff; color: #1a73e8;
            padding: 2px 6px; border-radius: 4px; margin: 1px 2px;
            font-size: 10px;
        }}

        .badge {{
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 11px; font-weight: 700; white-space: nowrap;
        }}
        .badge-danger {{ background: #ffeaea; color: #c0392b; }}
        .badge-warning {{ background: #fef5e7; color: #e67e22; }}
        .badge-success {{ background: #eafaf1; color: #27ae60; }}
        .badge-info {{ background: #eaf2f8; color: #2980b9; }}

        .progress-bar {{
            width: 100%; height: 6px; background: #ecf0f1; border-radius: 3px;
            overflow: hidden; margin-top: 2px;
        }}
        .progress-bar .fill {{ height: 100%; border-radius: 3px; transition: width 0.4s; }}
        .fill-danger {{ background: var(--danger); }}
        .fill-warning {{ background: var(--warning); }}
        .fill-info {{ background: var(--primary); }}

        .strategy-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}
        .strategy-card {{
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 20px 24px;
            border-top: 3px solid var(--danger);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .strategy-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }}
        .strategy-card.rank-1 {{ border-top-color: #c0392b; }}
        .strategy-card.rank-2 {{ border-top-color: #e74c3c; }}
        .strategy-card.rank-3 {{ border-top-color: #e67e22; }}
        .strategy-card .card-rank {{
            font-size: 11px; font-weight: 700; color: var(--text-secondary);
            text-transform: uppercase; letter-spacing: 1px;
        }}
        .strategy-card .card-rule {{
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px; line-height: 1.7; margin: 10px 0;
            background: #f8f9fb; padding: 12px 16px; border-radius: 8px;
            color: #2c3e50; word-break: break-all;
        }}
        .strategy-card .card-stats {{
            display: flex; gap: 20px; margin-top: 12px;
        }}
        .strategy-card .card-stat {{
            text-align: center;
        }}
        .strategy-card .card-stat .val {{
            font-size: 22px; font-weight: 700; color: var(--danger);
        }}
        .strategy-card .card-stat .lbl {{
            font-size: 11px; color: var(--text-secondary);
        }}

        .footer {{
            text-align: center; padding: 30px;
            color: var(--text-secondary); font-size: 12px;
        }}

        .tabs {{ display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }}
        .tab-btn {{
            padding: 8px 20px; border: 1px solid var(--border);
            background: white; border-radius: 20px; cursor: pointer;
            font-size: 13px; font-weight: 500; transition: all 0.2s;
        }}
        .tab-btn.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
        .tab-btn:hover:not(.active) {{ background: #f0f4ff; }}

        .risk-meter {{
            display: flex; align-items: center; gap: 8px;
        }}
        .risk-meter .dots {{
            display: flex; gap: 3px;
        }}
        .risk-meter .dot {{
            width: 8px; height: 8px; border-radius: 50%; background: #ddd;
        }}
        .risk-meter .dot.active.high {{ background: var(--danger); }}
        .risk-meter .dot.active.mid {{ background: var(--warning); }}
        .risk-meter .dot.active.low {{ background: var(--success); }}
    </style>
</head>
<body>

<div class="navbar">
    <h1>RiskRules · 风控策略挖掘报告</h1>
    <div class="subtitle">基于决策树算法的自动化策略发现与推荐系统</div>
    <div class="meta">
        <span>生成时间: {timestamp}</span>
        <span>max_depth={tree_params.get('max_depth', '-')}, min_samples_leaf={tree_params.get('min_samples_leaf', '-')}</span>
        <span>样本量: {data_info.get('n_samples', '-'):,}</span>
    </div>
</div>

<div class="container">

<div class="kpi-grid">
    <div class="kpi-card danger">
        <div class="kpi-value">{summary_stats['total_rules']}</div>
        <div class="kpi-label">挖掘规则总数</div>
        <div class="kpi-sub">筛选后有效策略</div>
    </div>
    <div class="kpi-card danger">
        <div class="kpi-value">{overall_br:.2%}</div>
        <div class="kpi-label">整体坏账率</div>
        <div class="kpi-sub">bad={data_info.get('n_bad', '-'):,}</div>
    </div>
    <div class="kpi-card danger">
        <div class="kpi-value">{summary_stats['max_bad_rate']:.1%}</div>
        <div class="kpi-label">最高坏账率</div>
        <div class="kpi-sub">最优策略风险水平</div>
    </div>
    <div class="kpi-card warning">
        <div class="kpi-value">{summary_stats['max_lift']:.1f}x</div>
        <div class="kpi-label">最高提升度</div>
        <div class="kpi-sub">相对整体坏账率倍数</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{summary_stats['avg_bad_rate']:.1%}</div>
        <div class="kpi-label">平均坏账率</div>
        <div class="kpi-sub">所有策略均值</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{summary_stats['avg_depth']:.1f}</div>
        <div class="kpi-label">平均规则深度</div>
        <div class="kpi-sub">条件数均值</div>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">&#x1F3AF;</span> 策略风险分级概览</div>
    <div style="display: flex; gap: 30px; align-items: center; flex-wrap: wrap;">
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 42px; font-weight: 700; color: #e74c3c;">{high_risk}</div>
            <div style="color: #7f8c8d;">高风险 (>=25%)</div>
        </div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 42px; font-weight: 700; color: #f39c12;">{mid_risk}</div>
            <div style="color: #7f8c8d;">中风险 (10-25%)</div>
        </div>
        <div style="text-align: center; flex: 1; min-width: 120px;">
            <div style="font-size: 42px; font-weight: 700; color: #27ae60;">{low_risk}</div>
            <div style="color: #7f8c8d;">低风险 (&lt;10%)</div>
        </div>
        <div style="flex: 2; min-width: 250px;">
            <div style="margin-bottom: 8px; font-size: 12px; color: #7f8c8d;">风险分布</div>
            <div style="display: flex; height: 22px; border-radius: 11px; overflow: hidden; background: #ecf0f1;">
                <div style="width: {high_risk/max(1,len(rules_df))*100:.0f}%; background: #e74c3c;" title="高风险 {high_risk}条"></div>
                <div style="width: {mid_risk/max(1,len(rules_df))*100:.0f}%; background: #f39c12;" title="中风险 {mid_risk}条"></div>
                <div style="width: {low_risk/max(1,len(rules_df))*100:.0f}%; background: #27ae60;" title="低风险 {low_risk}条"></div>
            </div>
        </div>
    </div>
</div>

{charts_section}

{static_section}

<div class="section">
    <div class="section-title"><span class="icon">&#x1F3C6;</span> 策略明细表 — Top 30 高风险策略</div>
    <div class="table-wrapper">
        <table class="strategy-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>策略规则</th>
                    <th>坏账率</th>
                    <th>风险等级</th>
                    <th>提升度</th>
                    <th>样本量</th>
                    <th>坏账数</th>
                    <th>深度</th>
                </tr>
            </thead>
            <tbody>
                {self._build_table_rows(top30)}
            </tbody>
        </table>
    </div>
</div>

<div class="section">
    <div class="section-title"><span class="icon">&#x1F4CE;</span> Top 6 策略详情</div>
    <div class="strategy-cards">
        {self._build_strategy_cards(rules_df.nlargest(6, 'bad_rate'))}
    </div>
</div>

{"".join(self._build_importance_section(importance_df)) if importance_df is not None and not any('feature_importance' in (c.get('id', '') or '') for c in (html_charts or [])) else ""}

<div class="footer">
    <p>RiskRules · 基于决策树XiaoWuGe算法 · 自动化策略发现与推荐</p>
    <p>Generated at {timestamp}</p>
</div>

</div>

<script>
document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', function() {{
        const tabGroup = this.parentElement;
        tabGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const targetId = this.dataset.target;
        const section = tabGroup.parentElement;
        section.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
        const target = section.querySelector('#' + targetId);
        if (target) target.style.display = 'block';
    }});
}});
</script>

</body>
</html>"""

    # ==================== HTML 组件构建 ====================

    def _build_table_rows(self, top30: pd.DataFrame) -> str:
        rows = []
        for i, (_, row) in enumerate(top30.iterrows()):
            br = row["bad_rate"]
            risk_class, risk_label = (
                ("danger", "高风险") if br >= 0.25
                else ("warning", "中风险") if br >= 0.10
                else ("success", "低风险")
            )
            rule_parts = row["rule"].split(" & ")
            rule_html = " ".join(
                f'<span class="cond">{p.strip()}</span>' for p in rule_parts
            )

            bar_class = "fill-danger" if br >= 0.25 else "fill-warning" if br >= 0.10 else "fill-info"
            br_display = f"""
                <div style="font-weight:600;">{br:.2%}</div>
                <div class="progress-bar"><div class="fill {bar_class}" style="width:{min(br*100, 100):.0f}%;"></div></div>
            """

            rows.append(f"""<tr>
                <td style="font-weight:700; color:#7f8c8d;">{i+1}</td>
                <td class="rule-cell">{rule_html}</td>
                <td>{br_display}</td>
                <td><span class="badge badge-{risk_class}">{risk_label}</span></td>
                <td style="font-weight:600;">{row['lift']:.1f}x</td>
                <td>{int(row['total']):,}</td>
                <td style="color:#c0392b; font-weight:600;">{int(row['bad'])}</td>
                <td>{int(row['depth'])}</td>
            </tr>""")
        return "\n".join(rows)

    def _build_strategy_cards(self, top6: pd.DataFrame) -> str:
        cards = []
        for i, (_, row) in enumerate(top6.iterrows()):
            rank_class = f"rank-{i+1}" if i < 3 else ""
            rule_parts = row["rule"].split(" & ")
            rule_html = "<br>".join(f"&nbsp;&nbsp;{'&#x2514;&#x2500;' if j>0 else '&#x251C;&#x2500;'} {p.strip()}" for j, p in enumerate(rule_parts))

            cards.append(f"""<div class="strategy-card {rank_class}">
                <div class="card-rank">策略 #{i+1}</div>
                <div class="card-rule">{rule_html}</div>
                <div class="card-stats">
                    <div class="card-stat">
                        <div class="val">{row['bad_rate']:.1%}</div>
                        <div class="lbl">坏账率</div>
                    </div>
                    <div class="card-stat">
                        <div class="val">{row['lift']:.1f}x</div>
                        <div class="lbl">提升度</div>
                    </div>
                    <div class="card-stat">
                        <div class="val">{int(row['total']):,}</div>
                        <div class="lbl">命中样本</div>
                    </div>
                    <div class="card-stat">
                        <div class="val">{int(row['bad'])}</div>
                        <div class="lbl">坏账数</div>
                    </div>
                    <div class="card-stat">
                        <div class="val">{int(row['depth'])}</div>
                        <div class="lbl">条件数</div>
                    </div>
                </div>
                <div style="margin-top:12px;">
                    <div class="risk-meter">
                        <span style="font-size:11px; color:#7f8c8d;">风险:</span>
                        <div class="dots">
                            {self._build_risk_dots(row['bad_rate'])}
                        </div>
                        <span style="font-size:11px; font-weight:600; color:{'#e74c3c' if row['bad_rate']>=0.25 else '#f39c12' if row['bad_rate']>=0.1 else '#27ae60'};">
                            {'极高' if row['bad_rate']>=0.5 else '高' if row['bad_rate']>=0.25 else '中' if row['bad_rate']>=0.1 else '低'}
                        </span>
                    </div>
                </div>
            </div>""")
        return "\n".join(cards)

    @staticmethod
    def _build_risk_dots(bad_rate: float) -> str:
        level = 5 if bad_rate >= 0.5 else 4 if bad_rate >= 0.25 else 3 if bad_rate >= 0.15 else 2 if bad_rate >= 0.08 else 1
        cls = "high" if level >= 4 else "mid" if level >= 3 else "low"
        return "".join(
            f'<div class="dot active {cls}"></div>' if i < level else '<div class="dot"></div>'
            for i in range(5)
        )

    def _build_charts_section(self, html_charts: List[dict]) -> str:
        if not html_charts:
            return ""

        parts = []
        for chart in html_charts:
            chart_id = chart.get("id", "")
            title = chart.get("title", "")
            html_content = chart.get("html", "")
            is_full = chart.get("full_width", False)

            if not html_content or len(html_content.strip()) < 100:
                continue

            div_start = html_content.find("<div")
            if div_start > 0:
                html_content = html_content[div_start:]

            box_class = "chart-box full" if is_full else "chart-box"
            parts.append(f"""<div class="{box_class}">
                <div class="section-title" style="margin-bottom:8px; border:none; padding-bottom:8px;">
                    <span class="icon">&#x1F4C8;</span> {title}
                </div>
                <div id="{chart_id}_container" style="width:100%; min-height:420px;">
                    {html_content}
                </div>
            </div>""")

        if not parts:
            return ""

        return f"""<div class="section">
            <div class="section-title"><span class="icon">&#x1F4C8;</span> 交互式可视化分析</div>
            <div class="chart-grid">
                {"".join(parts)}
            </div>
        </div>"""

    def _build_importance_section(self, importance_df: pd.DataFrame) -> List[str]:
        if importance_df is None or len(importance_df) == 0:
            return [""]

        top = importance_df.head(15)
        rows = []
        for i, (_, row) in enumerate(top.iterrows()):
            imp = row["importance"]
            bar_width = imp / importance_df["importance"].max() * 100
            rows.append(f"""<tr>
                <td style="font-weight:700; color:#7f8c8d;">{i+1}</td>
                <td style="font-weight:600;">{row['feature']}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div style="flex:1; height:8px; background:#ecf0f1; border-radius:4px; overflow:hidden;">
                            <div style="width:{bar_width:.0f}%; height:100%; background:linear-gradient(90deg, #3498db, #2980b9); border-radius:4px;"></div>
                        </div>
                        <span style="font-weight:600; font-size:12px; min-width:40px;">{imp:.4f}</span>
                    </div>
                </td>
            </tr>""")

        return [f"""<div class="section">
            <div class="section-title"><span class="icon">&#x1F511;</span> 特征重要性 Top 15</div>
            <div class="table-wrapper">
                <table class="strategy-table">
                    <thead><tr><th>#</th><th>特征名</th><th>重要性</th></tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
        </div>"""]

    def _build_static_images_section(self, static_images: List[dict]) -> str:
        """将静态PNG图片编码为base64嵌入HTML"""
        if not static_images:
            return ""

        valid_images = []
        for img in static_images:
            path = img.get("path", "")
            title = img.get("title", os.path.basename(path))
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(path)[1].lower()
                mime = "image/svg+xml" if ext == ".svg" else "image/png"
                valid_images.append({"title": title, "src": f"data:{mime};base64,{b64}"})
            except Exception:
                continue

        if not valid_images:
            return ""

        cards = []
        for img in valid_images:
            cards.append(f"""<div class="chart-box">
                <div class="section-title" style="margin-bottom:8px; border:none; padding-bottom:8px;">
                    <span class="icon">&#x1F4CA;</span> {img['title']}
                </div>
                <img src="{img['src']}" style="width:100%; height:auto; border-radius:8px;" />
            </div>""")

        return f"""<div class="section">
            <div class="section-title"><span class="icon">&#x1F4CA;</span> 静态可视化图表</div>
            <div class="chart-grid">
                {"".join(cards)}
            </div>
        </div>"""
