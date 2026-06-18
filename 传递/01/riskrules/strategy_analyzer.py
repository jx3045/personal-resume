"""
策略分析模块 — 策略筛选、评估、排序
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .config import DEFAULT_STRATEGY_FILTER


class StrategyAnalyzer:
    """策略分析器：筛选高质量策略，计算评估指标"""

    def __init__(self, rules_df: pd.DataFrame, filter_params: Optional[Dict] = None):
        """
        Parameters
        ----------
        rules_df : pd.DataFrame
            从RuleExtractor提取的规则DataFrame
        filter_params : dict, optional
            筛选参数，覆盖默认值
        """
        self.rules_df = rules_df.copy()
        self.filter_params = {**DEFAULT_STRATEGY_FILTER, **(filter_params or {})}

    def filter(
        self,
        min_bad_rate: Optional[float] = None,
        max_bad_rate: Optional[float] = None,
        min_samples: Optional[int] = None,
        min_lift: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        按阈值筛选策略。

        Returns
        -------
        filtered : pd.DataFrame
            筛选后的策略，按bad_rate降序排列
        """
        df = self.rules_df.copy()

        min_br = min_bad_rate if min_bad_rate is not None else self.filter_params["min_bad_rate"]
        max_br = max_bad_rate if max_bad_rate is not None else self.filter_params["max_bad_rate"]
        min_s = min_samples if min_samples is not None else self.filter_params["min_samples"]
        min_l = min_lift if min_lift is not None else self.filter_params["min_lift"]

        df = df[
            (df["bad_rate"] >= min_br)
            & (df["bad_rate"] <= max_br)
            & (df["total"] >= min_s)
            & (df["lift"] >= min_l)
        ]

        return df.sort_values("bad_rate", ascending=False).reset_index(drop=True)

    def top_strategies(self, n: int = 20, by: str = "bad_rate") -> pd.DataFrame:
        """获取top-n策略"""
        return self.rules_df.sort_values(by, ascending=False).head(n)

    def summary_stats(self) -> Dict[str, Any]:
        """输出策略汇总统计"""
        df = self.rules_df
        return {
            "total_rules": len(df),
            "avg_bad_rate": df["bad_rate"].mean(),
            "max_bad_rate": df["bad_rate"].max(),
            "min_bad_rate": df["bad_rate"].min(),
            "median_bad_rate": df["bad_rate"].median(),
            "avg_lift": df["lift"].mean(),
            "max_lift": df["lift"].max(),
            "avg_depth": df["depth"].mean(),
            "max_depth": df["depth"].max(),
            "avg_samples": df["total"].mean(),
            "total_bad_captured": df["bad"].sum(),
            "total_good_captured": df["good"].sum(),
        }

    def coverage_analysis(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        在数据集上计算每条策略的实际命中率和bad_rate。
        用于验证策略在新数据上的泛化能力。

        Returns
        -------
        result : pd.DataFrame
            包含hit_count、hit_bad、hit_rate、hit_bad_rate的DataFrame
        """
        results = []
        for _, row in self.rules_df.iterrows():
            try:
                mask = self._parse_rule_to_mask(row["rule"], X)
                hit_count = mask.sum()
                hit_bad = y[mask].sum() if hit_count > 0 else 0
                hit_bad_rate = hit_bad / hit_count if hit_count > 0 else 0

                results.append(
                    {
                        "rule": row["rule"],
                        "hit_count": hit_count,
                        "hit_bad": hit_bad,
                        "hit_rate": hit_count / len(X),
                        "hit_bad_rate": hit_bad_rate,
                    }
                )
            except Exception:
                results.append(
                    {
                        "rule": row["rule"],
                        "hit_count": 0,
                        "hit_bad": 0,
                        "hit_rate": 0,
                        "hit_bad_rate": 0,
                    }
                )

        coverage_df = pd.DataFrame(results)
        return self.rules_df.merge(coverage_df, on="rule")

    @staticmethod
    def _parse_rule_to_mask(rule: str, X: pd.DataFrame) -> pd.Series:
        """将规则字符串解析为pandas布尔掩码"""
        conditions = rule.split(" & ")
        mask = pd.Series(True, index=X.index)

        for cond in conditions:
            if "<=" in cond:
                feat, thresh = cond.split("<=")
                mask &= X[feat.strip()] <= float(thresh.strip())
            elif ">" in cond:
                feat, thresh = cond.split(">")
                mask &= X[feat.strip()] > float(thresh.strip())

        return mask
