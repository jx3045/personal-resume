"""
特征采样模块 — 支持随机采样和重要性采样
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier
from .tree_builder import TreeBuilder
from .rule_extractor import RuleExtractor
from .config import DEFAULT_SAMPLE_PARAMS


class FeatureSampler:
    """特征采样器：多种采样策略用于丰富规则挖掘"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def random_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        n_iterations: int = 5,
        tree_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        随机采样特征并分别训练决策树，汇总所有规则。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签列
        n_features : int
            每次采样的特征数量
        n_iterations : int
            采样迭代次数
        tree_params : dict, optional
            每棵树使用的决策树参数

        Returns
        -------
        all_rules_df : pd.DataFrame
            汇总的去重规则
        """
        all_rules = []

        for iteration in range(n_iterations):
            sampled_cols = np.random.choice(
                X.columns, size=min(n_features, X.shape[1]), replace=False
            )
            X_sample = X[sampled_cols]

            builder = TreeBuilder(tree_params)
            builder.fit(X_sample, y)

            extractor = RuleExtractor(builder.model, sampled_cols.tolist())
            rules_df = extractor.extract_to_dataframe()
            rules_df["iteration"] = iteration
            all_rules.append(rules_df)

        result = pd.concat(all_rules, ignore_index=True)
        result = result.drop_duplicates(subset=["rule"]).reset_index(drop=True)
        return result

    def importance_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 10,
        tree_params: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        基于特征重要性采样：先训练全特征树，按重要性取top-k再训练。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签列
        top_k : int
            选取top-k重要特征
        tree_params : dict, optional

        Returns
        -------
        importance_df : pd.DataFrame
            特征重要性排序
        rules_df : pd.DataFrame
            基于top-k特征提取的规则
        """
        # 第一轮：全特征训练获取重要性
        builder_full = TreeBuilder(tree_params)
        builder_full.fit(X, y)
        importance_df = builder_full.feature_importance

        # 第二轮：top-k特征训练
        top_features = importance_df.head(top_k)["feature"].tolist()
        X_top = X[top_features]

        builder_top = TreeBuilder(tree_params)
        builder_top.fit(X_top, y)

        extractor = RuleExtractor(builder_top.model, top_features)
        rules_df = extractor.extract_to_dataframe()

        return importance_df, rules_df

    def business_group_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_groups: dict,
        tree_params: Optional[dict] = None,
    ) -> dict:
        """
        按业务分组采样：对不同类别的特征分组分别挖掘。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签列
        feature_groups : dict
            特征分组，如 {"身份信息": ["XINGBIE","CSNY"], "财务信息": ["GRJCJS","GRZHYE"]}
        tree_params : dict, optional

        Returns
        -------
        results : dict
            {group_name: rules_df}
        """
        results = {}

        for group_name, features in feature_groups.items():
            valid_features = [f for f in features if f in X.columns]
            if len(valid_features) < 2:
                continue

            X_group = X[valid_features]
            builder = TreeBuilder(tree_params)
            builder.fit(X_group, y)

            extractor = RuleExtractor(builder.model, valid_features)
            results[group_name] = extractor.extract_to_dataframe()

        return results
