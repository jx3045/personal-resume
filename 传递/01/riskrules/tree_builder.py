"""
决策树构建模块
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing import Optional, Dict, Any
from .config import DEFAULT_TREE_PARAMS


class TreeBuilder:
    """决策树构建器：训练分类/回归决策树"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        params : dict, optional
            决策树参数，会覆盖默认参数
        """
        self.params = {**DEFAULT_TREE_PARAMS, **(params or {})}
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
        """
        训练分类决策树。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签列（0/1二分类）

        Returns
        -------
        model : DecisionTreeClassifier
            训练好的决策树模型
        """
        self.feature_names = X.columns.tolist()

        self.model = DecisionTreeClassifier(
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            criterion=self.params["criterion"],
            class_weight=self.params["class_weight"],
            random_state=self.params["random_state"],
        )
        self.model.fit(X, y)

        # 计算特征重要性
        self.feature_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return self.model

    def fit_with_params(
        self, X: pd.DataFrame, y: pd.Series, max_depth: int, min_samples_leaf: int
    ) -> DecisionTreeClassifier:
        """使用指定参数训练决策树"""
        self.params["max_depth"] = max_depth
        self.params["min_samples_leaf"] = min_samples_leaf
        return self.fit(X, y)

    def get_tree_info(self) -> Dict[str, Any]:
        """获取训练后决策树的结构信息"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 fit()")

        tree_ = self.model.tree_
        return {
            "node_count": tree_.node_count,
            "max_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
            "n_classes": self.model.n_classes_,
            "classes": self.model.classes_.tolist(),
            "feature_importance": self.feature_importance.to_dict("records"),
        }
