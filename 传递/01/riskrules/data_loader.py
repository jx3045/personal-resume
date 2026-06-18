"""
数据加载与预处理模块
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """数据加载器：读取CSV数据，完成基础预处理"""

    @staticmethod
    def load(
        path: str,
        label_col: str = "label",
        id_col: Optional[str] = None,
        feature_start: Optional[str] = None,
        feature_end: Optional[str] = None,
        fillna_value: float = -1,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
        """
        读取CSV文件并构建训练集。

        Parameters
        ----------
        path : str
            数据文件路径
        label_col : str
            标签列名
        id_col : str, optional
            主键列名（会被排除在特征之外）
        feature_start : str, optional
            特征列的起始列名（包含）
        feature_end : str, optional
            特征列的结束列名（包含）
        fillna_value : float
            缺失值填充值

        Returns
        -------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签列
        feature_names : pd.Index
            特征名称列表
        """
        df = pd.read_csv(path).fillna(fillna_value)

        # 确定特征列范围
        if feature_start and feature_end:
            feature_cols = df.loc[:, feature_start:feature_end].columns
        elif feature_start:
            feature_cols = df.loc[:, feature_start:].columns.drop(label_col, errors="ignore")
        else:
            exclude = [label_col]
            if id_col:
                exclude.append(id_col)
            feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df[label_col]

        return X, y, X.columns

    @staticmethod
    def summary(X: pd.DataFrame, y: pd.Series) -> dict:
        """输出数据摘要信息"""
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "bad_rate": y.mean(),
            "n_bad": int(y.sum()),
            "n_good": int(len(y) - y.sum()),
            "dtypes": X.dtypes.value_counts().to_dict(),
        }
