"""
决策树规则提取模块 — XiaoWuGe 算法

基于二叉树前序遍历 + 回溯，将决策树的所有叶子路径
提取为人类可读的风控策略规则。
"""

import numpy as np
import pandas as pd
from sklearn.tree import _tree
from typing import List, Optional


class RuleExtractor:
    """规则提取器：从决策树中提取所有叶子节点规则"""

    def __init__(self, model, feature_names: List[str]):
        """
        Parameters
        ----------
        model : DecisionTreeClassifier
            训练好的sklearn决策树模型
        feature_names : list of str
            特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.tree_ = model.tree_

    def extract(self) -> List[str]:
        """
        使用 XiaoWuGe 算法提取所有叶子规则。

        规则格式：'特征A<=阈值 & 特征B>阈值 & ... :good数:bad数'

        Returns
        -------
        rules : list of str
            规则字符串列表
        """
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        value = self.tree_.value
        n_node_samples = self.tree_.n_node_samples

        # 计算每个节点的深度
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]

        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # 构建特征名称映射
        feature_name = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in feature
        ]

        ways = []
        depth = []
        feat = []
        nodes = []
        rules = []

        for i in range(n_nodes):
            if is_leaves[i]:
                # 叶子节点：回溯到正确深度后生成规则
                while depth and depth[-1] >= node_depth[i]:
                    depth.pop()
                    ways.pop()
                    feat.pop()
                    nodes.pop()

                if children_left[i - 1] == i:
                    a = "{f}<={th}".format(
                        f=feat[-1], th=round(threshold[nodes[-1]], 4)
                    )
                    ways[-1] = a
                else:
                    a = "{f}>{th}".format(
                        f=feat[-1], th=round(threshold[nodes[-1]], 4)
                    )
                    ways[-1] = a

                good_cnt = int(round(value[i][0][0] * n_node_samples[i]))
                bad_cnt = int(round(value[i][0][1] * n_node_samples[i]))
                last = " & ".join(ways) + ":" + str(good_cnt) + ":" + str(bad_cnt)
                rules.append(last)

            else:
                # 非叶子节点：入栈
                if i == 0:
                    ways.append(round(threshold[i], 4))
                    depth.append(node_depth[i])
                    feat.append(feature_name[i])
                    nodes.append(i)
                else:
                    while depth and depth[-1] >= node_depth[i]:
                        depth.pop()
                        ways.pop()
                        feat.pop()
                        nodes.pop()

                    if i == children_left[nodes[-1]]:
                        w = "{f}<={th}".format(
                            f=feat[-1], th=round(threshold[nodes[-1]], 4)
                        )
                    else:
                        w = "{f}>{th}".format(
                            f=feat[-1], th=round(threshold[nodes[-1]], 4)
                        )

                    ways[-1] = w
                    ways.append(round(threshold[i], 4))
                    depth.append(node_depth[i])
                    feat.append(feature_name[i])
                    nodes.append(i)

        return rules

    def extract_to_dataframe(self) -> pd.DataFrame:
        """
        提取规则并直接转为DataFrame格式。

        Returns
        -------
        df : pd.DataFrame
            包含规则详情的数据框
        """
        rules = self.extract()
        return self._rules_to_df(rules)

    @staticmethod
    def _rules_to_df(rules: List[str]) -> pd.DataFrame:
        """将规则列表转为结构化DataFrame"""
        df = pd.DataFrame(rules, columns=["raw_rule"])

        df["rule"] = df["raw_rule"].str.rsplit(":", n=2).str.get(0)
        df["good"] = df["raw_rule"].str.rsplit(":", n=2).str.get(1).astype(float)
        df["bad"] = df["raw_rule"].str.rsplit(":", n=2).str.get(2).astype(float)
        df["total"] = df["bad"] + df["good"]
        df["bad_rate"] = df["bad"] / df["total"]

        # 计算提升度（lift = bad_rate / 整体bad_rate）
        overall_bad_rate = df["bad"].sum() / df["total"].sum()
        df["lift"] = df["bad_rate"] / overall_bad_rate

        # 计算规则深度（条件数）
        df["depth"] = df["rule"].str.count("&") + 1

        return df.drop(columns=["raw_rule"])
