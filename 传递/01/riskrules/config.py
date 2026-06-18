"""
全局配置参数
"""

# 默认决策树参数
DEFAULT_TREE_PARAMS = {
    "max_depth": 10,
    "min_samples_leaf": 50,
    "criterion": "gini",
    "class_weight": None,
    "random_state": 42,
}

# 策略筛选默认阈值
DEFAULT_STRATEGY_FILTER = {
    "min_bad_rate": 0.05,       # 最小坏账率
    "max_bad_rate": 0.80,       # 最大坏账率
    "min_samples": 30,           # 最小样本量
    "min_lift": 1.5,            # 最小提升度
}

# 可视化默认参数
DEFAULT_VIZ_PARAMS = {
    "figsize": (14, 10),
    "dpi": 150,
    "cmap": "RdYlGn_r",
    "max_rules_display": 30,    # 图表中最多展示的规则数
}

# 特征采样默认参数
DEFAULT_SAMPLE_PARAMS = {
    "n_features": 10,            # 随机采样特征数
    "n_iterations": 5,           # 采样迭代次数
    "top_k_importance": 10,      # 基于重要性选取top-k特征
}
