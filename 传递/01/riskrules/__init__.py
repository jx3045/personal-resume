"""
RiskRules - 基于决策树算法的自动化风控策略挖掘工具

支持规则提取、策略推荐、多维度可视化分析。

Author: XiaoWuGe
"""

from .data_loader import DataLoader
from .tree_builder import TreeBuilder
from .rule_extractor import RuleExtractor
from .strategy_analyzer import StrategyAnalyzer
from .visualizer import StrategyVisualizer
from .feature_sampler import FeatureSampler
from .exporter import StrategyExporter
from .premium_dashboard import PremiumDashboard

__version__ = "2.2.0"
__all__ = [
    "DataLoader",
    "TreeBuilder",
    "RuleExtractor",
    "StrategyAnalyzer",
    "StrategyVisualizer",
    "FeatureSampler",
    "StrategyExporter",
    "PremiumDashboard",
]
