# Midterm Draft — Team 6: AlphaWealth AI

**Course:** Quantitative Wealth and Investment Management (QWIM)
**Project:** Personalized Wealth Portfolios — Personalized Investing + LLMs
**Team Members:** Junyang Xu, Lei Zhao, Ziqian Zhu
**Date:** June 2026

## 1. Introduction

Traditional portfolio optimization, rooted in the mean-variance framework of Markowitz (1952), treats all investors as interchangeable agents distinguished only by a risk tolerance parameter. In practice, individual investors hold multiple financial goals simultaneously, each with its own time horizon, target amount, and acceptable probability of achievement. A thirty-year-old saving for a home down payment and retirement faces a fundamentally different problem than a sixty-year-old focused on generating sustainable income. This project asks how quantitative portfolio models can be made genuinely personalized, and how large language models can bridge the gap between sophisticated quantitative outputs and human-understandable investment advice. Our team approaches this from multiple angles: each member independently selects and implements a quantitative model aligned with the personalization objective, while sharing common infrastructure for data processing, benchmark comparison, and Shiny-based dashboard visualization. This midterm draft presents the report structure, preliminary literature analysis, and detailed implementation plans for the models selected to date.

## 2. Literature Review and Analysis

### 2.1 Goals-Based Wealth Management

The goals-based wealth management framework, formalized by Das, Evans, and Ashvin (2020, 2022), decomposes the investor's overall portfolio into independent "mental accounts," each corresponding to a specific financial goal. The theoretical justification draws on Thaler's (1985) mental accounting concept from behavioral economics. Under conditions proven by Das et al., the multi-goal optimization problem separates into independent single-goal sub-problems, each employing an S-shaped utility function inspired by Kahneman and Tversky's Prospect Theory, where investors exhibit risk-seeking behavior below the target amount and risk-averse behavior above it. This behavioral realism distinguishes GBWM from conventional frameworks and makes it particularly suitable for personalized wealth management, directly mirroring the advisory approach used at Bank of America and Merrill Lynch. The literature search followed the staged filtering methodology recommended by our mentor: an initial pass based on abstracts, followed by reading introductions, numerical results, and conclusions of retained papers, and finally assessing the availability of open-source implementations. Through this process, the Das et al. series emerged as the most directly relevant reference.

### 2.2 Reinforcement Learning and Robo-Advisory



### 2.3 Large Language Models in Financial Applications

Lo and Ross (2024), in "Can ChatGPT Plan Your Retirement?", conducted one of the earliest empirical studies of LLM capability in financial planning, finding that GPT-4 could generate retirement plans comparable to those of certified financial planners on completeness and coherence, though with weaknesses in tax-specific advice and individualized numerical precision. Takayanagi et al. (2025) found that LLM-based agents excel at information synthesis, explanation generation, and multi-turn goal clarification, but struggle with complex numerical optimization, providing strong motivation for the hybrid architecture adopted in our project where LLMs augment rather than replace quantitative engines. Oehler and Horn (2024) found that while LLMs produced more readable explanations, their portfolio recommendations were less diversified than those of purpose-built algorithms. Collectively, these studies support positioning LLMs as intelligent interfaces that complement quantitative optimization frameworks.

## 3. Description of Selected Models

### 3.1 Goals-Based Wealth Management with LLM Integration (Junyang Xu)

The GBWM framework is selected for its direct alignment with personalized investing. Unlike single-period frameworks such as Black-Litterman or Risk Parity, GBWM is inherently multi-period and goal-driven, decomposing the investor's portfolio into independent mental accounts optimized to maximize goal achievement probability.

Each sub-problem is formulated as a stochastic optimal control problem with an S-shaped utility function. The decomposition property proven by Das et al. (2020) allows the multi-goal problem to be solved as independent single-goal sub-problems via Monte Carlo simulation with 10,000 paths, supplemented by the analytical approximations from their work for validation.

The LLM integration operates at three levels. First, at goal elicitation, the LLM translates investors' natural language descriptions into structured parameters (time horizons, target amounts, priority weights, achievement probability thresholds) through prompts constrained by JSON Schema and Pydantic validation. Second, at market signal extraction, the LLM processes FOMC statements and economic reports to generate auxiliary inputs for updating market parameter estimates within each sub-portfolio. Third, at result explanation, the LLM generates investor-friendly narratives for portfolio weights, achievement probabilities, and rebalancing recommendations.

The asset universe consists of twelve ETFs: SPY, QQQ, IWM, AGG, TLT, LQD, GLD, USO, VNQ, EMB, TIP, and SHY, covering equities, fixed income, commodities, and real estate. Daily adjusted close prices are obtained via yfinance spanning January 2010 to present. Macroeconomic indicators (federal funds rate, CPI, GDP growth) are sourced from FRED via the fredapi package. Data processing uses Polars: missing values from market holidays are forward-filled, daily log returns are computed, and Winsorization at the 1st and 99th percentiles is applied to mitigate outlier influence. The signal layer uses LightGBM for short-term return and volatility forecasts as dynamic parameter updates, preserving GBWM's theoretical integrity while allowing data-driven refinement. The codebase follows the project's coding standards including Polars for tabular data, Pydantic for validation, and pytest coverage targeting 90 percent.

Evaluation uses goal achievement probability as the primary GBWM-specific metric, supplemented by Sharpe ratio, maximum drawdown, and Sortino ratio. Benchmarks include the 60/40 portfolio, equal-weight allocation, risk parity, and target-date fund glide paths. Performance is assessed across bull, bear, and high-volatility regimes with an ablation study to quantify the marginal contribution of LLM components.

### 3.2 Model Selected by Lei Zhao



### 3.3 Model Selected by Ziqian Zhu



## 4. Numerical Results

### 4.1 Experimental Design and Performance Metrics

The numerical evaluation follows a structured design applied uniformly across all team members' models to ensure comparability. The backtesting period spans from January 2010 to the present, covering multiple distinct market regimes: the post-crisis recovery (2010-2017), the late-cycle expansion (2018-2019), the COVID-19 shock and recovery (2020), the inflation surge and Fed tightening cycle (2021-2022), and the subsequent stabilization period (2023-2025). The training and testing split follows strict chronological ordering to prevent look-ahead bias: data from 2010 through 2017 is used for initial calibration, 2018 through 2020 for validation, and 2021 through the present for out-of-sample evaluation. For the GBWM model, Monte Carlo simulation uses 10,000 paths per sub-problem with return distributions estimated from rolling 252-day windows. Each model is evaluated using standard risk-return metrics (annualized return, volatility, Sharpe ratio, maximum drawdown, Sortino ratio, Calmar ratio) along with model-specific metrics. For GBWM, the primary metric is goal achievement probability, the fraction of simulated paths where terminal wealth exceeds the target. Benchmark strategies include the classic 60/40 portfolio (rebalanced quarterly), equal-weight allocation (rebalanced monthly), risk parity via inverse-volatility weighting, and a target-date fund glide path.

### 4.2 Regime-Conditional Analysis and Ablation Study

Performance is reported conditionally on market regimes identified using volatility and return criteria. Bull markets are defined as rolling twelve-month windows with positive returns and below-median volatility; bear markets as windows with negative returns; high-volatility periods as those where realized twenty-day volatility exceeds the 80th historical percentile. This analysis is critical because a strategy that performs well on average but fails during stress periods is less useful for long-term wealth management. For GBWM specifically, we examine how goal achievement probabilities evolve as horizons shorten during adverse conditions. To isolate the marginal contribution of each component, an ablation study compares the full GBWM-LLM framework against reduced versions: GBWM with static parameters (no signal layer), GBWM without LLM text sentiment inputs, and GBWM with fixed rather than LLM-elicited goal parameters. The difference in goal achievement probability and Sharpe ratio between versions quantifies whether the added complexity of LLM integration is justified by measurable improvement.

### 4.3 Dashboard Visualization

All numerical results are presented through the team's shared interactive Shiny Dashboard, which serves as both the primary deliverable and the analytical interface. The Dashboard is structured into several modules. The Clients module, provided by our mentor, captures investor profile information through form inputs and natural language text fields. Each team member's model occupies an independent Tab containing Sub-tabs for parameter configuration, allocation visualization, performance tracking, and benchmark comparison. The GBWM Tab includes a Goal Setting Sub-tab displaying LLM-parsed specifications with a confirmation interface, a Portfolio Allocation Sub-tab with interactive pie charts and stacked bar charts showing weight distributions across the twelve ETFs per sub-account with historical sliders, an Achievement Probability Sub-tab with gauge indicators and time-series tracking charts, and a Performance Comparison Sub-tab overlaying cumulative return curves for the GBWM strategy and all benchmarks on interactive Plotly charts with hover tooltips and zoom functionality. A shared Model Comparison Tab aggregates results across all models into comparison tables and radar charts. The Dashboard uses Shiny for Python with reactive programming so that any input change triggers immediate recomputation and visualization update.

## 5. Analysis of Results Including Narratives

### 5.1 Market Regime Interpretation

The numerical results are analyzed through a framework connecting quantitative metrics to market dynamics, economic conditions, and model structural properties. The regime-conditional analysis focuses on stress periods and transitions where model differences are most revealing. The COVID-19 shock of February through March 2020 provides an instructive case: equity markets declined by approximately 30 percent within weeks while bond markets experienced simultaneous sell-offs, creating a scenario where traditional diversification temporarily failed. For GBWM, we will examine whether the dynamic rebalancing mechanism successfully protected near-term goals while allowing longer-term goals to maintain equity exposure for eventual recovery. The 2022 inflation surge presents a different challenge: rising rates caused simultaneous equity and bond declines, but GBWM's inclusion of commodities (GLD, USO) and inflation-protected securities (TIP) may have provided partial hedging unavailable in simpler constructions. Each regime analysis connects observed performance to economic mechanisms and to specific model design choices that either capitalized on or were vulnerable to those mechanisms.

### 5.2 Structural Properties and LLM Impact

The analysis also examines how each model's structural properties translate into behavioral implications. For GBWM, the S-shaped utility function implies that allocation strategies change non-linearly as wealth approaches or diverges from targets, fundamentally different from a static 60/40 portfolio. We will track allocation trajectories over time and compute metrics such as average equity exposure, turnover rate, and rebalancing frequency to quantify these differences. The analysis also explores whether GBWM's probability-based communication, enhanced by LLM explanations, produces more actionable investor experiences than traditional performance reporting. A dedicated assessment evaluates LLM integration impact across three layers: whether LLM-parsed goal parameters differ systematically from questionnaire-based inputs, whether LLM-derived sentiment signals improve parameter estimates versus purely statistical forecasts, and whether LLM-generated explanations enhance investor comprehension through a small-scale user study rating clarity, actionability, and trustworthiness. The Dashboard enables further exploratory analysis through reactive sensitivity testing, allowing users to modify goal parameters and immediately observe allocation and probability responses, revealing robustness to specification and identifying instability regions. The analysis concludes with a candid discussion of limitations including simplified transaction cost assumptions, the twelve-ETF asset universe excluding alternatives, log-normal return assumptions that may understate tail risk, and LLM-related considerations around cost, latency, and hallucination risk.

## 6. Conclusion

This section will synthesize findings across all team members' models and draw overarching conclusions about the feasibility, effectiveness, and practical implications of integrating quantitative portfolio optimization with large language models for personalized wealth management. The conclusion is expected to address several key questions that the numerical analysis aims to answer. First, to what extent do different quantitative frameworks serve distinct dimensions of personalization, and does any single approach demonstrate broad superiority or do models complement each other across different investor profiles and market conditions? Second, does the LLM integration produce measurable value beyond what traditional quantitative models achieve on their own, and if so, which components of the LLM architecture (goal elicitation, signal extraction, or result explanation) contribute most to this incremental value? Third, how does the interactive Dashboard and natural language explanation capability affect investor engagement and comprehension compared to conventional reporting formats?

The conclusion will also discuss the broader implications for the wealth management industry, examining whether the hybrid architecture demonstrated in this project, in which LLMs serve as intelligent interfaces while specialized engines handle optimization, represents a viable path toward AI-enhanced advisory systems. This discussion will reference the robo-advisory literature (Lehner 2025) on trust-building and explanation gaps, and assess whether our findings support or challenge existing conclusions. Finally, the section will identify specific limitations of the current work, including the scope of the asset universe, return distribution assumptions, and the preliminary nature of the user study, and will propose directions for future research such as extending to alternative investments, incorporating regime-switching return models, exploring fine-tuned financial LLMs, and conducting larger-scale behavioral studies on probability-based goal tracking.

## 7. List of References

Das, S. R., Evans, P., & Ashvin, V. (2020). Goals-Based Investing: Goal Direction and the Architecture of the Mind. Journal of Wealth Management.

Das, S. R., Evans, P., & Ashvin, V. (2022). Dynamic Goals-Based Wealth Management: Extensions and Applications. Journal of Portfolio Management.

Golts, M., & Jones, D. (2025). Goal Parity: Integrating Risk Parity with Goals-Based Investing.

Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263-291.

Lo, A. W., & Ross, S. (2024). Can ChatGPT Plan Your Retirement? MIT Sloan Working Paper.

Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.

Oehler, A., & Horn, C. (2024). Does ChatGPT Provide Better Advice than Robo-Advisors? Finance Research Letters.

Takayanagi, K., Uchida, S., et al. (2025). Are Generative AI Agents Effective Personalized Financial Advisors?

Thaler, R. H. (1985). Mental Accounting and Consumer Choice. Marketing Science, 4(3), 199-214.

[Additional references to be added by team members for their respective model sections.]



本报告大纲共七个章节，结构如下：

第1节"引言"阐述项目核心动机：传统均值-方差优化将所有投资者视为同质个体，无法处理多目标、多期限的个性化需求。本项目提出将量化组合优化与投资者真实目标对齐，并通过大语言模型（LLM）弥合复杂量化输出与人类可理解投资建议之间的鸿沟。团队成员各自选择独立的量化模型，共享数据、基准和Dashboard基础设施，形成横向对比视角。

第2节"文献综述"分三个子节。2.1梳理了目标导向财富管理（GBWM）的理论基础，包括Das et al.（2020, 2022）的心理账户分解定理和S型效用函数，以及其与BofA实际顾问业务的对应关系。2.2为强化学习与智能投顾方向的文献，由队友负责填写。2.3综述了LLM在金融规划中的应用现状，包括Lo and Ross（2024）、Takayanagi et al.（2025）、Oehler and Horn（2024）的关键发现，论证了LLM作为增强层而非替代层的定位。

第3节"模型描述"中，3.1详细描述了本人选择的GBWM+LLM方案：模型选择理由（多期动态、目标驱动特性）、数学框架（随机最优控制+可分解性质）、LLM三层融合架构（目标提取、信号提取、结果解释）、12个ETF标的及数据来源（yfinance/FRED）、Polars数据清洗流程、LightGBM信号层，以及评估指标和基准设定。3.2和3.3为队友的模型部分，待完成。

第4节"数值结果"规划了实验设计（2010至今回测、时间序列切分防泄露）、绩效指标体系（通用风险收益指标+GBWM特有的达标概率）、四个基准策略、市场周期条件分析方法、消融实验设计（逐层剥离LLM组件），以及Shiny Dashboard的可视化结构（Clients模块、各模型独立Tab、Goal Setting/Allocation/Achievement Probability/Performance Comparison四个Sub-tab、Plotly交互图表）。

第5节"结果分析与叙述"规划了分析框架：将绩效数据与市场动态和模型结构特性关联解读，重点分析COVID-19冲击和2022通胀周期等压力情景下的表现差异，考察S型效用函数带来的非线性动态再平衡行为，评估LLM三层整合的实际影响，以及讨论局限性（交易成本简化假设、资产范围、对数正态收益假设、LLM成本与幻觉风险）。

第6节"结论"将以前瞻性口吻说明预期探讨的核心问题：不同量化框架在个性化各维度上的互补性、LLM整合是否产生可量化增量价值、Dashboard交互体验对投资者理解力的影响，以及混合架构对财富管理行业的启示和未来研究方向。

第7节"参考文献"列出了目前已引用的9篇核心文献，队友部分待补充。
