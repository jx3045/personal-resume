# 技术方案补充说明：数据源、LLM选型与调用、Dashboard搭建

## 一、数据源

本项目涉及的数据分为两大类：金融资产价格数据（供GBWM回测引擎使用）和非结构化文本数据（供LLM提取信号使用）。

### 1.1 金融资产价格数据

GBWM模型的回测需要多资产类别的长期历史日频收益率数据。由于项目要求选取5-20种资产构建多资产组合，且投资期限为中期到长期（3个月至数年），我们需要覆盖股票、债券、大宗商品、REITs等主要资产类别的长期数据。

主要数据源及获取方式如下。股票与ETF数据使用`yfinance`包获取，这是最常用的免费金融数据接口，支持美股、港股等市场的日频OHLCV数据下载。计划选取的ETF标的包括SPY（标普500）、QQQ（纳斯达克100）、AGG/TLT（美国债券，分别代表综合债和长期国债）、GLD（黄金）、VNQ（美国REITs）、EFA/VWO（发达市场/新兴市场国际股票）、DBC（大宗商品）等，这些ETF的历史数据普遍可追溯至2003-2007年，足以覆盖多个完整市场周期。宏观经济指标数据使用`fredapi`（FRED的Python接口）获取，包括联邦基金利率、CPI同比、GDP增速、失业率等，这些数据可作为LLM信号层和GBWM市场参数估计的辅助输入。

数据获取的典型代码流程是：通过`yfinance`批量下载各ETF的Adj Close价格序列，计算日对数收益率，用`polars`（项目要求使用polars而非pandas）进行清洗和对齐，处理缺失值（ETF成立前的交易日用NaN标记，回测时按可用资产动态调整），最终形成一个T×N的收益率矩阵（T为交易日数，N为资产数）。

### 1.2 非结构化文本数据（供LLM使用）

如果方案中包含LLM从新闻/报告中提取市场信号的模块，则需要文本数据源。可选来源包括NewsAPI（提供全球新闻的REST API，免费版有调用频次限制）、SEC EDGAR（美国上市公司财报10-K/10-Q的官方数据库，免费）、美联储FOMC会议纪要和声明（可从federalreserve.gov抓取，也可用FRED获取部分结构化数据）。在初始阶段，为控制工作量，可以仅使用FOMC声明和少数几类核心文本数据源，后续再视效果扩展。

### 1.3 投资者画像数据

GBWM需要投资者的个人信息（年龄、收入、目标等）作为输入。项目Dashboard已有Clients Tab提供客户信息输入功能，因此这部分数据由用户在Dashboard界面中实时输入，不需要预置数据集。如果需要做批量回测对比（不同投资者画像下的策略表现），可以程序化生成模拟的投资者画像数据。

---

## 二、LLM模型选型与调用方式

### 2.1 模型选择

本方案中LLM承担的任务是结构化信息提取和自然语言解释，不涉及大规模代码生成或复杂推理，因此不需要最贵的旗舰模型，中等能力的模型即可胜任。推荐选择如下。

**首选：Claude Sonnet 4（claude-sonnet-4-6）。** 理由：在结构化输出（JSON格式遵循）方面表现优异，支持JSON Schema约束输出格式，价格适中（输入$3/M tokens，输出$15/M tokens），单次目标解析调用的token量通常在500-2000 tokens之间，成本极低。Anthropic的API原生支持structured output功能，可以强制LLM输出符合预定义JSON Schema的结果，这对于我们需要将LLM输出直接传递给下游GBWM计算引擎的场景非常关键。

**备选：GPT-4o 或 GPT-4o-mini。** OpenAI的API同样支持structured output（通过response_format参数），且生态系统更成熟，文档和社区资源更丰富。GPT-4o-mini成本更低，适合开发调试阶段大量调用。

在开发和测试阶段，建议先使用成本更低的模型（如Claude Haiku或GPT-4o-mini）验证prompt设计和JSON schema的合理性，确认效果后再切换到更强的模型进行正式实验。

### 2.2 调用方式与工程实现

LLM的调用封装为一个独立的Python模块（如`src/models/llm_interface.py`），对外暴露清晰的函数接口。核心设计思路是：将prompt模板、API调用、输出解析、错误重试封装在一个类中，上游（Dashboard交互层）只需调用函数并传入自然语言文本，下游（GBWM计算层）只需接收标准化的Python字典或Pydantic模型对象。

典型的调用流程如下。首先，Dashboard前端的用户输入（自然语言文本）通过Shiny的reactive机制传递到后端Python函数。后端函数调用LLM接口，传入prompt和原始文本。LLM返回结构化JSON。后端对JSON进行schema校验（使用`pydantic`进行类型和范围检查），校验通过后转化为Pydantic模型对象，传递给GBWM计算引擎。

一个简化的代码结构示例如下（伪代码，仅说明数据流向）：

```
# 定义输出结构
class InvestmentGoal(BaseModel):
    goal_name: str           # 目标名称
    horizon_years: float     # 投资期限（年）
    target_amount: float     # 目标金额
    min_probability: float   # 最低达标概率（0.5-0.99）
    priority_weight: float   # 优先级权重（0-1）
    goal_type: str           # "lump_sum" 或 "cash_flow"

class GoalsManifest(BaseModel):
    goals: list[InvestmentGoal]
    risk_profile: str        # "conservative" / "balanced" / "aggressive"
    total_wealth: float      # 总投资金额

# LLM调用函数
def parse_investor_goals(natural_language_input: str) -> GoalsManifest:
    prompt = GOAL_EXTRACTION_PROMPT.format(input=natural_language_input)
    response = llm_client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": prompt}],
        # 强制JSON输出，schema对应GoalsManifest
    )
    parsed = GoalsManifest.model_validate_json(response.content)
    return parsed

# GBWM计算引擎接收标准化输入
def run_gbwm_optimization(goals: GoalsManifest, market_data: DataFrame) -> PortfolioResult:
    # 此处为GBWM核心计算逻辑
    ...
```

关键在于`pydantic`的使用——它不仅做JSON解析，还做类型校验和业务规则校验（如horizon_years不能为负数、min_probability必须在合理范围内），如果校验失败可以触发重试或返回错误信息给前端要求用户补充输入。这种设计确保了LLM输出的不确定性在进入量化计算层之前被完全消除。

### 2.3 LLM输出如何传递到下一层

数据流的完整链路是：

```
用户自然语言输入
    → Shiny reactive input（前端到后端的桥梁）
    → parse_investor_goals() 函数调用LLM
    → LLM返回JSON字符串
    → pydantic校验并转化为GoalsManifest对象
    → 传递给run_gbwm_optimization()函数
    → GBWM输出组合权重、达标概率等结果
    → Shiny reactive output渲染到前端展示
```

在Shiny框架中，这条链路通过reactive表达式（`@reactive.Calc`、`@render`等装饰器）自动管理依赖关系和数据流，不需要手动编写回调或事件监听代码。当用户在输入框修改文本时，Shiny会自动触发重新计算并更新所有依赖该输入的展示模块。

---

## 三、Dashboard搭建方案

### 3.1 工具选择

项目导师明确推荐使用**Shiny for Python**（即`shiny`包），备选方案是Streamlit。建议使用Shiny，原因是：导师会在私有GitHub仓库中提供Shiny Dashboard的基础Python代码（包含Clients Tab等已有模块），使用Shiny可以直接在导师代码基础上扩展，避免框架不一致导致的集成问题。此外，Shiny的reactive编程模型对于金融Dashboard这类"输入变化→重计算→更新图表"的场景比Streamlit更适合。

核心依赖包：`shiny`（Dashboard框架）、`shinyswatch`（主题美化，可选）、`plotly`（交互式图表）、`polars`（数据处理）、`pydantic`（数据校验）。

### 3.2 Dashboard整体结构

Dashboard的整体架构由公共模块和个人模块组成。公共模块是团队共享的基础功能，个人模块是每位成员各自模型的独立Tab。

```
Dashboard (main_App.py)
│
├── Clients Tab（公共，已有）
│   └── 客户信息输入：年龄、收入、总资产、风险偏好问卷等
│
├── 数据概览 Tab（公共）
│   ├── 资产价格走势图
│   ├── 收益率统计摘要（年化收益、波动率、相关性矩阵）
│   └── 市场环境判断（牛/熊/高波动）
│
├── GBWM模型 Tab（我的个人Tab）
│   ├── Sub-tab 1: 目标设定
│   │   ├── 自然语言输入框（LLM解析投资目标）
│   │   ├── 解析结果展示与确认/修改
│   │   └── 目标清单表格
│   │
│   ├── Sub-tab 2: 组合配置
│   │   ├── 各目标账户的资产配置权重（饼图/柱状图）
│   │   ├── 权重随时间的动态调整轨迹（折线图）
│   │   └── 与基准组合的权重对比
│   │
│   ├── Sub-tab 3: 达标概率追踪
│   │   ├── 各目标的当前达标概率（仪表盘式展示）
│   │   ├── 达标概率的历史变化轨迹
│   │   └── 蒙特卡洛模拟的终端财富分布直方图
│   │
│   └── Sub-tab 4: 绩效比较
│       ├── 累计收益曲线（GBWM vs 基准）
│       ├── 回撤曲线
│       ├── 关键指标对比表格（年化收益、波动率、夏普、最大回撤、达标概率）
│       └── 不同市场环境下的分阶段绩效
│
├── 其他成员模型 Tab × 4（各自独立）
│
├── 模型比较 Tab（公共）
│   ├── 各模型绩效指标横向对比表格
│   ├── 累计收益曲线叠加图
│   └── 不同市场情景下各模型表现雷达图
│
└── LLM交互 Tab（公共，如适用）
    └── 自然语言查询界面（支持对任意模型的结果提问）
```

### 3.3 具体搭建方法

Shiny for Python的Dashboard本质上是一个Python脚本（`src/dashboard/main_App.py`），其中定义了UI布局和服务端逻辑。搭建步骤如下。

**第一步：运行导师提供的基础代码。** 导师会在私有GitHub仓库中提供包含Clients Tab和基础框架的Shiny代码。首先确认能成功运行`shiny run --launch-browser src/dashboard/main_App.py`。

**第二步：在现有框架上添加自己的Tab。** Shiny的Tab结构通过`ui.nav_panel()`和`ui.navset_tab()`定义。添加新的模型Tab只需要在`ui`定义中增加一个`nav_panel`，在`server`函数中增加对应的reactive逻辑。每个成员在各自的feature branch上开发，完成后通过Pull Request合并到main分支。

**第三步：开发GBWM Tab的具体内容。** 每个Sub-tab对应Shiny中的一个子页面，使用`ui.navset_tab()`嵌套实现。图表使用`plotly`库生成交互式图表（支持缩放、悬浮提示等），通过`shiny.render.plotly`渲染到页面。数据计算逻辑放在独立的Python模块中（如`src/models/gbwm_engine.py`），Dashboard只负责调用和展示，保持关注点分离。

**第四步：连接客户端输入。** GBWM Tab需要读取Clients Tab中用户输入的个人信息（年龄、总资产等），Shiny通过reactive values实现跨Tab的数据共享——Clients Tab的输入存储在一个共享的reactive value中，GBWM Tab读取该value并在其变化时自动重新计算。

**第五步：测试与调试。** 使用`pytest`对核心计算逻辑编写单元测试，Dashboard层面的交互测试可以手动进行。开发过程中使用`shiny run --reload`实现热重载，修改代码后自动刷新页面。

### 3.4 运行与管理

Dashboard的运行命令为`shiny run --launch-browser src/dashboard/main_App.py`，会在本地启动一个Web服务器并自动打开浏览器。项目使用`uv`管理Python依赖，`kedro`管理数据pipeline（数据获取→清洗→存储的自动化流程），`ruff`格式化代码，`pytest`运行测试。所有代码变更遵循Feature Branching策略：每位成员在自己的feature branch上开发，通过Pull Request合并到main分支，导师（AlexPars）和团队成员进行代码审查。

---

## 附：相关知识与工具速查

**yfinance：** Python包，通过Yahoo Finance的非官方API获取股票、ETF、指数等金融数据。使用方式为`import yfinance as yf; data = yf.download("SPY", start="2010-01-01")`，返回包含Open/High/Low/Close/Volume等列的DataFrame。免费、无需API key，但数据质量和稳定性不如Bloomberg等付费数据源，适合学术项目使用。

**FRED（Federal Reserve Economic Data）：** 美联储维护的宏观经济数据库，包含数十万条经济时间序列（利率、通胀、就业、GDP等）。Python中通过`fredapi`包调用，需要申请免费的API key。使用方式为`from fredapi import Fred; fred = Fred(api_key='xxx'); cpi = fred.get_series('CPIAUCSL')`。

**polars：** 高性能DataFrame库，语法与pandas类似但执行速度快数倍至数十倍，底层用Rust实现。项目要求使用polars替代pandas。核心概念包括DataFrame、Series、LazyFrame（延迟计算，类似SQL查询优化）。常见操作如`df.filter(pl.col("date") > "2020-01-01").group_by("asset").agg(pl.col("return").mean())`。

**Pydantic：** Python的数据验证库，通过定义数据模型类（继承`BaseModel`）自动进行类型检查、范围验证和JSON序列化/反序列化。在LLM调用场景中特别有用——可以强制LLM输出符合特定schema的JSON，并在解析时自动校验。是FastAPI框架的核心依赖之一，生态成熟。

**Shiny for Python：** R Shiny的Python移植版，由Posit（原RStudio）公司开发。核心概念包括：UI定义（页面布局）、Server函数（响应式逻辑）、Reactive Values（响应式数据，输入变化时自动触发依赖它的计算重新执行）。与Streamlit的区别在于Shiny采用声明式reactive编程模型（定义"什么依赖什么"），Streamlit采用命令式脚本模型（从上到下依次执行）。对于复杂的多Tab交互Dashboard，Shiny的reactive模型更高效。

**Plotly：** 交互式数据可视化库，支持折线图、柱状图、饼图、散点图、热力图等常见图表类型，生成的图表支持鼠标悬浮显示数值、缩放、平移等交互操作。在Shiny中通过`shiny.render.plotly`渲染。适合金融Dashboard场景的图表包括：累计收益曲线、回撤面积图、资产配置饼图、达标概率仪表盘等。

**kedro：** 数据科学项目框架，将项目拆分为"节点（Node）"和"管道（Pipeline）"，每个节点是一个Python函数，管道是节点的有序组合。kedro管理数据在各节点之间的流转，支持数据缓存、依赖管理和可视化。在本项目中用于管理"数据获取→清洗→特征工程→模型输入"的数据pipeline。

**Structured Output：** LLM API的一项功能，允许开发者指定输出必须遵循的JSON Schema。OpenAI通过`response_format={"type": "json_schema", "json_schema": {...}}`参数实现，Anthropic通过tool_use或tool_choice参数实现。这确保了LLM输出可以被程序可靠地解析，是将LLM集成到自动化流程中的关键技术。

**Reactive Programming（响应式编程）：** Shiny的核心编程范式。开发者声明"数据A变化时，计算B需要重新执行，图表C需要重新渲染"，框架自动管理依赖关系和执行顺序。在Python Shiny中通过`@reactive.Calc`、`@reactive.Effect`、`@render.plotly`等装饰器实现。这与传统的回调函数或事件监听模式不同，代码更声明式、更易维护。
