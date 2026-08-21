# Enhanced Financial Portfolio Optimization with Risk Management using GraphSAGE-PPO

This project presents a hybrid **Graph Neural Network (GNN) and Deep Reinforcement Learning (DRL)** approach for financial portfolio optimization.

The main idea is simple: financial assets do not move independently. Stocks within the same sector, market indices, bonds, and other financial instruments can have strong relationships with each other. Instead of looking at each asset separately, this project represents the market as a **graph** and allows the model to learn these relationships.

The framework combines **GraphSAGE** to learn relationships between assets and **Proximal Policy Optimization (PPO)** to learn how portfolio weights should be adjusted over time. It also uses **SHAP** to identify important features and an **equal-weight fallback strategy** to provide additional protection during periods of poor performance or high market instability.

---

## 📌 Project Overview

Traditional portfolio optimization methods generally rely on predefined assumptions about returns, risk, and relationships between assets. These assumptions may not work well when market conditions change rapidly.

This project takes a different approach by treating the financial market as a network:

* **Nodes** represent financial assets such as stocks, market indices, ETFs, and treasury yields.
* **Edges** represent relationships between these assets, primarily based on their correlations.
* **GraphSAGE** learns useful representations of each asset by considering both its own features and the features of related assets.
* **PPO** uses these representations to learn dynamic portfolio allocation strategies.
* **SHAP** helps identify which financial features are most important.
* A **fallback equal-weight strategy** is used when the portfolio faces significant losses or sustained underperformance.

The goal is not simply to maximize returns, but to develop a portfolio strategy that is **adaptive, risk-aware, and more robust to changing market conditions**.

---

## 🧠 How the Model Works

The complete pipeline can be summarized as:

```text
Financial Market Data
        │
        ├── Stock Prices & Volume
        ├── Technical Indicators
        ├── Macroeconomic Indicators
        ├── Global Market Indices
        └── Treasury Yields
        │
        ▼
Data Preprocessing
        │
        ├── Missing Value Handling
        ├── Forward Filling
        ├── Outlier Treatment
        └── Normalization
        │
        ▼
SHAP-Based Feature Selection
        │
        ▼
Financial Graph Construction
        │
        ├── Nodes → Financial Assets
        └── Edges → Asset Relationships
        │
        ▼
GraphSAGE
        │
        ▼
Asset Embeddings
        │
        ▼
PPO Agent
        │
        ▼
Dynamic Portfolio Allocation
        │
        ├── Normal Conditions
        │       ↓
        │   Learned Allocation
        │
        └── Significant Drawdown
                ↓
        Equal-Weight Fallback
```

In short:

**Financial Data → Feature Selection → Financial Graph → GraphSAGE → PPO → Portfolio Allocation**

---

## 📊 Data Used

The framework uses several types of financial information so that the model can look at the market from different perspectives.

### Market Data

Historical **OHLCV (Open, High, Low, Close, Volume)** data are collected using the `yfinance` API.

### Technical Indicators

The project uses technical indicators such as:

* Simple Moving Average (SMA)
* Exponential Moving Average (EMA)
* Raw Stochastic Value (RSV)
* Relative Strength Index (RSI)

These indicators provide information about price trends and market momentum.

### Macroeconomic Data

Macroeconomic variables are obtained from the **Federal Reserve Economic Data (FRED)** database.

The data include indicators related to:

* Employment
* Inflation
* GDP growth
* Consumer sentiment
* Labor productivity

### Global Market Information

The framework also considers broader market conditions through information such as:

* S&P 500
* SSE Composite
* Nikkei 225
* FTSE 100
* Sector ETFs
* 5-year Treasury yields
* 10-year Treasury yields

Using these different sources allows the model to consider not only individual stocks but also the wider economic and market environment.

---

## 🔧 Methodology

### 1. Data Preprocessing

Before feeding the data into the model, the raw financial data are cleaned and prepared.

The preprocessing pipeline includes:

* Handling missing values
* Forward filling
* Outlier treatment
* Feature normalization using z-score standardization
* SHAP-based feature selection

Feature selection is particularly useful because financial datasets can contain many indicators, some of which may add little useful information.

---

### 2. Building the Financial Graph

The financial market is represented as a graph.

Each financial asset becomes a **node**, while relationships between assets are represented by **edges**.

A correlation matrix is used to measure relationships between assets. In this study, an edge is created when the correlation between two assets exceeds **0.7**.

For example:

```text
        Stock A
        /     \
       /       \
  Stock B ---- Stock C
      |
      |
   Sector ETF
```

This allows the model to capture relationships that may be missed when every asset is treated independently.

---

### 3. GraphSAGE

Once the financial graph has been created, **GraphSAGE** is used to learn an embedding for every asset.

Instead of looking only at the features of one stock, GraphSAGE also considers information from its neighboring assets.

For example, if two stocks have strong relationships, information from one stock can contribute to the representation learned for the other.

The resulting embeddings contain information about both:

* The individual characteristics of an asset
* Its relationships with other assets

These embeddings are then passed to the reinforcement learning agent.

---

### 4. PPO-Based Portfolio Optimization

The learned GraphSAGE embeddings are given to a **Proximal Policy Optimization (PPO)** agent.

The PPO agent learns through interaction with a simulated trading environment.

At each step, the agent decides how the available capital should be distributed among the assets.

Conceptually:

```text
Market State
     ↓
GraphSAGE
     ↓
Asset Embeddings
     ↓
PPO Agent
     ↓
Portfolio Weights
     ↓
Portfolio Return
     ↓
Reward
     ↓
PPO Updates Strategy
```

Over time, the agent learns an allocation strategy that attempts to improve **risk-adjusted portfolio performance** rather than simply maximizing raw returns.

---

### 5. Risk-Controlled Fallback Strategy

One of the important parts of the framework is the fallback mechanism.

A reinforcement learning model can sometimes make aggressive decisions during unusual market conditions. To reduce this risk, the system includes a simple and conservative backup strategy.

If the portfolio experiences a predefined level of loss or sustained underperformance, the system switches to an **equal-weight allocation**.

For example, instead of allowing the RL agent to make highly concentrated allocations during a difficult market period:

```text
PPO Allocation
      ↓
Significant Loss / Drawdown?
      │
   ┌──┴──┐
   │     │
  No    Yes
   │     │
   ↓     ↓
PPO     Equal Weight
```

The idea is to provide an additional layer of protection when market conditions become unstable.

---

## 🔍 Explainability with SHAP

Financial models can be difficult to interpret, especially when they combine graph neural networks and reinforcement learning.

To address this, the project uses **SHAP (SHapley Additive exPlanations)**.

SHAP is used mainly for:

### Feature Selection

SHAP values help identify which financial indicators contribute the most useful information.

This can help reduce unnecessary features and noise in the dataset.

### Model Interpretation

SHAP also provides a way to understand how different financial features influence the model.

This makes the overall framework easier to analyze instead of treating it as a complete black box.

---

## 📈 Evaluation

The portfolio strategy is evaluated using several performance and risk metrics:

* **Cumulative Return**
* **Sharpe Ratio**
* **Skewness**
* **Kurtosis**
* **Win Rate**
* **Volatility**
* **Tail Risk**
* **Portfolio Drawdown**

The proposed approach is compared with conventional portfolio strategies, including an **equal-weight portfolio**.

The reported results show that the GraphSAGE-PPO approach achieved higher cumulative returns and stronger risk-adjusted performance than the equal-weight strategy. The model also showed lower volatility and reduced exposure to severe losses in the reported experiments.

