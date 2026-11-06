# Part 5 — Portfolio Analytics

**Summary.** The MSCI USA Value Index has the advertised positive value exposure (HML loading 0.322) and negative small-cap exposure (SMB loading -0.174). A Fama–French factor replication captures 94.84% of its monthly return variation, but MSCI's additional style-selection rules do not generate statistically significant positive alpha. The replicating portfolio has a higher gross Sharpe ratio. Separately, the equal-weighted four-factor MSCI blend earns only 0.15% annualized active return against 3.39% tracking error (information ratio 0.04). Minimum volatility is the largest contributor to active risk.

## 5A. What is MSCI actually selling? A value-definition audit

**The non-obvious distinction is that MSCI Value is not simply a portfolio of the cheapest stocks.** MSCI scores value using **book value/price, 12-month forward earnings/price and dividend yield**, while separately measuring growth through **long-term and short-term forward EPS growth, current internal growth, historical long-term EPS growth and historical long-term sales-per-share growth**. Because value and growth are evaluated separately, a stock can score well on both dimensions; the style indexes are not a binary cheap-versus-expensive split. Securities can receive partial style inclusion weights.

This is economically consequential. **Book/price** emphasizes accounting assets, **forward earnings/price** incorporates analysts' expectations of profitability, and **dividend yield** captures cash distributed to shareholders. A stock with weak book value but strong expected earnings need not be treated like a conventional high-book-to-market stock. The index is therefore a *multi-signal valuation screen*, not a pure bet on balance-sheet cheapness. MSCI also recognizes that a universal growth signal is not appropriate for every industry: the cited methodology excludes historical long-term sales-per-share growth for banks and diversified financials (subject to specified sub-industry exceptions), and excludes long-term forward EPS growth in Small Cap indexes. These are **sector/universe-specific measurement choices**, not evidence of superior performance.

**Exhibit 5A. What is being measured?**

| Due-diligence question | MSCI USA Value | Fama–French HML |
|---|---|---|
| What defines cheapness? | Book/price + forward earnings/price + dividend yield | High-minus-low book-to-market portfolio spread |
| Is growth assessed separately? | Yes; five forecast, internal and historical growth indicators | No comparable multi-indicator growth classification |
| Does classification vary by security type? | Yes; sector and index-size exceptions | Standard academic size/book-to-market sorts |
| What is the investable object? | Long-only, large/mid-cap style index | Long-short factor return spread, not a stand-alone long-only index |

**Investment implication.** MSCI's methodology may improve *classification* and investability, but that does not establish *alpha*. The test for a manager-selection team is whether the extra signals deliver returns that cannot be explained by inexpensive market, size and value exposures. Sections 5B–5D test precisely that. **Sources:** MSCI Global Investable Market Value and Growth Index Methodology (February 2021), §§1.2, 2.1–2.3; Fama–French factor definitions.

## 5B. The index's quantitative fingerprint: advertised exposure or hidden bets?

We regress MSCI USA Standard Value **monthly gross total returns** on contemporaneous RF, Mkt−RF, HML and SMB, including an intercept, over **343 months (January 1995–July 2023)**. This is one index time series, so it is a **time-series OLS regression**, notwithstanding the assignment's “pooled panel” wording. Returns and the intercept are in monthly percentage points. Reported t-statistics are conventional OLS, as requested (no Newey–West correction).

| Factor | Loading | t-statistic | Economic interpretation |
|---|---:|---:|---|
| Intercept (monthly %) | −0.0713 | −0.91 | No statistically detectable positive residual return |
| RF | 0.8646 | 2.79 | Cash-rate sensitivity estimated rather than fixed at one |
| Market excess return | 0.9480 | 76.99 | Near-market equity exposure, not market neutrality |
| HML (value) | **+0.3216** | **19.68** | Clear, statistically strong value tilt |
| SMB (small minus big) | **−0.1744** | **−9.90** | Large-cap tilt consistent with the Standard universe |

**Exhibit 5B.** Factor fingerprint of MSCI USA Standard Value; OLS R² = **94.84%** (n = 343). The 95% confidence interval for HML is **[0.289, 0.354]**, and for SMB **[−0.209, −0.140]**. Source: supplied Fama–French factors and MSCI gross USD index levels.

**Translate the coefficients into a stress test.** Holding other regressors fixed, a **+1 percentage-point HML month** predicts approximately **+0.322 percentage points** of MSCI Value return; a **+1 percentage-point SMB month** predicts **−0.174 percentage points**. Thus, an episode in which *small caps outperform large caps* is a headwind for this nominally “value” index even if the pure value spread is unchanged. That is a hidden implementation choice an asset owner should understand: **the index delivers value exposure bundled with a large/mid-cap bias**.

**Verdict: as advertised, but not a pure HML trade.** The positive HML and negative SMB coefficients match the index's style and universe; the near-unit market loading confirms it remains an equity portfolio. The 94.84% fit means standard factors explain most *in-sample return variation*, not that 94.84% of performance is attributable to stock-picking skill. The remaining variation could reflect sector weights, reconstitution, valuation signals beyond book/price and measurement differences. It is not automatically alpha; the intercept in 5C tests that narrower claim.

## 5C. Enhancing factor performance

The estimated intercept is **-0.0713% per month** (t = -0.91), approximately **-0.86% per year** under simple annualization. It is negative and statistically indistinguishable from zero (p = 0.366). Thus the data provide **no evidence that MSCI's more elaborate value definition improved risk-adjusted performance** over the fitted generic FF factor combination. This is regression alpha, not a causal test of index-design superiority; the sample includes historical methodology changes and is gross of costs.

## 5D. Fund replication

The zero-intercept replicating return is 0.8646 RF + 0.9480(Mkt−RF) + 0.3216 HML -0.1744 SMB. The in-sample correlation with MSCI Value is **0.974**; replication is strong in explaining fluctuations, but is not a feasible self-financing investment by itself without specifying factor instruments, financing and collateral.

| Metric | MSCI Value | FF replication |
|---|---:|---:|
| Annualized arithmetic return | 9.87% | 10.73% |
| Annualized compounded return (CAGR) | 9.05% | 10.05% |
| Annualized volatility | 15.27% | 14.87% |
| Annualized Sharpe ratio | 0.505 | 0.576 |

**Exhibit 5D.** MSCI Value versus in-sample FF replication, Jan 1995–Jul 2023, gross monthly USD returns. Sharpe = sqrt(12) × mean(monthly return − RF) / std(monthly total return); annualized volatility = sqrt(12) × monthly sample standard deviation. The FF replication has higher gross return and Sharpe in-sample, but the regression weights were fitted on the same evaluation period. Before implementation, assess (i) factor-leg trading, shorting and financing costs, (ii) turnover, liquidity and rebalancing frictions, and (iii) availability of suitable investable factor products and their tracking error. A rolling out-of-sample test is needed before selecting the replication in practice.

![Exhibit 5D: Growth of $1](5d_replication_growth.png)

## 5E. Relative performance analysis

Each component's monthly active return equals its MSCI gross total return minus MSCI USA Standard's same-month return. The multi-factor index is the equal-weighted monthly average of Value, Minimum Volatility, Equal Weighted and Momentum. Its annualized active return is 12 × mean(active monthly return), and its annualized tracking error is sqrt(12) × sample standard deviation of monthly active return.

| Index | Annualized active return | Annualized tracking error | Information ratio |
|---|---:|---:|---:|
| Value | -1.34% | 5.15% | -0.261 |
| MinVol | -0.80% | 6.38% | -0.125 |
| EqualWeight | +0.59% | 5.40% | 0.109 |
| Momentum | +2.16% | 7.73% | 0.279 |
| MultiFactor | +0.15% | 3.39% | 0.045 |

**Exhibit 5E.** Benchmark-relative performance, Jan 1995–Jul 2023. The multi-factor blend's annualized excess return is only **0.15%** for **3.39%** tracking error, producing an information ratio of **0.045**. This is economically weak active efficiency: the blend improves some absolute-risk characteristics relative to individual factors, but adds little return per unit of benchmark-relative risk.

## 5F. Risk decomposition

With active factor returns aᵢ and equally weighted active blend aₚ = ¼Σᵢaᵢ, its variance satisfies Var(aₚ) = ¼ΣᵢCov(aₚ,aᵢ). We attribute each component's share of the blend's active-return variance as Cov(aₚ,aᵢ)/[4 Var(aₚ)]. These signed contributions sum to 100% and reflect both standalone volatility and cross-factor correlations.

| Factor | Share of active-return variance |
|---|---:|
| Value | 23.26% |
| MinVol | 34.08% |
| EqualWeight | 17.86% |
| Momentum | 24.81% |


**Exhibit 5F.** Covariance-based tracking-error variance attribution, Jan 1995–Jul 2023. **Minimum volatility (34.1%)** is the largest contributor, followed by Momentum (24.8%). Although minimum volatility is attractive as a stand-alone defensive allocation, it can create substantial benchmark-relative risk because its defensive tilt differs from the cap-weighted equity benchmark. Under limits to arbitrage, low-beta stocks can offer attractive risk-adjusted standalone returns, but that does not imply low tracking error for a benchmark-constrained investor. Thus the appropriate choice depends on the investor's mandate: standalone for absolute-risk control, or a more carefully sized tilt for benchmark-relative mandates.

![Exhibit 5F: Tracking-error decomposition](5f_tracking_error_decomposition.png)

**Sources and notes.** Supplied workbook `class-project-data-ff-msci-parts-1-and-5.xlsx`: FF monthly RF, Mkt−RF, SMB, HML and MSCI gross USD index levels; MSCI methodology: https://www.msci.com/eqb/methodology/meth_docs/MSCI_GIMIVGMethod_Feb2021.pdf. MSCI monthly returns are computed from adjacent month-end index levels. The factor replicating return omits the fitted intercept by design. All results gross of fees/costs, using in-sample OLS. The arithmetic return is mean monthly return × 12; CAGR is compounded monthly growth annualized. The supplied MSCI series and FF factor data end in July 2023.
