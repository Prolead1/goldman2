# Part 5 - Portfolio Analytics

**Summary.** MSCI Value delivers measurable HML exposure (+0.322), but its more elaborate classification does not generate significant alpha (t = -0.91). A generic factor replication historically delivered a higher Sharpe (0.529 versus 0.450). The equal-weighted multi-factor index improves absolute performance, but its benchmark-relative information ratio is only 0.045. Minimum Volatility is the better stand-alone risk-adjusted investment (Sharpe 0.66), although its low market beta makes it challenging for benchmark-constrained managers.

## 5A. MSCI Value and Fama-French value definitions

MSCI uses three value indicators: book value/price, 12-month forward earnings/price, and dividend yield. It separately measures growth using long- and short-term forward EPS growth, current internal growth, historical long-term EPS growth, and historical long-term sales-per-share growth. Value and growth membership can overlap, with partial style weights. MSCI's industry and universe exceptions apply **only to growth variables**: for example, historical long-term sales-per-share growth is excluded for Banks and Diversified Financials (except Multi-Sector Holdings and Financial Exchanges & Data), while long-term forward EPS growth is excluded for small-cap indexes.

Fama-French HML instead uses **one cheapness measure, book-to-market (B/M)**. Stocks are sorted into a 2-by-3 grid by size (small/big) and B/M (low/neutral/high), rebalanced each June. HML is the average return of high-B/M portfolios minus low-B/M portfolios across the two size groups. The low-B/M end represents growth without MSCI's separate five-signal growth score.

| Design choice | MSCI USA Value | Fama-French HML |
|---|---|---|
| Value signal | Book/price, forward earnings/price, yield | Single book-to-market ratio |
| Growth treatment | Separate five-indicator score | Low book-to-market end of sort |
| Construction | Long-only large/mid-cap style weights | Long-short 2x3 size/B/M sorts; June rebalance |
| Implication | Investable but blends style and universe decisions | Long-short factor return; not an investable index |

**Exhibit 5A.** Value construction comparison, methodology applicable to the US equity universe; MSCI USA Standard and Fama-French US factor portfolios. Source: MSCI Global Investable Market Value and Growth Index Methodology (February 2021); Kenneth French Data Library factor definitions.

**Investment question:** Does the extra classification complexity earn a return premium, or merely repackage exposures? The following regression provides a direct diagnostic, not a causal answer.

## 5B. Factor exposure regression

Using 343 monthly gross USD returns (January 1995-July 2023), regress MSCI USA Value on an intercept, RF, market excess return, HML and SMB. Coefficients measure exposure; conventional OLS t-statistics measure estimation precision.

| Regressor | Loading | t-statistic | Interpretation |
|---|---:|---:|---|
| Intercept (monthly %) | -0.0713 | -0.91 | No significant positive alpha |
| RF | 0.8646 | 2.79 | Cash-rate exposure estimated, not fixed |
| Market excess | 0.9480 | 76.99 | Near-market equity sensitivity |
| HML | +0.3216 | 19.68 | Strong value exposure |
| SMB | -0.1744 | -9.90 | Large-cap bias embedded in style index |

**Exhibit 5B.** OLS factor fingerprint of MSCI USA Value, January 1995-July 2023, 343 monthly observations, MSCI USA Standard and Fama-French US factors. Source: supplied MSCI and Fama-French workbook.

R-squared is **94.84%**. In a month when HML gains one percentage point, the fitted index return increases by 0.322 percentage points, holding other inputs fixed; a one-point SMB gain reduces it by 0.174 points. Verdict: MSCI Value behaves as advertised, with strong HML exposure (+0.32; t = 19.7) and a large-cap tilt shown by SMB (-0.17). Its negative SMB loading can hurt when small caps outperform. 

## 5C. Enhancing factor performance

The intercept is **-0.0713% monthly** (t = -0.91, p = 0.366), equivalent to -0.86% under simple annualization. There is no statistically significant evidence that MSCI's multi-indicator value methodology adds positive risk-adjusted alpha beyond these conventional factors. 

As a robustness check following the **May 2003 methodology change**, the June 2003-July 2023 subsample gives alpha **-0.0378% monthly (t = -0.55; n = 242)**. 

**Exhibit 5C.** Full-sample and post-May-2003 intercept tests, January 1995-July 2023 and June 2003-July 2023, MSCI USA Value and Fama-French US factors. Source: supplied workbook; authors' OLS calculations.

## 5D. Fund replication

The fitted zero-alpha replication is 0.8646 RF + 0.9480(Mkt-RF) + 0.3216 HML - 0.1744 SMB. Replication was effective (correlation **0.974**). The replica historically achieved a higher compounded return and Sharpe. Before investing, assess shorting and trading costs for long-short factors, fees relative to an index fund, and the risk of in-sample fitting.

| Metric | MSCI Value | FF replication |
|---|---:|---:|
| Arithmetic return, annualized | 9.87% | 10.73% |
| CAGR | 9.05% | 10.05% |
| Volatility, annualized | 15.27% | 14.87% |
| Geometric-return Sharpe | 0.450 | 0.529 |

**Exhibit 5D.** Historical return, risk and growth-of-$1 comparison, January 1995-July 2023, MSCI USA Value versus fitted Fama-French US factor replication, gross USD. Source: supplied workbook; authors' calculations. Sharpe = (CAGR - geometric annualized RF)/annualized monthly volatility, consistent with Part 1.

The replica's arithmetic return advantage of **0.86 percentage points annually** equals the regression intercept with its sign reversed and annualized; that intercept is **not significant (t = -0.91)**. 

![Exhibit 5D: Growth of $1](5d_replication_growth.png)

## 5E. Relative performance analysis

For each MSCI style index, active return equals its monthly total return less MSCI USA Standard. The multi-factor index is rebalanced to equal weights monthly. Annualized tracking error is monthly active-return sample standard deviation times sqrt(12); IR is annualized arithmetic active return divided by tracking error.

| Index | Active return (annual) | Tracking error | IR |
|---|---:|---:|---:|
| Value | -1.34% | 5.15% | -0.261 |
| Minimum Volatility | -0.80% | 6.38% | -0.125 |
| Equal Weight | +0.59% | 5.40% | 0.109 |
| Momentum | +2.16% | 7.73% | 0.279 |
| Multi-factor index | +0.15% | 3.39% | **0.045** |

**Exhibit 5E.** Benchmark-relative returns, tracking errors and information ratios, January 1995-July 2023, MSCI USA Value, Minimum Volatility, Equal Weight, Momentum and equal-weight blend versus MSCI USA Standard, gross USD. Source: supplied MSCI index data; authors' calculations.

**IR 0.045 is very low**, indicating limited benchmark-relative efficiency. Part 1 showed that the multi-factor index beat the market on Sharpe (**0.60 versus 0.54**), mainly through lower volatility. Its **0.15% annual arithmetic active return** averages monthly differences; its **0.34 percentage-point geometric CAGR advantage** compares two separately compounded wealth paths. 

![Exhibit 5E: Cumulative active returns relative to MSCI USA Standard](5e_active_returns.png)

**Exhibit 5E chart.** Cumulative active returns relative to MSCI USA Standard, January 1995-July 2023. Source: supplied MSCI index data; authors' calculations.

## 5F. Risk decomposition and investment decision

Let a_i represent each index's active return and a_p the equal-weight portfolio's active return. Then a_p = (1/4) sum(a_i). Each index's variance contribution is Cov(a_i, a_p) / [4 Var(a_p)]. Contributions sum to 100%. Minimum Volatility is the largest driver (**34.08%**). 

| Factor | Share of active-return variance |
|---|---:|
| Value | 23.26% |
| Minimum Volatility | 34.08% |
| Equal Weight | 17.86% |
| Momentum | 24.81% |

**Exhibit 5F.** Covariance contributions to multi-factor index tracking-error variance, January 1995-July 2023, MSCI USA smart-beta indexes relative to MSCI USA Standard, gross USD. Source: supplied MSCI data; authors' calculations.

![Exhibit 5F: Contributions to active-return variance](5f_tracking_error_decomposition.png)

**Exhibit 5F chart.** Covariance-based contributions to the multi-factor index's active-return variance. Source: supplied MSCI index data; authors' calculations.

**Decision: Minimum Volatility is better as a stand-alone investment.** Its Sharpe of **0.66** is the highest among the MSCI indexes examined in Part 1, even though its IR is **-0.125** with **6.38% tracking error**. Its market beta is approximately **0.72**, so it tends to lag when the equity market rises sharply. This creates a **limits-to-arbitrage** problem: a benchmarked manager can recognize superior stand-alone risk-adjusted returns yet be penalized for tracking error and relative underperformance before the thesis pays off. 

**Methods and caveats.** MSCI index levels are converted to monthly returns; all results use January 1995-July 2023 gross USD data. Factor replication is fitted and evaluated on the same observations. Conventional OLS inference is unadjusted for autocorrelation. All investment comparisons exclude trading and financing costs. The limits-to-arbitrage interpretation is qualitative and should be considered alongside the reported tracking-error evidence.
