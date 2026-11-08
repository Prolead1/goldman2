"""Part 5: reproducible factor attribution and benchmark-relative analysis.

Run with: python -m part5.analysis (from repository root).
All returns are percentage points per month, not decimal fractions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from goldman2.data import load_ff3, load_msci_returns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'part5' / 'outputs'
FACTORS = ['Value', 'MinVol', 'EqualWeight', 'Momentum']


def load_sample() -> pd.DataFrame:
    msci = load_msci_returns().copy()
    msci.columns = ['Market', *FACTORS]
    ff = load_ff3()
    data = msci.join(ff, how='inner').loc['1995-01':'2023-07'].dropna()
    if len(data) != 343:
        raise ValueError(f'Expected 343 monthly observations, found {len(data)}')
    return data


def fit_factors(data: pd.DataFrame):
    return sm.OLS(data['Value'], sm.add_constant(data[['RF', 'Mkt-RF', 'HML', 'SMB']])).fit()


def analyze(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model = fit_factors(data)
    d = data.copy()
    d['Replication'] = d[['RF', 'Mkt-RF', 'HML', 'SMB']] @ model.params[['RF', 'Mkt-RF', 'HML', 'SMB']]
    d['MultiFactor'] = d[FACTORS].mean(axis=1)
    active = d[FACTORS + ['MultiFactor']].sub(d['Market'], axis=0)
    rf_geo=((1+d['RF']/100).prod()**(12/len(d))-1)*100
    def stats(r):
        return dict(arithmetic_ann=float(r.mean()*12), geometric_ann=float(((1+r/100).prod()**(12/len(r))-1)*100),
                    vol_ann=float(r.std(ddof=1)*np.sqrt(12)),
                    sharpe=float((((1+r/100).prod()**(12/len(r))-1)*100-rf_geo)/(r.std(ddof=1)*np.sqrt(12))))
    post_model=fit_factors(d.loc['2003-06':])
    v = active['MultiFactor'].var(ddof=1)
    shares = {c:float(active[c].cov(active['MultiFactor'])/(4*v)) for c in FACTORS}
    result = dict(n=len(d), start=str(d.index.min()), end=str(d.index.max()),
                  params=model.params.to_dict(), t=model.tvalues.to_dict(), p=model.pvalues.to_dict(),
                  confidence_intervals=model.conf_int().to_dict('index'), r2=float(model.rsquared),
                  replication_correlation=float(d['Value'].corr(d['Replication'])),
                  portfolios={c:stats(d[c]) for c in ['Value', 'Replication']},
                  active={c:dict(annual_excess=float(active[c].mean()*12),
                                 tracking_error=float(active[c].std(ddof=1)*np.sqrt(12)),
                                 information_ratio=float(active[c].mean()/active[c].std(ddof=1)*np.sqrt(12))) for c in active},
                  variance_contributions=shares,
                  post_2003_alpha=dict(start='2003-06',n=int(post_model.nobs),monthly_alpha=float(post_model.params['const']),t_stat=float(post_model.tvalues['const'])),
                  minvol_beta=float(d['MinVol'].cov(d['Market'])/d['Market'].var(ddof=1)),
                  geometric_active_spread=float(((1+d['MultiFactor']/100).prod()**(12/len(d))-(1+d['Market']/100).prod()**(12/len(d)))*100))
    assert np.isclose(sum(shares.values()),1,atol=1e-10)
    assert np.isclose(active['MultiFactor'].var(ddof=1),sum(active[c].cov(active['MultiFactor'])/4 for c in FACTORS))
    return result,d,active


def save_charts(d: pd.DataFrame, active: pd.DataFrame, result: dict):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    wealth=(1+d[['Value','Replication']]/100).cumprod()
    fig,ax=plt.subplots(figsize=(8.1,3.5)); wealth.plot(ax=ax,lw=2)
    ax.set(title='Growth of $1: MSCI USA Value vs FF replication',ylabel='Growth of $1',xlabel='Year');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'5d_replication_growth.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.1,3.5));
    pd.Series(result['variance_contributions']).mul(100).sort_values().plot.barh(ax=ax)
    ax.set(title='Share of active-return variance by index',xlabel='Share of active-return variance (%)',ylabel='Factor')
    fig.tight_layout();fig.savefig(OUT/'5f_tracking_error_decomposition.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.1,3.5));
    active.cumsum().plot(ax=ax,lw=1.5)
    ax.set(title='Active return: factor tilts versus the market',ylabel='Cumulative arithmetic active return (pp)',xlabel='Year')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'5e_active_returns.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result,d,active=analyze(load_sample())
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    d.to_csv(OUT/'monthly_returns.csv');active.to_csv(OUT/'monthly_active_returns.csv')
    save_charts(d,active,result)
    print(f"PASS: {result['n']} months, R2={result['r2']:.4f}, IR={result['active']['MultiFactor']['information_ratio']:.4f}")

if __name__=='__main__':main()
