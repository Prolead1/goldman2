import numpy as np
from part5.analysis import load_sample, analyze, fit_factors

def test_sample_and_regression():
    data=load_sample();result,returns,active=analyze(data)
    assert len(data)==343
    assert result['params']['HML']>0 and result['params']['SMB']<0
    assert np.isclose(result['r2'],0.948414224732985,atol=1e-9)
    assert np.isclose(result['replication_correlation'],0.9738656091745855,atol=1e-9)
    assert 'const' in fit_factors(data).params.index

def test_portfolio_and_risk_identity():
    result,returns,active=analyze(load_sample())
    assert np.allclose(returns['MultiFactor'],returns[['Value','MinVol','EqualWeight','Momentum']].mean(axis=1))
    assert np.allclose(active['MultiFactor'],active[['Value','MinVol','EqualWeight','Momentum']].mean(axis=1))
    assert np.isclose(sum(result['variance_contributions'].values()),1)
    assert max(result['variance_contributions'],key=result['variance_contributions'].get)=='MinVol'
    assert np.isclose(result['active']['MultiFactor']['information_ratio'],0.04484617791112214)
