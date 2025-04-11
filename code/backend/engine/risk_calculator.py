import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import norm

class AdvancedRiskMetrics:
    def calculate_cvar(self, returns, confidence=0.95):
        var = np.percentile(returns, 100*(1-confidence))
        return returns[returns <= var].mean()

    def liquidity_at_risk(self, volume_history, confidence=0.99):
        log_volumes = np.log(volume_history)
        mu = np.mean(log_volumes)
        sigma = np.std(log_volumes)
        return np.exp(mu + sigma * norm.ppf(1-confidence))

    def portfolio_skewness(self, returns):
        return skew(returns, bias=False)

    def portfolio_kurtosis(self, returns):
        return kurtosis(returns, fisher=False)
