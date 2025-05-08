import pytest
import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurtosis, norm
from backend.engine.risk_calculator import AdvancedRiskMetrics

@pytest.fixture
def arm_instance():
    return AdvancedRiskMetrics()

class TestAdvancedRiskMetrics:
    def test_calculate_cvar(self, arm_instance):
        returns_1 = np.array([-0.1, -0.05, 0.01, 0.02, 0.05]) # Sorted for easier manual check
        confidence_1 = 0.95
        # For 95% confidence, VaR is the 5th percentile. With 5 points, it's the lowest value: -0.1
        # CVaR is the mean of returns <= VaR. Here, only -0.1.
        expected_cvar_1 = -0.1
        assert arm_instance.calculate_cvar(returns_1, confidence_1) == pytest.approx(expected_cvar_1)

        returns_2 = np.array([-0.2, -0.15, -0.1, -0.05, 0.01, 0.02, 0.05, 0.1]) # 8 points
        confidence_2 = 0.90 # VaR is 10th percentile. 0.1*8 = 0.8 -> 1st element: -0.2
        # CVaR is mean of returns <= -0.2. Here, only -0.2
        expected_cvar_2 = -0.2 
        assert arm_instance.calculate_cvar(returns_2, confidence_2) == pytest.approx(expected_cvar_2)
        
        confidence_3 = 0.75 # VaR is 25th percentile. 0.25*8 = 2 -> 2nd element: -0.15
        # CVaR is mean of returns <= -0.15. Here, [-0.2, -0.15]. Mean = -0.175
        var_3 = np.percentile(returns_2, 100 * (1 - confidence_3))
        expected_cvar_3 = returns_2[returns_2 <= var_3].mean()
        assert arm_instance.calculate_cvar(returns_2, confidence_3) == pytest.approx(expected_cvar_3)

        # Edge case: all positive returns
        returns_positive = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        var_positive = np.percentile(returns_positive, 100 * (1 - 0.95))
        expected_cvar_positive = returns_positive[returns_positive <= var_positive].mean()
        assert arm_instance.calculate_cvar(returns_positive, 0.95) == pytest.approx(expected_cvar_positive)

        # Edge case: empty list
        with pytest.raises(IndexError): # Or specific error handling if implemented
            arm_instance.calculate_cvar(np.array([]), 0.95)

        # New test cases
        # Test with large dataset
        returns_large = np.random.normal(0, 0.02, 10000)
        cvar_large = arm_instance.calculate_cvar(returns_large, 0.99)
        assert isinstance(cvar_large, float)
        assert cvar_large < 0  # Should be negative for normal distribution

        # Test with extreme values
        returns_extreme = np.array([-0.5, -0.4, -0.3, 0.1, 0.2, 0.3, 0.4, 0.5])
        cvar_extreme = arm_instance.calculate_cvar(returns_extreme, 0.95)
        assert cvar_extreme <= -0.5  # Should be at most -0.5

        # Test confidence level boundaries
        with pytest.raises(ValueError):
            arm_instance.calculate_cvar(returns_1, 0)  # Invalid confidence level
        with pytest.raises(ValueError):
            arm_instance.calculate_cvar(returns_1, 1.1)  # Invalid confidence level

    def test_liquidity_at_risk(self, arm_instance):
        volume_history_1 = np.array([1000, 1200, 900, 1500, 1100])
        confidence_1 = 0.99
        log_volumes_1 = np.log(volume_history_1)
        mu_1 = np.mean(log_volumes_1)
        sigma_1 = np.std(log_volumes_1)
        expected_lar_1 = np.exp(mu_1 + sigma_1 * norm.ppf(1 - confidence_1))
        assert arm_instance.liquidity_at_risk(volume_history_1, confidence_1) == pytest.approx(expected_lar_1)

        # Edge case: constant volume
        volume_history_const = np.array([1000, 1000, 1000, 1000])
        log_volumes_const = np.log(volume_history_const)
        mu_const = np.mean(log_volumes_const)
        sigma_const = np.std(log_volumes_const) # Should be 0
        expected_lar_const = np.exp(mu_const + sigma_const * norm.ppf(1 - 0.99))
        assert arm_instance.liquidity_at_risk(volume_history_const, 0.99) == pytest.approx(expected_lar_const)
        assert sigma_const == 0 # verify std is 0 for constant input

        # Edge case: empty list
        with pytest.raises(ValueError): # numpy mean of empty slice raises RuntimeWarning and returns nan
            arm_instance.liquidity_at_risk(np.array([]), 0.99)

        # New test cases
        # Test with large dataset
        volume_large = np.random.lognormal(10, 1, 1000)
        lar_large = arm_instance.liquidity_at_risk(volume_large, 0.99)
        assert isinstance(lar_large, float)
        assert lar_large > 0

        # Test with extreme volatility
        volume_volatile = np.array([100, 1000, 100, 1000, 100])
        lar_volatile = arm_instance.liquidity_at_risk(volume_volatile, 0.99)
        assert lar_volatile < np.min(volume_volatile)  # Should be below minimum volume

        # Test confidence level boundaries
        with pytest.raises(ValueError):
            arm_instance.liquidity_at_risk(volume_history_1, 0)  # Invalid confidence level
        with pytest.raises(ValueError):
            arm_instance.liquidity_at_risk(volume_history_1, 1.1)  # Invalid confidence level

    def test_portfolio_skewness(self, arm_instance):
        returns_symmetric = np.array([-2, -1, 0, 1, 2])
        expected_skew_symmetric = skew(returns_symmetric, bias=False)
        assert arm_instance.portfolio_skewness(returns_symmetric) == pytest.approx(expected_skew_symmetric)
        assert expected_skew_symmetric == pytest.approx(0) # Symmetric distribution

        returns_positive_skew = np.array([1, 2, 3, 4, 10]) # Positively skewed
        expected_skew_positive = skew(returns_positive_skew, bias=False)
        assert arm_instance.portfolio_skewness(returns_positive_skew) == pytest.approx(expected_skew_positive)
        assert expected_skew_positive > 0

        # New test cases
        # Test with large dataset
        returns_large = np.random.normal(0, 1, 1000)
        skew_large = arm_instance.portfolio_skewness(returns_large)
        assert isinstance(skew_large, float)
        assert abs(skew_large) < 0.1  # Should be close to 0 for normal distribution

        # Test with extreme skew
        returns_extreme_skew = np.concatenate([
            np.random.normal(-1, 0.1, 990),  # Many small negative returns
            np.random.normal(10, 0.1, 10)    # Few large positive returns
        ])
        skew_extreme = arm_instance.portfolio_skewness(returns_extreme_skew)
        assert skew_extreme > 0  # Should be positively skewed

        # Test with single value
        with pytest.raises(ValueError):
            arm_instance.portfolio_skewness(np.array([1.0]))

    def test_portfolio_kurtosis(self, arm_instance):
        # Using scipy.stats.kurtosis with fisher=False gives Pearson's kurtosis
        returns_normal_ish = norm.rvs(size=1000, random_state=42) # Sample from normal distribution
        expected_kurtosis_normal = scipy_kurtosis(returns_normal_ish, fisher=False, bias=False)
        assert arm_instance.portfolio_kurtosis(returns_normal_ish) == pytest.approx(expected_kurtosis_normal)
        # For a normal distribution, Pearson's kurtosis is approx 3
        assert arm_instance.portfolio_kurtosis(returns_normal_ish) == pytest.approx(3, abs=0.5) 

        returns_leptokurtic = np.array([-10, -1, 0, 1, 10]) # More peaked, fatter tails
        expected_kurtosis_lepto = scipy_kurtosis(returns_leptokurtic, fisher=False, bias=False)
        assert arm_instance.portfolio_kurtosis(returns_leptokurtic) == pytest.approx(expected_kurtosis_lepto)
        # Leptokurtic should have kurtosis > 3
        assert arm_instance.portfolio_kurtosis(returns_leptokurtic) > 3

        # New test cases
        # Test with large dataset
        returns_large = np.random.normal(0, 1, 1000)
        kurt_large = arm_instance.portfolio_kurtosis(returns_large)
        assert isinstance(kurt_large, float)
        assert abs(kurt_large - 3) < 0.5  # Should be close to 3 for normal distribution

        # Test with platykurtic distribution (less peaked than normal)
        returns_platykurtic = np.random.uniform(-1, 1, 1000)
        kurt_platy = arm_instance.portfolio_kurtosis(returns_platykurtic)
        assert kurt_platy < 3  # Should be less than 3 for uniform distribution

        # Test with extreme kurtosis
        returns_extreme_kurt = np.concatenate([
            np.random.normal(0, 0.1, 980),  # Many small returns
            np.random.normal(0, 10, 20)     # Few extreme returns
        ])
        kurt_extreme = arm_instance.portfolio_kurtosis(returns_extreme_kurt)
        assert kurt_extreme > 3  # Should be leptokurtic

        # Test with single value
        with pytest.raises(ValueError):
            arm_instance.portfolio_kurtosis(np.array([1.0]))

