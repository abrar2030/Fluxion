import pytest
import numpy as np
from backend.engine.pricing_model import CurvePricing, APR_model, OptimizedPricing

class TestCurvePricing:
    def test_initialization(self):
        # Test with default values
        cp_default = CurvePricing()
        assert cp_default.A == 100
        assert cp_default.D == 1_000_000

        # Test with custom values
        cp_custom = CurvePricing(A=50, D=500_000)
        assert cp_custom.A == 50
        assert cp_custom.D == 500_000

    def test_xy_to_d(self):
        cp = CurvePricing(A=100, D=1_000_000)
        # Using the formula: d = (4*A*x*y + D^2)**0.5 - D
        # Let x = 500_000, y = 500_000. D is 1_000_000, A is 100
        # d = (4*100*500000*500000 + (10^6)^2)**0.5 - 10^6
        # d = (10^14 + 10^12)**0.5 - 10^6 = (100*10^12 + 10^12)**0.5 - 10^6
        # d = (101 * 10^12)**0.5 - 10^6 = 10^6 * (sqrt(101) - 1) approx 9049875.6
        # This formula seems to be for a different invariant. The one in the code is simpler.
        # The provided formula in the code is: (4 * self.A * x * y + self.D**2)**0.5 - self.D
        # This seems to be a simplified or incorrect representation of Curve's invariant.
        # Let's test based on the provided formula, assuming it's a specific model.
        # If x=D/2 and y=D/2, then d = (A*D^2 + D^2)**0.5 - D = D * ( (A+1)**0.5 -1)
        # This doesn't seem right for Curve's D invariant. Let's assume the formula is as written for testing purposes.
        # For x=250_000, y=250_000, A=100, D=1_000_000
        # d = (4*100*250000*250000 + 1000000**2)**0.5 - 1000000
        # d = (400 * 6.25e10 + 1e12)**0.5 - 1e6
        # d = (2.5e13 + 1e12)**0.5 - 1e6
        # d = (25e12 + 1e12)**0.5 - 1e6 = (26e12)**0.5 - 1e6 = 1e6 * (sqrt(26) - 1) approx 4.099e6
        # The xy_to_d method as written in the problem seems to be a custom invariant, not standard Curve.
        # Let's test it as is.
        x, y = 250000, 250000
        expected_d = (4 * cp.A * x * y + cp.D**2)**0.5 - cp.D
        assert cp.xy_to_d(x, y) == pytest.approx(expected_d)

        x_zero, y_val = 0, 500000
        expected_d_x_zero = (cp.D**2)**0.5 - cp.D # Should be 0
        assert cp.xy_to_d(x_zero, y_val) == pytest.approx(expected_d_x_zero)

        # New test cases
        # Test with very small values
        x_small, y_small = 1, 1
        d_small = cp.xy_to_d(x_small, y_small)
        assert d_small > 0
        assert d_small < cp.D

        # Test with very large values
        x_large, y_large = 1_000_000_000, 1_000_000_000
        d_large = cp.xy_to_d(x_large, y_large)
        assert d_large > 0
        assert d_large > cp.D

        # Test with equal values
        x_equal, y_equal = 1_000_000, 1_000_000
        d_equal = cp.xy_to_d(x_equal, y_equal)
        assert d_equal > 0
        assert d_equal == pytest.approx(cp.xy_to_d(y_equal, x_equal))  # Should be symmetric

    def test_calculate_swap(self):
        cp = CurvePricing(A=100, D=1_000_000)
        x, y = 500_000, 500_000 # Initial state, assuming balanced pool for simplicity
        dx = 10_000
        
        # Formula: new_y = (D**2) / (4 * A * new_x)
        # dy = y - new_y
        new_x_calc = x + dx
        expected_new_y = (cp.D**2) / (4 * cp.A * new_x_calc)
        expected_dy = y - expected_new_y
        
        assert cp.calculate_swap(dx, x, y) == pytest.approx(expected_dy)

        # Test selling y (dx is negative if x is the token being sold to get y)
        # To avoid confusion, let's assume dx is always for the first token (x)
        # If we want to test receiving x by giving y, we'd need a different perspective or a reverse function.
        # The current function calculates how much y is received for dx of x.
        dx_negative = -10_000 # Selling x
        new_x_neg_calc = x + dx_negative
        if new_x_neg_calc <= 0: # Avoid division by zero or negative sqrt if model implies
            with pytest.raises(ValueError): # Or appropriate error
                 cp.calculate_swap(dx_negative, x, y)
        else:
            expected_new_y_neg = (cp.D**2) / (4 * cp.A * new_x_neg_calc)
            expected_dy_neg = y - expected_new_y_neg
            assert cp.calculate_swap(dx_negative, x, y) == pytest.approx(expected_dy_neg)

        # New test cases
        # Test with very small swap
        dx_small = 1
        dy_small = cp.calculate_swap(dx_small, x, y)
        assert dy_small > 0
        assert dy_small < y

        # Test with very large swap
        dx_large = x * 0.5  # 50% of pool
        dy_large = cp.calculate_swap(dx_large, x, y)
        assert dy_large > 0
        assert dy_large < y

        # Test price impact
        dx_1 = 10_000
        dx_2 = 20_000
        dy_1 = cp.calculate_swap(dx_1, x, y)
        dy_2 = cp.calculate_swap(dx_2, x, y)
        # Price impact should be non-linear (slippage)
        assert (dy_2 / dx_2) < (dy_1 / dx_1)

class TestAPRModel:
    def test_apr_model_calculation(self):
        # Base case
        assert APR_model(A=100, D=1_000_000) == pytest.approx(0.10 - (0.0001*100) - (0.00000001*1000000))
        # Min APR check
        assert APR_model(A=1000, D=10_000_000) == 0.01
        # Intermediate value
        assert APR_model(A=50, D=500_000) == pytest.approx(0.10 - (0.0001*50) - (0.00000001*500000))
        # Zero A and D
        assert APR_model(A=0, D=0) == 0.10

        # New test cases
        # Test with very small values
        apr_small = APR_model(A=1, D=1_000)
        assert 0.01 <= apr_small <= 0.10

        # Test with very large values
        apr_large = APR_model(A=10_000, D=100_000_000)
        assert apr_large == 0.01  # Should hit minimum

        # Test monotonicity
        apr_1 = APR_model(A=100, D=1_000_000)
        apr_2 = APR_model(A=200, D=1_000_000)
        apr_3 = APR_model(A=100, D=2_000_000)
        assert apr_2 < apr_1  # Higher A should give lower APR
        assert apr_3 < apr_1  # Higher D should give lower APR

class TestOptimizedPricing:
    def test_optimize_pool_runs(self, mocker):
        # Mock the minimize function to avoid actual optimization during unit test
        mock_result = mocker.Mock()
        mock_result.x = [100, 1_000_000] # Example return
        mocker.patch("backend.engine.pricing_model.minimize", return_value=mock_result)
        
        op = OptimizedPricing()
        target_apr = 0.05
        params = op.optimize_pool(target_apr)
        assert isinstance(params, list) or isinstance(params, np.ndarray)
        assert len(params) == 2
        # Further assertions could check if minimize was called with expected arguments

        # New test cases
        # Test with different target APRs
        target_aprs = [0.03, 0.05, 0.07]
        for target in target_aprs:
            params = op.optimize_pool(target)
            assert len(params) == 2
            assert all(p > 0 for p in params)

        # Test optimization constraints
        mock_result.x = [0, 0]  # Invalid parameters
        with pytest.raises(ValueError):
            op.optimize_pool(0.05)

        # Test optimization convergence
        mock_result.success = False
        with pytest.raises(RuntimeError):
            op.optimize_pool(0.05)

