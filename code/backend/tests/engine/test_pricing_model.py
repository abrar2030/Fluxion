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

