"""
Comprehensive test suite for risk management services
Tests all risk assessment, monitoring, and mitigation functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from services.risk.enhanced_risk_management_service import (
    EnhancedRiskManagementService,
    RiskAssessment,
    RiskLevel,
    RiskType,
    RiskMetric,
    PortfolioRisk,
    MarketRisk,
    CreditRisk,
    LiquidityRisk,
    OperationalRisk
)
from models.user import User
from models.portfolio import Portfolio, PortfolioAsset
from models.transaction import Transaction, TransactionStatus, TransactionType


class TestEnhancedRiskManagementService:
    """Test suite for Enhanced Risk Management Service"""
    
    @pytest.fixture
    def risk_service(self):
        """Create risk management service instance"""
        return EnhancedRiskManagementService()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing"""
        return User(
            id=uuid4(),
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            country="US",
            risk_tolerance="medium",
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_portfolio(self, sample_user):
        """Create sample portfolio for testing"""
        return Portfolio(
            id=uuid4(),
            user_id=sample_user.id,
            name="Test Portfolio",
            total_value=Decimal("100000.00"),
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_assets(self, sample_portfolio):
        """Create sample portfolio assets"""
        return [
            PortfolioAsset(
                id=uuid4(),
                portfolio_id=sample_portfolio.id,
                symbol="BTC",
                quantity=Decimal("2.5"),
                current_price=Decimal("40000.00"),
                current_value=Decimal("100000.00"),
                cost_basis=Decimal("35000.00")
            ),
            PortfolioAsset(
                id=uuid4(),
                portfolio_id=sample_portfolio.id,
                symbol="ETH",
                quantity=Decimal("50.0"),
                current_price=Decimal("2500.00"),
                current_value=Decimal("125000.00"),
                cost_basis=Decimal("2000.00")
            )
        ]
    
    @pytest.fixture
    def sample_transactions(self, sample_user, sample_portfolio):
        """Create sample transactions for testing"""
        return [
            Transaction(
                id=uuid4(),
                user_id=sample_user.id,
                portfolio_id=sample_portfolio.id,
                transaction_type=TransactionType.BUY,
                asset_symbol="BTC",
                quantity=Decimal("1.0"),
                price=Decimal("40000.00"),
                usd_value=Decimal("40000.00"),
                status=TransactionStatus.CONFIRMED,
                created_at=datetime.utcnow() - timedelta(days=1)
            ),
            Transaction(
                id=uuid4(),
                user_id=sample_user.id,
                portfolio_id=sample_portfolio.id,
                transaction_type=TransactionType.SELL,
                asset_symbol="ETH",
                quantity=Decimal("10.0"),
                price=Decimal("2500.00"),
                usd_value=Decimal("25000.00"),
                status=TransactionStatus.CONFIRMED,
                created_at=datetime.utcnow() - timedelta(hours=12)
            )
        ]
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk(self, risk_service, mock_db_session, 
                                          sample_portfolio, sample_assets):
        """Test portfolio risk calculation"""
        # Mock database queries
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_assets
        
        # Calculate portfolio risk
        portfolio_risk = await risk_service.calculate_portfolio_risk(
            mock_db_session, sample_portfolio.id
        )
        
        # Assertions
        assert isinstance(portfolio_risk, PortfolioRisk)
        assert portfolio_risk.portfolio_id == str(sample_portfolio.id)
        assert portfolio_risk.total_value > 0
        assert 0 <= portfolio_risk.volatility <= 1
        assert 0 <= portfolio_risk.var_95 <= 1
        assert portfolio_risk.sharpe_ratio is not None
        assert len(portfolio_risk.asset_allocation) > 0
        assert portfolio_risk.risk_level in [level.value for level in RiskLevel]
    
    @pytest.mark.asyncio
    async def test_assess_market_risk(self, risk_service, mock_db_session, sample_portfolio):
        """Test market risk assessment"""
        # Mock market data
        with patch.object(risk_service, '_get_market_data') as mock_market_data:
            mock_market_data.return_value = {
                'volatility_index': 0.25,
                'correlation_matrix': [[1.0, 0.7], [0.7, 1.0]],
                'market_trend': 'bearish',
                'fear_greed_index': 30
            }
            
            market_risk = await risk_service.assess_market_risk(
                mock_db_session, sample_portfolio.id
            )
            
            # Assertions
            assert isinstance(market_risk, MarketRisk)
            assert market_risk.portfolio_id == str(sample_portfolio.id)
            assert 0 <= market_risk.market_beta <= 5  # Reasonable beta range
            assert 0 <= market_risk.correlation_risk <= 1
            assert market_risk.volatility_risk >= 0
            assert market_risk.trend_risk is not None
    
    @pytest.mark.asyncio
    async def test_assess_credit_risk(self, risk_service, mock_db_session, 
                                    sample_user, sample_portfolio):
        """Test credit risk assessment"""
        # Mock credit data
        with patch.object(risk_service, '_get_credit_data') as mock_credit_data:
            mock_credit_data.return_value = {
                'credit_score': 750,
                'debt_to_income': 0.3,
                'payment_history': 0.95,
                'credit_utilization': 0.25
            }
            
            credit_risk = await risk_service.assess_credit_risk(
                mock_db_session, sample_user.id, sample_portfolio.id
            )
            
            # Assertions
            assert isinstance(credit_risk, CreditRisk)
            assert credit_risk.user_id == str(sample_user.id)
            assert 0 <= credit_risk.default_probability <= 1
            assert credit_risk.credit_score > 0
            assert 0 <= credit_risk.debt_ratio <= 1
            assert credit_risk.risk_grade in ['A', 'B', 'C', 'D', 'F']
    
    @pytest.mark.asyncio
    async def test_assess_liquidity_risk(self, risk_service, mock_db_session, 
                                       sample_portfolio, sample_assets):
        """Test liquidity risk assessment"""
        # Mock liquidity data
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_assets
        
        with patch.object(risk_service, '_get_liquidity_data') as mock_liquidity_data:
            mock_liquidity_data.return_value = {
                'BTC': {'daily_volume': 1000000000, 'bid_ask_spread': 0.001},
                'ETH': {'daily_volume': 500000000, 'bid_ask_spread': 0.002}
            }
            
            liquidity_risk = await risk_service.assess_liquidity_risk(
                mock_db_session, sample_portfolio.id
            )
            
            # Assertions
            assert isinstance(liquidity_risk, LiquidityRisk)
            assert liquidity_risk.portfolio_id == str(sample_portfolio.id)
            assert 0 <= liquidity_risk.liquidity_score <= 1
            assert liquidity_risk.time_to_liquidate > 0
            assert 0 <= liquidity_risk.price_impact <= 1
            assert len(liquidity_risk.asset_liquidity) > 0
    
    @pytest.mark.asyncio
    async def test_assess_operational_risk(self, risk_service, mock_db_session, 
                                         sample_user, sample_transactions):
        """Test operational risk assessment"""
        # Mock transaction data
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_transactions
        
        operational_risk = await risk_service.assess_operational_risk(
            mock_db_session, sample_user.id
        )
        
        # Assertions
        assert isinstance(operational_risk, OperationalRisk)
        assert operational_risk.user_id == str(sample_user.id)
        assert 0 <= operational_risk.system_risk <= 1
        assert 0 <= operational_risk.fraud_risk <= 1
        assert 0 <= operational_risk.compliance_risk <= 1
        assert operational_risk.operational_score >= 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_assessment(self, risk_service, mock_db_session, 
                                               sample_user, sample_portfolio):
        """Test comprehensive risk assessment"""
        # Mock all sub-assessments
        with patch.object(risk_service, 'calculate_portfolio_risk') as mock_portfolio_risk, \
             patch.object(risk_service, 'assess_market_risk') as mock_market_risk, \
             patch.object(risk_service, 'assess_credit_risk') as mock_credit_risk, \
             patch.object(risk_service, 'assess_liquidity_risk') as mock_liquidity_risk, \
             patch.object(risk_service, 'assess_operational_risk') as mock_operational_risk:
            
            # Setup mock returns
            mock_portfolio_risk.return_value = PortfolioRisk(
                portfolio_id=str(sample_portfolio.id),
                total_value=Decimal("100000"),
                volatility=0.2,
                var_95=0.05,
                expected_shortfall=0.08,
                sharpe_ratio=1.2,
                max_drawdown=0.15,
                beta=1.1,
                asset_allocation={'BTC': 0.6, 'ETH': 0.4},
                concentration_risk=0.3,
                risk_level=RiskLevel.MEDIUM,
                calculated_at=datetime.utcnow()
            )
            
            mock_market_risk.return_value = MarketRisk(
                portfolio_id=str(sample_portfolio.id),
                market_beta=1.1,
                correlation_risk=0.3,
                volatility_risk=0.25,
                trend_risk=0.4,
                sector_risk=0.2,
                calculated_at=datetime.utcnow()
            )
            
            mock_credit_risk.return_value = CreditRisk(
                user_id=str(sample_user.id),
                credit_score=750,
                default_probability=0.02,
                debt_ratio=0.3,
                payment_history_score=0.95,
                risk_grade='B',
                calculated_at=datetime.utcnow()
            )
            
            mock_liquidity_risk.return_value = LiquidityRisk(
                portfolio_id=str(sample_portfolio.id),
                liquidity_score=0.8,
                time_to_liquidate=2.5,
                price_impact=0.03,
                asset_liquidity={'BTC': 0.9, 'ETH': 0.85},
                calculated_at=datetime.utcnow()
            )
            
            mock_operational_risk.return_value = OperationalRisk(
                user_id=str(sample_user.id),
                system_risk=0.1,
                fraud_risk=0.05,
                compliance_risk=0.08,
                operational_score=0.92,
                calculated_at=datetime.utcnow()
            )
            
            # Perform comprehensive assessment
            assessment = await risk_service.perform_comprehensive_assessment(
                mock_db_session, sample_user.id, sample_portfolio.id
            )
            
            # Assertions
            assert isinstance(assessment, RiskAssessment)
            assert assessment.user_id == str(sample_user.id)
            assert assessment.portfolio_id == str(sample_portfolio.id)
            assert assessment.overall_risk_score >= 0
            assert assessment.risk_level in [level.value for level in RiskLevel]
            assert assessment.portfolio_risk is not None
            assert assessment.market_risk is not None
            assert assessment.credit_risk is not None
            assert assessment.liquidity_risk is not None
            assert assessment.operational_risk is not None
            assert len(assessment.risk_factors) > 0
            assert len(assessment.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_risk_limits(self, risk_service, mock_db_session, 
                                     sample_user, sample_portfolio):
        """Test risk limit monitoring"""
        # Setup risk limits
        risk_limits = {
            'max_portfolio_var': 0.1,
            'max_concentration': 0.4,
            'max_leverage': 2.0,
            'min_liquidity_score': 0.7
        }
        
        # Mock portfolio risk with limit breach
        portfolio_risk = PortfolioRisk(
            portfolio_id=str(sample_portfolio.id),
            total_value=Decimal("100000"),
            volatility=0.3,
            var_95=0.12,  # Exceeds limit
            expected_shortfall=0.18,
            sharpe_ratio=0.8,
            max_drawdown=0.25,
            beta=1.5,
            asset_allocation={'BTC': 0.8, 'ETH': 0.2},  # High concentration
            concentration_risk=0.5,  # Exceeds limit
            risk_level=RiskLevel.HIGH,
            calculated_at=datetime.utcnow()
        )
        
        with patch.object(risk_service, 'calculate_portfolio_risk') as mock_portfolio_risk:
            mock_portfolio_risk.return_value = portfolio_risk
            
            violations = await risk_service.monitor_risk_limits(
                mock_db_session, sample_portfolio.id, risk_limits
            )
            
            # Assertions
            assert len(violations) > 0
            assert any('VaR' in violation['description'] for violation in violations)
            assert any('concentration' in violation['description'].lower() for violation in violations)
    
    @pytest.mark.asyncio
    async def test_calculate_stress_test_scenarios(self, risk_service, mock_db_session, 
                                                 sample_portfolio, sample_assets):
        """Test stress testing scenarios"""
        # Mock portfolio data
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_assets
        
        scenarios = {
            'market_crash': {'market_shock': -0.3, 'volatility_shock': 2.0},
            'liquidity_crisis': {'liquidity_shock': -0.5, 'spread_shock': 3.0},
            'interest_rate_shock': {'rate_shock': 0.02, 'duration_shock': 1.5}
        }
        
        stress_results = await risk_service.calculate_stress_test_scenarios(
            mock_db_session, sample_portfolio.id, scenarios
        )
        
        # Assertions
        assert len(stress_results) == len(scenarios)
        for scenario_name, result in stress_results.items():
            assert scenario_name in scenarios
            assert 'portfolio_value_change' in result
            assert 'var_change' in result
            assert 'risk_level_change' in result
            assert isinstance(result['portfolio_value_change'], (int, float, Decimal))
    
    @pytest.mark.asyncio
    async def test_generate_risk_report(self, risk_service, mock_db_session, 
                                      sample_user, sample_portfolio):
        """Test risk report generation"""
        # Mock comprehensive assessment
        mock_assessment = RiskAssessment(
            user_id=str(sample_user.id),
            portfolio_id=str(sample_portfolio.id),
            overall_risk_score=0.65,
            risk_level=RiskLevel.MEDIUM,
            portfolio_risk=None,  # Would be populated in real scenario
            market_risk=None,
            credit_risk=None,
            liquidity_risk=None,
            operational_risk=None,
            risk_factors=[
                RiskMetric(
                    metric_type=RiskType.MARKET,
                    name="Market Volatility",
                    value=0.25,
                    threshold=0.3,
                    status="normal"
                )
            ],
            recommendations=[
                "Consider diversifying portfolio to reduce concentration risk",
                "Monitor market volatility closely"
            ],
            assessed_at=datetime.utcnow()
        )
        
        with patch.object(risk_service, 'perform_comprehensive_assessment') as mock_assessment_func:
            mock_assessment_func.return_value = mock_assessment
            
            report = await risk_service.generate_risk_report(
                mock_db_session, sample_user.id, sample_portfolio.id
            )
            
            # Assertions
            assert 'executive_summary' in report
            assert 'risk_assessment' in report
            assert 'recommendations' in report
            assert 'risk_metrics' in report
            assert 'generated_at' in report
            assert report['risk_assessment']['overall_risk_score'] == 0.65
    
    def test_risk_level_classification(self, risk_service):
        """Test risk level classification logic"""
        # Test low risk
        low_risk_score = 0.2
        assert risk_service._classify_risk_level(low_risk_score) == RiskLevel.LOW
        
        # Test medium risk
        medium_risk_score = 0.5
        assert risk_service._classify_risk_level(medium_risk_score) == RiskLevel.MEDIUM
        
        # Test high risk
        high_risk_score = 0.8
        assert risk_service._classify_risk_level(high_risk_score) == RiskLevel.HIGH
        
        # Test critical risk
        critical_risk_score = 0.95
        assert risk_service._classify_risk_level(critical_risk_score) == RiskLevel.CRITICAL
    
    def test_calculate_portfolio_concentration(self, risk_service):
        """Test portfolio concentration calculation"""
        # Test balanced portfolio
        balanced_allocation = {'BTC': 0.3, 'ETH': 0.3, 'ADA': 0.2, 'DOT': 0.2}
        concentration = risk_service._calculate_concentration_risk(balanced_allocation)
        assert concentration < 0.5  # Should be relatively low
        
        # Test concentrated portfolio
        concentrated_allocation = {'BTC': 0.8, 'ETH': 0.2}
        concentration = risk_service._calculate_concentration_risk(concentrated_allocation)
        assert concentration > 0.5  # Should be high
    
    def test_calculate_sharpe_ratio(self, risk_service):
        """Test Sharpe ratio calculation"""
        returns = [0.1, 0.05, -0.02, 0.08, 0.12]
        volatility = 0.15
        risk_free_rate = 0.02
        
        sharpe_ratio = risk_service._calculate_sharpe_ratio(returns, volatility, risk_free_rate)
        
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio > 0  # Should be positive for profitable strategy
    
    def test_calculate_var(self, risk_service):
        """Test Value at Risk calculation"""
        returns = [-0.05, -0.02, 0.01, 0.03, -0.08, 0.02, -0.01, 0.04, -0.03, 0.01]
        confidence_level = 0.95
        
        var = risk_service._calculate_var(returns, confidence_level)
        
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative (loss)
        assert var >= min(returns)  # Should not exceed worst historical loss
    
    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring(self, risk_service, mock_db_session, 
                                           sample_portfolio):
        """Test real-time risk monitoring"""
        # Mock real-time data
        with patch.object(risk_service, '_get_real_time_market_data') as mock_real_time_data:
            mock_real_time_data.return_value = {
                'price_changes': {'BTC': -0.05, 'ETH': -0.03},
                'volume_changes': {'BTC': 0.2, 'ETH': 0.15},
                'volatility_changes': {'BTC': 0.1, 'ETH': 0.08}
            }
            
            monitoring_result = await risk_service.start_real_time_monitoring(
                mock_db_session, sample_portfolio.id
            )
            
            # Assertions
            assert 'monitoring_id' in monitoring_result
            assert 'status' in monitoring_result
            assert monitoring_result['status'] == 'active'
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, risk_service, mock_db_session, 
                                       sample_user, sample_portfolio):
        """Test risk alert generation"""
        # Create high-risk scenario
        high_risk_metrics = [
            RiskMetric(
                metric_type=RiskType.MARKET,
                name="Portfolio VaR",
                value=0.15,
                threshold=0.1,
                status="breach"
            ),
            RiskMetric(
                metric_type=RiskType.LIQUIDITY,
                name="Liquidity Score",
                value=0.5,
                threshold=0.7,
                status="breach"
            )
        ]
        
        alerts = await risk_service.generate_risk_alerts(
            mock_db_session, sample_user.id, sample_portfolio.id, high_risk_metrics
        )
        
        # Assertions
        assert len(alerts) > 0
        for alert in alerts:
            assert 'alert_id' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'recommended_action' in alert
    
    @pytest.mark.asyncio
    async def test_risk_mitigation_suggestions(self, risk_service, sample_portfolio):
        """Test risk mitigation suggestions"""
        # High concentration risk scenario
        portfolio_risk = PortfolioRisk(
            portfolio_id=str(sample_portfolio.id),
            total_value=Decimal("100000"),
            volatility=0.4,
            var_95=0.12,
            expected_shortfall=0.18,
            sharpe_ratio=0.6,
            max_drawdown=0.3,
            beta=1.8,
            asset_allocation={'BTC': 0.9, 'ETH': 0.1},
            concentration_risk=0.8,
            risk_level=RiskLevel.HIGH,
            calculated_at=datetime.utcnow()
        )
        
        suggestions = risk_service.generate_mitigation_suggestions(portfolio_risk)
        
        # Assertions
        assert len(suggestions) > 0
        assert any('diversif' in suggestion.lower() for suggestion in suggestions)
        assert any('concentration' in suggestion.lower() for suggestion in suggestions)


class TestRiskManagementAPI:
    """Test suite for Risk Management API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}
    
    def test_get_risk_assessment_endpoint(self, client, auth_headers):
        """Test GET /api/v1/risk/assessment/{user_id} endpoint"""
        user_id = str(uuid4())
        
        with patch('services.risk.enhanced_risk_management_service.EnhancedRiskManagementService') as mock_service:
            mock_service.return_value.perform_comprehensive_assessment.return_value = {
                'user_id': user_id,
                'overall_risk_score': 0.65,
                'risk_level': 'medium'
            }
            
            response = client.get(
                f"/api/v1/risk/assessment/{user_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['user_id'] == user_id
            assert 'overall_risk_score' in data
    
    def test_start_risk_monitoring_endpoint(self, client, auth_headers):
        """Test POST /api/v1/risk/monitor endpoint"""
        portfolio_id = str(uuid4())
        
        with patch('services.risk.enhanced_risk_management_service.EnhancedRiskManagementService') as mock_service:
            mock_service.return_value.start_real_time_monitoring.return_value = {
                'monitoring_id': str(uuid4()),
                'status': 'active'
            }
            
            response = client.post(
                "/api/v1/risk/monitor",
                json={"portfolio_id": portfolio_id},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'monitoring_id' in data
            assert data['status'] == 'active'
    
    def test_get_risk_alerts_endpoint(self, client, auth_headers):
        """Test GET /api/v1/risk/alerts endpoint"""
        with patch('services.risk.enhanced_risk_management_service.EnhancedRiskManagementService') as mock_service:
            mock_service.return_value.get_active_alerts.return_value = [
                {
                    'alert_id': str(uuid4()),
                    'severity': 'high',
                    'message': 'Portfolio VaR exceeded threshold'
                }
            ]
            
            response = client.get("/api/v1/risk/alerts", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            if data:
                assert 'alert_id' in data[0]
                assert 'severity' in data[0]
    
    def test_generate_risk_report_endpoint(self, client, auth_headers):
        """Test POST /api/v1/risk/report endpoint"""
        user_id = str(uuid4())
        portfolio_id = str(uuid4())
        
        with patch('services.risk.enhanced_risk_management_service.EnhancedRiskManagementService') as mock_service:
            mock_service.return_value.generate_risk_report.return_value = {
                'executive_summary': 'Portfolio shows medium risk profile',
                'risk_assessment': {'overall_risk_score': 0.65},
                'recommendations': ['Diversify holdings']
            }
            
            response = client.post(
                "/api/v1/risk/report",
                json={"user_id": user_id, "portfolio_id": portfolio_id},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'executive_summary' in data
            assert 'risk_assessment' in data
            assert 'recommendations' in data


class TestRiskCalculations:
    """Test suite for risk calculation utilities"""
    
    def test_portfolio_volatility_calculation(self):
        """Test portfolio volatility calculation"""
        # Sample returns data
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03]
        
        # Calculate volatility (standard deviation)
        import numpy as np
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation"""
        import numpy as np
        
        # Sample price data for two assets
        asset1_returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        asset2_returns = [0.01, -0.02, 0.02, -0.01, 0.02]
        
        correlation = np.corrcoef(asset1_returns, asset2_returns)[0, 1]
        
        assert -1 <= correlation <= 1
        assert isinstance(correlation, float)
    
    def test_beta_calculation(self):
        """Test beta calculation"""
        import numpy as np
        
        # Sample returns
        portfolio_returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        market_returns = [0.015, -0.008, 0.025, -0.015, 0.008]
        
        # Calculate beta
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance if market_variance != 0 else 1.0
        
        assert isinstance(beta, float)
        assert beta > 0  # Assuming positive correlation with market
    
    def test_maximum_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        import numpy as np
        
        # Sample portfolio values
        portfolio_values = [100, 105, 98, 102, 95, 110, 108]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (np.array(portfolio_values) - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        assert max_drawdown <= 0  # Should be negative or zero
        assert isinstance(max_drawdown, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

