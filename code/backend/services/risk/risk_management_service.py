"""
Comprehensive Risk Management Service for Fluxion Backend
Implements advanced risk assessment, monitoring, and management for financial services
including market risk, credit risk, operational risk, and regulatory compliance.
"""

import asyncio
import json
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
import uuid
import statistics

from config.settings import settings
from services.security.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class RiskType(Enum):
    """Types of financial risks"""
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CURRENCY_RISK = "currency_risk"
    INTEREST_RATE_RISK = "interest_rate_risk"
    REGULATORY_RISK = "regulatory_risk"
    COUNTERPARTY_RISK = "counterparty_risk"


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Risk alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class RiskMetric:
    """Individual risk metric"""
    metric_id: str
    name: str
    risk_type: RiskType
    value: Decimal
    threshold: Decimal
    limit: Decimal
    unit: str
    confidence_level: Decimal
    calculation_method: str
    last_calculated: datetime
    historical_values: List[Tuple[datetime, Decimal]]


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    assessment_id: str
    entity_id: str
    entity_type: str  # portfolio, user, transaction, etc.
    overall_risk_score: Decimal
    risk_level: RiskLevel
    risk_metrics: List[RiskMetric]
    risk_factors: List[str]
    recommendations: List[str]
    assessment_date: datetime
    valid_until: datetime
    assessor: str
    methodology: str
    confidence_score: Decimal


@dataclass
class RiskAlert:
    """Risk alert/warning"""
    alert_id: str
    entity_id: str
    entity_type: str
    risk_type: RiskType
    risk_level: RiskLevel
    title: str
    description: str
    threshold_breached: str
    current_value: Decimal
    threshold_value: Decimal
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    assigned_to: Optional[str]
    escalation_level: int
    metadata: Dict[str, Any]


@dataclass
class RiskLimit:
    """Risk limit configuration"""
    limit_id: str
    entity_id: str
    entity_type: str
    risk_type: RiskType
    metric_name: str
    limit_value: Decimal
    warning_threshold: Decimal
    currency: str
    time_period: str
    effective_date: datetime
    expiry_date: Optional[datetime]
    created_by: str
    approved_by: str
    metadata: Dict[str, Any]


@dataclass
class StressTestScenario:
    """Stress testing scenario"""
    scenario_id: str
    name: str
    description: str
    scenario_type: str
    parameters: Dict[str, Any]
    market_shocks: Dict[str, Decimal]
    expected_impact: Dict[str, Decimal]
    probability: Decimal
    created_at: datetime
    created_by: str


@dataclass
class StressTestResult:
    """Stress test results"""
    test_id: str
    scenario_id: str
    entity_id: str
    entity_type: str
    baseline_value: Decimal
    stressed_value: Decimal
    impact_amount: Decimal
    impact_percentage: Decimal
    risk_metrics_impact: Dict[str, Decimal]
    passed: bool
    test_date: datetime
    methodology: str


class RiskManagementService:
    """
    Comprehensive risk management service providing:
    - Risk assessment and scoring
    - Risk monitoring and alerting
    - Risk limit management
    - Stress testing and scenario analysis
    - Value at Risk (VaR) calculations
    - Expected Shortfall (ES) calculations
    - Concentration risk analysis
    - Regulatory risk reporting
    - Risk dashboard and analytics
    """
    
    def __init__(self):
        self.encryption_service = EncryptionService()
        
        # Risk management configuration
        self.confidence_levels = [Decimal('0.95'), Decimal('0.99'), Decimal('0.999')]
        self.var_calculation_methods = ['historical', 'parametric', 'monte_carlo']
        self.stress_test_frequency_days = 30
        
        # Risk thresholds and limits
        self.default_risk_limits = {
            RiskType.MARKET_RISK: {
                'var_95': Decimal('0.05'),  # 5% of portfolio value
                'var_99': Decimal('0.10'),  # 10% of portfolio value
                'concentration_single_asset': Decimal('0.20'),  # 20% max in single asset
                'concentration_sector': Decimal('0.30'),  # 30% max in single sector
                'leverage_ratio': Decimal('2.0')  # 2:1 max leverage
            },
            RiskType.CREDIT_RISK: {
                'single_counterparty': Decimal('0.15'),  # 15% max exposure
                'credit_rating_minimum': 'BBB-',
                'default_probability_max': Decimal('0.05')  # 5% max default probability
            },
            RiskType.LIQUIDITY_RISK: {
                'liquidity_ratio_min': Decimal('0.10'),  # 10% min liquid assets
                'funding_concentration_max': Decimal('0.25')  # 25% max from single source
            }
        }
        
        # Market volatility assumptions (annualized)
        self.market_volatilities = {
            'stocks': Decimal('0.20'),  # 20%
            'bonds': Decimal('0.05'),   # 5%
            'commodities': Decimal('0.30'),  # 30%
            'currencies': Decimal('0.15'),   # 15%
            'crypto': Decimal('0.80')   # 80%
        }
        
        # Correlation matrix (simplified)
        self.correlation_matrix = {
            ('stocks', 'bonds'): Decimal('-0.2'),
            ('stocks', 'commodities'): Decimal('0.3'),
            ('stocks', 'crypto'): Decimal('0.1'),
            ('bonds', 'commodities'): Decimal('-0.1'),
            ('bonds', 'crypto'): Decimal('0.0'),
            ('commodities', 'crypto'): Decimal('0.2')
        }
        
        # In-memory storage (in production, use database)
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        self.risk_alerts: Dict[str, RiskAlert] = {}
        self.risk_limits: Dict[str, List[RiskLimit]] = {}
        self.stress_test_scenarios: Dict[str, StressTestScenario] = {}
        self.stress_test_results: Dict[str, List[StressTestResult]] = {}
        self.risk_metrics_history: Dict[str, List[RiskMetric]] = {}
        
        # Initialize default scenarios
        self._initialize_stress_test_scenarios()
    
    def _initialize_stress_test_scenarios(self):
        """Initialize default stress test scenarios"""
        scenarios = [
            {
                'name': 'Market Crash 2008',
                'description': 'Severe market downturn similar to 2008 financial crisis',
                'scenario_type': 'historical',
                'market_shocks': {
                    'stocks': Decimal('-0.40'),  # 40% decline
                    'bonds': Decimal('0.10'),    # 10% increase (flight to quality)
                    'commodities': Decimal('-0.30'),  # 30% decline
                    'currencies': Decimal('0.05')     # 5% USD strengthening
                },
                'probability': Decimal('0.01')  # 1% annual probability
            },
            {
                'name': 'Interest Rate Shock',
                'description': 'Sudden 300 basis point increase in interest rates',
                'scenario_type': 'hypothetical',
                'market_shocks': {
                    'bonds': Decimal('-0.15'),   # 15% decline in bond prices
                    'stocks': Decimal('-0.10'),  # 10% decline in stocks
                    'currencies': Decimal('0.08') # 8% currency impact
                },
                'probability': Decimal('0.05')  # 5% annual probability
            },
            {
                'name': 'Liquidity Crisis',
                'description': 'Severe liquidity shortage in financial markets',
                'scenario_type': 'hypothetical',
                'market_shocks': {
                    'stocks': Decimal('-0.25'),
                    'bonds': Decimal('-0.08'),
                    'commodities': Decimal('-0.20'),
                    'crypto': Decimal('-0.50')
                },
                'probability': Decimal('0.02')  # 2% annual probability
            }
        ]
        
        for scenario_data in scenarios:
            scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"
            scenario = StressTestScenario(
                scenario_id=scenario_id,
                name=scenario_data['name'],
                description=scenario_data['description'],
                scenario_type=scenario_data['scenario_type'],
                parameters={},
                market_shocks=scenario_data['market_shocks'],
                expected_impact={},
                probability=scenario_data['probability'],
                created_at=datetime.now(timezone.utc),
                created_by='system'
            )
            self.stress_test_scenarios[scenario_id] = scenario
    
    async def assess_portfolio_risk(self, portfolio_id: str, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment for a portfolio"""
        assessment_id = f"assessment_{uuid.uuid4().hex[:12]}"
        
        # Calculate various risk metrics
        risk_metrics = []
        
        # Market Risk Metrics
        var_95 = await self._calculate_var(portfolio_data, Decimal('0.95'))
        var_99 = await self._calculate_var(portfolio_data, Decimal('0.99'))
        expected_shortfall = await self._calculate_expected_shortfall(portfolio_data, Decimal('0.95'))
        
        risk_metrics.extend([
            RiskMetric(
                metric_id=f"var_95_{uuid.uuid4().hex[:8]}",
                name="Value at Risk (95%)",
                risk_type=RiskType.MARKET_RISK,
                value=var_95,
                threshold=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.03'),  # 3% threshold
                limit=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.05'),     # 5% limit
                unit="USD",
                confidence_level=Decimal('0.95'),
                calculation_method="historical_simulation",
                last_calculated=datetime.now(timezone.utc),
                historical_values=[]
            ),
            RiskMetric(
                metric_id=f"var_99_{uuid.uuid4().hex[:8]}",
                name="Value at Risk (99%)",
                risk_type=RiskType.MARKET_RISK,
                value=var_99,
                threshold=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.05'),  # 5% threshold
                limit=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.10'),     # 10% limit
                unit="USD",
                confidence_level=Decimal('0.99'),
                calculation_method="historical_simulation",
                last_calculated=datetime.now(timezone.utc),
                historical_values=[]
            ),
            RiskMetric(
                metric_id=f"es_95_{uuid.uuid4().hex[:8]}",
                name="Expected Shortfall (95%)",
                risk_type=RiskType.MARKET_RISK,
                value=expected_shortfall,
                threshold=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.07'),  # 7% threshold
                limit=portfolio_data.get('total_value', Decimal('0')) * Decimal('0.12'),     # 12% limit
                unit="USD",
                confidence_level=Decimal('0.95'),
                calculation_method="historical_simulation",
                last_calculated=datetime.now(timezone.utc),
                historical_values=[]
            )
        ])
        
        # Concentration Risk Metrics
        concentration_metrics = await self._calculate_concentration_risk(portfolio_data)
        risk_metrics.extend(concentration_metrics)
        
        # Liquidity Risk Metrics
        liquidity_metrics = await self._calculate_liquidity_risk(portfolio_data)
        risk_metrics.extend(liquidity_metrics)
        
        # Calculate overall risk score
        overall_risk_score = await self._calculate_overall_risk_score(risk_metrics)
        risk_level = self._determine_risk_level(overall_risk_score)
        
        # Generate risk factors and recommendations
        risk_factors = await self._identify_risk_factors(risk_metrics, portfolio_data)
        recommendations = await self._generate_risk_recommendations(risk_metrics, risk_factors)
        
        # Create risk assessment
        assessment = RiskAssessment(
            assessment_id=assessment_id,
            entity_id=portfolio_id,
            entity_type="portfolio",
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            risk_metrics=risk_metrics,
            risk_factors=risk_factors,
            recommendations=recommendations,
            assessment_date=datetime.now(timezone.utc),
            valid_until=datetime.now(timezone.utc) + timedelta(days=1),
            assessor="risk_management_service",
            methodology="comprehensive_quantitative_analysis",
            confidence_score=Decimal('0.85')
        )
        
        self.risk_assessments[assessment_id] = assessment
        
        # Check for risk limit breaches and create alerts
        await self._check_risk_limits(assessment)
        
        logger.info(f"Risk assessment completed for portfolio {portfolio_id}: {risk_level.value}")
        
        return {
            'assessment_id': assessment_id,
            'portfolio_id': portfolio_id,
            'overall_risk_score': str(overall_risk_score),
            'risk_level': risk_level.value,
            'var_95': str(var_95),
            'var_99': str(var_99),
            'expected_shortfall': str(expected_shortfall),
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'assessment_date': assessment.assessment_date.isoformat(),
            'valid_until': assessment.valid_until.isoformat(),
            'confidence_score': str(assessment.confidence_score)
        }
    
    async def run_stress_test(self, entity_id: str, entity_type: str, 
                            entity_data: Dict[str, Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Run stress test on portfolio or position"""
        if scenario_id:
            scenarios = [self.stress_test_scenarios[scenario_id]]
        else:
            scenarios = list(self.stress_test_scenarios.values())
        
        stress_test_results = []
        
        for scenario in scenarios:
            test_id = f"stress_test_{uuid.uuid4().hex[:12]}"
            
            # Calculate baseline value
            baseline_value = entity_data.get('total_value', Decimal('0'))
            
            # Apply stress scenario
            stressed_value = await self._apply_stress_scenario(entity_data, scenario)
            
            # Calculate impact
            impact_amount = stressed_value - baseline_value
            impact_percentage = (impact_amount / baseline_value * 100) if baseline_value > 0 else Decimal('0')
            
            # Calculate impact on risk metrics
            risk_metrics_impact = await self._calculate_stress_impact_on_metrics(entity_data, scenario)
            
            # Determine if test passed (based on risk tolerance)
            risk_tolerance = entity_data.get('risk_tolerance', 'medium')
            max_acceptable_loss = self._get_max_acceptable_loss(risk_tolerance)
            passed = abs(impact_percentage) <= max_acceptable_loss
            
            result = StressTestResult(
                test_id=test_id,
                scenario_id=scenario.scenario_id,
                entity_id=entity_id,
                entity_type=entity_type,
                baseline_value=baseline_value,
                stressed_value=stressed_value,
                impact_amount=impact_amount,
                impact_percentage=impact_percentage,
                risk_metrics_impact=risk_metrics_impact,
                passed=passed,
                test_date=datetime.now(timezone.utc),
                methodology="scenario_analysis"
            )
            
            stress_test_results.append(result)
            
            # Store result
            if entity_id not in self.stress_test_results:
                self.stress_test_results[entity_id] = []
            self.stress_test_results[entity_id].append(result)
        
        # Generate summary
        total_tests = len(stress_test_results)
        passed_tests = sum(1 for result in stress_test_results if result.passed)
        worst_case_loss = min(result.impact_percentage for result in stress_test_results)
        
        logger.info(f"Stress test completed for {entity_type} {entity_id}: {passed_tests}/{total_tests} passed")
        
        return {
            'entity_id': entity_id,
            'entity_type': entity_type,
            'total_scenarios_tested': total_tests,
            'scenarios_passed': passed_tests,
            'pass_rate': str((Decimal(passed_tests) / Decimal(total_tests) * 100).quantize(Decimal('0.01'))),
            'worst_case_loss_percentage': str(worst_case_loss),
            'test_date': datetime.now(timezone.utc).isoformat(),
            'detailed_results': [
                {
                    'scenario_name': self.stress_test_scenarios[result.scenario_id].name,
                    'baseline_value': str(result.baseline_value),
                    'stressed_value': str(result.stressed_value),
                    'impact_amount': str(result.impact_amount),
                    'impact_percentage': str(result.impact_percentage),
                    'passed': result.passed
                }
                for result in stress_test_results
            ]
        }
    
    async def get_risk_alerts(self, entity_id: Optional[str] = None, 
                            risk_type: Optional[RiskType] = None,
                            status: Optional[AlertStatus] = None) -> Dict[str, Any]:
        """Get risk alerts with optional filtering"""
        alerts = list(self.risk_alerts.values())
        
        # Apply filters
        if entity_id:
            alerts = [alert for alert in alerts if alert.entity_id == entity_id]
        
        if risk_type:
            alerts = [alert for alert in alerts if alert.risk_type == risk_type]
        
        if status:
            alerts = [alert for alert in alerts if alert.status == status]
        
        # Sort by creation date (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                'alert_id': alert.alert_id,
                'entity_id': alert.entity_id,
                'entity_type': alert.entity_type,
                'risk_type': alert.risk_type.value,
                'risk_level': alert.risk_level.value,
                'title': alert.title,
                'description': alert.description,
                'current_value': str(alert.current_value),
                'threshold_value': str(alert.threshold_value),
                'status': alert.status.value,
                'created_at': alert.created_at.isoformat(),
                'escalation_level': alert.escalation_level
            })
        
        return {
            'alerts': formatted_alerts,
            'total_alerts': len(formatted_alerts),
            'active_alerts': len([a for a in alerts if a.status == AlertStatus.ACTIVE]),
            'critical_alerts': len([a for a in alerts if a.risk_level == RiskLevel.CRITICAL])
        }
    
    async def acknowledge_risk_alert(self, alert_id: str, acknowledged_by: str) -> Dict[str, Any]:
        """Acknowledge a risk alert"""
        alert = self.risk_alerts.get(alert_id)
        if not alert:
            raise ValueError("Alert not found")
        
        if alert.status != AlertStatus.ACTIVE:
            raise ValueError(f"Alert is not active (current status: {alert.status.value})")
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.assigned_to = acknowledged_by
        
        logger.info(f"Risk alert acknowledged: {alert_id} by {acknowledged_by}")
        
        return {
            'alert_id': alert_id,
            'status': alert.status.value,
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': alert.acknowledged_at.isoformat()
        }
    
    async def set_risk_limit(self, entity_id: str, entity_type: str, risk_type: RiskType,
                           metric_name: str, limit_value: Decimal, warning_threshold: Decimal,
                           created_by: str, approved_by: str) -> Dict[str, Any]:
        """Set risk limit for entity"""
        limit_id = f"limit_{uuid.uuid4().hex[:12]}"
        
        risk_limit = RiskLimit(
            limit_id=limit_id,
            entity_id=entity_id,
            entity_type=entity_type,
            risk_type=risk_type,
            metric_name=metric_name,
            limit_value=limit_value,
            warning_threshold=warning_threshold,
            currency="USD",
            time_period="daily",
            effective_date=datetime.now(timezone.utc),
            expiry_date=None,
            created_by=created_by,
            approved_by=approved_by,
            metadata={}
        )
        
        if entity_id not in self.risk_limits:
            self.risk_limits[entity_id] = []
        self.risk_limits[entity_id].append(risk_limit)
        
        logger.info(f"Risk limit set: {metric_name} = {limit_value} for {entity_type} {entity_id}")
        
        return {
            'limit_id': limit_id,
            'entity_id': entity_id,
            'risk_type': risk_type.value,
            'metric_name': metric_name,
            'limit_value': str(limit_value),
            'warning_threshold': str(warning_threshold),
            'effective_date': risk_limit.effective_date.isoformat()
        }
    
    # Private calculation methods
    
    async def _calculate_var(self, portfolio_data: Dict[str, Any], confidence_level: Decimal) -> Decimal:
        """Calculate Value at Risk using historical simulation"""
        positions = portfolio_data.get('positions', [])
        total_value = portfolio_data.get('total_value', Decimal('0'))
        
        if not positions or total_value == 0:
            return Decimal('0')
        
        # Simulate historical returns (simplified)
        # In production, use actual historical price data
        portfolio_returns = []
        
        for i in range(252):  # One year of daily returns
            daily_return = Decimal('0')
            
            for position in positions:
                weight = position.get('weight', Decimal('0'))
                asset_type = position.get('asset_type', 'stocks')
                
                # Simulate daily return based on asset type volatility
                volatility = self.market_volatilities.get(asset_type, Decimal('0.20'))
                daily_volatility = volatility / Decimal('15.87')  # sqrt(252)
                
                # Simple random return simulation (in production, use historical data)
                simulated_return = daily_volatility * Decimal('0.5')  # Simplified
                daily_return += weight * simulated_return
            
            portfolio_returns.append(daily_return)
        
        # Sort returns and find VaR
        portfolio_returns.sort()
        var_index = int((1 - confidence_level) * len(portfolio_returns))
        var_return = portfolio_returns[var_index] if var_index < len(portfolio_returns) else portfolio_returns[0]
        
        # Convert to dollar amount
        var_amount = abs(var_return * total_value)
        
        return var_amount
    
    async def _calculate_expected_shortfall(self, portfolio_data: Dict[str, Any], confidence_level: Decimal) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)"""
        positions = portfolio_data.get('positions', [])
        total_value = portfolio_data.get('total_value', Decimal('0'))
        
        if not positions or total_value == 0:
            return Decimal('0')
        
        # Simulate returns (same as VaR calculation)
        portfolio_returns = []
        
        for i in range(252):
            daily_return = Decimal('0')
            
            for position in positions:
                weight = position.get('weight', Decimal('0'))
                asset_type = position.get('asset_type', 'stocks')
                volatility = self.market_volatilities.get(asset_type, Decimal('0.20'))
                daily_volatility = volatility / Decimal('15.87')
                simulated_return = daily_volatility * Decimal('0.5')
                daily_return += weight * simulated_return
            
            portfolio_returns.append(daily_return)
        
        # Sort returns and calculate ES
        portfolio_returns.sort()
        var_index = int((1 - confidence_level) * len(portfolio_returns))
        
        # Expected Shortfall is the average of returns worse than VaR
        tail_returns = portfolio_returns[:var_index] if var_index > 0 else [portfolio_returns[0]]
        expected_shortfall_return = sum(tail_returns) / len(tail_returns) if tail_returns else Decimal('0')
        
        # Convert to dollar amount
        es_amount = abs(expected_shortfall_return * total_value)
        
        return es_amount
    
    async def _calculate_concentration_risk(self, portfolio_data: Dict[str, Any]) -> List[RiskMetric]:
        """Calculate concentration risk metrics"""
        positions = portfolio_data.get('positions', [])
        metrics = []
        
        if not positions:
            return metrics
        
        # Single asset concentration
        max_position_weight = max(position.get('weight', Decimal('0')) for position in positions)
        
        metrics.append(RiskMetric(
            metric_id=f"concentration_single_{uuid.uuid4().hex[:8]}",
            name="Single Asset Concentration",
            risk_type=RiskType.CONCENTRATION_RISK,
            value=max_position_weight,
            threshold=Decimal('0.15'),  # 15% warning threshold
            limit=Decimal('0.20'),      # 20% limit
            unit="percentage",
            confidence_level=Decimal('1.0'),
            calculation_method="position_weight_analysis",
            last_calculated=datetime.now(timezone.utc),
            historical_values=[]
        ))
        
        # Sector concentration (simplified)
        sector_weights = {}
        for position in positions:
            sector = position.get('sector', 'Unknown')
            weight = position.get('weight', Decimal('0'))
            sector_weights[sector] = sector_weights.get(sector, Decimal('0')) + weight
        
        max_sector_weight = max(sector_weights.values()) if sector_weights else Decimal('0')
        
        metrics.append(RiskMetric(
            metric_id=f"concentration_sector_{uuid.uuid4().hex[:8]}",
            name="Sector Concentration",
            risk_type=RiskType.CONCENTRATION_RISK,
            value=max_sector_weight,
            threshold=Decimal('0.25'),  # 25% warning threshold
            limit=Decimal('0.30'),      # 30% limit
            unit="percentage",
            confidence_level=Decimal('1.0'),
            calculation_method="sector_weight_analysis",
            last_calculated=datetime.now(timezone.utc),
            historical_values=[]
        ))
        
        return metrics
    
    async def _calculate_liquidity_risk(self, portfolio_data: Dict[str, Any]) -> List[RiskMetric]:
        """Calculate liquidity risk metrics"""
        positions = portfolio_data.get('positions', [])
        total_value = portfolio_data.get('total_value', Decimal('0'))
        cash_balance = portfolio_data.get('cash_balance', Decimal('0'))
        
        metrics = []
        
        # Liquidity ratio
        liquid_assets = cash_balance
        for position in positions:
            asset_type = position.get('asset_type', 'stocks')
            market_value = position.get('market_value', Decimal('0'))
            
            # Assign liquidity scores based on asset type
            liquidity_scores = {
                'cash': Decimal('1.0'),
                'stocks': Decimal('0.9'),
                'bonds': Decimal('0.8'),
                'etf': Decimal('0.9'),
                'mutual_fund': Decimal('0.7'),
                'real_estate': Decimal('0.3'),
                'crypto': Decimal('0.6')
            }
            
            liquidity_score = liquidity_scores.get(asset_type, Decimal('0.5'))
            liquid_assets += market_value * liquidity_score
        
        liquidity_ratio = liquid_assets / total_value if total_value > 0 else Decimal('0')
        
        metrics.append(RiskMetric(
            metric_id=f"liquidity_ratio_{uuid.uuid4().hex[:8]}",
            name="Liquidity Ratio",
            risk_type=RiskType.LIQUIDITY_RISK,
            value=liquidity_ratio,
            threshold=Decimal('0.15'),  # 15% warning threshold
            limit=Decimal('0.10'),      # 10% minimum limit
            unit="percentage",
            confidence_level=Decimal('1.0'),
            calculation_method="weighted_liquidity_analysis",
            last_calculated=datetime.now(timezone.utc),
            historical_values=[]
        ))
        
        return metrics
    
    async def _calculate_overall_risk_score(self, risk_metrics: List[RiskMetric]) -> Decimal:
        """Calculate overall risk score from individual metrics"""
        if not risk_metrics:
            return Decimal('0')
        
        total_score = Decimal('0')
        total_weight = Decimal('0')
        
        # Weight different risk types
        risk_weights = {
            RiskType.MARKET_RISK: Decimal('0.4'),
            RiskType.CONCENTRATION_RISK: Decimal('0.3'),
            RiskType.LIQUIDITY_RISK: Decimal('0.2'),
            RiskType.CREDIT_RISK: Decimal('0.1')
        }
        
        for metric in risk_metrics:
            weight = risk_weights.get(metric.risk_type, Decimal('0.1'))
            
            # Normalize metric value to 0-10 scale
            if metric.limit > 0:
                normalized_score = min(metric.value / metric.limit * 10, Decimal('10'))
            else:
                normalized_score = Decimal('5')  # Default moderate score
            
            total_score += normalized_score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else Decimal('5')
        return overall_score.quantize(Decimal('0.01'))
    
    def _determine_risk_level(self, risk_score: Decimal) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _identify_risk_factors(self, risk_metrics: List[RiskMetric], 
                                   portfolio_data: Dict[str, Any]) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []
        
        for metric in risk_metrics:
            if metric.value > metric.threshold:
                if metric.risk_type == RiskType.MARKET_RISK:
                    risk_factors.append(f"High market risk: {metric.name} exceeds threshold")
                elif metric.risk_type == RiskType.CONCENTRATION_RISK:
                    risk_factors.append(f"Concentration risk: {metric.name} above recommended level")
                elif metric.risk_type == RiskType.LIQUIDITY_RISK:
                    risk_factors.append(f"Liquidity concern: {metric.name} below minimum threshold")
        
        # Additional risk factors based on portfolio characteristics
        positions = portfolio_data.get('positions', [])
        if len(positions) < 5:
            risk_factors.append("Portfolio under-diversified with fewer than 5 positions")
        
        return risk_factors
    
    async def _generate_risk_recommendations(self, risk_metrics: List[RiskMetric], 
                                           risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        for metric in risk_metrics:
            if metric.value > metric.limit:
                if metric.risk_type == RiskType.CONCENTRATION_RISK:
                    recommendations.append(f"Reduce {metric.name} by diversifying holdings")
                elif metric.risk_type == RiskType.MARKET_RISK:
                    recommendations.append(f"Consider hedging strategies to reduce {metric.name}")
                elif metric.risk_type == RiskType.LIQUIDITY_RISK:
                    recommendations.append(f"Increase liquid assets to improve {metric.name}")
        
        # General recommendations
        if len(risk_factors) > 3:
            recommendations.append("Consider comprehensive portfolio rebalancing")
        
        if not recommendations:
            recommendations.append("Portfolio risk levels are within acceptable ranges")
        
        return recommendations
    
    async def _check_risk_limits(self, assessment: RiskAssessment):
        """Check risk limits and create alerts if breached"""
        entity_limits = self.risk_limits.get(assessment.entity_id, [])
        
        for limit in entity_limits:
            # Find corresponding metric
            matching_metric = None
            for metric in assessment.risk_metrics:
                if (metric.risk_type == limit.risk_type and 
                    metric.name.lower().replace(' ', '_') == limit.metric_name.lower()):
                    matching_metric = metric
                    break
            
            if matching_metric:
                # Check if limit is breached
                if matching_metric.value > limit.limit_value:
                    await self._create_risk_alert(
                        assessment.entity_id,
                        assessment.entity_type,
                        limit.risk_type,
                        RiskLevel.CRITICAL,
                        f"Risk Limit Breached: {limit.metric_name}",
                        f"{limit.metric_name} value {matching_metric.value} exceeds limit {limit.limit_value}",
                        matching_metric.value,
                        limit.limit_value
                    )
                elif matching_metric.value > limit.warning_threshold:
                    await self._create_risk_alert(
                        assessment.entity_id,
                        assessment.entity_type,
                        limit.risk_type,
                        RiskLevel.HIGH,
                        f"Risk Warning: {limit.metric_name}",
                        f"{limit.metric_name} value {matching_metric.value} exceeds warning threshold {limit.warning_threshold}",
                        matching_metric.value,
                        limit.warning_threshold
                    )
    
    async def _create_risk_alert(self, entity_id: str, entity_type: str, risk_type: RiskType,
                               risk_level: RiskLevel, title: str, description: str,
                               current_value: Decimal, threshold_value: Decimal):
        """Create a risk alert"""
        alert_id = f"alert_{uuid.uuid4().hex[:12]}"
        
        alert = RiskAlert(
            alert_id=alert_id,
            entity_id=entity_id,
            entity_type=entity_type,
            risk_type=risk_type,
            risk_level=risk_level,
            title=title,
            description=description,
            threshold_breached=f"{current_value} > {threshold_value}",
            current_value=current_value,
            threshold_value=threshold_value,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            acknowledged_at=None,
            resolved_at=None,
            assigned_to=None,
            escalation_level=1 if risk_level == RiskLevel.HIGH else 2,
            metadata={}
        )
        
        self.risk_alerts[alert_id] = alert
        
        logger.warning(f"Risk alert created: {title} for {entity_type} {entity_id}")
    
    async def _apply_stress_scenario(self, entity_data: Dict[str, Any], 
                                   scenario: StressTestScenario) -> Decimal:
        """Apply stress scenario to calculate stressed value"""
        positions = entity_data.get('positions', [])
        cash_balance = entity_data.get('cash_balance', Decimal('0'))
        
        stressed_value = cash_balance  # Cash is unaffected
        
        for position in positions:
            market_value = position.get('market_value', Decimal('0'))
            asset_type = position.get('asset_type', 'stocks')
            
            # Apply shock based on asset type
            shock = scenario.market_shocks.get(asset_type, Decimal('0'))
            stressed_position_value = market_value * (Decimal('1') + shock)
            stressed_value += stressed_position_value
        
        return stressed_value
    
    async def _calculate_stress_impact_on_metrics(self, entity_data: Dict[str, Any],
                                                scenario: StressTestScenario) -> Dict[str, Decimal]:
        """Calculate impact of stress scenario on risk metrics"""
        # Simplified calculation - in production, recalculate all metrics
        impact = {}
        
        for asset_type, shock in scenario.market_shocks.items():
            volatility_impact = abs(shock) * Decimal('0.5')  # Simplified
            impact[f"{asset_type}_volatility_increase"] = volatility_impact
        
        return impact
    
    def _get_max_acceptable_loss(self, risk_tolerance: str) -> Decimal:
        """Get maximum acceptable loss percentage based on risk tolerance"""
        tolerance_limits = {
            'conservative': Decimal('5'),    # 5%
            'moderate': Decimal('10'),       # 10%
            'aggressive': Decimal('20'),     # 20%
            'very_aggressive': Decimal('30') # 30%
        }
        return tolerance_limits.get(risk_tolerance, Decimal('10'))
    
    def get_risk_management_statistics(self) -> Dict[str, Any]:
        """Get risk management service statistics"""
        alert_counts = {}
        assessment_counts = {}
        
        for alert in self.risk_alerts.values():
            alert_counts[alert.status.value] = alert_counts.get(alert.status.value, 0) + 1
        
        for assessment in self.risk_assessments.values():
            assessment_counts[assessment.risk_level.value] = assessment_counts.get(assessment.risk_level.value, 0) + 1
        
        return {
            'total_assessments': len(self.risk_assessments),
            'total_alerts': len(self.risk_alerts),
            'active_alerts': alert_counts.get('active', 0),
            'total_stress_tests': sum(len(results) for results in self.stress_test_results.values()),
            'assessment_distribution': assessment_counts,
            'alert_distribution': alert_counts,
            'stress_test_scenarios': len(self.stress_test_scenarios)
        }

