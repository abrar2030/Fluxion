"""
Comprehensive Analytics and Reporting Service for Fluxion Backend
Implements advanced analytics, business intelligence, reporting, and data insights
for financial services platform with real-time and historical analysis.
"""

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from services.security.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class AnalyticsType(Enum):
    """Types of analytics"""

    USER_ANALYTICS = "user_analytics"
    TRANSACTION_ANALYTICS = "transaction_analytics"
    PORTFOLIO_ANALYTICS = "portfolio_analytics"
    RISK_ANALYTICS = "risk_analytics"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    MARKET_ANALYTICS = "market_analytics"
    COMPLIANCE_ANALYTICS = "compliance_analytics"
    OPERATIONAL_ANALYTICS = "operational_analytics"


class ReportType(Enum):
    """Types of reports"""

    DASHBOARD = "dashboard"
    SUMMARY = "summary"
    DETAILED = "detailed"
    REGULATORY = "regulatory"
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    CUSTOM = "custom"


class TimeFrame(Enum):
    """Time frame for analytics"""

    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of metrics"""

    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    VOLATILITY = "volatility"


@dataclass
class AnalyticsMetric:
    """Individual analytics metric"""

    metric_id: str
    name: str
    analytics_type: AnalyticsType
    metric_type: MetricType
    value: Decimal
    previous_value: Optional[Decimal]
    change: Optional[Decimal]
    change_percentage: Optional[Decimal]
    unit: str
    time_frame: TimeFrame
    calculation_date: datetime
    metadata: Dict[str, Any]


@dataclass
class AnalyticsReport:
    """Analytics report"""

    report_id: str
    name: str
    report_type: ReportType
    analytics_type: AnalyticsType
    time_frame: TimeFrame
    start_date: datetime
    end_date: datetime
    metrics: List[AnalyticsMetric]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    generated_by: str
    generated_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class Dashboard:
    """Analytics dashboard"""

    dashboard_id: str
    name: str
    description: str
    user_id: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int
    last_updated: datetime
    created_at: datetime
    is_public: bool
    metadata: Dict[str, Any]


@dataclass
class KPI:
    """Key Performance Indicator"""

    kpi_id: str
    name: str
    description: str
    category: str
    current_value: Decimal
    target_value: Decimal
    threshold_warning: Decimal
    threshold_critical: Decimal
    unit: str
    trend: str  # up, down, stable
    last_updated: datetime
    historical_values: List[Tuple[datetime, Decimal]]


class AnalyticsService:
    """
    Comprehensive analytics service providing:
    - Real-time and historical analytics
    - Business intelligence and insights
    - Custom dashboard creation
    - KPI monitoring and tracking
    - Automated report generation
    - Data visualization support
    - Trend analysis and forecasting
    - Comparative analysis
    - Regulatory reporting
    - Performance benchmarking
    """

    def __init__(self):
        self.encryption_service = EncryptionService()

        # Analytics configuration
        self.default_time_frames = ["1D", "1W", "1M", "3M", "6M", "1Y"]
        self.cache_duration = timedelta(minutes=15)
        self.max_data_points = 1000

        # KPI thresholds and targets
        self.default_kpis = {
            "user_growth_rate": {
                "target": Decimal("10.0"),  # 10% monthly growth
                "warning": Decimal("5.0"),  # 5% warning threshold
                "critical": Decimal("0.0"),  # 0% critical threshold
            },
            "transaction_volume": {
                "target": Decimal("1000000.00"),  # $1M daily volume
                "warning": Decimal("500000.00"),  # $500K warning
                "critical": Decimal("100000.00"),  # $100K critical
            },
            "portfolio_performance": {
                "target": Decimal("8.0"),  # 8% annual return
                "warning": Decimal("4.0"),  # 4% warning
                "critical": Decimal("0.0"),  # 0% critical
            },
            "risk_score": {
                "target": Decimal("5.0"),  # Target risk score
                "warning": Decimal("7.0"),  # Warning threshold
                "critical": Decimal("9.0"),  # Critical threshold
            },
        }

        # In-memory storage (in production, use time-series database)
        self.analytics_cache: Dict[str, Dict[str, Any]] = {}
        self.reports: Dict[str, AnalyticsReport] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.kpis: Dict[str, KPI] = {}
        self.metrics_history: Dict[str, List[AnalyticsMetric]] = {}

        # Sample data for demonstration
        self._initialize_sample_data()

        # Initialize default KPIs
        self._initialize_default_kpis()

    def _initialize_sample_data(self):
        """Initialize sample analytics data"""
        # This would be populated from actual service data in production
        self.sample_data = {
            "users": {
                "total_users": 1250,
                "active_users_daily": 890,
                "active_users_monthly": 1100,
                "new_users_today": 15,
                "new_users_this_month": 180,
                "user_retention_rate": 0.85,
            },
            "transactions": {
                "total_transactions": 15420,
                "daily_volume": Decimal("2500000.00"),
                "monthly_volume": Decimal("75000000.00"),
                "average_transaction_size": Decimal("1620.00"),
                "transaction_success_rate": 0.987,
                "failed_transactions": 203,
            },
            "portfolios": {
                "total_portfolios": 980,
                "total_aum": Decimal("125000000.00"),  # Assets Under Management
                "average_portfolio_size": Decimal("127551.02"),
                "top_performing_portfolios": 45,
                "underperforming_portfolios": 12,
            },
            "risk": {
                "high_risk_portfolios": 23,
                "medium_risk_portfolios": 456,
                "low_risk_portfolios": 501,
                "risk_alerts_active": 8,
                "compliance_violations": 2,
            },
        }

    def _initialize_default_kpis(self):
        """Initialize default KPIs"""
        for kpi_name, config in self.default_kpis.items():
            kpi_id = f"kpi_{uuid.uuid4().hex[:8]}"

            # Generate sample current value
            current_value = config["target"] * Decimal("0.8")  # 80% of target

            kpi = KPI(
                kpi_id=kpi_id,
                name=kpi_name.replace("_", " ").title(),
                description=f"Key performance indicator for {kpi_name.replace('_', ' ')}",
                category=kpi_name.split("_")[0],
                current_value=current_value,
                target_value=config["target"],
                threshold_warning=config["warning"],
                threshold_critical=config["critical"],
                unit=(
                    "%"
                    if "rate" in kpi_name or "performance" in kpi_name
                    else "$" if "volume" in kpi_name else "score"
                ),
                trend="up",
                last_updated=datetime.now(timezone.utc),
                historical_values=[],
            )

            self.kpis[kpi_id] = kpi

    async def generate_analytics_report(
        self,
        analytics_type: AnalyticsType,
        report_type: ReportType,
        time_frame: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        generated_by: str = "system",
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        report_id = f"report_{uuid.uuid4().hex[:12]}"

        # Set default date range if not provided
        if not start_date or not end_date:
            end_date = datetime.now(timezone.utc)
            if time_frame == TimeFrame.DAILY:
                start_date = end_date - timedelta(days=1)
            elif time_frame == TimeFrame.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
            elif time_frame == TimeFrame.MONTHLY:
                start_date = end_date - timedelta(days=30)
            elif time_frame == TimeFrame.QUARTERLY:
                start_date = end_date - timedelta(days=90)
            elif time_frame == TimeFrame.YEARLY:
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=7)  # Default to weekly

        # Generate metrics based on analytics type
        metrics = await self._generate_metrics(
            analytics_type, time_frame, start_date, end_date, filters
        )

        # Generate charts
        charts = await self._generate_charts(analytics_type, metrics, time_frame)

        # Generate insights
        insights = await self._generate_insights(analytics_type, metrics)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            analytics_type, metrics, insights
        )

        # Create report
        report = AnalyticsReport(
            report_id=report_id,
            name=f"{analytics_type.value.replace('_', ' ').title()} Report",
            report_type=report_type,
            analytics_type=analytics_type,
            time_frame=time_frame,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            charts=charts,
            insights=insights,
            recommendations=recommendations,
            generated_by=generated_by,
            generated_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            metadata=filters or {},
        )

        self.reports[report_id] = report

        logger.info(f"Analytics report generated: {report_id} ({analytics_type.value})")

        return {
            "report_id": report_id,
            "name": report.name,
            "type": report_type.value,
            "analytics_type": analytics_type.value,
            "time_frame": time_frame.value,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics_count": len(metrics),
            "charts_count": len(charts),
            "insights_count": len(insights),
            "recommendations_count": len(recommendations),
            "generated_at": report.generated_at.isoformat(),
        }

    async def get_real_time_analytics(
        self, analytics_type: AnalyticsType, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get real-time analytics data"""
        cache_key = f"{analytics_type.value}_realtime"

        # Check cache
        if (
            cache_key in self.analytics_cache
            and datetime.now(timezone.utc)
            - self.analytics_cache[cache_key]["timestamp"]
            < self.cache_duration
        ):
            return self.analytics_cache[cache_key]["data"]

        # Generate real-time metrics
        real_time_data = {}

        if analytics_type == AnalyticsType.USER_ANALYTICS:
            real_time_data = {
                "active_users_now": self.sample_data["users"]["active_users_daily"],
                "new_registrations_today": self.sample_data["users"]["new_users_today"],
                "user_sessions_active": 234,
                "average_session_duration": "12m 34s",
                "bounce_rate": "15.2%",
                "conversion_rate": "3.8%",
            }

        elif analytics_type == AnalyticsType.TRANSACTION_ANALYTICS:
            real_time_data = {
                "transactions_per_minute": 12.5,
                "volume_per_minute": str(Decimal("45000.00")),
                "success_rate": "98.7%",
                "average_processing_time": "2.3s",
                "pending_transactions": 45,
                "failed_transactions_today": 12,
            }

        elif analytics_type == AnalyticsType.PORTFOLIO_ANALYTICS:
            real_time_data = {
                "total_aum": str(self.sample_data["portfolios"]["total_aum"]),
                "portfolios_rebalanced_today": 23,
                "top_performing_asset": "AAPL (+2.3%)",
                "worst_performing_asset": "TSLA (-1.8%)",
                "market_sentiment": "Bullish",
                "volatility_index": "18.5",
            }

        elif analytics_type == AnalyticsType.RISK_ANALYTICS:
            real_time_data = {
                "active_risk_alerts": self.sample_data["risk"]["risk_alerts_active"],
                "high_risk_portfolios": self.sample_data["risk"][
                    "high_risk_portfolios"
                ],
                "var_95_breach_count": 2,
                "stress_test_failures": 1,
                "compliance_score": "94.2%",
                "risk_adjusted_return": "8.7%",
            }

        # Cache the data
        self.analytics_cache[cache_key] = {
            "data": real_time_data,
            "timestamp": datetime.now(timezone.utc),
        }

        return real_time_data

    async def create_dashboard(
        self,
        name: str,
        description: str,
        user_id: str,
        widgets: List[Dict[str, Any]],
        layout: Dict[str, Any],
        is_public: bool = False,
    ) -> Dict[str, Any]:
        """Create custom analytics dashboard"""
        dashboard_id = f"dashboard_{uuid.uuid4().hex[:12]}"

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            user_id=user_id,
            widgets=widgets,
            layout=layout,
            refresh_interval=300,  # 5 minutes default
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            is_public=is_public,
            metadata={},
        )

        self.dashboards[dashboard_id] = dashboard

        logger.info(f"Dashboard created: {dashboard_id} by user {user_id}")

        return {
            "dashboard_id": dashboard_id,
            "name": name,
            "widgets_count": len(widgets),
            "created_at": dashboard.created_at.isoformat(),
            "is_public": is_public,
        }

    async def get_dashboard_data(
        self, dashboard_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get dashboard data with all widget information"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            raise ValueError("Dashboard not found")

        # Check access permissions
        if not dashboard.is_public and dashboard.user_id != user_id:
            raise ValueError("Unauthorized access to dashboard")

        # Get data for each widget
        widget_data = []
        for widget in dashboard.widgets:
            widget_info = await self._get_widget_data(widget)
            widget_data.append(widget_info)

        return {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "layout": dashboard.layout,
            "widgets": widget_data,
            "last_updated": dashboard.last_updated.isoformat(),
            "refresh_interval": dashboard.refresh_interval,
        }

    async def get_kpi_summary(self) -> Dict[str, Any]:
        """Get summary of all KPIs"""
        kpi_summary = {
            "total_kpis": len(self.kpis),
            "on_target": 0,
            "warning": 0,
            "critical": 0,
            "kpis": [],
        }

        for kpi in self.kpis.values():
            # Determine status
            if kpi.current_value >= kpi.target_value:
                status = "on_target"
                kpi_summary["on_target"] += 1
            elif kpi.current_value >= kpi.threshold_warning:
                status = "warning"
                kpi_summary["warning"] += 1
            else:
                status = "critical"
                kpi_summary["critical"] += 1

            # Calculate progress percentage
            progress = (
                (kpi.current_value / kpi.target_value * 100)
                if kpi.target_value > 0
                else Decimal("0")
            )

            kpi_summary["kpis"].append(
                {
                    "kpi_id": kpi.kpi_id,
                    "name": kpi.name,
                    "category": kpi.category,
                    "current_value": str(kpi.current_value),
                    "target_value": str(kpi.target_value),
                    "unit": kpi.unit,
                    "status": status,
                    "progress_percentage": str(progress.quantize(Decimal("0.1"))),
                    "trend": kpi.trend,
                    "last_updated": kpi.last_updated.isoformat(),
                }
            )

        return kpi_summary

    async def get_comparative_analysis(
        self,
        entity_type: str,
        entity_ids: List[str],
        metrics: List[str],
        time_frame: TimeFrame,
    ) -> Dict[str, Any]:
        """Get comparative analysis between entities"""
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"

        comparison_data = {}

        for entity_id in entity_ids:
            entity_metrics = {}

            # Generate sample comparative data
            for metric in metrics:
                if metric == "total_return":
                    entity_metrics[metric] = str(
                        Decimal("8.5") + Decimal(str(hash(entity_id) % 100)) / 10
                    )
                elif metric == "volatility":
                    entity_metrics[metric] = str(
                        Decimal("15.0") + Decimal(str(hash(entity_id) % 50)) / 10
                    )
                elif metric == "sharpe_ratio":
                    entity_metrics[metric] = str(
                        Decimal("1.2") + Decimal(str(hash(entity_id) % 30)) / 100
                    )
                elif metric == "max_drawdown":
                    entity_metrics[metric] = str(
                        Decimal("5.0") + Decimal(str(hash(entity_id) % 20)) / 10
                    )
                else:
                    entity_metrics[metric] = str(
                        Decimal("100.0") + Decimal(str(hash(entity_id) % 200))
                    )

            comparison_data[entity_id] = entity_metrics

        # Generate rankings
        rankings = {}
        for metric in metrics:
            metric_values = [
                (entity_id, Decimal(data[metric]))
                for entity_id, data in comparison_data.items()
            ]

            # Sort based on metric (higher is better for most metrics except volatility and drawdown)
            reverse_sort = metric not in ["volatility", "max_drawdown"]
            metric_values.sort(key=lambda x: x[1], reverse=reverse_sort)

            rankings[metric] = [
                {"entity_id": entity_id, "value": str(value), "rank": i + 1}
                for i, (entity_id, value) in enumerate(metric_values)
            ]

        return {
            "analysis_id": analysis_id,
            "entity_type": entity_type,
            "entities_compared": len(entity_ids),
            "metrics_analyzed": len(metrics),
            "time_frame": time_frame.value,
            "comparison_data": comparison_data,
            "rankings": rankings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Private helper methods

    async def _generate_metrics(
        self,
        analytics_type: AnalyticsType,
        time_frame: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        filters: Optional[Dict[str, Any]],
    ) -> List[AnalyticsMetric]:
        """Generate metrics for analytics report"""
        metrics = []

        if analytics_type == AnalyticsType.USER_ANALYTICS:
            metrics.extend(
                [
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Total Users",
                        analytics_type=analytics_type,
                        metric_type=MetricType.COUNT,
                        value=Decimal(str(self.sample_data["users"]["total_users"])),
                        previous_value=Decimal("1180"),
                        change=Decimal("70"),
                        change_percentage=Decimal("5.93"),
                        unit="users",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="User Retention Rate",
                        analytics_type=analytics_type,
                        metric_type=MetricType.PERCENTAGE,
                        value=Decimal(
                            str(self.sample_data["users"]["user_retention_rate"] * 100)
                        ),
                        previous_value=Decimal("82.5"),
                        change=Decimal("2.5"),
                        change_percentage=Decimal("3.03"),
                        unit="%",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                ]
            )

        elif analytics_type == AnalyticsType.TRANSACTION_ANALYTICS:
            metrics.extend(
                [
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Transaction Volume",
                        analytics_type=analytics_type,
                        metric_type=MetricType.SUM,
                        value=self.sample_data["transactions"]["daily_volume"],
                        previous_value=Decimal("2300000.00"),
                        change=Decimal("200000.00"),
                        change_percentage=Decimal("8.70"),
                        unit="USD",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Success Rate",
                        analytics_type=analytics_type,
                        metric_type=MetricType.PERCENTAGE,
                        value=Decimal(
                            str(
                                self.sample_data["transactions"][
                                    "transaction_success_rate"
                                ]
                                * 100
                            )
                        ),
                        previous_value=Decimal("98.2"),
                        change=Decimal("0.5"),
                        change_percentage=Decimal("0.51"),
                        unit="%",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                ]
            )

        elif analytics_type == AnalyticsType.PORTFOLIO_ANALYTICS:
            metrics.extend(
                [
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Assets Under Management",
                        analytics_type=analytics_type,
                        metric_type=MetricType.SUM,
                        value=self.sample_data["portfolios"]["total_aum"],
                        previous_value=Decimal("120000000.00"),
                        change=Decimal("5000000.00"),
                        change_percentage=Decimal("4.17"),
                        unit="USD",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Average Portfolio Performance",
                        analytics_type=analytics_type,
                        metric_type=MetricType.PERCENTAGE,
                        value=Decimal("8.5"),
                        previous_value=Decimal("7.8"),
                        change=Decimal("0.7"),
                        change_percentage=Decimal("8.97"),
                        unit="%",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                ]
            )

        elif analytics_type == AnalyticsType.RISK_ANALYTICS:
            metrics.extend(
                [
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="Average Risk Score",
                        analytics_type=analytics_type,
                        metric_type=MetricType.AVERAGE,
                        value=Decimal("5.2"),
                        previous_value=Decimal("5.8"),
                        change=Decimal("-0.6"),
                        change_percentage=Decimal("-10.34"),
                        unit="score",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                    AnalyticsMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name="High Risk Portfolios",
                        analytics_type=analytics_type,
                        metric_type=MetricType.COUNT,
                        value=Decimal(
                            str(self.sample_data["risk"]["high_risk_portfolios"])
                        ),
                        previous_value=Decimal("28"),
                        change=Decimal("-5"),
                        change_percentage=Decimal("-17.86"),
                        unit="portfolios",
                        time_frame=time_frame,
                        calculation_date=datetime.now(timezone.utc),
                        metadata={},
                    ),
                ]
            )

        return metrics

    async def _generate_charts(
        self,
        analytics_type: AnalyticsType,
        metrics: List[AnalyticsMetric],
        time_frame: TimeFrame,
    ) -> List[Dict[str, Any]]:
        """Generate chart configurations for metrics"""
        charts = []

        # Time series chart for main metrics
        time_series_data = []
        for i in range(30):  # Last 30 data points
            date = datetime.now(timezone.utc) - timedelta(days=29 - i)
            value = float(metrics[0].value) * (
                0.9 + (i % 10) * 0.02
            )  # Simulate variation
            time_series_data.append(
                {"date": date.isoformat(), "value": round(value, 2)}
            )

        charts.append(
            {
                "chart_id": f"chart_{uuid.uuid4().hex[:8]}",
                "type": "line",
                "title": f"{metrics[0].name} Trend",
                "data": time_series_data,
                "x_axis": "date",
                "y_axis": "value",
                "unit": metrics[0].unit,
            }
        )

        # Pie chart for distribution metrics
        if analytics_type == AnalyticsType.RISK_ANALYTICS:
            charts.append(
                {
                    "chart_id": f"chart_{uuid.uuid4().hex[:8]}",
                    "type": "pie",
                    "title": "Risk Distribution",
                    "data": [
                        {
                            "label": "Low Risk",
                            "value": self.sample_data["risk"]["low_risk_portfolios"],
                        },
                        {
                            "label": "Medium Risk",
                            "value": self.sample_data["risk"]["medium_risk_portfolios"],
                        },
                        {
                            "label": "High Risk",
                            "value": self.sample_data["risk"]["high_risk_portfolios"],
                        },
                    ],
                }
            )

        # Bar chart for comparative metrics
        if len(metrics) > 1:
            bar_data = []
            for metric in metrics[:5]:  # Top 5 metrics
                bar_data.append(
                    {
                        "label": metric.name,
                        "value": float(metric.value),
                        "change": float(metric.change) if metric.change else 0,
                    }
                )

            charts.append(
                {
                    "chart_id": f"chart_{uuid.uuid4().hex[:8]}",
                    "type": "bar",
                    "title": "Key Metrics Comparison",
                    "data": bar_data,
                }
            )

        return charts

    async def _generate_insights(
        self, analytics_type: AnalyticsType, metrics: List[AnalyticsMetric]
    ) -> List[str]:
        """Generate insights from metrics"""
        insights = []

        for metric in metrics:
            if metric.change_percentage and metric.change_percentage > 10:
                insights.append(
                    f"{metric.name} has increased significantly by {metric.change_percentage}%"
                )
            elif metric.change_percentage and metric.change_percentage < -10:
                insights.append(
                    f"{metric.name} has decreased significantly by {abs(metric.change_percentage)}%"
                )

        # Analytics type specific insights
        if analytics_type == AnalyticsType.USER_ANALYTICS:
            insights.append(
                "User engagement is trending upward with improved retention rates"
            )
            insights.append("New user acquisition is meeting monthly targets")

        elif analytics_type == AnalyticsType.TRANSACTION_ANALYTICS:
            insights.append("Transaction processing efficiency has improved")
            insights.append("Peak transaction hours are between 9 AM and 11 AM EST")

        elif analytics_type == AnalyticsType.PORTFOLIO_ANALYTICS:
            insights.append(
                "Portfolio diversification has improved across all risk categories"
            )
            insights.append(
                "Technology sector allocation is above average market exposure"
            )

        elif analytics_type == AnalyticsType.RISK_ANALYTICS:
            insights.append(
                "Overall portfolio risk has decreased due to better diversification"
            )
            insights.append(
                "Stress test results show improved resilience to market shocks"
            )

        return insights

    async def _generate_recommendations(
        self,
        analytics_type: AnalyticsType,
        metrics: List[AnalyticsMetric],
        insights: List[str],
    ) -> List[str]:
        """Generate recommendations based on metrics and insights"""
        recommendations = []

        # Metric-based recommendations
        for metric in metrics:
            if metric.change_percentage and metric.change_percentage < -5:
                recommendations.append(
                    f"Investigate factors causing decline in {metric.name}"
                )

        # Analytics type specific recommendations
        if analytics_type == AnalyticsType.USER_ANALYTICS:
            recommendations.append("Consider implementing user engagement campaigns")
            recommendations.append("Optimize onboarding process to improve retention")

        elif analytics_type == AnalyticsType.TRANSACTION_ANALYTICS:
            recommendations.append("Monitor transaction processing during peak hours")
            recommendations.append("Consider implementing dynamic fee structures")

        elif analytics_type == AnalyticsType.PORTFOLIO_ANALYTICS:
            recommendations.append(
                "Review asset allocation strategies for underperforming portfolios"
            )
            recommendations.append(
                "Consider rebalancing recommendations for concentrated positions"
            )

        elif analytics_type == AnalyticsType.RISK_ANALYTICS:
            recommendations.append("Maintain current risk management practices")
            recommendations.append("Consider stress testing with additional scenarios")

        return recommendations

    async def _get_widget_data(self, widget: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for a specific dashboard widget"""
        widget_type = widget.get("type", "metric")

        if widget_type == "metric":
            # Single metric widget
            return {
                "widget_id": widget.get("id"),
                "type": widget_type,
                "title": widget.get("title", "Metric"),
                "value": "1,250",
                "change": "+5.2%",
                "trend": "up",
            }

        elif widget_type == "chart":
            # Chart widget
            return {
                "widget_id": widget.get("id"),
                "type": widget_type,
                "title": widget.get("title", "Chart"),
                "chart_type": widget.get("chart_type", "line"),
                "data": [
                    {"x": "2024-01-01", "y": 100},
                    {"x": "2024-01-02", "y": 105},
                    {"x": "2024-01-03", "y": 103},
                    {"x": "2024-01-04", "y": 108},
                    {"x": "2024-01-05", "y": 112},
                ],
            }

        elif widget_type == "table":
            # Table widget
            return {
                "widget_id": widget.get("id"),
                "type": widget_type,
                "title": widget.get("title", "Table"),
                "headers": ["Name", "Value", "Change"],
                "rows": [
                    ["Metric 1", "1,250", "+5.2%"],
                    ["Metric 2", "890", "-2.1%"],
                    ["Metric 3", "2,340", "+8.7%"],
                ],
            }

        return widget

    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics service statistics"""
        return {
            "total_reports": len(self.reports),
            "total_dashboards": len(self.dashboards),
            "total_kpis": len(self.kpis),
            "cache_entries": len(self.analytics_cache),
            "metrics_tracked": sum(
                len(history) for history in self.metrics_history.values()
            ),
            "active_dashboards": len(
                [
                    d
                    for d in self.dashboards.values()
                    if datetime.now(timezone.utc) - d.last_updated < timedelta(days=7)
                ]
            ),
        }
