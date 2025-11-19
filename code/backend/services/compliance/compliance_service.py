"""
Compliance service for Fluxion backend
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from config.settings import settings
from models.compliance import (
    AMLCheck,
    AMLRiskLevel,
    AuditLog,
    AuditLogLevel,
    ComplianceAlert,
    ComplianceAlertStatus,
    ComplianceAlertType,
    KYCRecord,
    KYCStatus,
)
from models.risk import RiskLevel, RiskProfile
from models.transaction import Transaction, TransactionStatus, TransactionType
from models.user import User
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class ComplianceService:
    """Comprehensive compliance service"""

    def __init__(self):
        self.transaction_threshold_usd = Decimal("10000")  # $10,000 threshold
        self.daily_threshold_usd = Decimal("50000")  # $50,000 daily threshold
        self.suspicious_velocity_threshold = 10  # 10 transactions per hour
        self.high_risk_countries = [
            "AF",
            "IR",
            "KP",
            "SY",
        ]  # Example high-risk countries

    async def check_transaction_compliance(
        self, db: AsyncSession, transaction: Transaction, user: User
    ) -> Dict[str, Any]:
        """Comprehensive transaction compliance check"""
        try:
            compliance_result = {
                "compliant": True,
                "risk_score": 0.0,
                "alerts": [],
                "recommendations": [],
            }

            # Check transaction amount thresholds
            if (
                transaction.usd_value
                and transaction.usd_value >= self.transaction_threshold_usd
            ):
                compliance_result["risk_score"] += 20
                compliance_result["alerts"].append(
                    {
                        "type": "high_value_transaction",
                        "message": f"Transaction exceeds ${self.transaction_threshold_usd} threshold",
                    }
                )

            # Check daily transaction volume
            daily_volume = await self._get_user_daily_volume(db, user.id)
            if daily_volume >= self.daily_threshold_usd:
                compliance_result["risk_score"] += 30
                compliance_result["alerts"].append(
                    {
                        "type": "daily_threshold_breach",
                        "message": f"Daily transaction volume exceeds ${self.daily_threshold_usd}",
                    }
                )

            # Check transaction velocity
            hourly_count = await self._get_user_hourly_transaction_count(db, user.id)
            if hourly_count >= self.suspicious_velocity_threshold:
                compliance_result["risk_score"] += 25
                compliance_result["alerts"].append(
                    {
                        "type": "suspicious_velocity",
                        "message": f"High transaction velocity: {hourly_count} transactions in last hour",
                    }
                )

            # Check user risk profile
            user_risk = await self._assess_user_risk(db, user)
            compliance_result["risk_score"] += user_risk["score"]
            if user_risk["alerts"]:
                compliance_result["alerts"].extend(user_risk["alerts"])

            # Check KYC status
            kyc_check = await self._check_kyc_compliance(db, user)
            if not kyc_check["compliant"]:
                compliance_result["compliant"] = False
                compliance_result["risk_score"] += 50
                compliance_result["alerts"].extend(kyc_check["alerts"])

            # Check sanctions and watchlists
            sanctions_check = await self._check_sanctions(db, user, transaction)
            compliance_result["risk_score"] += sanctions_check["score"]
            if sanctions_check["alerts"]:
                compliance_result["alerts"].extend(sanctions_check["alerts"])

            # Determine overall compliance
            if compliance_result["risk_score"] >= 70:
                compliance_result["compliant"] = False
                compliance_result["recommendations"].append(
                    "Transaction requires manual review"
                )
            elif compliance_result["risk_score"] >= 50:
                compliance_result["recommendations"].append(
                    "Enhanced monitoring required"
                )

            # Create compliance alert if necessary
            if (
                not compliance_result["compliant"]
                or compliance_result["risk_score"] >= 50
            ):
                await self._create_compliance_alert(
                    db, user.id, transaction.id, compliance_result
                )

            logger.info(
                f"Compliance check completed for transaction {transaction.id}: "
                f"Score {compliance_result['risk_score']}, "
                f"Compliant: {compliance_result['compliant']}"
            )

            return compliance_result

        except Exception as e:
            logger.error(
                f"Compliance check failed for transaction {transaction.id}: {str(e)}"
            )
            raise

    async def monitor_user_activity(
        self, db: AsyncSession, user_id: UUID, activity_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Monitor user activity for suspicious patterns"""
        try:
            monitoring_result = {
                "risk_score": 0.0,
                "patterns": [],
                "recommendations": [],
            }

            # Get recent transactions
            cutoff_time = datetime.utcnow() - timedelta(hours=activity_window_hours)

            result = await db.execute(
                select(Transaction)
                .where(
                    and_(
                        Transaction.user_id == user_id,
                        Transaction.created_at >= cutoff_time,
                        Transaction.status == TransactionStatus.CONFIRMED,
                    )
                )
                .order_by(Transaction.created_at.desc())
            )
            transactions = result.scalars().all()

            if not transactions:
                return monitoring_result

            # Analyze transaction patterns
            patterns = await self._analyze_transaction_patterns(transactions)
            monitoring_result["patterns"] = patterns

            # Calculate risk score based on patterns
            for pattern in patterns:
                monitoring_result["risk_score"] += pattern.get("risk_score", 0)

            # Check for round number bias
            round_number_count = sum(
                1 for tx in transactions if tx.amount and float(tx.amount) % 1000 == 0
            )
            if round_number_count > len(transactions) * 0.5:
                monitoring_result["risk_score"] += 15
                monitoring_result["patterns"].append(
                    {
                        "type": "round_number_bias",
                        "description": "High frequency of round number transactions",
                        "risk_score": 15,
                    }
                )

            # Check for structuring (amounts just below reporting thresholds)
            structuring_count = sum(
                1
                for tx in transactions
                if tx.usd_value and 9000 <= tx.usd_value <= 9999
            )
            if structuring_count >= 3:
                monitoring_result["risk_score"] += 40
                monitoring_result["patterns"].append(
                    {
                        "type": "potential_structuring",
                        "description": "Multiple transactions just below $10,000 threshold",
                        "risk_score": 40,
                    }
                )

            # Generate recommendations
            if monitoring_result["risk_score"] >= 50:
                monitoring_result["recommendations"].append(
                    "Enhanced due diligence required"
                )
            if monitoring_result["risk_score"] >= 30:
                monitoring_result["recommendations"].append(
                    "Increased monitoring frequency"
                )

            return monitoring_result

        except Exception as e:
            logger.error(
                f"User activity monitoring failed for user {user_id}: {str(e)}"
            )
            raise

    async def generate_compliance_report(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "monthly",
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        try:
            report = {
                "period": {"start": start_date, "end": end_date, "type": report_type},
                "summary": {},
                "metrics": {},
                "alerts": {},
                "recommendations": [],
            }

            # Transaction metrics
            tx_result = await db.execute(
                select(
                    func.count(Transaction.id).label("total_transactions"),
                    func.sum(Transaction.usd_value).label("total_volume"),
                    func.count(Transaction.id)
                    .filter(Transaction.usd_value >= self.transaction_threshold_usd)
                    .label("high_value_transactions"),
                ).where(
                    and_(
                        Transaction.created_at >= start_date,
                        Transaction.created_at <= end_date,
                        Transaction.status == TransactionStatus.CONFIRMED,
                    )
                )
            )
            tx_metrics = tx_result.first()

            report["metrics"]["transactions"] = {
                "total_count": tx_metrics.total_transactions or 0,
                "total_volume_usd": float(tx_metrics.total_volume or 0),
                "high_value_count": tx_metrics.high_value_transactions or 0,
            }

            # KYC metrics
            kyc_result = await db.execute(
                select(
                    func.count(KYCRecord.id).label("total_kyc"),
                    func.count(KYCRecord.id)
                    .filter(KYCRecord.status == KYCStatus.APPROVED)
                    .label("approved_kyc"),
                    func.count(KYCRecord.id)
                    .filter(KYCRecord.status == KYCStatus.REJECTED)
                    .label("rejected_kyc"),
                ).where(
                    and_(
                        KYCRecord.created_at >= start_date,
                        KYCRecord.created_at <= end_date,
                    )
                )
            )
            kyc_metrics = kyc_result.first()

            report["metrics"]["kyc"] = {
                "total_submissions": kyc_metrics.total_kyc or 0,
                "approved_count": kyc_metrics.approved_kyc or 0,
                "rejected_count": kyc_metrics.rejected_kyc or 0,
                "approval_rate": (kyc_metrics.approved_kyc or 0)
                / max(kyc_metrics.total_kyc or 1, 1)
                * 100,
            }

            # AML checks
            aml_result = await db.execute(
                select(
                    func.count(AMLCheck.id).label("total_checks"),
                    func.count(AMLCheck.id)
                    .filter(AMLCheck.risk_level == AMLRiskLevel.HIGH)
                    .label("high_risk_checks"),
                    func.count(AMLCheck.id)
                    .filter(
                        or_(
                            AMLCheck.sanctions_match == True, AMLCheck.pep_match == True
                        )
                    )
                    .label("matches_found"),
                ).where(
                    and_(
                        AMLCheck.created_at >= start_date,
                        AMLCheck.created_at <= end_date,
                    )
                )
            )
            aml_metrics = aml_result.first()

            report["metrics"]["aml"] = {
                "total_checks": aml_metrics.total_checks or 0,
                "high_risk_count": aml_metrics.high_risk_checks or 0,
                "matches_count": aml_metrics.matches_found or 0,
            }

            # Compliance alerts
            alert_result = await db.execute(
                select(
                    func.count(ComplianceAlert.id).label("total_alerts"),
                    func.count(ComplianceAlert.id)
                    .filter(ComplianceAlert.status == ComplianceAlertStatus.OPEN)
                    .label("open_alerts"),
                    func.count(ComplianceAlert.id)
                    .filter(ComplianceAlert.status == ComplianceAlertStatus.RESOLVED)
                    .label("resolved_alerts"),
                ).where(
                    and_(
                        ComplianceAlert.created_at >= start_date,
                        ComplianceAlert.created_at <= end_date,
                    )
                )
            )
            alert_metrics = alert_result.first()

            report["alerts"] = {
                "total_count": alert_metrics.total_alerts or 0,
                "open_count": alert_metrics.open_alerts or 0,
                "resolved_count": alert_metrics.resolved_alerts or 0,
                "resolution_rate": (alert_metrics.resolved_alerts or 0)
                / max(alert_metrics.total_alerts or 1, 1)
                * 100,
            }

            # Generate summary
            report["summary"] = {
                "total_users_monitored": await self._count_active_users(
                    db, start_date, end_date
                ),
                "compliance_score": await self._calculate_compliance_score(
                    report["metrics"]
                ),
                "key_findings": await self._generate_key_findings(report["metrics"]),
            }

            # Generate recommendations
            report["recommendations"] = await self._generate_compliance_recommendations(
                report
            )

            logger.info(
                f"Compliance report generated for period {start_date} to {end_date}"
            )
            return report

        except Exception as e:
            logger.error(f"Compliance report generation failed: {str(e)}")
            raise

    # Private helper methods

    async def _get_user_daily_volume(self, db: AsyncSession, user_id: UUID) -> Decimal:
        """Get user's daily transaction volume"""
        today = datetime.utcnow().date()
        result = await db.execute(
            select(func.sum(Transaction.usd_value)).where(
                and_(
                    Transaction.user_id == user_id,
                    func.date(Transaction.created_at) == today,
                    Transaction.status == TransactionStatus.CONFIRMED,
                )
            )
        )
        return result.scalar() or Decimal("0")

    async def _get_user_hourly_transaction_count(
        self, db: AsyncSession, user_id: UUID
    ) -> int:
        """Get user's transaction count in the last hour"""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        result = await db.execute(
            select(func.count(Transaction.id)).where(
                and_(
                    Transaction.user_id == user_id,
                    Transaction.created_at >= one_hour_ago,
                    Transaction.status == TransactionStatus.CONFIRMED,
                )
            )
        )
        return result.scalar() or 0

    async def _assess_user_risk(self, db: AsyncSession, user: User) -> Dict[str, Any]:
        """Assess user risk factors"""
        risk_assessment = {"score": 0, "alerts": []}

        # Check user country risk
        if user.country in self.high_risk_countries:
            risk_assessment["score"] += 30
            risk_assessment["alerts"].append(
                {
                    "type": "high_risk_jurisdiction",
                    "message": f"User from high-risk country: {user.country}",
                }
            )

        # Check account age
        account_age = datetime.utcnow() - user.created_at
        if account_age.days < 30:
            risk_assessment["score"] += 15
            risk_assessment["alerts"].append(
                {
                    "type": "new_account",
                    "message": f"Account created {account_age.days} days ago",
                }
            )

        # Check KYC status
        if user.kyc_status != KYCStatus.APPROVED:
            risk_assessment["score"] += 25
            risk_assessment["alerts"].append(
                {
                    "type": "kyc_incomplete",
                    "message": f"KYC status: {user.kyc_status.value}",
                }
            )

        return risk_assessment

    async def _check_kyc_compliance(
        self, db: AsyncSession, user: User
    ) -> Dict[str, Any]:
        """Check KYC compliance"""
        kyc_result = {"compliant": True, "alerts": []}

        if user.kyc_status != KYCStatus.APPROVED:
            kyc_result["compliant"] = False
            kyc_result["alerts"].append(
                {
                    "type": "kyc_required",
                    "message": "KYC verification required for this transaction",
                }
            )

        # Check KYC expiry
        result = await db.execute(select(KYCRecord).where(KYCRecord.user_id == user.id))
        kyc_record = result.scalar_one_or_none()

        if kyc_record and kyc_record.is_expired():
            kyc_result["compliant"] = False
            kyc_result["alerts"].append(
                {"type": "kyc_expired", "message": "KYC verification has expired"}
            )

        return kyc_result

    async def _check_sanctions(
        self, db: AsyncSession, user: User, transaction: Transaction
    ) -> Dict[str, Any]:
        """Check sanctions and watchlists"""
        sanctions_result = {"score": 0, "alerts": []}

        # Check recent AML results
        result = await db.execute(
            select(AMLCheck)
            .where(AMLCheck.user_id == user.id)
            .order_by(AMLCheck.created_at.desc())
            .limit(1)
        )
        latest_aml = result.scalar_one_or_none()

        if latest_aml:
            if latest_aml.sanctions_match:
                sanctions_result["score"] += 100
                sanctions_result["alerts"].append(
                    {
                        "type": "sanctions_match",
                        "message": "User matches sanctions list",
                    }
                )

            if latest_aml.pep_match:
                sanctions_result["score"] += 50
                sanctions_result["alerts"].append(
                    {"type": "pep_match", "message": "User matches PEP list"}
                )

            if latest_aml.adverse_media:
                sanctions_result["score"] += 25
                sanctions_result["alerts"].append(
                    {"type": "adverse_media", "message": "Adverse media found for user"}
                )

        return sanctions_result

    async def _create_compliance_alert(
        self,
        db: AsyncSession,
        user_id: UUID,
        transaction_id: UUID,
        compliance_result: Dict[str, Any],
    ) -> None:
        """Create compliance alert"""
        alert_type = ComplianceAlertType.SUSPICIOUS_TRANSACTION
        severity = "high" if compliance_result["risk_score"] >= 70 else "medium"

        alert = ComplianceAlert(
            user_id=user_id,
            transaction_id=transaction_id,
            alert_type=alert_type,
            severity=severity,
            title="Transaction Compliance Alert",
            description=f"Transaction flagged with risk score {compliance_result['risk_score']}",
            details={
                "risk_score": compliance_result["risk_score"],
                "alerts": compliance_result["alerts"],
                "recommendations": compliance_result["recommendations"],
            },
            risk_score=compliance_result["risk_score"],
            triggered_at=datetime.utcnow(),
        )

        db.add(alert)

    async def _analyze_transaction_patterns(
        self, transactions: List[Transaction]
    ) -> List[Dict[str, Any]]:
        """Analyze transaction patterns for suspicious activity"""
        patterns = []

        if len(transactions) < 2:
            return patterns

        # Check for rapid succession
        time_diffs = []
        for i in range(1, len(transactions)):
            diff = (
                transactions[i - 1].created_at - transactions[i].created_at
            ).total_seconds() / 60
            time_diffs.append(diff)

        avg_time_diff = sum(time_diffs) / len(time_diffs)
        if avg_time_diff < 5:  # Less than 5 minutes between transactions
            patterns.append(
                {
                    "type": "rapid_succession",
                    "description": f"Average time between transactions: {avg_time_diff:.1f} minutes",
                    "risk_score": 20,
                }
            )

        # Check for similar amounts
        amounts = [float(tx.amount) for tx in transactions if tx.amount]
        if amounts:
            amount_variance = max(amounts) - min(amounts)
            if amount_variance < max(amounts) * 0.1:  # Less than 10% variance
                patterns.append(
                    {
                        "type": "similar_amounts",
                        "description": "Transactions have very similar amounts",
                        "risk_score": 15,
                    }
                )

        return patterns

    async def _count_active_users(
        self, db: AsyncSession, start_date: datetime, end_date: datetime
    ) -> int:
        """Count active users in period"""
        result = await db.execute(
            select(func.count(func.distinct(Transaction.user_id))).where(
                and_(
                    Transaction.created_at >= start_date,
                    Transaction.created_at <= end_date,
                )
            )
        )
        return result.scalar() or 0

    def _calculate_compliance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        # Simple scoring algorithm - can be enhanced
        score = 100.0

        # Deduct points for high-risk transactions
        tx_metrics = metrics.get("transactions", {})
        total_tx = tx_metrics.get("total_count", 0)
        high_value_tx = tx_metrics.get("high_value_count", 0)

        if total_tx > 0:
            high_value_ratio = high_value_tx / total_tx
            if high_value_ratio > 0.1:  # More than 10% high-value
                score -= (high_value_ratio - 0.1) * 200

        # Deduct points for KYC issues
        kyc_metrics = metrics.get("kyc", {})
        approval_rate = kyc_metrics.get("approval_rate", 100)
        if approval_rate < 90:
            score -= (90 - approval_rate) * 0.5

        # Deduct points for AML matches
        aml_metrics = metrics.get("aml", {})
        total_checks = aml_metrics.get("total_checks", 0)
        matches = aml_metrics.get("matches_count", 0)

        if total_checks > 0:
            match_ratio = matches / total_checks
            score -= match_ratio * 50

        return max(0, min(100, score))

    def _generate_key_findings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate key findings from metrics"""
        findings = []

        tx_metrics = metrics.get("transactions", {})
        if tx_metrics.get("high_value_count", 0) > 0:
            findings.append(
                f"{tx_metrics['high_value_count']} high-value transactions detected"
            )

        kyc_metrics = metrics.get("kyc", {})
        if kyc_metrics.get("approval_rate", 100) < 90:
            findings.append(
                f"KYC approval rate below 90%: {kyc_metrics['approval_rate']:.1f}%"
            )

        aml_metrics = metrics.get("aml", {})
        if aml_metrics.get("matches_count", 0) > 0:
            findings.append(f"{aml_metrics['matches_count']} AML matches found")

        return findings

    async def _generate_compliance_recommendations(
        self, report: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        compliance_score = report["summary"].get("compliance_score", 100)
        if compliance_score < 80:
            recommendations.append("Enhance compliance monitoring and controls")

        alert_metrics = report.get("alerts", {})
        open_alerts = alert_metrics.get("open_count", 0)
        if open_alerts > 0:
            recommendations.append(
                f"Review and resolve {open_alerts} open compliance alerts"
            )

        resolution_rate = alert_metrics.get("resolution_rate", 100)
        if resolution_rate < 90:
            recommendations.append("Improve alert resolution processes")

        return recommendations
