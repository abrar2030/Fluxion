"""
Database models package for Fluxion backend
"""

from code.backend.models.base import BaseModel, SoftDeleteMixin, TimestampMixin
from code.backend.models.blockchain import (
    BlockchainNetwork,
    ContractEvent,
    SmartContract,
)
from code.backend.models.compliance import (
    AMLCheck,
    AuditLog,
    ComplianceAlert,
    KYCRecord,
)
from code.backend.models.portfolio import AssetHolding, Portfolio, PortfolioAsset
from code.backend.models.risk import RiskAlert, RiskAssessment, RiskProfile
from code.backend.models.transaction import (
    Transaction,
    TransactionStatus,
    TransactionType,
)
from code.backend.models.user import User, UserActivity, UserProfile, UserSession

__all__ = [
    # Base models
    "BaseModel",
    "TimestampMixin",
    "SoftDeleteMixin",
    # User models
    "User",
    "UserProfile",
    "UserSession",
    "UserActivity",
    # Transaction models
    "Transaction",
    "TransactionStatus",
    "TransactionType",
    # Portfolio models
    "Portfolio",
    "PortfolioAsset",
    "AssetHolding",
    # Compliance models
    "KYCRecord",
    "AMLCheck",
    "ComplianceAlert",
    "AuditLog",
    # Risk models
    "RiskProfile",
    "RiskAssessment",
    "RiskAlert",
    # Blockchain models
    "BlockchainNetwork",
    "SmartContract",
    "ContractEvent",
]
