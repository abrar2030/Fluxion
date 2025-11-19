"""
Pydantic schemas package for Fluxion backend
"""

from .auth import (MFASetup, MFAVerify, PasswordChange, TokenResponse,
                   UserLogin, UserRegister, UserResponse)
from .base import BaseResponse, ErrorResponse, PaginatedResponse
from .blockchain import (AssetResponse, ContractResponse, NetworkResponse,
                         TransferResponse)
from .compliance import (AMLCheckResponse, ComplianceAlertResponse,
                         KYCResponse, KYCSubmission)
from .portfolio import (AssetHoldingResponse, PortfolioCreate,
                        PortfolioResponse, PortfolioSummary, PortfolioUpdate)
from .risk import (RiskAlertResponse, RiskAssessmentResponse,
                   RiskProfileCreate, RiskProfileResponse)
from .transaction import (TransactionCreate, TransactionFilter,
                          TransactionResponse, TransactionSummary)
from .user import UserPreferences, UserProfile, UserUpdate

__all__ = [
    # Base schemas
    "BaseResponse",
    "PaginatedResponse",
    "ErrorResponse",
    # Auth schemas
    "UserRegister",
    "UserLogin",
    "UserResponse",
    "TokenResponse",
    "PasswordChange",
    "MFASetup",
    "MFAVerify",
    # User schemas
    "UserProfile",
    "UserUpdate",
    "UserPreferences",
    # Portfolio schemas
    "PortfolioCreate",
    "PortfolioUpdate",
    "PortfolioResponse",
    "AssetHoldingResponse",
    "PortfolioSummary",
    # Transaction schemas
    "TransactionCreate",
    "TransactionResponse",
    "TransactionSummary",
    "TransactionFilter",
    # Compliance schemas
    "KYCSubmission",
    "KYCResponse",
    "AMLCheckResponse",
    "ComplianceAlertResponse",
    # Risk schemas
    "RiskProfileCreate",
    "RiskProfileResponse",
    "RiskAssessmentResponse",
    "RiskAlertResponse",
    # Blockchain schemas
    "NetworkResponse",
    "ContractResponse",
    "AssetResponse",
    "TransferResponse",
]
