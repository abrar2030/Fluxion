"""
Pydantic schemas package for Fluxion backend
"""

from .base import BaseResponse, PaginatedResponse, ErrorResponse
from .auth import (
    UserRegister, UserLogin, UserResponse, TokenResponse,
    PasswordChange, MFASetup, MFAVerify
)
from .user import UserProfile, UserUpdate, UserPreferences
from .portfolio import (
    PortfolioCreate, PortfolioUpdate, PortfolioResponse,
    AssetHoldingResponse, PortfolioSummary
)
from .transaction import (
    TransactionCreate, TransactionResponse, TransactionSummary,
    TransactionFilter
)
from .compliance import (
    KYCSubmission, KYCResponse, AMLCheckResponse,
    ComplianceAlertResponse
)
from .risk import (
    RiskProfileCreate, RiskProfileResponse, RiskAssessmentResponse,
    RiskAlertResponse
)
from .blockchain import (
    NetworkResponse, ContractResponse, AssetResponse,
    TransferResponse
)

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

