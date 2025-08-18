"""
Comprehensive KYC (Know Your Customer) Service for Fluxion Backend
Implements advanced customer verification, identity validation, and compliance
monitoring for financial services regulatory requirements.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import re
import hashlib
from pathlib import Path

from config.settings import settings
from services.security.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class KYCStatus(Enum):
    """KYC verification status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REQUIRES_UPDATE = "requires_update"


class DocumentType(Enum):
    """Types of identity documents"""
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"
    BUSINESS_LICENSE = "business_license"
    ARTICLES_OF_INCORPORATION = "articles_of_incorporation"


class RiskLevel(Enum):
    """Customer risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


class VerificationMethod(Enum):
    """Identity verification methods"""
    DOCUMENT_UPLOAD = "document_upload"
    BIOMETRIC = "biometric"
    VIDEO_CALL = "video_call"
    THIRD_PARTY_VERIFICATION = "third_party_verification"
    BANK_VERIFICATION = "bank_verification"


@dataclass
class PersonalInfo:
    """Personal information structure"""
    first_name: str
    last_name: str
    middle_name: Optional[str]
    date_of_birth: str
    nationality: str
    country_of_residence: str
    address_line1: str
    address_line2: Optional[str]
    city: str
    state_province: str
    postal_code: str
    phone_number: str
    email: str
    occupation: str
    employer: Optional[str]
    annual_income: Optional[float]
    source_of_funds: str


@dataclass
class BusinessInfo:
    """Business information structure"""
    business_name: str
    business_type: str
    registration_number: str
    tax_id: str
    incorporation_date: str
    country_of_incorporation: str
    business_address: str
    industry: str
    annual_revenue: Optional[float]
    number_of_employees: Optional[int]
    beneficial_owners: List[Dict[str, Any]]
    authorized_representatives: List[Dict[str, Any]]


@dataclass
class Document:
    """Document information structure"""
    document_id: str
    document_type: DocumentType
    file_path: str
    file_hash: str
    upload_timestamp: datetime
    expiry_date: Optional[datetime]
    verification_status: str
    extracted_data: Dict[str, Any]
    confidence_score: float


@dataclass
class KYCRecord:
    """Complete KYC record"""
    user_id: str
    customer_type: str  # individual or business
    status: KYCStatus
    risk_level: RiskLevel
    personal_info: Optional[PersonalInfo]
    business_info: Optional[BusinessInfo]
    documents: List[Document]
    verification_methods: List[VerificationMethod]
    verification_date: Optional[datetime]
    expiry_date: Optional[datetime]
    last_updated: datetime
    compliance_notes: List[str]
    sanctions_check_result: Dict[str, Any]
    pep_check_result: Dict[str, Any]
    adverse_media_check: Dict[str, Any]
    verification_history: List[Dict[str, Any]]


class KYCService:
    """
    Comprehensive KYC service providing:
    - Customer identity verification
    - Document validation and OCR
    - Risk assessment and scoring
    - Sanctions and PEP screening
    - Ongoing monitoring and updates
    - Regulatory compliance reporting
    """
    
    def __init__(self):
        self.encryption_service = EncryptionService()
        
        # KYC configuration
        self.verification_expiry_days = 365
        self.document_retention_days = 2555  # 7 years
        self.risk_score_threshold = {
            RiskLevel.LOW: 30,
            RiskLevel.MEDIUM: 70,
            RiskLevel.HIGH: 90
        }
        
        # Document validation patterns
        self.document_patterns = {
            'passport': r'^[A-Z]{1,2}[0-9]{6,9}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'ein': r'^\d{2}-\d{7}$',
            'phone': r'^\+?1?[2-9]\d{2}[2-9]\d{2}\d{4}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        }
        
        # Sanctions lists (in production, integrate with OFAC, UN, EU lists)
        self.sanctions_lists = {
            'OFAC_SDN': [],
            'UN_SANCTIONS': [],
            'EU_SANCTIONS': [],
            'PEP_LIST': []
        }
        
        # High-risk countries (FATF list)
        self.high_risk_countries = {
            'AF', 'IR', 'KP', 'MM', 'PK', 'UG', 'YE'  # Example codes
        }
        
        # In-memory storage (in production, use database)
        self.kyc_records: Dict[str, KYCRecord] = {}
        self.document_storage: Dict[str, bytes] = {}
    
    async def initiate_kyc(self, user_id: str, customer_type: str = 'individual') -> str:
        """Initiate KYC process for a user"""
        if user_id in self.kyc_records:
            existing_record = self.kyc_records[user_id]
            if existing_record.status in [KYCStatus.VERIFIED, KYCStatus.IN_PROGRESS]:
                return f"KYC already {existing_record.status.value} for user {user_id}"
        
        # Create new KYC record
        kyc_record = KYCRecord(
            user_id=user_id,
            customer_type=customer_type,
            status=KYCStatus.PENDING,
            risk_level=RiskLevel.MEDIUM,  # Default until assessment
            personal_info=None,
            business_info=None,
            documents=[],
            verification_methods=[],
            verification_date=None,
            expiry_date=None,
            last_updated=datetime.now(timezone.utc),
            compliance_notes=[],
            sanctions_check_result={},
            pep_check_result={},
            adverse_media_check={},
            verification_history=[]
        )
        
        self.kyc_records[user_id] = kyc_record
        
        logger.info(f"Initiated KYC process for user {user_id} as {customer_type}")
        return f"KYC process initiated for user {user_id}"
    
    async def submit_personal_info(self, user_id: str, personal_info: PersonalInfo) -> Dict[str, Any]:
        """Submit personal information for KYC verification"""
        if user_id not in self.kyc_records:
            raise ValueError(f"KYC process not initiated for user {user_id}")
        
        kyc_record = self.kyc_records[user_id]
        
        # Validate personal information
        validation_result = await self._validate_personal_info(personal_info)
        if not validation_result['valid']:
            return {
                'success': False,
                'errors': validation_result['errors'],
                'status': kyc_record.status.value
            }
        
        # Encrypt and store personal information
        encrypted_info = await self._encrypt_personal_info(personal_info)
        kyc_record.personal_info = encrypted_info
        kyc_record.status = KYCStatus.IN_PROGRESS
        kyc_record.last_updated = datetime.now(timezone.utc)
        
        # Perform initial risk assessment
        risk_assessment = await self._assess_personal_info_risk(personal_info)
        kyc_record.risk_level = risk_assessment['risk_level']
        kyc_record.compliance_notes.extend(risk_assessment['notes'])
        
        # Add to verification history
        kyc_record.verification_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'personal_info_submitted',
            'details': 'Personal information submitted and validated'
        })
        
        logger.info(f"Personal information submitted for user {user_id}")
        
        return {
            'success': True,
            'status': kyc_record.status.value,
            'risk_level': kyc_record.risk_level.value,
            'next_steps': await self._get_next_steps(kyc_record)
        }
    
    async def submit_business_info(self, user_id: str, business_info: BusinessInfo) -> Dict[str, Any]:
        """Submit business information for KYC verification"""
        if user_id not in self.kyc_records:
            raise ValueError(f"KYC process not initiated for user {user_id}")
        
        kyc_record = self.kyc_records[user_id]
        
        if kyc_record.customer_type != 'business':
            raise ValueError("Business information can only be submitted for business customers")
        
        # Validate business information
        validation_result = await self._validate_business_info(business_info)
        if not validation_result['valid']:
            return {
                'success': False,
                'errors': validation_result['errors'],
                'status': kyc_record.status.value
            }
        
        # Encrypt and store business information
        encrypted_info = await self._encrypt_business_info(business_info)
        kyc_record.business_info = encrypted_info
        kyc_record.status = KYCStatus.IN_PROGRESS
        kyc_record.last_updated = datetime.now(timezone.utc)
        
        # Perform business risk assessment
        risk_assessment = await self._assess_business_info_risk(business_info)
        kyc_record.risk_level = risk_assessment['risk_level']
        kyc_record.compliance_notes.extend(risk_assessment['notes'])
        
        # Add to verification history
        kyc_record.verification_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'business_info_submitted',
            'details': 'Business information submitted and validated'
        })
        
        logger.info(f"Business information submitted for user {user_id}")
        
        return {
            'success': True,
            'status': kyc_record.status.value,
            'risk_level': kyc_record.risk_level.value,
            'next_steps': await self._get_next_steps(kyc_record)
        }
    
    async def upload_document(self, user_id: str, document_type: DocumentType,
                            file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload and process identity document"""
        if user_id not in self.kyc_records:
            raise ValueError(f"KYC process not initiated for user {user_id}")
        
        kyc_record = self.kyc_records[user_id]
        
        # Generate document ID
        document_id = self._generate_document_id(user_id, document_type)
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check for duplicate documents
        for existing_doc in kyc_record.documents:
            if existing_doc.file_hash == file_hash:
                return {
                    'success': False,
                    'error': 'Document already uploaded',
                    'document_id': existing_doc.document_id
                }
        
        # Encrypt and store document
        encrypted_content = self.encryption_service.encrypt_field(
            file_content.hex(), 'sensitive'
        )
        file_path = f"documents/{user_id}/{document_id}"
        self.document_storage[file_path] = encrypted_content
        
        # Process document (OCR, validation)
        processing_result = await self._process_document(
            file_content, document_type, filename
        )
        
        # Create document record
        document = Document(
            document_id=document_id,
            document_type=document_type,
            file_path=file_path,
            file_hash=file_hash,
            upload_timestamp=datetime.now(timezone.utc),
            expiry_date=processing_result.get('expiry_date'),
            verification_status=processing_result['status'],
            extracted_data=processing_result['extracted_data'],
            confidence_score=processing_result['confidence_score']
        )
        
        kyc_record.documents.append(document)
        kyc_record.verification_methods.append(VerificationMethod.DOCUMENT_UPLOAD)
        kyc_record.last_updated = datetime.now(timezone.utc)
        
        # Update verification status if document is valid
        if processing_result['status'] == 'verified':
            await self._update_verification_progress(kyc_record)
        
        # Add to verification history
        kyc_record.verification_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'document_uploaded',
            'details': f'{document_type.value} document uploaded and processed',
            'document_id': document_id,
            'verification_status': processing_result['status']
        })
        
        logger.info(f"Document {document_type.value} uploaded for user {user_id}")
        
        return {
            'success': True,
            'document_id': document_id,
            'verification_status': processing_result['status'],
            'confidence_score': processing_result['confidence_score'],
            'extracted_data': processing_result['extracted_data'],
            'next_steps': await self._get_next_steps(kyc_record)
        }
    
    async def perform_sanctions_screening(self, user_id: str) -> Dict[str, Any]:
        """Perform sanctions and PEP screening"""
        if user_id not in self.kyc_records:
            raise ValueError(f"KYC record not found for user {user_id}")
        
        kyc_record = self.kyc_records[user_id]
        
        if not kyc_record.personal_info and not kyc_record.business_info:
            raise ValueError("Personal or business information required for screening")
        
        screening_results = {
            'sanctions_match': False,
            'pep_match': False,
            'adverse_media_match': False,
            'matches': [],
            'risk_score': 0
        }
        
        # Perform sanctions screening
        if kyc_record.personal_info:
            personal_info = await self._decrypt_personal_info(kyc_record.personal_info)
            sanctions_result = await self._screen_sanctions_individual(personal_info)
            screening_results.update(sanctions_result)
        
        if kyc_record.business_info:
            business_info = await self._decrypt_business_info(kyc_record.business_info)
            business_sanctions_result = await self._screen_sanctions_business(business_info)
            # Merge results
            screening_results['sanctions_match'] = (
                screening_results['sanctions_match'] or 
                business_sanctions_result['sanctions_match']
            )
            screening_results['matches'].extend(business_sanctions_result['matches'])
            screening_results['risk_score'] = max(
                screening_results['risk_score'],
                business_sanctions_result['risk_score']
            )
        
        # Store screening results
        kyc_record.sanctions_check_result = screening_results
        kyc_record.last_updated = datetime.now(timezone.utc)
        
        # Update risk level based on screening
        if screening_results['sanctions_match']:
            kyc_record.risk_level = RiskLevel.PROHIBITED
            kyc_record.status = KYCStatus.REJECTED
            kyc_record.compliance_notes.append("Sanctions match detected")
        elif screening_results['pep_match']:
            kyc_record.risk_level = RiskLevel.HIGH
            kyc_record.compliance_notes.append("PEP match detected - enhanced due diligence required")
        
        # Add to verification history
        kyc_record.verification_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'sanctions_screening',
            'details': f'Sanctions screening completed - Risk score: {screening_results["risk_score"]}',
            'results': screening_results
        })
        
        logger.info(f"Sanctions screening completed for user {user_id}")
        
        return screening_results
    
    async def complete_verification(self, user_id: str, manual_review: bool = False) -> Dict[str, Any]:
        """Complete KYC verification process"""
        if user_id not in self.kyc_records:
            raise ValueError(f"KYC record not found for user {user_id}")
        
        kyc_record = self.kyc_records[user_id]
        
        # Check if all required information is provided
        verification_check = await self._check_verification_completeness(kyc_record)
        if not verification_check['complete']:
            return {
                'success': False,
                'status': kyc_record.status.value,
                'missing_requirements': verification_check['missing'],
                'next_steps': await self._get_next_steps(kyc_record)
            }
        
        # Perform final risk assessment
        final_risk_assessment = await self._perform_final_risk_assessment(kyc_record)
        kyc_record.risk_level = final_risk_assessment['risk_level']
        kyc_record.compliance_notes.extend(final_risk_assessment['notes'])
        
        # Determine verification outcome
        if kyc_record.risk_level == RiskLevel.PROHIBITED:
            kyc_record.status = KYCStatus.REJECTED
            outcome = 'rejected'
        elif manual_review or kyc_record.risk_level == RiskLevel.HIGH:
            # High-risk customers require manual review
            outcome = 'manual_review_required'
            kyc_record.compliance_notes.append("Manual review required due to high risk level")
        else:
            kyc_record.status = KYCStatus.VERIFIED
            kyc_record.verification_date = datetime.now(timezone.utc)
            kyc_record.expiry_date = datetime.now(timezone.utc) + timedelta(days=self.verification_expiry_days)
            outcome = 'verified'
        
        kyc_record.last_updated = datetime.now(timezone.utc)
        
        # Add to verification history
        kyc_record.verification_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'verification_completed',
            'details': f'KYC verification completed with outcome: {outcome}',
            'risk_level': kyc_record.risk_level.value,
            'final_status': kyc_record.status.value
        })
        
        logger.info(f"KYC verification completed for user {user_id} with outcome: {outcome}")
        
        return {
            'success': True,
            'outcome': outcome,
            'status': kyc_record.status.value,
            'risk_level': kyc_record.risk_level.value,
            'verification_date': kyc_record.verification_date.isoformat() if kyc_record.verification_date else None,
            'expiry_date': kyc_record.expiry_date.isoformat() if kyc_record.expiry_date else None,
            'compliance_notes': kyc_record.compliance_notes
        }
    
    async def get_kyc_status(self, user_id: str) -> Dict[str, Any]:
        """Get current KYC status for a user"""
        if user_id not in self.kyc_records:
            return {
                'user_id': user_id,
                'status': 'not_initiated',
                'verified': False
            }
        
        kyc_record = self.kyc_records[user_id]
        
        # Check if verification has expired
        if (kyc_record.status == KYCStatus.VERIFIED and 
            kyc_record.expiry_date and 
            datetime.now(timezone.utc) > kyc_record.expiry_date):
            kyc_record.status = KYCStatus.EXPIRED
            kyc_record.last_updated = datetime.now(timezone.utc)
        
        return {
            'user_id': user_id,
            'status': kyc_record.status.value,
            'verified': kyc_record.status == KYCStatus.VERIFIED,
            'risk_level': kyc_record.risk_level.value,
            'verification_date': kyc_record.verification_date.isoformat() if kyc_record.verification_date else None,
            'expiry_date': kyc_record.expiry_date.isoformat() if kyc_record.expiry_date else None,
            'last_updated': kyc_record.last_updated.isoformat(),
            'documents_count': len(kyc_record.documents),
            'verification_methods': [method.value for method in kyc_record.verification_methods],
            'next_steps': await self._get_next_steps(kyc_record)
        }
    
    # Private helper methods
    
    async def _validate_personal_info(self, personal_info: PersonalInfo) -> Dict[str, Any]:
        """Validate personal information"""
        errors = []
        
        # Required field validation
        required_fields = ['first_name', 'last_name', 'date_of_birth', 'nationality',
                          'country_of_residence', 'address_line1', 'city', 'postal_code',
                          'phone_number', 'email']
        
        for field in required_fields:
            if not getattr(personal_info, field):
                errors.append(f"{field} is required")
        
        # Format validation
        if personal_info.email and not re.match(self.document_patterns['email'], personal_info.email):
            errors.append("Invalid email format")
        
        if personal_info.phone_number and not re.match(self.document_patterns['phone'], personal_info.phone_number):
            errors.append("Invalid phone number format")
        
        # Date validation
        try:
            birth_date = datetime.fromisoformat(personal_info.date_of_birth)
            age = (datetime.now() - birth_date).days / 365.25
            if age < 18:
                errors.append("Customer must be at least 18 years old")
            elif age > 120:
                errors.append("Invalid date of birth")
        except ValueError:
            errors.append("Invalid date of birth format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _validate_business_info(self, business_info: BusinessInfo) -> Dict[str, Any]:
        """Validate business information"""
        errors = []
        
        # Required field validation
        required_fields = ['business_name', 'business_type', 'registration_number',
                          'tax_id', 'incorporation_date', 'country_of_incorporation',
                          'business_address', 'industry']
        
        for field in required_fields:
            if not getattr(business_info, field):
                errors.append(f"{field} is required")
        
        # Beneficial owners validation
        if not business_info.beneficial_owners:
            errors.append("At least one beneficial owner is required")
        else:
            for i, owner in enumerate(business_info.beneficial_owners):
                if not owner.get('name') or not owner.get('ownership_percentage'):
                    errors.append(f"Beneficial owner {i+1} missing required information")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _encrypt_personal_info(self, personal_info: PersonalInfo) -> PersonalInfo:
        """Encrypt sensitive personal information"""
        # In production, encrypt PII fields
        return personal_info
    
    async def _decrypt_personal_info(self, encrypted_info: PersonalInfo) -> PersonalInfo:
        """Decrypt personal information"""
        # In production, decrypt PII fields
        return encrypted_info
    
    async def _encrypt_business_info(self, business_info: BusinessInfo) -> BusinessInfo:
        """Encrypt sensitive business information"""
        # In production, encrypt sensitive business fields
        return business_info
    
    async def _decrypt_business_info(self, encrypted_info: BusinessInfo) -> BusinessInfo:
        """Decrypt business information"""
        # In production, decrypt business fields
        return encrypted_info
    
    async def _assess_personal_info_risk(self, personal_info: PersonalInfo) -> Dict[str, Any]:
        """Assess risk based on personal information"""
        risk_score = 0
        notes = []
        
        # Country risk assessment
        if personal_info.country_of_residence in self.high_risk_countries:
            risk_score += 30
            notes.append(f"High-risk country of residence: {personal_info.country_of_residence}")
        
        if personal_info.nationality in self.high_risk_countries:
            risk_score += 20
            notes.append(f"High-risk nationality: {personal_info.nationality}")
        
        # Age-based risk
        try:
            birth_date = datetime.fromisoformat(personal_info.date_of_birth)
            age = (datetime.now() - birth_date).days / 365.25
            if age < 25:
                risk_score += 10
                notes.append("Young customer - increased monitoring")
        except ValueError:
            pass
        
        # Determine risk level
        if risk_score >= self.risk_score_threshold[RiskLevel.HIGH]:
            risk_level = RiskLevel.HIGH
        elif risk_score >= self.risk_score_threshold[RiskLevel.MEDIUM]:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'notes': notes
        }
    
    async def _assess_business_info_risk(self, business_info: BusinessInfo) -> Dict[str, Any]:
        """Assess risk based on business information"""
        risk_score = 0
        notes = []
        
        # Country risk
        if business_info.country_of_incorporation in self.high_risk_countries:
            risk_score += 40
            notes.append(f"High-risk incorporation country: {business_info.country_of_incorporation}")
        
        # Industry risk
        high_risk_industries = ['money_services', 'gambling', 'adult_entertainment', 'cryptocurrency']
        if business_info.industry.lower() in high_risk_industries:
            risk_score += 25
            notes.append(f"High-risk industry: {business_info.industry}")
        
        # Business age
        try:
            incorporation_date = datetime.fromisoformat(business_info.incorporation_date)
            business_age = (datetime.now() - incorporation_date).days / 365.25
            if business_age < 2:
                risk_score += 15
                notes.append("New business - increased monitoring")
        except ValueError:
            pass
        
        # Determine risk level
        if risk_score >= self.risk_score_threshold[RiskLevel.HIGH]:
            risk_level = RiskLevel.HIGH
        elif risk_score >= self.risk_score_threshold[RiskLevel.MEDIUM]:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'notes': notes
        }
    
    def _generate_document_id(self, user_id: str, document_type: DocumentType) -> str:
        """Generate unique document ID"""
        timestamp = int(datetime.now().timestamp())
        return f"doc_{user_id}_{document_type.value}_{timestamp}"
    
    async def _process_document(self, file_content: bytes, document_type: DocumentType,
                              filename: str) -> Dict[str, Any]:
        """Process uploaded document (OCR, validation)"""
        # Simulated document processing
        # In production, integrate with OCR services like AWS Textract, Google Vision API
        
        processing_result = {
            'status': 'verified',
            'confidence_score': 0.95,
            'extracted_data': {
                'document_number': 'SIMULATED123456',
                'expiry_date': '2025-12-31',
                'name': 'John Doe',
                'date_of_birth': '1990-01-01'
            },
            'expiry_date': datetime(2025, 12, 31)
        }
        
        # Basic file validation
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            processing_result['status'] = 'rejected'
            processing_result['confidence_score'] = 0.0
            processing_result['error'] = 'File size too large'
        
        return processing_result
    
    async def _screen_sanctions_individual(self, personal_info: PersonalInfo) -> Dict[str, Any]:
        """Screen individual against sanctions lists"""
        # Simulated sanctions screening
        # In production, integrate with OFAC, UN, EU sanctions APIs
        
        full_name = f"{personal_info.first_name} {personal_info.last_name}".lower()
        
        # Check against simulated sanctions list
        sanctions_matches = []
        for sanctions_name in ['john terrorist', 'jane criminal']:  # Simulated list
            if sanctions_name in full_name:
                sanctions_matches.append({
                    'list': 'OFAC_SDN',
                    'name': sanctions_name,
                    'match_score': 0.95
                })
        
        return {
            'sanctions_match': len(sanctions_matches) > 0,
            'pep_match': False,  # Simulated
            'adverse_media_match': False,  # Simulated
            'matches': sanctions_matches,
            'risk_score': 100 if sanctions_matches else 0
        }
    
    async def _screen_sanctions_business(self, business_info: BusinessInfo) -> Dict[str, Any]:
        """Screen business against sanctions lists"""
        # Simulated business sanctions screening
        return {
            'sanctions_match': False,
            'matches': [],
            'risk_score': 0
        }
    
    async def _update_verification_progress(self, kyc_record: KYCRecord):
        """Update verification progress based on completed steps"""
        # Check if all required documents are uploaded and verified
        required_docs = self._get_required_documents(kyc_record.customer_type)
        verified_docs = [doc for doc in kyc_record.documents if doc.verification_status == 'verified']
        
        if len(verified_docs) >= len(required_docs):
            # All documents verified, ready for final review
            pass
    
    def _get_required_documents(self, customer_type: str) -> List[DocumentType]:
        """Get required documents for customer type"""
        if customer_type == 'individual':
            return [DocumentType.PASSPORT, DocumentType.UTILITY_BILL]
        elif customer_type == 'business':
            return [DocumentType.BUSINESS_LICENSE, DocumentType.ARTICLES_OF_INCORPORATION]
        else:
            return []
    
    async def _check_verification_completeness(self, kyc_record: KYCRecord) -> Dict[str, Any]:
        """Check if KYC verification is complete"""
        missing = []
        
        # Check personal/business info
        if kyc_record.customer_type == 'individual' and not kyc_record.personal_info:
            missing.append('personal_information')
        elif kyc_record.customer_type == 'business' and not kyc_record.business_info:
            missing.append('business_information')
        
        # Check required documents
        required_docs = self._get_required_documents(kyc_record.customer_type)
        verified_docs = [doc.document_type for doc in kyc_record.documents 
                        if doc.verification_status == 'verified']
        
        for required_doc in required_docs:
            if required_doc not in verified_docs:
                missing.append(f'{required_doc.value}_document')
        
        # Check sanctions screening
        if not kyc_record.sanctions_check_result:
            missing.append('sanctions_screening')
        
        return {
            'complete': len(missing) == 0,
            'missing': missing
        }
    
    async def _perform_final_risk_assessment(self, kyc_record: KYCRecord) -> Dict[str, Any]:
        """Perform final comprehensive risk assessment"""
        total_risk_score = 0
        notes = []
        
        # Sanctions screening risk
        if kyc_record.sanctions_check_result.get('sanctions_match'):
            return {
                'risk_level': RiskLevel.PROHIBITED,
                'risk_score': 100,
                'notes': ['Sanctions match - customer prohibited']
            }
        
        # Document verification risk
        verified_docs = [doc for doc in kyc_record.documents if doc.verification_status == 'verified']
        if len(verified_docs) < len(self._get_required_documents(kyc_record.customer_type)):
            total_risk_score += 20
            notes.append('Incomplete document verification')
        
        # PEP risk
        if kyc_record.sanctions_check_result.get('pep_match'):
            total_risk_score += 30
            notes.append('PEP match detected')
        
        # Determine final risk level
        if total_risk_score >= self.risk_score_threshold[RiskLevel.HIGH]:
            risk_level = RiskLevel.HIGH
        elif total_risk_score >= self.risk_score_threshold[RiskLevel.MEDIUM]:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            'risk_level': risk_level,
            'risk_score': total_risk_score,
            'notes': notes
        }
    
    async def _get_next_steps(self, kyc_record: KYCRecord) -> List[str]:
        """Get next steps for KYC completion"""
        next_steps = []
        
        if kyc_record.status == KYCStatus.PENDING:
            if kyc_record.customer_type == 'individual' and not kyc_record.personal_info:
                next_steps.append('Submit personal information')
            elif kyc_record.customer_type == 'business' and not kyc_record.business_info:
                next_steps.append('Submit business information')
        
        if kyc_record.status == KYCStatus.IN_PROGRESS:
            # Check for missing documents
            required_docs = self._get_required_documents(kyc_record.customer_type)
            uploaded_doc_types = [doc.document_type for doc in kyc_record.documents]
            
            for required_doc in required_docs:
                if required_doc not in uploaded_doc_types:
                    next_steps.append(f'Upload {required_doc.value} document')
            
            # Check for sanctions screening
            if not kyc_record.sanctions_check_result:
                next_steps.append('Complete sanctions screening')
            
            # If all requirements met, ready for completion
            if not next_steps:
                next_steps.append('Ready for verification completion')
        
        return next_steps
    
    def get_kyc_statistics(self) -> Dict[str, Any]:
        """Get KYC service statistics"""
        status_counts = {}
        risk_level_counts = {}
        
        for record in self.kyc_records.values():
            status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
            risk_level_counts[record.risk_level.value] = risk_level_counts.get(record.risk_level.value, 0) + 1
        
        return {
            'total_records': len(self.kyc_records),
            'status_distribution': status_counts,
            'risk_level_distribution': risk_level_counts,
            'documents_stored': len(self.document_storage)
        }

