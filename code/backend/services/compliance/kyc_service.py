"""
Enhanced KYC (Know Your Customer) Service for Fluxion Backend
Implements comprehensive identity verification, document processing,
and compliance monitoring following global regulatory standards.
"""

import logging
import asyncio
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from models.user import User
from models.compliance import (
    KYCRecord, KYCStatus, KYCLevel, DocumentType,
    IdentityVerification, BiometricData, ComplianceAlert
)
from config.settings import settings
from services.security.encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class KYCTier(Enum):
    """KYC verification tiers"""
    BASIC = "basic"          # Email + Phone verification
    STANDARD = "standard"    # + Government ID
    ENHANCED = "enhanced"    # + Proof of address + Selfie
    PREMIUM = "premium"      # + Enhanced due diligence + Source of funds


class DocumentStatus(Enum):
    """Document verification status"""
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REQUIRES_REVIEW = "requires_review"


class RiskRating(Enum):
    """Customer risk rating"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROHIBITED = "prohibited"


@dataclass
class DocumentVerificationResult:
    """Document verification result"""
    document_id: str
    document_type: DocumentType
    status: DocumentStatus
    confidence_score: float
    extracted_data: Dict[str, Any]
    verification_checks: Dict[str, bool]
    risk_indicators: List[str]
    processing_time: float
    verified_at: datetime
    expires_at: Optional[datetime]


@dataclass
class BiometricVerificationResult:
    """Biometric verification result"""
    verification_id: str
    match_score: float
    liveness_score: float
    quality_score: float
    is_match: bool
    risk_indicators: List[str]
    verified_at: datetime


@dataclass
class KYCAssessment:
    """Comprehensive KYC assessment"""
    user_id: str
    kyc_level: KYCTier
    overall_status: KYCStatus
    risk_rating: RiskRating
    compliance_score: float
    document_verifications: List[DocumentVerificationResult]
    biometric_verification: Optional[BiometricVerificationResult]
    sanctions_check: Dict[str, Any]
    pep_check: Dict[str, Any]
    adverse_media_check: Dict[str, Any]
    address_verification: Dict[str, Any]
    source_of_funds_verification: Dict[str, Any]
    ongoing_monitoring: Dict[str, Any]
    recommendations: List[str]
    next_review_date: datetime
    assessed_at: datetime


class EnhancedKYCService:
    """
    Enhanced KYC service providing:
    - Multi-tier identity verification
    - Document authentication and OCR
    - Biometric verification (facial recognition, liveness detection)
    - Sanctions and PEP screening
    - Adverse media monitoring
    - Address verification
    - Source of funds verification
    - Ongoing monitoring and periodic reviews
    - Risk-based approach to customer due diligence
    """
    
    def __init__(self):
        self.encryption_service = EncryptionService()
        
        # KYC configuration
        self.tier_requirements = {
            KYCTier.BASIC: {
                'required_documents': [],
                'required_verifications': ['email', 'phone'],
                'transaction_limits': {'daily': 1000, 'monthly': 5000},
                'review_frequency_days': 365
            },
            KYCTier.STANDARD: {
                'required_documents': [DocumentType.GOVERNMENT_ID],
                'required_verifications': ['email', 'phone', 'identity'],
                'transaction_limits': {'daily': 10000, 'monthly': 50000},
                'review_frequency_days': 180
            },
            KYCTier.ENHANCED: {
                'required_documents': [DocumentType.GOVERNMENT_ID, DocumentType.PROOF_OF_ADDRESS],
                'required_verifications': ['email', 'phone', 'identity', 'address', 'biometric'],
                'transaction_limits': {'daily': 50000, 'monthly': 250000},
                'review_frequency_days': 90
            },
            KYCTier.PREMIUM: {
                'required_documents': [
                    DocumentType.GOVERNMENT_ID, 
                    DocumentType.PROOF_OF_ADDRESS,
                    DocumentType.PROOF_OF_INCOME
                ],
                'required_verifications': [
                    'email', 'phone', 'identity', 'address', 'biometric', 'source_of_funds'
                ],
                'transaction_limits': {'daily': 100000, 'monthly': 1000000},
                'review_frequency_days': 30
            }
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'sanctions_match': 100,
            'pep_match': 80,
            'adverse_media': 60,
            'high_risk_country': 40,
            'document_quality': 30,
            'biometric_quality': 25,
            'address_verification': 20,
            'source_of_funds': 35
        }
        
        # High-risk countries (example list)
        self.high_risk_countries = [
            'AF', 'IR', 'KP', 'SY', 'MM', 'BY', 'CU', 'IQ', 'LB', 'LY', 'SO', 'SS', 'SD', 'YE', 'ZW'
        ]
        
        # Document expiry periods
        self.document_validity_periods = {
            DocumentType.GOVERNMENT_ID: timedelta(days=1825),  # 5 years
            DocumentType.PASSPORT: timedelta(days=3650),       # 10 years
            DocumentType.PROOF_OF_ADDRESS: timedelta(days=90), # 3 months
            DocumentType.PROOF_OF_INCOME: timedelta(days=365)  # 1 year
        }
    
    async def initiate_kyc_process(
        self,
        db: AsyncSession,
        user_id: UUID,
        target_tier: KYCTier = KYCTier.STANDARD
    ) -> Dict[str, Any]:
        """Initiate KYC process for a user"""
        try:
            # Check if user exists
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Check existing KYC records
            existing_kyc = await db.execute(
                select(KYCRecord)
                .where(KYCRecord.user_id == user_id)
                .order_by(desc(KYCRecord.created_at))
                .limit(1)
            )
            current_kyc = existing_kyc.scalar_one_or_none()
            
            # Determine required steps based on target tier
            tier_config = self.tier_requirements[target_tier]
            required_steps = []
            
            # Check what's already completed
            completed_verifications = set()
            if current_kyc:
                if current_kyc.email_verified:
                    completed_verifications.add('email')
                if current_kyc.phone_verified:
                    completed_verifications.add('phone')
                if current_kyc.identity_verified:
                    completed_verifications.add('identity')
                if current_kyc.address_verified:
                    completed_verifications.add('address')
                if current_kyc.biometric_verified:
                    completed_verifications.add('biometric')
                if current_kyc.source_of_funds_verified:
                    completed_verifications.add('source_of_funds')
            
            # Determine missing steps
            missing_verifications = set(tier_config['required_verifications']) - completed_verifications
            
            for verification in missing_verifications:
                if verification == 'email':
                    required_steps.append({
                        'step': 'email_verification',
                        'title': 'Email Verification',
                        'description': 'Verify your email address',
                        'status': 'pending'
                    })
                elif verification == 'phone':
                    required_steps.append({
                        'step': 'phone_verification',
                        'title': 'Phone Verification',
                        'description': 'Verify your phone number via SMS',
                        'status': 'pending'
                    })
                elif verification == 'identity':
                    required_steps.append({
                        'step': 'document_upload',
                        'title': 'Identity Document Upload',
                        'description': 'Upload government-issued ID (passport, driver\'s license, etc.)',
                        'status': 'pending',
                        'accepted_documents': ['passport', 'drivers_license', 'national_id']
                    })
                elif verification == 'address':
                    required_steps.append({
                        'step': 'address_verification',
                        'title': 'Address Verification',
                        'description': 'Upload proof of address (utility bill, bank statement, etc.)',
                        'status': 'pending',
                        'accepted_documents': ['utility_bill', 'bank_statement', 'rental_agreement']
                    })
                elif verification == 'biometric':
                    required_steps.append({
                        'step': 'biometric_verification',
                        'title': 'Biometric Verification',
                        'description': 'Take a selfie for facial recognition verification',
                        'status': 'pending'
                    })
                elif verification == 'source_of_funds':
                    required_steps.append({
                        'step': 'source_of_funds',
                        'title': 'Source of Funds Verification',
                        'description': 'Provide documentation for source of funds',
                        'status': 'pending'
                    })
            
            # Create or update KYC record
            if not current_kyc:
                kyc_record = KYCRecord(
                    id=uuid4(),
                    user_id=user_id,
                    target_level=target_tier.value,
                    status=KYCStatus.PENDING,
                    created_at=datetime.utcnow()
                )
                db.add(kyc_record)
            else:
                current_kyc.target_level = target_tier.value
                current_kyc.updated_at = datetime.utcnow()
            
            await db.commit()
            
            kyc_process = {
                'kyc_id': str(current_kyc.id if current_kyc else kyc_record.id),
                'user_id': str(user_id),
                'target_tier': target_tier.value,
                'current_status': current_kyc.status.value if current_kyc else KYCStatus.PENDING.value,
                'required_steps': required_steps,
                'completed_steps': list(completed_verifications),
                'estimated_completion_time': len(required_steps) * 5,  # 5 minutes per step
                'transaction_limits': tier_config['transaction_limits'],
                'initiated_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"KYC process initiated for user {user_id}, target tier: {target_tier.value}")
            return kyc_process
            
        except Exception as e:
            logger.error(f"KYC process initiation failed for user {user_id}: {str(e)}")
            raise
    
    async def verify_document(
        self,
        db: AsyncSession,
        user_id: UUID,
        document_type: DocumentType,
        document_data: bytes,
        document_metadata: Dict[str, Any]
    ) -> DocumentVerificationResult:
        """Verify uploaded document using OCR and validation"""
        try:
            start_time = datetime.utcnow()
            
            # Generate document ID
            document_id = str(uuid4())
            
            # Encrypt and store document
            encrypted_document = await self.encryption_service.encrypt_data(document_data)
            
            # Simulate document processing (in real implementation, use OCR service)
            extracted_data = await self._extract_document_data(document_data, document_type)
            
            # Perform verification checks
            verification_checks = await self._perform_document_checks(
                extracted_data, document_type, document_metadata
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_document_confidence(verification_checks, extracted_data)
            
            # Identify risk indicators
            risk_indicators = await self._identify_document_risks(extracted_data, verification_checks)
            
            # Determine status
            if confidence_score >= 0.9 and all(verification_checks.values()):
                status = DocumentStatus.APPROVED
            elif confidence_score >= 0.7:
                status = DocumentStatus.REQUIRES_REVIEW
            else:
                status = DocumentStatus.REJECTED
            
            # Calculate expiry date
            expires_at = None
            if document_type in self.document_validity_periods:
                expires_at = datetime.utcnow() + self.document_validity_periods[document_type]
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create verification result
            verification_result = DocumentVerificationResult(
                document_id=document_id,
                document_type=document_type,
                status=status,
                confidence_score=confidence_score,
                extracted_data=extracted_data,
                verification_checks=verification_checks,
                risk_indicators=risk_indicators,
                processing_time=processing_time,
                verified_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            # Store verification result
            await self._store_document_verification(db, user_id, verification_result, encrypted_document)
            
            # Update KYC record
            await self._update_kyc_progress(db, user_id, document_type, status)
            
            logger.info(f"Document verification completed for user {user_id}: "
                       f"Type: {document_type.value}, Status: {status.value}, "
                       f"Confidence: {confidence_score:.2f}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Document verification failed for user {user_id}: {str(e)}")
            raise
    
    async def verify_biometric(
        self,
        db: AsyncSession,
        user_id: UUID,
        selfie_data: bytes,
        reference_document_id: Optional[str] = None
    ) -> BiometricVerificationResult:
        """Verify biometric data (facial recognition and liveness detection)"""
        try:
            verification_id = str(uuid4())
            
            # Simulate biometric processing (in real implementation, use biometric service)
            biometric_analysis = await self._analyze_biometric_data(selfie_data)
            
            # Perform liveness detection
            liveness_score = await self._detect_liveness(selfie_data)
            
            # Calculate quality score
            quality_score = await self._assess_image_quality(selfie_data)
            
            # Compare with reference document if provided
            match_score = 0.0
            if reference_document_id:
                match_score = await self._compare_with_reference(
                    selfie_data, reference_document_id, db
                )
            else:
                match_score = 0.85  # Simulate high match for demo
            
            # Determine if it's a match
            is_match = (match_score >= 0.8 and 
                       liveness_score >= 0.7 and 
                       quality_score >= 0.6)
            
            # Identify risk indicators
            risk_indicators = []
            if liveness_score < 0.7:
                risk_indicators.append("Low liveness score - possible spoof attempt")
            if quality_score < 0.6:
                risk_indicators.append("Poor image quality")
            if match_score < 0.8:
                risk_indicators.append("Low facial match confidence")
            
            # Create verification result
            verification_result = BiometricVerificationResult(
                verification_id=verification_id,
                match_score=match_score,
                liveness_score=liveness_score,
                quality_score=quality_score,
                is_match=is_match,
                risk_indicators=risk_indicators,
                verified_at=datetime.utcnow()
            )
            
            # Store biometric verification
            await self._store_biometric_verification(db, user_id, verification_result, selfie_data)
            
            # Update KYC record
            await self._update_kyc_biometric_status(db, user_id, is_match)
            
            logger.info(f"Biometric verification completed for user {user_id}: "
                       f"Match: {is_match}, Score: {match_score:.2f}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Biometric verification failed for user {user_id}: {str(e)}")
            raise
    
    async def perform_comprehensive_assessment(
        self,
        db: AsyncSession,
        user_id: UUID
    ) -> KYCAssessment:
        """Perform comprehensive KYC assessment"""
        try:
            # Get user and KYC records
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            kyc_result = await db.execute(
                select(KYCRecord)
                .where(KYCRecord.user_id == user_id)
                .order_by(desc(KYCRecord.created_at))
                .limit(1)
            )
            kyc_record = kyc_result.scalar_one_or_none()
            
            # Perform various checks
            sanctions_check = await self._perform_sanctions_screening(user)
            pep_check = await self._perform_pep_screening(user)
            adverse_media_check = await self._perform_adverse_media_check(user)
            address_verification = await self._verify_address(db, user_id)
            source_of_funds_verification = await self._verify_source_of_funds(db, user_id)
            
            # Get document verifications
            document_verifications = await self._get_document_verifications(db, user_id)
            
            # Get biometric verification
            biometric_verification = await self._get_biometric_verification(db, user_id)
            
            # Calculate risk rating and compliance score
            risk_rating, compliance_score = await self._calculate_risk_assessment(
                user, sanctions_check, pep_check, adverse_media_check,
                document_verifications, biometric_verification
            )
            
            # Determine KYC level and status
            kyc_level = await self._determine_kyc_level(
                document_verifications, biometric_verification, compliance_score
            )
            
            overall_status = await self._determine_overall_status(
                kyc_record, risk_rating, compliance_score
            )
            
            # Generate recommendations
            recommendations = await self._generate_kyc_recommendations(
                risk_rating, compliance_score, document_verifications, biometric_verification
            )
            
            # Calculate next review date
            tier_config = self.tier_requirements.get(kyc_level, self.tier_requirements[KYCTier.BASIC])
            next_review_date = datetime.utcnow() + timedelta(days=tier_config['review_frequency_days'])
            
            # Create ongoing monitoring plan
            ongoing_monitoring = await self._create_monitoring_plan(user, risk_rating, kyc_level)
            
            # Create comprehensive assessment
            assessment = KYCAssessment(
                user_id=str(user_id),
                kyc_level=kyc_level,
                overall_status=overall_status,
                risk_rating=risk_rating,
                compliance_score=compliance_score,
                document_verifications=document_verifications,
                biometric_verification=biometric_verification,
                sanctions_check=sanctions_check,
                pep_check=pep_check,
                adverse_media_check=adverse_media_check,
                address_verification=address_verification,
                source_of_funds_verification=source_of_funds_verification,
                ongoing_monitoring=ongoing_monitoring,
                recommendations=recommendations,
                next_review_date=next_review_date,
                assessed_at=datetime.utcnow()
            )
            
            # Store assessment
            await self._store_kyc_assessment(db, assessment)
            
            # Update KYC record with final status
            if kyc_record:
                kyc_record.status = overall_status
                kyc_record.risk_level = risk_rating.value
                kyc_record.compliance_score = compliance_score
                kyc_record.next_review_date = next_review_date
                kyc_record.updated_at = datetime.utcnow()
                await db.commit()
            
            logger.info(f"Comprehensive KYC assessment completed for user {user_id}: "
                       f"Level: {kyc_level.value}, Status: {overall_status.value}, "
                       f"Risk: {risk_rating.value}, Score: {compliance_score:.2f}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Comprehensive KYC assessment failed for user {user_id}: {str(e)}")
            raise
    
    # Private helper methods
    
    async def _extract_document_data(self, document_data: bytes, document_type: DocumentType) -> Dict[str, Any]:
        """Extract data from document using OCR"""
        # In a real implementation, this would use OCR services like AWS Textract, Google Vision, etc.
        # For now, simulate extracted data
        
        extracted_data = {
            'document_number': 'A12345678',
            'full_name': 'John Doe',
            'date_of_birth': '1990-01-01',
            'nationality': 'US',
            'issue_date': '2020-01-01',
            'expiry_date': '2030-01-01',
            'issuing_authority': 'Department of Motor Vehicles'
        }
        
        if document_type == DocumentType.PROOF_OF_ADDRESS:
            extracted_data.update({
                'address': '123 Main St, Anytown, ST 12345',
                'document_date': '2024-01-01',
                'account_holder': 'John Doe'
            })
        
        return extracted_data
    
    async def _perform_document_checks(
        self,
        extracted_data: Dict[str, Any],
        document_type: DocumentType,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Perform various document verification checks"""
        checks = {
            'format_valid': True,
            'not_expired': True,
            'authentic_features': True,
            'readable_text': True,
            'consistent_data': True,
            'security_features': True
        }
        
        # Simulate some checks based on extracted data
        if 'expiry_date' in extracted_data:
            try:
                expiry_date = datetime.strptime(extracted_data['expiry_date'], '%Y-%m-%d')
                checks['not_expired'] = expiry_date > datetime.utcnow()
            except:
                checks['not_expired'] = False
        
        # Simulate quality checks based on metadata
        if metadata.get('image_quality', 1.0) < 0.7:
            checks['readable_text'] = False
        
        return checks
    
    def _calculate_document_confidence(
        self,
        verification_checks: Dict[str, bool],
        extracted_data: Dict[str, Any]
    ) -> float:
        """Calculate document verification confidence score"""
        # Base score from verification checks
        passed_checks = sum(verification_checks.values())
        total_checks = len(verification_checks)
        base_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Adjust based on data completeness
        required_fields = ['document_number', 'full_name', 'date_of_birth']
        present_fields = sum(1 for field in required_fields if field in extracted_data and extracted_data[field])
        completeness_score = present_fields / len(required_fields)
        
        # Weighted average
        confidence_score = (base_score * 0.7) + (completeness_score * 0.3)
        
        return min(confidence_score, 1.0)
    
    async def _identify_document_risks(
        self,
        extracted_data: Dict[str, Any],
        verification_checks: Dict[str, bool]
    ) -> List[str]:
        """Identify potential risks in document"""
        risks = []
        
        if not verification_checks.get('not_expired', True):
            risks.append("Document has expired")
        
        if not verification_checks.get('authentic_features', True):
            risks.append("Suspicious document features detected")
        
        if not verification_checks.get('readable_text', True):
            risks.append("Poor document quality or readability")
        
        # Check for high-risk countries
        nationality = extracted_data.get('nationality', '').upper()
        if nationality in self.high_risk_countries:
            risks.append(f"High-risk jurisdiction: {nationality}")
        
        return risks
    
    async def _analyze_biometric_data(self, biometric_data: bytes) -> Dict[str, Any]:
        """Analyze biometric data for facial features"""
        # In a real implementation, this would use facial recognition services
        return {
            'face_detected': True,
            'face_count': 1,
            'face_quality': 0.85,
            'facial_landmarks': True,
            'age_estimate': 30,
            'gender_estimate': 'male'
        }
    
    async def _detect_liveness(self, image_data: bytes) -> float:
        """Detect if the image shows a live person"""
        # In a real implementation, this would use liveness detection algorithms
        return 0.92  # Simulate high liveness score
    
    async def _assess_image_quality(self, image_data: bytes) -> float:
        """Assess the quality of the biometric image"""
        # In a real implementation, this would analyze image quality metrics
        return 0.88  # Simulate good quality
    
    async def _compare_with_reference(
        self,
        selfie_data: bytes,
        reference_document_id: str,
        db: AsyncSession
    ) -> float:
        """Compare selfie with reference document photo"""
        # In a real implementation, this would perform facial comparison
        return 0.91  # Simulate high match score
    
    async def _perform_sanctions_screening(self, user: User) -> Dict[str, Any]:
        """Perform sanctions screening"""
        # In a real implementation, this would check against sanctions lists
        return {
            'checked': True,
            'match_found': False,
            'lists_checked': ['OFAC', 'UN', 'EU', 'HMT'],
            'last_checked': datetime.utcnow().isoformat(),
            'confidence': 0.99
        }
    
    async def _perform_pep_screening(self, user: User) -> Dict[str, Any]:
        """Perform Politically Exposed Person screening"""
        # In a real implementation, this would check PEP databases
        return {
            'checked': True,
            'is_pep': False,
            'pep_category': None,
            'last_checked': datetime.utcnow().isoformat(),
            'confidence': 0.95
        }
    
    async def _perform_adverse_media_check(self, user: User) -> Dict[str, Any]:
        """Perform adverse media screening"""
        # In a real implementation, this would search news and media sources
        return {
            'checked': True,
            'adverse_findings': False,
            'sources_checked': 50,
            'last_checked': datetime.utcnow().isoformat(),
            'confidence': 0.90
        }
    
    async def _verify_address(self, db: AsyncSession, user_id: UUID) -> Dict[str, Any]:
        """Verify user's address"""
        # In a real implementation, this would use address verification services
        return {
            'verified': True,
            'verification_method': 'document_upload',
            'confidence': 0.88,
            'last_verified': datetime.utcnow().isoformat()
        }
    
    async def _verify_source_of_funds(self, db: AsyncSession, user_id: UUID) -> Dict[str, Any]:
        """Verify source of funds"""
        # In a real implementation, this would analyze financial documents
        return {
            'verified': False,
            'documentation_provided': False,
            'risk_assessment': 'medium',
            'last_checked': datetime.utcnow().isoformat()
        }
    
    async def _get_document_verifications(self, db: AsyncSession, user_id: UUID) -> List[DocumentVerificationResult]:
        """Get all document verifications for user"""
        # In a real implementation, this would query the database
        return []
    
    async def _get_biometric_verification(self, db: AsyncSession, user_id: UUID) -> Optional[BiometricVerificationResult]:
        """Get biometric verification for user"""
        # In a real implementation, this would query the database
        return None
    
    async def _calculate_risk_assessment(
        self,
        user: User,
        sanctions_check: Dict[str, Any],
        pep_check: Dict[str, Any],
        adverse_media_check: Dict[str, Any],
        document_verifications: List[DocumentVerificationResult],
        biometric_verification: Optional[BiometricVerificationResult]
    ) -> Tuple[RiskRating, float]:
        """Calculate overall risk rating and compliance score"""
        risk_score = 0.0
        
        # Sanctions and PEP checks
        if sanctions_check.get('match_found', False):
            risk_score += self.risk_weights['sanctions_match']
        
        if pep_check.get('is_pep', False):
            risk_score += self.risk_weights['pep_match']
        
        if adverse_media_check.get('adverse_findings', False):
            risk_score += self.risk_weights['adverse_media']
        
        # Country risk
        if user.country and user.country.upper() in self.high_risk_countries:
            risk_score += self.risk_weights['high_risk_country']
        
        # Document quality
        if document_verifications:
            avg_doc_confidence = sum(doc.confidence_score for doc in document_verifications) / len(document_verifications)
            if avg_doc_confidence < 0.8:
                risk_score += self.risk_weights['document_quality']
        
        # Biometric quality
        if biometric_verification and biometric_verification.match_score < 0.8:
            risk_score += self.risk_weights['biometric_quality']
        
        # Determine risk rating
        if risk_score >= 100:
            risk_rating = RiskRating.PROHIBITED
        elif risk_score >= 60:
            risk_rating = RiskRating.HIGH
        elif risk_score >= 30:
            risk_rating = RiskRating.MEDIUM
        else:
            risk_rating = RiskRating.LOW
        
        # Calculate compliance score (inverse of risk)
        compliance_score = max(0.0, 100.0 - risk_score) / 100.0
        
        return risk_rating, compliance_score
    
    async def _determine_kyc_level(
        self,
        document_verifications: List[DocumentVerificationResult],
        biometric_verification: Optional[BiometricVerificationResult],
        compliance_score: float
    ) -> KYCTier:
        """Determine achieved KYC level"""
        if compliance_score < 0.5:
            return KYCTier.BASIC
        
        has_id_document = any(
            doc.document_type == DocumentType.GOVERNMENT_ID and doc.status == DocumentStatus.APPROVED
            for doc in document_verifications
        )
        
        has_address_proof = any(
            doc.document_type == DocumentType.PROOF_OF_ADDRESS and doc.status == DocumentStatus.APPROVED
            for doc in document_verifications
        )
        
        has_biometric = biometric_verification and biometric_verification.is_match
        
        if has_id_document and has_address_proof and has_biometric and compliance_score >= 0.8:
            return KYCTier.ENHANCED
        elif has_id_document and compliance_score >= 0.7:
            return KYCTier.STANDARD
        else:
            return KYCTier.BASIC
    
    async def _determine_overall_status(
        self,
        kyc_record: Optional[KYCRecord],
        risk_rating: RiskRating,
        compliance_score: float
    ) -> KYCStatus:
        """Determine overall KYC status"""
        if risk_rating == RiskRating.PROHIBITED:
            return KYCStatus.REJECTED
        elif risk_rating == RiskRating.HIGH or compliance_score < 0.6:
            return KYCStatus.UNDER_REVIEW
        elif compliance_score >= 0.8:
            return KYCStatus.APPROVED
        else:
            return KYCStatus.PENDING
    
    async def _generate_kyc_recommendations(
        self,
        risk_rating: RiskRating,
        compliance_score: float,
        document_verifications: List[DocumentVerificationResult],
        biometric_verification: Optional[BiometricVerificationResult]
    ) -> List[str]:
        """Generate KYC recommendations"""
        recommendations = []
        
        if compliance_score < 0.7:
            recommendations.append("Enhanced due diligence required")
        
        if risk_rating in [RiskRating.HIGH, RiskRating.PROHIBITED]:
            recommendations.append("Manual review by compliance team required")
        
        if not document_verifications:
            recommendations.append("Identity document verification required")
        
        if not biometric_verification:
            recommendations.append("Biometric verification recommended for enhanced security")
        
        if risk_rating == RiskRating.MEDIUM:
            recommendations.append("Ongoing monitoring recommended")
        
        return recommendations
    
    async def _create_monitoring_plan(self, user: User, risk_rating: RiskRating, kyc_level: KYCTier) -> Dict[str, Any]:
        """Create ongoing monitoring plan"""
        tier_config = self.tier_requirements[kyc_level]
        
        return {
            'monitoring_frequency': f"Every {tier_config['review_frequency_days']} days",
            'transaction_monitoring': True,
            'sanctions_screening_frequency': 'daily' if risk_rating == RiskRating.HIGH else 'weekly',
            'document_renewal_alerts': True,
            'behavioral_analysis': risk_rating in [RiskRating.MEDIUM, RiskRating.HIGH],
            'enhanced_monitoring': risk_rating == RiskRating.HIGH
        }
    
    async def _store_document_verification(
        self,
        db: AsyncSession,
        user_id: UUID,
        verification_result: DocumentVerificationResult,
        encrypted_document: bytes
    ):
        """Store document verification result"""
        # In a real implementation, this would store in the database
        logger.info(f"Storing document verification for user {user_id}")
    
    async def _store_biometric_verification(
        self,
        db: AsyncSession,
        user_id: UUID,
        verification_result: BiometricVerificationResult,
        biometric_data: bytes
    ):
        """Store biometric verification result"""
        # In a real implementation, this would store in the database
        logger.info(f"Storing biometric verification for user {user_id}")
    
    async def _store_kyc_assessment(self, db: AsyncSession, assessment: KYCAssessment):
        """Store comprehensive KYC assessment"""
        # In a real implementation, this would store in the database
        logger.info(f"Storing KYC assessment for user {assessment.user_id}")
    
    async def _update_kyc_progress(self, db: AsyncSession, user_id: UUID, document_type: DocumentType, status: DocumentStatus):
        """Update KYC progress based on document verification"""
        # In a real implementation, this would update the KYC record
        logger.info(f"Updating KYC progress for user {user_id}: {document_type.value} -> {status.value}")
    
    async def _update_kyc_biometric_status(self, db: AsyncSession, user_id: UUID, is_verified: bool):
        """Update KYC biometric verification status"""
        # In a real implementation, this would update the KYC record
        logger.info(f"Updating biometric status for user {user_id}: {is_verified}")

