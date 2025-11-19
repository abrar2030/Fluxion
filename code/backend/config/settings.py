"""
Comprehensive settings configuration for Fluxion backend
"""

from typing import List, Optional

from pydantic import BaseSettings, Field, PostgresDsn, RedisDsn, validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings"""

    APP_NAME: str = Field(default="Fluxion API", env="APP_NAME")
    APP_VERSION: str = Field(default="2.0.0", env="APP_VERSION")
    APP_DESCRIPTION: str = Field(
        default="Production-ready DeFi Supply Chain Platform", env="APP_DESCRIPTION"
    )
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")

    # API Configuration
    API_V1_PREFIX: str = Field(default="/api/v1", env="API_V1_PREFIX")
    DOCS_URL: Optional[str] = Field(default="/docs", env="DOCS_URL")
    REDOC_URL: Optional[str] = Field(default="/redoc", env="REDOC_URL")

    @validator("WORKERS")
    def validate_workers(cls, v, values):
        if values.get("ENVIRONMENT") == "production" and v < 2:
            return 4
        return v


class DatabaseSettings(BaseSettings):
    """Database settings"""

    DATABASE_URL: PostgresDsn = Field(..., env="DATABASE_URL")
    DATABASE_READ_URL: Optional[PostgresDsn] = Field(None, env="DATABASE_READ_URL")
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=30, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    DB_POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")


class RedisSettings(BaseSettings):
    """Redis settings"""

    REDIS_URL: RedisDsn = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    SESSION_TTL: int = Field(default=86400, env="SESSION_TTL")  # 24 hours


class SecuritySettings(BaseSettings):
    """Security settings"""

    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")

    # Password security
    PASSWORD_MIN_LENGTH: int = Field(default=8, env="PASSWORD_MIN_LENGTH")
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(
        default=True, env="PASSWORD_REQUIRE_UPPERCASE"
    )
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(
        default=True, env="PASSWORD_REQUIRE_LOWERCASE"
    )
    PASSWORD_REQUIRE_NUMBERS: bool = Field(default=True, env="PASSWORD_REQUIRE_NUMBERS")
    PASSWORD_REQUIRE_SPECIAL: bool = Field(default=True, env="PASSWORD_REQUIRE_SPECIAL")

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=100, env="RATE_LIMIT_BURST")

    # CORS
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # Encryption
    ENCRYPTION_KEY: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    FIELD_ENCRYPTION_ENABLED: bool = Field(default=True, env="FIELD_ENCRYPTION_ENABLED")

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v


class BlockchainSettings(BaseSettings):
    """Blockchain settings"""

    # Ethereum
    ETH_RPC_URL: str = Field(..., env="ETH_RPC_URL")
    ETH_WEBSOCKET_URL: Optional[str] = Field(None, env="ETH_WEBSOCKET_URL")
    ETH_CHAIN_ID: int = Field(default=1, env="ETH_CHAIN_ID")

    # Polygon
    POLYGON_RPC_URL: Optional[str] = Field(None, env="POLYGON_RPC_URL")
    POLYGON_CHAIN_ID: int = Field(default=137, env="POLYGON_CHAIN_ID")

    # BSC
    BSC_RPC_URL: Optional[str] = Field(None, env="BSC_RPC_URL")
    BSC_CHAIN_ID: int = Field(default=56, env="BSC_CHAIN_ID")

    # Gas settings
    GAS_PRICE_STRATEGY: str = Field(default="medium", env="GAS_PRICE_STRATEGY")
    MAX_GAS_PRICE: int = Field(default=100, env="MAX_GAS_PRICE")  # in gwei

    # Contract addresses
    SUPPLY_CHAIN_ADDRESS: Optional[str] = Field(None, env="SUPPLY_CHAIN_ADDRESS")
    LIQUIDITY_POOL_ADDRESS: Optional[str] = Field(None, env="LIQUIDITY_POOL_ADDRESS")
    SYNTHETIC_ASSET_ADDRESS: Optional[str] = Field(None, env="SYNTHETIC_ASSET_ADDRESS")
    GOVERNANCE_TOKEN_ADDRESS: Optional[str] = Field(
        None, env="GOVERNANCE_TOKEN_ADDRESS"
    )
    ASSET_VAULT_ADDRESS: Optional[str] = Field(None, env="ASSET_VAULT_ADDRESS")

    # API Keys
    ETHERSCAN_API_KEY: Optional[str] = Field(None, env="ETHERSCAN_API_KEY")
    POLYGONSCAN_API_KEY: Optional[str] = Field(None, env="POLYGONSCAN_API_KEY")


class ComplianceSettings(BaseSettings):
    """Compliance settings"""

    # KYC
    KYC_ENABLED: bool = Field(default=True, env="KYC_ENABLED")
    KYC_PROVIDER: str = Field(default="jumio", env="KYC_PROVIDER")
    KYC_API_KEY: Optional[str] = Field(None, env="KYC_API_KEY")
    KYC_API_SECRET: Optional[str] = Field(None, env="KYC_API_SECRET")

    # AML
    AML_ENABLED: bool = Field(default=True, env="AML_ENABLED")
    AML_PROVIDER: str = Field(default="chainalysis", env="AML_PROVIDER")
    AML_API_KEY: Optional[str] = Field(None, env="AML_API_KEY")

    # Transaction monitoring
    TRANSACTION_MONITORING_ENABLED: bool = Field(
        default=True, env="TRANSACTION_MONITORING_ENABLED"
    )
    SUSPICIOUS_AMOUNT_THRESHOLD: float = Field(
        default=10000.0, env="SUSPICIOUS_AMOUNT_THRESHOLD"
    )
    DAILY_TRANSACTION_LIMIT: float = Field(
        default=50000.0, env="DAILY_TRANSACTION_LIMIT"
    )

    # Regulatory reporting
    REGULATORY_REPORTING_ENABLED: bool = Field(
        default=True, env="REGULATORY_REPORTING_ENABLED"
    )
    AUDIT_LOG_RETENTION_DAYS: int = Field(
        default=2555, env="AUDIT_LOG_RETENTION_DAYS"
    )  # 7 years


class MonitoringSettings(BaseSettings):
    """Monitoring and logging settings"""

    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Sentry
    SENTRY_DSN: Optional[str] = Field(None, env="SENTRY_DSN")
    SENTRY_ENVIRONMENT: Optional[str] = Field(None, env="SENTRY_ENVIRONMENT")


class ExternalServicesSettings(BaseSettings):
    """External services settings"""

    NOTIFICATION_SERVICE_URL: Optional[str] = Field(
        None, env="NOTIFICATION_SERVICE_URL"
    )
    ANALYTICS_SERVICE_URL: Optional[str] = Field(None, env="ANALYTICS_SERVICE_URL")


class Settings(BaseSettings):
    """Main settings class combining all setting groups"""

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    blockchain: BlockchainSettings = BlockchainSettings()
    compliance: ComplianceSettings = ComplianceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    external: ExternalServicesSettings = ExternalServicesSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings
