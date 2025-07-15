"""
Authentication models for DuoPet AI Service

This module defines data models for API key management and authentication.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, validator
from bson import ObjectId


class APIKeyStatus(str, Enum):
    """API key status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKeyScope(str, Enum):
    """API key permission scopes"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    
    # Service-specific scopes
    FACE_LOGIN = "face_login"
    CHATBOT = "chatbot"
    VIDEO_RECOMMEND = "video_recommend"
    HEALTH_DIAGNOSIS = "health_diagnosis"
    BEHAVIOR_ANALYSIS = "behavior_analysis"


class APIKeyModel(BaseModel):
    """API key database model"""
    id: Optional[str] = Field(default=None, alias="_id")
    key_id: str = Field(..., description="Unique key identifier (public)")
    key_hash: str = Field(..., description="Hashed API key (private)")
    name: str = Field(..., description="Descriptive name for the key")
    description: Optional[str] = Field(None, description="Key description")
    
    # Ownership
    user_id: str = Field(..., description="User or service ID that owns this key")
    organization_id: Optional[str] = Field(None, description="Organization ID if applicable")
    
    # Permissions
    scopes: List[APIKeyScope] = Field(
        default=[APIKeyScope.READ], 
        description="Permission scopes"
    )
    allowed_ips: Optional[List[str]] = Field(
        None, 
        description="IP whitelist (None means all IPs allowed)"
    )
    allowed_origins: Optional[List[str]] = Field(
        None,
        description="Allowed origins for CORS"
    )
    
    # Status and lifecycle
    status: APIKeyStatus = Field(
        default=APIKeyStatus.ACTIVE,
        description="Current status of the key"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration timestamp (None means never expires)"
    )
    last_used_at: Optional[datetime] = Field(
        None,
        description="Last usage timestamp"
    )
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Total number of uses")
    rate_limit: Optional[int] = Field(
        None,
        description="Custom rate limit (requests per minute)"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @validator("id", pre=True)
    def validate_id(cls, v):
        """Convert ObjectId to string"""
        if isinstance(v, ObjectId):
            return str(v)
        return v
    
    @validator("expires_at")
    def validate_expiration(cls, v, values):
        """Validate expiration date"""
        if v and v <= datetime.utcnow():
            values["status"] = APIKeyStatus.EXPIRED
        return v
    
    def is_valid(self) -> bool:
        """Check if API key is currently valid"""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        
        if self.expires_at and self.expires_at <= datetime.utcnow():
            return False
        
        return True
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if key has specific scope"""
        return scope in self.scopes or APIKeyScope.ADMIN in self.scopes
    
    def has_any_scope(self, scopes: List[APIKeyScope]) -> bool:
        """Check if key has any of the specified scopes"""
        if APIKeyScope.ADMIN in self.scopes:
            return True
        
        return any(scope in self.scopes for scope in scopes)
    
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP address is allowed"""
        if not self.allowed_ips:
            return True
        
        return ip in self.allowed_ips
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not self.allowed_origins:
            return True
        
        return origin in self.allowed_origins
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }
        allow_population_by_field_name = True


class APIKeyCreate(BaseModel):
    """Model for creating a new API key"""
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    scopes: List[APIKeyScope] = Field(default=[APIKeyScope.READ])
    allowed_ips: Optional[List[str]] = None
    allowed_origins: Optional[List[str]] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("scopes")
    def validate_scopes(cls, v):
        """Ensure at least one scope is specified"""
        if not v:
            raise ValueError("At least one scope must be specified")
        return v


class APIKeyUpdate(BaseModel):
    """Model for updating an API key"""
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    scopes: Optional[List[APIKeyScope]] = None
    allowed_ips: Optional[List[str]] = None
    allowed_origins: Optional[List[str]] = None
    status: Optional[APIKeyStatus] = None
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    metadata: Optional[Dict[str, Any]] = None


class APIKeyResponse(BaseModel):
    """Model for API key response (without sensitive data)"""
    key_id: str
    name: str
    description: Optional[str]
    scopes: List[APIKeyScope]
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: Optional[int]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyWithSecret(APIKeyResponse):
    """Model for API key response with secret (only shown once)"""
    api_key: str = Field(..., description="The actual API key (show only once)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "key_abc123",
                "api_key": "sk_live_abcdefghijklmnopqrstuvwxyz123456",
                "name": "Production API Key",
                "scopes": ["read", "write"],
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }


class APIKeyValidation(BaseModel):
    """Model for API key validation result"""
    valid: bool
    key_id: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[APIKeyScope] = []
    rate_limit: Optional[int] = None
    metadata: Dict[str, Any] = {}
    reason: Optional[str] = None