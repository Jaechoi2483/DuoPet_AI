"""
Test API Authentication System
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from common.auth import (
    APIKeyService,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyStatus,
    APIKeyScope
)
from common.database import get_database, get_redis_client


@pytest.mark.asyncio
async def test_api_key_lifecycle():
    """Test creating, validating, updating, and revoking API keys"""
    # Get database connections
    db = await get_database()
    redis_client = await get_redis_client()
    service = APIKeyService(db, redis_client)
    
    # Test user ID
    test_user_id = "test_user_123"
    
    # 1. Create an API key
    key_data = APIKeyCreate(
        name="Test API Key",
        description="Key for testing",
        scopes=[APIKeyScope.READ, APIKeyScope.WRITE],
        allowed_ips=["127.0.0.1", "192.168.1.0/24"],
        allowed_origins=["http://localhost:3000"],
        expires_in_days=30,
        rate_limit=100,
        metadata={"purpose": "testing"}
    )
    
    created_key = await service.create_api_key(
        user_id=test_user_id,
        key_data=key_data
    )
    
    assert created_key.key_id.startswith("key_")
    assert created_key.api_key.startswith("sk_live_")
    assert created_key.name == "Test API Key"
    assert APIKeyScope.READ in created_key.scopes
    assert APIKeyScope.WRITE in created_key.scopes
    
    # 2. Validate the API key
    validation = await service.validate_api_key(
        api_key=created_key.api_key,
        client_ip="127.0.0.1",
        origin="http://localhost:3000"
    )
    
    assert validation.valid == True
    assert validation.key_id == created_key.key_id
    assert validation.user_id == test_user_id
    assert validation.rate_limit == 100
    
    # 3. Test invalid IP
    validation_bad_ip = await service.validate_api_key(
        api_key=created_key.api_key,
        client_ip="10.0.0.1"  # Not in allowed list
    )
    
    assert validation_bad_ip.valid == False
    assert "not allowed" in validation_bad_ip.reason
    
    # 4. Test scope validation
    validation_with_scope = await service.validate_api_key(
        api_key=created_key.api_key,
        required_scopes=[APIKeyScope.READ],
        client_ip="127.0.0.1"
    )
    
    assert validation_with_scope.valid == True
    
    validation_missing_scope = await service.validate_api_key(
        api_key=created_key.api_key,
        required_scopes=[APIKeyScope.ADMIN],  # Key doesn't have this
        client_ip="127.0.0.1"
    )
    
    assert validation_missing_scope.valid == False
    assert "Missing required scopes" in validation_missing_scope.reason
    
    # 5. Update the key
    update_data = APIKeyUpdate(
        name="Updated Test Key",
        scopes=[APIKeyScope.READ],  # Remove WRITE scope
        rate_limit=50
    )
    
    updated_key = await service.update_api_key(
        key_id=created_key.key_id,
        user_id=test_user_id,
        update_data=update_data
    )
    
    assert updated_key.name == "Updated Test Key"
    assert updated_key.rate_limit == 50
    assert APIKeyScope.WRITE not in updated_key.scopes
    
    # 6. List user's keys
    user_keys = await service.list_user_keys(test_user_id)
    assert len(user_keys) >= 1
    assert any(k.key_id == created_key.key_id for k in user_keys)
    
    # 7. Revoke the key
    success = await service.revoke_api_key(
        key_id=created_key.key_id,
        user_id=test_user_id
    )
    
    assert success == True
    
    # 8. Validate revoked key
    validation_revoked = await service.validate_api_key(
        api_key=created_key.api_key,
        client_ip="127.0.0.1"
    )
    
    assert validation_revoked.valid == False
    assert "revoked" in validation_revoked.reason.lower()
    
    print("✅ All API key tests passed!")


@pytest.mark.asyncio
async def test_api_key_expiration():
    """Test API key expiration"""
    db = await get_database()
    redis_client = await get_redis_client()
    service = APIKeyService(db, redis_client)
    
    # Create key that expires in 1 day
    key_data = APIKeyCreate(
        name="Expiring Key",
        description="Key that will expire",
        scopes=[APIKeyScope.READ],
        expires_in_days=1
    )
    
    created_key = await service.create_api_key(
        user_id="test_user_exp",
        key_data=key_data
    )
    
    assert created_key.expires_at is not None
    assert created_key.expires_at > datetime.utcnow()
    assert created_key.expires_at < datetime.utcnow() + timedelta(days=2)
    
    print("✅ API key expiration test passed!")


if __name__ == "__main__":
    asyncio.run(test_api_key_lifecycle())
    asyncio.run(test_api_key_expiration())