#!/usr/bin/env python3
"""
Test API Authentication Flow

This script demonstrates the complete API authentication flow:
1. Create an API key
2. Use the key to make authenticated requests
3. Test various authentication scenarios
"""

import asyncio
import httpx
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
ADMIN_API_KEY = "sk_live_ADMIN_KEY_HERE"  # Replace with actual admin key


async def create_api_key(client: httpx.AsyncClient, admin_key: str) -> Optional[str]:
    """Create a new API key"""
    print("\n🔑 Creating new API key...")
    
    response = await client.post(
        f"{BASE_URL}/auth/keys",
        headers={"X-API-Key": admin_key},
        json={
            "name": "Test Integration Key",
            "description": "Key for testing API integration",
            "scopes": ["read", "write", "face_login", "chatbot"],
            "rate_limit": 100
        }
    )
    
    if response.status_code == 201:
        data = response.json()
        api_key = data["data"]["api_key"]
        key_id = data["data"]["key_id"]
        print(f"✅ Created API key: {key_id}")
        print(f"   API Key: {api_key}")
        return api_key
    else:
        print(f"❌ Failed to create API key: {response.status_code}")
        print(f"   Response: {response.text}")
        return None


async def test_auth_endpoints(client: httpx.AsyncClient, api_key: str):
    """Test various authenticated endpoints"""
    print("\n🧪 Testing authenticated endpoints...")
    
    # Test 1: Get current key info
    print("\n1️⃣ Get current API key info:")
    response = await client.get(
        f"{BASE_URL}/auth/me",
        headers={"X-API-Key": api_key}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()["data"]
        print(f"   Key ID: {data['key_id']}")
        print(f"   Scopes: {data['scopes']}")
        print(f"   Rate Limit: {data['rate_limit']}")
    
    # Test 2: Try face login endpoint
    print("\n2️⃣ Test face login endpoint:")
    response = await client.get(
        f"{BASE_URL}/face-login/status",
        headers={"X-API-Key": api_key}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Access granted to face login")
    
    # Test 3: Try without API key
    print("\n3️⃣ Test without API key:")
    response = await client.get(f"{BASE_URL}/face-login/status")
    print(f"   Status: {response.status_code}")
    if response.status_code == 401:
        print("   ✅ Correctly rejected unauthorized request")
    
    # Test 4: Test rate limiting
    print("\n4️⃣ Test rate limiting (making 5 rapid requests):")
    for i in range(5):
        response = await client.get(
            f"{BASE_URL}/auth/me",
            headers={"X-API-Key": api_key}
        )
        print(f"   Request {i+1}: Status {response.status_code}")
        if "X-RateLimit-Remaining" in response.headers:
            print(f"   Rate limit remaining: {response.headers['X-RateLimit-Remaining']}")


async def test_scope_restrictions(client: httpx.AsyncClient, api_key: str):
    """Test scope-based access control"""
    print("\n🔒 Testing scope restrictions...")
    
    # This key doesn't have admin scope, so should fail
    print("\n1️⃣ Try to create another API key (requires admin):")
    response = await client.post(
        f"{BASE_URL}/auth/keys",
        headers={"X-API-Key": api_key},
        json={
            "name": "Should Fail",
            "description": "This should fail",
            "scopes": ["read"]
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 403:
        print("   ✅ Correctly denied due to insufficient permissions")


async def list_api_keys(client: httpx.AsyncClient, api_key: str):
    """List all API keys for the user"""
    print("\n📋 Listing API keys...")
    
    response = await client.get(
        f"{BASE_URL}/auth/keys",
        headers={"X-API-Key": api_key}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        print(f"   Found {len(data)} API keys")
        for key in data:
            print(f"   - {key['name']} ({key['key_id']})")
            print(f"     Status: {key['status']}, Scopes: {key['scopes']}")


async def main():
    """Run all tests"""
    print("🚀 DuoPet AI API Authentication Test")
    print("====================================")
    
    async with httpx.AsyncClient() as client:
        # First check if the API is running
        try:
            response = await client.get(f"{BASE_URL.replace('/api/v1', '')}/health")
            if response.status_code != 200:
                print("❌ API is not responding. Please start the server first.")
                return
        except httpx.ConnectError:
            print("❌ Cannot connect to API. Please start the server first.")
            print(f"   Run: cd /mnt/d/final_project/DuoPet_AI && python api/main.py")
            return
        
        # Create a test API key (requires admin key)
        if ADMIN_API_KEY == "sk_live_ADMIN_KEY_HERE":
            print("\n⚠️  Please set ADMIN_API_KEY in this script first!")
            print("   You need to manually create an admin API key in the database.")
            return
        
        # Create test key
        test_api_key = await create_api_key(client, ADMIN_API_KEY)
        if not test_api_key:
            return
        
        # Run tests
        await test_auth_endpoints(client, test_api_key)
        await test_scope_restrictions(client, test_api_key)
        await list_api_keys(client, test_api_key)
        
        print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())