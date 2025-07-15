#!/usr/bin/env python3
"""
Create Initial Admin API Key

This script creates the first admin API key for bootstrapping the system.
Run this once after setting up the database.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from common.auth import APIKeyService, APIKeyCreate, APIKeyScope
from common.database import get_database, get_redis_client, close_connections


async def create_admin_key():
    """Create an admin API key"""
    print("🔐 Creating Admin API Key for DuoPet AI Service")
    print("=" * 50)
    
    try:
        # Connect to database
        print("\nConnecting to database...")
        db = await get_database()
        redis_client = await get_redis_client()
        
        # Create service
        service = APIKeyService(db, redis_client)
        
        # Create admin key
        print("\nCreating admin API key...")
        key_data = APIKeyCreate(
            name="Admin Key",
            description="Master admin key for system administration",
            scopes=[APIKeyScope.ADMIN],  # Admin has all permissions
            rate_limit=1000,  # Higher rate limit for admin
            metadata={
                "purpose": "system_administration",
                "created_by": "bootstrap_script"
            }
        )
        
        # Create the key
        admin_key = await service.create_api_key(
            user_id="admin",
            key_data=key_data,
            organization_id="duopet"
        )
        
        print("\n✅ Admin API Key Created Successfully!")
        print("\n" + "=" * 50)
        print("IMPORTANT: Save this key securely - it won't be shown again!")
        print("=" * 50)
        print(f"\nAPI Key: {admin_key.api_key}")
        print(f"Key ID: {admin_key.key_id}")
        print(f"Scopes: {admin_key.scopes}")
        print("\n" + "=" * 50)
        
        # Save to file for convenience (delete after copying!)
        with open("admin_api_key.txt", "w") as f:
            f.write(f"API Key: {admin_key.api_key}\n")
            f.write(f"Key ID: {admin_key.key_id}\n")
            f.write(f"Created: {admin_key.created_at}\n")
        
        print("\n📝 Key also saved to admin_api_key.txt (DELETE THIS FILE AFTER COPYING!)")
        
    except Exception as e:
        print(f"\n❌ Error creating admin key: {str(e)}")
        print("\nMake sure:")
        print("1. MongoDB is running")
        print("2. Environment variables are set (.env file)")
        print("3. Database connection settings are correct")
        
    finally:
        # Close connections
        await close_connections()


async def list_existing_admin_keys():
    """List existing admin keys"""
    print("\n📋 Checking for existing admin keys...")
    
    try:
        db = await get_database()
        redis_client = await get_redis_client()
        service = APIKeyService(db, redis_client)
        
        # List admin user's keys
        keys = await service.list_user_keys("admin", include_inactive=True)
        
        if keys:
            print(f"\nFound {len(keys)} existing admin keys:")
            for key in keys:
                print(f"  - {key.name} ({key.key_id})")
                print(f"    Status: {key.status}, Created: {key.created_at}")
                print(f"    Scopes: {key.scopes}")
        else:
            print("\nNo existing admin keys found.")
        
        return len(keys) > 0
        
    except Exception as e:
        print(f"Error checking existing keys: {str(e)}")
        return False
    finally:
        await close_connections()


async def main():
    """Main function"""
    # Check for existing admin keys
    has_admin_keys = await list_existing_admin_keys()
    
    if has_admin_keys:
        print("\n⚠️  Admin keys already exist!")
        response = input("\nDo you want to create another admin key? (y/N): ")
        if response.lower() != 'y':
            print("Exiting without creating new key.")
            return
    
    # Create new admin key
    await create_admin_key()


if __name__ == "__main__":
    asyncio.run(main())