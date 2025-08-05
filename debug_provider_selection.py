#!/usr/bin/env python3
"""
Debug script to help users diagnose provider selection issues.

This script tests:
1. API key validity 
2. Provider selection logic
3. User settings propagation
4. Environment variable state

Run this script to debug why real conversations aren't happening.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/home/green/freeagentics')

# Get dev user ID at module level
try:
    from auth.dev_bypass import _DEV_USER
    DEV_USER_ID = _DEV_USER.user_id
except ImportError:
    DEV_USER_ID = "dev_user"

def test_api_key_directly():
    """Test the user's API key directly with OpenAI."""
    print("🔑 Testing API key directly...")
    
    # Try to get from various sources
    api_key = None
    source = None
    
    # Check environment first
    if os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        source = "environment variable"
    
    # Check user settings
    if not api_key:
        try:
            from database.models import UserSettings
            from database.session import SessionLocal
            
            db = SessionLocal()
            user_settings = db.query(UserSettings).filter(UserSettings.user_id == DEV_USER_ID).first()
            if user_settings and user_settings.get_openai_key():
                api_key = user_settings.get_openai_key()
                source = "user settings database"
            db.close()
        except Exception as e:
            print(f"❌ Could not check user settings: {e}")
    
    if not api_key:
        print("❌ No OpenAI API key found in environment or user settings")
        return False
    
    print(f"✅ Found API key from {source}: {api_key[:15]}...{api_key[-4:]}")
    
    # Test the key
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        print("🧪 Testing API key with simple completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API key works'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ API key is valid! Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_provider_factories():
    """Test different provider factory paths."""
    print("\n🏭 Testing provider factories...")
    
    # Test core.providers.get_llm()
    try:
        from core.providers import get_llm
        
        print("Testing core.providers.get_llm()...")
        provider = get_llm(user_id=DEV_USER_ID)
        print(f"✅ core.providers.get_llm() returned: {type(provider).__name__}")
        
        # Test if it's mock or real
        if "Mock" in type(provider).__name__:
            print("⚠️  WARNING: get_llm() returned a Mock provider!")
        else:
            print("✅ get_llm() returned a real provider")
            
    except Exception as e:
        print(f"❌ core.providers.get_llm() failed: {e}")
    
    # Test inference.llm.provider_factory
    try:
        from inference.llm.provider_factory import LLMProviderFactory
        
        print("Testing inference.llm.provider_factory.LLMProviderFactory...")
        factory = LLMProviderFactory()
        provider_manager = factory.create_from_config(user_id=DEV_USER_ID)
        healthy_providers = provider_manager.registry.get_healthy_providers()
        
        print(f"✅ LLMProviderFactory created provider_manager: {type(provider_manager).__name__}")
        print(f"✅ Healthy providers: {[getattr(p, 'name', type(p).__name__) for p in healthy_providers] if healthy_providers else 'NONE'}")
        
        if not healthy_providers:
            print("⚠️  WARNING: No healthy providers found!")
        elif "Mock" in str(healthy_providers[0]):
            print("⚠️  WARNING: First provider is Mock!")
        else:
            print("✅ First provider appears to be real")
            
    except Exception as e:
        print(f"❌ inference.llm.provider_factory test failed: {e}")

def test_user_settings():
    """Test user settings persistence and retrieval."""
    print("\n💾 Testing user settings...")
    
    try:
        from database.models import UserSettings
        from database.session import SessionLocal
        
        db = SessionLocal()
        user_settings = db.query(UserSettings).filter(UserSettings.user_id == DEV_USER_ID).first()
        
        if not user_settings:
            print("❌ No user settings found in database")
            return
        
        print(f"✅ User settings found for user: {DEV_USER_ID}")
        print(f"   Provider: {user_settings.llm_provider}")
        print(f"   Model: {user_settings.llm_model}")
        print(f"   Has OpenAI key: {bool(user_settings.encrypted_openai_key)}")
        print(f"   Has Anthropic key: {bool(user_settings.encrypted_anthropic_key)}")
        
        # Test decryption
        if user_settings.encrypted_openai_key:
            decrypted_key = user_settings.get_openai_key()
            if decrypted_key:
                print(f"✅ OpenAI key decryption successful: {decrypted_key[:15]}...{decrypted_key[-4:]}")
            else:
                print("❌ OpenAI key decryption failed")
        
        db.close()
        
    except Exception as e:
        print(f"❌ User settings test failed: {e}")

def test_environment_state():
    """Test current environment variable state."""
    print("\n📊 Testing environment state...")
    
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "LLM_PROVIDER",
        "DATABASE_URL",
        "PRODUCTION"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys
            if "API_KEY" in var and len(value) > 10:
                display_value = f"{value[:10]}...{value[-4:]}"
            else:
                display_value = value
            print(f"✅ {var}={display_value}")
        else:
            print(f"❌ {var}=UNSET")

def provide_debugging_commands():
    """Provide user with debugging commands they can run."""
    print("\n🔧 Debugging Commands for Browser Console:")
    print("""
// 1. Check current settings
fetch('/api/v1/settings')
  .then(r => r.json())
  .then(d => console.log('Settings:', d));

// 2. Test API key validation
fetch('/api/v1/settings/validate-key', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    provider: 'openai',
    api_key: 'YOUR_API_KEY_HERE'
  })
}).then(r => r.json()).then(d => console.log('Validation:', d));

// 3. Check WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/agent-conversation');
ws.onopen = () => console.log('WebSocket connected');
ws.onerror = (e) => console.log('WebSocket error:', e);
ws.onmessage = (e) => console.log('WebSocket message:', e.data);
    """)
    
    print("\n🔧 Backend Log Patterns to Look For:")
    print("""
# Look for these patterns in backend logs when starting a conversation:
grep "🔍 PROVIDER DEBUG" logs/app.log
grep "🏭 LLMProviderFactory" logs/app.log  
grep "💾 User settings" logs/app.log
grep "📊 Environment state" logs/app.log

# Or run the backend in debug mode:
LOG_LEVEL=DEBUG python -m uvicorn api.main:app --reload
    """)

async def main():
    """Run all diagnostic tests."""
    print("🚀 FreeAgentics Provider Selection Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Direct API key test
    api_key_works = test_api_key_directly()
    
    # Test 2: Provider factories
    test_provider_factories()
    
    # Test 3: User settings
    test_user_settings()
    
    # Test 4: Environment state
    test_environment_state()
    
    # Test 5: Debugging commands
    provide_debugging_commands()
    
    print("\n" + "=" * 50)
    if api_key_works:
        print("✅ Your API key is valid - the issue is in provider selection logic")
        print("🔍 Check the backend logs when creating a conversation to see which provider path is taken")
    else:
        print("❌ Your API key is not working - fix this first")
        print("💡 Make sure you've saved your API key in the FreeAgentics settings UI")

if __name__ == "__main__":
    asyncio.run(main())