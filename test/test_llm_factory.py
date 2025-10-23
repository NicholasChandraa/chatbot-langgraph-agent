"""
Test LLM Provider Factory
"""
import asyncio
from app.database.connection.connection import get_db, db_manager
from app.config.agent_config.agent_config_manager import get_agent_config
from app.llm.provider_factory import LLMProviderFactory

async def test_create_from_db():
    """Test creating LLM from database config"""
    
    # Initialize database first
    print("🔧 Initializing database...")
    db_manager.init()
    
    # Check database health
    is_healthy = await db_manager.health_check()
    if is_healthy:
        print("✅ Database connection verified")
    else:
        print("❌ Database connection failed")
        return

    # Get database session (get_db is a generator, not context manager)
    async for db in get_db():
        # Load config from database
        config = await get_agent_config("supervisor", db)
        print(f"✅ Loaded config: {config}\n")

        # Create LLM
        print("🔧 Creating LLM from config...")
        llm = LLMProviderFactory.create_from_config(config)
        print(f"✅ LLM created: {llm}\n")

        # Test invoke
        print("🤖 Testing LLM invoke...")
        response = await llm.ainvoke("Hello! Who are you?")
        print(f"✅ Response received!")
        print(f"🤖 Response: {response.content}\n")
        
        # Only process one session
        break
    
    # Cleanup
    print("\n🧹 Closing database connections...")
    await db_manager.close()
    print("✅ Database connections closed")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing LLM Provider Factory")
    print("=" * 60)
    
    asyncio.run(test_create_from_db())
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)