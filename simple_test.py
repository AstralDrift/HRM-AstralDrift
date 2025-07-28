#!/usr/bin/env python3
"""
Simple test for HRM multi-agent basic functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path  
sys.path.append(str(Path(__file__).parent))

from agent_runtime.swe_rex_integration import (
    HRMAgentRuntime,
    AgentType,
    AgentCapabilities,
    AgentTask
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-test")

async def test_single_agent():
    """Test single agent functionality."""
    logger.info("🧪 Testing single HRM agent")
    
    # Create agent capabilities
    capabilities = AgentCapabilities(
        agent_type=AgentType.LANGUAGE_PYTHON,
        languages=["python"],
        max_concurrent_tasks=3
    )
    
    # Create agent
    agent = HRMAgentRuntime(
        agent_id="test_python_agent",
        agent_type=AgentType.LANGUAGE_PYTHON,
        capabilities=capabilities,
        deployment_config={"type": "local"},
        model_path=None
    )
    
    try:
        # Initialize agent
        logger.info("Initializing agent...")
        try:
            success = await asyncio.wait_for(agent.initialize(), timeout=5.0)
            assert success, "Failed to initialize agent"
            logger.info("✅ Agent initialized")
        except asyncio.TimeoutError:
            logger.error("❌ Agent initialization timed out")
            return False
        
        # Check if agent is active
        assert agent.is_active, "Agent not active"
        logger.info("✅ Agent is active")
        
        # Health check
        logger.info("Running health check...")
        try:
            health = await asyncio.wait_for(agent.health_check(), timeout=5.0)
            assert health["is_active"], "Health check shows agent not active"
            logger.info("✅ Health check passed")
        except asyncio.TimeoutError:
            logger.error("❌ Health check timed out")
            return False
        
        # Create test task
        task = AgentTask(
            task_id="test_task_001",
            agent_type=AgentType.LANGUAGE_PYTHON,
            task_data={
                "problem_statement": "Write a hello world function",
                "language": "python"
            },
            language="python"
        )
        
        # Execute task
        logger.info("Executing test task...")
        result = await agent.execute_task(task)
        assert result["success"], "Task execution failed"
        logger.info("✅ Task executed successfully")
        logger.info(f"   Result: {result}")
        
        # Shutdown agent
        await agent.shutdown()
        logger.info("✅ Agent shutdown complete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Run simple test."""
    logger.info("🚀 Starting simple HRM agent test")
    
    success = await test_single_agent()
    
    if success:
        logger.info("🎉 Simple test passed!")
        return 0
    else:
        logger.error("❌ Simple test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))