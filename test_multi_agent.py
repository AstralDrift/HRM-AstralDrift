#!/usr/bin/env python3
"""
Quick test script for HRM multi-agent system

This script performs basic functionality tests to ensure the multi-agent
architecture is working correctly without requiring full SWE-ReX setup.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agent_runtime.swe_rex_integration import (
    HRMMultiAgentOrchestrator,
    AgentType,
    AgentCapabilities,
    AgentTask
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-multi-agent")


async def test_basic_functionality():
    """Test basic multi-agent functionality."""
    logger.info("🧪 Testing HRM Multi-Agent System")
    
    # Create minimal test config
    test_config = {
        "max_agents": 4,
        "agent_types": {
            "language_python": {"count": 2, "model_path": None},
            "tool_git": {"count": 1, "model_path": None},
            "meta_coordinator": {"count": 1, "model_path": None}
        },
        "deployment": {
            "type": "local",
            "local": {}
        }
    }
    
    # Save test config
    import yaml
    test_config_path = "/tmp/hrm_test_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # Initialize orchestrator
        orchestrator = HRMMultiAgentOrchestrator(test_config_path)
        
        logger.info("✅ Test 1: Orchestrator initialization")
        success = await orchestrator.initialize()
        assert success, "Failed to initialize orchestrator"
        
        logger.info("✅ Test 2: Starting orchestrator")
        await orchestrator.start()
        
        logger.info("✅ Test 3: System status check")
        status = orchestrator.get_system_status()
        assert status['is_running'], "Orchestrator not running"
        assert status['active_agents'] > 0, "No active agents"
        
        logger.info("✅ Test 4: Task submission")
        task_id = orchestrator.submit_task({
            "problem_statement": "Test problem: implement hello world",
            "language": "python",
            "complexity_score": 0.1
        })
        assert task_id, "Failed to submit task"
        
        logger.info("✅ Test 5: Agent health check")
        health_results = await orchestrator.agent_pool.health_check_all()
        assert len(health_results) > 0, "No health results"
        logger.info(f"   Health results: {len(health_results)} agents")
        
        # Wait a bit for task processing
        logger.info("   Waiting for task processing...")
        await asyncio.sleep(2)
        
        logger.info("✅ Test 6: Updated system status")
        final_status = orchestrator.get_system_status()
        logger.info(f"   Processed tasks: {final_status['metrics']['total_tasks_processed']}")
        
        logger.info("✅ Test 7: Stopping orchestrator")
        await orchestrator.stop()
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_agent_capabilities():
    """Test agent capability system."""
    logger.info("🔧 Testing Agent Capabilities")
    
    # Test capability creation
    python_caps = AgentCapabilities(
        agent_type=AgentType.LANGUAGE_PYTHON,
        languages=["python"],
        max_concurrent_tasks=3
    )
    assert python_caps.agent_type == AgentType.LANGUAGE_PYTHON
    assert "python" in python_caps.languages
    logger.info("✅ Capability creation works")
    
    # Test task creation
    test_task = AgentTask(
        task_id="test_001",
        agent_type=AgentType.LANGUAGE_PYTHON,
        task_data={"problem": "test problem"},
        language="python"
    )
    assert test_task.task_id == "test_001"
    assert test_task.language == "python"
    logger.info("✅ Task creation works")
    
    return True


async def main():
    """Run all tests."""
    logger.info("🚀 Starting HRM multi-agent system tests")
    
    try:
        # Test 1: Basic functionality
        test1_success = await test_basic_functionality()
        
        # Test 2: Agent capabilities
        test2_success = await test_agent_capabilities()
        
        if test1_success and test2_success:
            logger.info("✅ All tests passed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))