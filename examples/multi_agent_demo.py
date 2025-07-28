#!/usr/bin/env python3
"""
HRM Multi-Agent System Demonstration

This script demonstrates the capabilities of the HRM multi-agent architecture
by running multiple specialized agents in parallel to solve different types
of coding and development tasks.

Usage:
    python examples/multi_agent_demo.py [--config agent_config.yaml] [--mock]

Examples:
    # Run with mock agents (no SWE-ReX dependency)
    python examples/multi_agent_demo.py --mock
    
    # Run with SWE-ReX integration
    python examples/multi_agent_demo.py --config agent_runtime/example_config.yaml
    
    # Run interactive demo
    python examples/multi_agent_demo.py --interactive
"""

import asyncio
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agent_runtime.swe_rex_integration import (
    HRMMultiAgentOrchestrator,
    AgentType,
    EXAMPLE_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi-agent-demo")


class MultiAgentDemo:
    """Demonstration of HRM multi-agent capabilities."""
    
    def __init__(self, config_path: str = None, mock_mode: bool = True):
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.orchestrator = None
        
        # Demo task sets
        self.demo_tasks = self._create_demo_tasks()
    
    def _create_demo_tasks(self) -> List[Dict[str, Any]]:
        """Create a comprehensive set of demo tasks."""
        return [
            # Python coding tasks
            {
                "name": "Binary Search Implementation",
                "task_data": {
                    "problem_statement": "Implement an efficient binary search algorithm that works on sorted arrays",
                    "language": "python",
                    "complexity_score": 0.6,
                    "expected_functions": ["binary_search"],
                    "test_cases": [
                        {"input": "[1,2,3,4,5], 3", "expected": "2"},
                        {"input": "[1,3,5,7,9], 7", "expected": "3"}
                    ]
                },
                "agent_type": AgentType.LANGUAGE_PYTHON,
                "priority": 2
            },
            {
                "name": "Data Structure - Red-Black Tree",
                "task_data": {
                    "problem_statement": "Implement a Red-Black Tree with insertion and deletion operations",
                    "language": "python",
                    "complexity_score": 0.9,
                    "expected_classes": ["RedBlackTree", "Node"],
                    "operations": ["insert", "delete", "search", "traverse"]
                },
                "agent_type": AgentType.LANGUAGE_PYTHON,
                "priority": 1
            },
            
            # JavaScript tasks
            {
                "name": "React Component - Todo List",
                "task_data": {
                    "problem_statement": "Create a React component for managing a todo list with add, delete, and toggle functionality",
                    "language": "javascript",
                    "framework": "react",
                    "complexity_score": 0.7,
                    "components": ["TodoList", "TodoItem", "AddTodo"],
                    "features": ["add_todo", "delete_todo", "toggle_complete", "filter_todos"]
                },
                "agent_type": AgentType.LANGUAGE_JAVASCRIPT,
                "priority": 2
            },
            
            # Rust systems programming
            {
                "name": "Concurrent Web Server",
                "task_data": {
                    "problem_statement": "Implement a multi-threaded HTTP web server in Rust with connection pooling",
                    "language": "rust",
                    "complexity_score": 0.8,
                    "features": ["http_parsing", "thread_pool", "connection_handling", "static_files"],
                    "dependencies": ["tokio", "hyper"]
                },
                "agent_type": AgentType.LANGUAGE_RUST,
                "priority": 1
            },
            
            # Go microservice
            {
                "name": "REST API Service",
                "task_data": {
                    "problem_statement": "Create a RESTful API service in Go with user authentication and CRUD operations",
                    "language": "go",
                    "complexity_score": 0.75,
                    "endpoints": ["/users", "/auth/login", "/auth/register", "/profile"],
                    "features": ["jwt_auth", "database_integration", "middleware", "logging"]
                },
                "agent_type": AgentType.LANGUAGE_GO,
                "priority": 2
            },
            
            # Git workflow tasks
            {
                "name": "Git Branch Strategy Setup",
                "task_data": {
                    "commands": [
                        "git checkout -b feature/multi-agent-integration",
                        "git push -u origin feature/multi-agent-integration",
                        "git checkout main",
                        "git branch --set-upstream-to=origin/main main"
                    ],
                    "tools_required": ["git"],
                    "description": "Set up a feature branch workflow for multi-agent development"
                },
                "agent_type": AgentType.TOOL_GIT,
                "priority": 3
            },
            {
                "name": "Repository Analysis",
                "task_data": {
                    "commands": [
                        "git log --oneline -10",
                        "git branch -a",
                        "git status --porcelain",
                        "git diff --stat HEAD~5..HEAD"
                    ],
                    "tools_required": ["git"],
                    "description": "Analyze repository state and recent changes"
                },
                "agent_type": AgentType.TOOL_GIT,
                "priority": 2
            },
            
            # Docker containerization tasks
            {
                "name": "Multi-Service Docker Setup",
                "task_data": {
                    "dockerfile_content": '''
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
''',
                    "docker_compose": {
                        "version": "3.8",
                        "services": {
                            "web": {"build": ".", "ports": ["8000:8000"]},
                            "redis": {"image": "redis:alpine", "ports": ["6379:6379"]},
                            "db": {"image": "postgres:13", "environment": ["POSTGRES_DB=myapp"]}
                        }
                    },
                    "tools_required": ["docker"],
                    "description": "Create Docker setup for web application with dependencies"
                },
                "agent_type": AgentType.TOOL_DOCKER,
                "priority": 2
            },
            
            # Testing tasks
            {
                "name": "Comprehensive Test Suite",
                "task_data": {
                    "test_types": ["unit", "integration", "end-to-end"],
                    "frameworks": ["pytest", "unittest", "selenium"],
                    "coverage_target": 90,
                    "test_files": [
                        "test_binary_search.py",
                        "test_red_black_tree.py", 
                        "test_web_server.py",
                        "test_api_service.py"
                    ],
                    "tools_required": ["testing"],
                    "description": "Create comprehensive test suite for all implemented components"
                },
                "agent_type": AgentType.TOOL_TESTING,
                "priority": 1
            },
            
            # Web development domain task
            {
                "name": "Full-Stack E-commerce Feature",
                "task_data": {
                    "type": "feature_development",
                    "domain": "web",
                    "complexity_score": 0.85,
                    "components": {
                        "frontend": {
                            "framework": "react",
                            "components": ["ProductList", "ShoppingCart", "Checkout"],
                            "styling": "tailwindcss"
                        },
                        "backend": {
                            "framework": "fastapi",
                            "endpoints": ["/products", "/cart", "/checkout", "/orders"],
                            "database": "postgresql"
                        },
                        "integration": {
                            "payment": "stripe",
                            "authentication": "oauth2",
                            "deployment": "docker"
                        }
                    },
                    "description": "Implement complete e-commerce shopping cart and checkout flow"
                },
                "agent_type": AgentType.DOMAIN_WEB,
                "priority": 1
            },
            
            # Machine Learning domain task
            {
                "name": "Predictive Analytics Pipeline",
                "task_data": {
                    "type": "ml_pipeline",
                    "domain": "ml",
                    "complexity_score": 0.8,
                    "pipeline_stages": [
                        "data_ingestion",
                        "feature_engineering", 
                        "model_training",
                        "model_evaluation",
                        "deployment"
                    ],
                    "models": ["random_forest", "xgboost", "neural_network"],
                    "frameworks": ["scikit-learn", "pandas", "numpy", "tensorflow"],
                    "deployment": "mlflow",
                    "description": "Build end-to-end ML pipeline for predictive analytics"
                },
                "agent_type": AgentType.DOMAIN_ML,
                "priority": 1
            },
            
            # Meta-coordination task
            {
                "name": "Multi-Agent Task Coordination",
                "task_data": {
                    "type": "coordination",
                    "sub_tasks": [
                        {"id": "code_generation", "agent_type": "language_python", "dependencies": []},
                        {"id": "testing", "agent_type": "tool_testing", "dependencies": ["code_generation"]},
                        {"id": "containerization", "agent_type": "tool_docker", "dependencies": ["testing"]},
                        {"id": "deployment", "agent_type": "domain_web", "dependencies": ["containerization"]}
                    ],
                    "coordination_strategy": "dependency_graph",
                    "success_criteria": "all_tasks_completed",
                    "description": "Coordinate execution of interdependent tasks across multiple agents"
                },
                "agent_type": AgentType.META_COORDINATOR,
                "priority": 1
            }
        ]
    
    async def run_demo(self, interactive: bool = False):
        """Run the multi-agent demonstration."""
        logger.info("🚀 Starting HRM Multi-Agent System Demonstration")
        
        try:
            # Initialize orchestrator
            if not await self._initialize_orchestrator():
                logger.error("Failed to initialize orchestrator")
                return False
            
            # Start the orchestrator
            await self.orchestrator.start()
            logger.info("✅ Multi-agent orchestrator started successfully")
            
            if interactive:
                await self._run_interactive_demo()
            else:
                await self._run_automated_demo()
            
            return True
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False
        
        finally:
            if self.orchestrator:
                await self.orchestrator.stop()
                logger.info("🛑 Multi-agent orchestrator stopped")
    
    async def _initialize_orchestrator(self) -> bool:
        """Initialize the multi-agent orchestrator."""
        try:
            # Create config if needed
            if not self.config_path:
                if self.mock_mode:
                    # Use minimal config for mock mode
                    config = {
                        "max_agents": 8,
                        "agent_types": {
                            "language_python": {"count": 2, "model_path": None},
                            "language_javascript": {"count": 1, "model_path": None},
                            "language_rust": {"count": 1, "model_path": None},
                            "tool_git": {"count": 1, "model_path": None},
                            "tool_docker": {"count": 1, "model_path": None},
                            "tool_testing": {"count": 1, "model_path": None},
                            "meta_coordinator": {"count": 1, "model_path": None}
                        },
                        "deployment": {"type": "local", "local": {}}
                    }
                    
                    # Save temporary config
                    import yaml
                    temp_config_path = "/tmp/hrm_demo_config.yaml"
                    with open(temp_config_path, 'w') as f:
                        yaml.dump(config, f)
                    self.config_path = temp_config_path
                else:
                    self.config_path = "agent_runtime/example_config.yaml"
            
            # Initialize orchestrator
            self.orchestrator = HRMMultiAgentOrchestrator(self.config_path)
            success = await self.orchestrator.initialize()
            
            if success:
                logger.info(f"✅ Initialized orchestrator with config: {self.config_path}")
                return True
            else:
                logger.error("❌ Failed to initialize orchestrator")
                return False
                
        except Exception as e:
            logger.error(f"Orchestrator initialization error: {e}")
            return False
    
    async def _run_automated_demo(self):
        """Run automated demonstration with all demo tasks."""
        logger.info("🎯 Running automated multi-agent demonstration")
        
        # Submit all demo tasks
        task_ids = []
        for task_config in self.demo_tasks:
            task_id = self.orchestrator.submit_task(
                task_data=task_config["task_data"],
                agent_type=task_config["agent_type"],
                priority=task_config["priority"]
            )
            task_ids.append((task_id, task_config["name"]))
            logger.info(f"📋 Submitted: {task_config['name']} (ID: {task_id})")
        
        logger.info(f"📊 Total tasks submitted: {len(task_ids)}")
        
        # Monitor system status
        monitoring_duration = 60  # Monitor for 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < monitoring_duration:
            status = self.orchestrator.get_system_status()
            
            logger.info(
                f"📈 System Status: "
                f"{status['active_agents']} active agents, "
                f"{status['pending_tasks']} pending tasks, "
                f"{status['metrics']['total_tasks_processed']} completed, "
                f"Success Rate: {status['metrics']['total_tasks_successful']}/{status['metrics']['total_tasks_processed']}"
            )
            
            # Show agent breakdown
            if status['agent_breakdown']:
                logger.info("🤖 Agent Activity:")
                for agent_type, stats in status['agent_breakdown'].items():
                    logger.info(
                        f"  {agent_type}: {stats['count']} agents, "
                        f"{stats['active_tasks']} active tasks, "
                        f"{stats['total_completed']} completed"
                    )
            
            await asyncio.sleep(5)
        
        # Wait for some tasks to complete
        logger.info("⏳ Waiting for task completions...")
        completed_tasks = []
        
        for task_id, task_name in task_ids[:5]:  # Wait for first 5 tasks
            result = await self.orchestrator.wait_for_task_completion(task_id, timeout=30.0)
            if result:
                completed_tasks.append((task_name, result))
                logger.info(
                    f"✅ Completed: {task_name} "
                    f"(Status: {result['status']}, Time: {result['execution_time']:.2f}s)"
                )
            else:
                logger.warning(f"⏰ Timeout: {task_name}")
        
        # Final system report
        final_status = self.orchestrator.get_system_status()
        logger.info("📊 Final System Report:")
        logger.info(f"  Total Tasks Processed: {final_status['metrics']['total_tasks_processed']}")
        logger.info(f"  Successful Tasks: {final_status['metrics']['total_tasks_successful']}")
        logger.info(f"  Average Task Time: {final_status['metrics']['average_task_time']:.2f}s")
        logger.info(f"  System Throughput: {final_status['metrics']['system_throughput']:.2f} tasks/min")
        
        # Show completed task results
        if completed_tasks:
            logger.info("🎉 Completed Task Summary:")
            for task_name, result in completed_tasks:
                if result['result']:
                    logger.info(f"  {task_name}: {json.dumps(result['result'], indent=2)[:200]}...")
    
    async def _run_interactive_demo(self):
        """Run interactive demonstration where user can submit custom tasks."""
        logger.info("🎮 Starting interactive multi-agent demonstration")
        logger.info("Commands: submit, status, health, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit":
                    break
                    
                elif command == "status":
                    await self._show_status()
                    
                elif command == "health":
                    await self._show_health()
                    
                elif command == "submit":
                    await self._interactive_task_submission()
                    
                elif command == "demo":
                    # Submit one of the demo tasks
                    task_config = self.demo_tasks[0]  # Binary search task
                    task_id = self.orchestrator.submit_task(
                        task_data=task_config["task_data"],
                        agent_type=task_config["agent_type"],
                        priority=task_config["priority"]
                    )
                    logger.info(f"📋 Submitted demo task: {task_config['name']} (ID: {task_id})")
                    
                else:
                    logger.info("Available commands: submit, status, health, demo, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Command error: {e}")
    
    async def _show_status(self):
        """Show current system status."""
        status = self.orchestrator.get_system_status()
        
        print("\n📊 System Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Active Agents: {status['active_agents']}/{status['total_agents']}")
        print(f"  Pending Tasks: {status['pending_tasks']}")
        print(f"  Total Processed: {status['metrics']['total_tasks_processed']}")
        print(f"  Success Rate: {status['metrics']['total_tasks_successful']}/{status['metrics']['total_tasks_processed']}")
        
        if status['agent_breakdown']:
            print("\n🤖 Agent Breakdown:")
            for agent_type, stats in status['agent_breakdown'].items():
                print(f"  {agent_type}: {stats['count']} agents, {stats['active_tasks']} active")
    
    async def _show_health(self):
        """Show agent health status."""
        health_results = await self.orchestrator.agent_pool.health_check_all()
        
        print("\n🏥 Agent Health:")
        for agent_id, health in health_results.items():
            status_icon = "✅" if health.get('is_active', False) else "❌"
            print(f"  {status_icon} {agent_id}: {health.get('current_tasks', 0)} tasks, "
                  f"Success Rate: {health.get('success_rate', 0):.2f}")
    
    async def _interactive_task_submission(self):
        """Interactive task submission."""
        print("\n📝 Task Submission:")
        print("1. Python coding task")
        print("2. Git operation")
        print("3. Custom task")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            problem = input("Enter problem statement: ").strip()
            task_id = self.orchestrator.submit_task({
                "problem_statement": problem,
                "language": "python",
                "complexity_score": 0.5
            })
            logger.info(f"📋 Submitted Python task (ID: {task_id})")
            
        elif choice == "2":
            command = input("Enter git command: ").strip()
            task_id = self.orchestrator.submit_task({
                "commands": [command],
                "tools_required": ["git"]
            })
            logger.info(f"📋 Submitted Git task (ID: {task_id})")
            
        elif choice == "3":
            task_data = {}
            task_data["description"] = input("Enter task description: ").strip()
            task_id = self.orchestrator.submit_task(task_data)
            logger.info(f"📋 Submitted custom task (ID: {task_id})")
        
        else:
            logger.info("Invalid option")


async def main():
    """Main demonstration entry point."""
    parser = argparse.ArgumentParser(description="HRM Multi-Agent System Demonstration")
    parser.add_argument("--config", help="Path to agent configuration file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no SWE-ReX dependency)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demonstration")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create demo instance
    demo = MultiAgentDemo(
        config_path=args.config,
        mock_mode=args.mock or not args.config
    )
    
    # Run demonstration
    success = await demo.run_demo(interactive=args.interactive)
    
    if success:
        logger.info("🎉 Demonstration completed successfully!")
        return 0
    else:
        logger.error("❌ Demonstration failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))