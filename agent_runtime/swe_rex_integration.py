"""
SWE-ReX Runtime Integration for HRM Multi-Agent Architecture

This module provides the runtime infrastructure for executing multiple specialized HRM agents
in parallel using SWE-ReX's sandboxed execution environments. It enables the HRM system
to scale from single-agent operation to 30+ parallel specialized agents.

Key Components:
- HRMAgentRuntime: Individual agent execution environment
- HRMMultiAgentOrchestrator: Coordinates multiple agents with load balancing
- HRMAgentPool: Manages lifecycle of agent instances
- HRMTaskDistributor: Distributes tasks across available agents

Architecture:
                    ┌─────────────────────────────────┐
                    │     HRM Meta-Agent              │
                    │   (Task Planning & Coordination)│
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │   HRMMultiAgentOrchestrator     │
                    │   - Load balancing              │
                    │   - Agent health monitoring     │
                    │   - Task distribution           │
                    └─────────────┬───────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
     ┌─────▼─────┐         ┌─────▼─────┐         ┌─────▼─────┐
     │Language   │         │Tool       │         │Domain     │
     │Specialists│         │Specialists│         │Specialists│
     │Python,Go, │         │Git,Docker,│         │Web,ML,    │
     │Rust,JS... │         │NPM,Test...│         │Systems... │
     └───────────┘         └───────────┘         └───────────┘
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml

# SWE-ReX imports
try:
    from swerex.deployment.local import LocalDeployment
    from swerex.deployment.docker import DockerDeployment
    from swerex.deployment.modal import ModalDeployment
    from swerex.runtime.abstract import (
        AbstractRuntime, BashAction, CreateBashSessionRequest,
        Command, ReadFileRequest, WriteFileRequest
    )
    SWEREX_AVAILABLE = True
except ImportError:
    SWEREX_AVAILABLE = False
    logging.warning("SWE-ReX not available, using mock implementations")

# HRM imports (optional for mock mode)
try:
    import torch
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, HRM model features disabled")


class AgentType(Enum):
    """Specialized agent types for different code generation tasks."""
    LANGUAGE_PYTHON = "language_python"
    LANGUAGE_JAVASCRIPT = "language_javascript" 
    LANGUAGE_RUST = "language_rust"
    LANGUAGE_GO = "language_go"
    LANGUAGE_JAVA = "language_java"
    LANGUAGE_CPP = "language_cpp"
    
    TOOL_GIT = "tool_git"
    TOOL_DOCKER = "tool_docker"
    TOOL_NPM = "tool_npm"
    TOOL_TESTING = "tool_testing"
    TOOL_BUILD = "tool_build"
    TOOL_DEBUG = "tool_debug"
    
    DOMAIN_WEB = "domain_web"
    DOMAIN_ML = "domain_ml"
    DOMAIN_SYSTEMS = "domain_systems"
    DOMAIN_DATABASE = "domain_database"
    DOMAIN_API = "domain_api"
    DOMAIN_CLI = "domain_cli"
    
    META_COORDINATOR = "meta_coordinator"
    META_REVIEWER = "meta_reviewer"
    META_PLANNER = "meta_planner"


@dataclass
class AgentCapabilities:
    """Defines the capabilities and specialization of an agent."""
    agent_type: AgentType
    languages: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    specialization_score: float = 1.0
    
    # Performance characteristics
    avg_processing_time: float = 10.0  # seconds
    success_rate: float = 0.85
    complexity_threshold: float = 0.7
    
    # Resource requirements
    memory_requirement: str = "2GB"
    cpu_requirement: float = 1.0
    gpu_requirement: bool = False


@dataclass
class AgentTask:
    """Represents a task assigned to a specific agent."""
    task_id: str
    agent_type: AgentType
    task_data: Dict[str, Any]
    priority: int = 1
    timeout: float = 300.0
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    # Task requirements
    language: Optional[str] = None
    tools_required: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    complexity_score: float = 0.5
    
    # Execution state
    status: str = "pending"  # pending, assigned, running, completed, failed
    assigned_agent: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HRMAgentRuntime:
    """Individual HRM agent execution environment using SWE-ReX runtime."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        deployment_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.deployment_config = deployment_config or {}
        
        # Runtime state
        self.is_active = False
        self.current_tasks: List[AgentTask] = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.last_heartbeat = time.time()
        
        # SWE-ReX deployment
        self.deployment: Optional[Any] = None  # AbstractDeployment
        self.runtime: Optional[Any] = None     # AbstractRuntime
        
        # HRM model (loaded on demand)
        self.model: Optional[Any] = None  # HierarchicalReasoningModel_ACTV1 when available
        self.model_path = model_path
        
        # Logging
        self.logger = logging.getLogger(f"hrm-agent-{agent_id}")
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize the agent runtime environment."""
        try:
            if not SWEREX_AVAILABLE:
                self.logger.warning("SWE-ReX not available, using mock runtime")
                self.is_active = True
                return True
            
            # Create SWE-ReX deployment based on config
            deployment_type = self.deployment_config.get("type", "local")
            
            if deployment_type == "local":
                self.deployment = LocalDeployment(**self.deployment_config.get("local", {}))
            elif deployment_type == "docker":
                self.deployment = DockerDeployment(**self.deployment_config.get("docker", {}))
            elif deployment_type == "modal":
                self.deployment = ModalDeployment(**self.deployment_config.get("modal", {}))
            else:
                raise ValueError(f"Unsupported deployment type: {deployment_type}")
            
            # Start the deployment
            await self.deployment.start()
            self.runtime = self.deployment.runtime
            
            # Create bash session for this agent
            session_request = CreateBashSessionRequest(
                session=self.agent_id,
                startup_source=self._get_startup_sources()
            )
            await self.runtime.create_session(session_request)
            
            self.is_active = True
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            return False
    
    def _get_startup_sources(self) -> List[str]:
        """Get startup source files based on agent specialization."""
        sources = []
        
        # Add language-specific environments
        if self.agent_type.value.startswith("language_"):
            lang = self.agent_type.value.split("_")[1]
            if lang == "python":
                sources.extend(["/opt/conda/etc/profile.d/conda.sh"])
            elif lang == "rust":
                sources.extend(["$HOME/.cargo/env"])
            elif lang == "go":
                sources.extend(["/usr/local/go/bin"])
        
        # Add tool-specific configurations
        if self.agent_type.value.startswith("tool_"):
            tool = self.agent_type.value.split("_")[1]
            if tool == "docker":
                sources.extend(["/etc/docker/daemon.json"])
            elif tool == "git":
                sources.extend(["$HOME/.gitconfig"])
        
        return sources
    
    async def load_hrm_model(self) -> bool:
        """Load the HRM model for this agent."""
        try:
            if self.model is not None:
                return True
            
            if not TORCH_AVAILABLE:
                self.logger.warning("PyTorch not available, using mock model")
                return True
            
            if self.model_path is None:
                self.logger.warning("No model path specified, using default")
                return True
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location="cpu")
            config = checkpoint.get("config", {})
            
            # Create model instance
            self.model = HierarchicalReasoningModel_ACTV1(**config)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            
            # Move to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
            self.logger.info(f"HRM model loaded successfully for agent {self.agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load HRM model: {e}")
            return False
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task using this agent."""
        task.status = "running"
        task.assigned_agent = self.agent_id
        task.started_at = time.time()
        
        try:
            self.current_tasks.append(task)
            
            # Load model if needed
            if not await self.load_hrm_model():
                raise RuntimeError("Failed to load HRM model")
            
            # Execute based on task type
            if task.language:
                result = await self._execute_language_task(task)
            elif task.tools_required:
                result = await self._execute_tool_task(task)
            elif task.domain:
                result = await self._execute_domain_task(task)
            else:
                result = await self._execute_general_task(task)
            
            # Update task status
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            self.completed_tasks += 1
            self._update_performance_metrics(task, success=True)
            
            return result
            
        except Exception as e:
            task.status = "failed"
            task.completed_at = time.time()
            task.error = str(e)
            
            self.failed_tasks += 1
            self._update_performance_metrics(task, success=False)
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            raise
        
        finally:
            if task in self.current_tasks:
                self.current_tasks.remove(task)
    
    async def _execute_language_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a language-specific code generation task."""
        language = task.language
        problem_statement = task.task_data.get("problem_statement", "")
        
        if not SWEREX_AVAILABLE:
            # Mock execution for testing
            return {
                "generated_code": f"# Generated {language} code for: {problem_statement[:50]}...",
                "success": True,
                "execution_time": 5.0
            }
        
        # Use HRM model for code generation
        if self.model:
            # Prepare input for HRM
            input_data = self._prepare_hrm_input(task)
            
            # Generate code using hierarchical reasoning
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    output = self.model(input_data)
                    generated_code = self._decode_hrm_output(output, language)
            else:
                generated_code = self._decode_hrm_output(None, language)
        else:
            generated_code = f"// TODO: Implement {problem_statement}"
        
        # Execute code validation in runtime environment
        validation_result = await self._validate_generated_code(generated_code, language)
        
        return {
            "generated_code": generated_code,
            "validation": validation_result,
            "language": language,
            "success": validation_result.get("valid", False)
        }
    
    async def _execute_tool_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a tool-specific task (git, docker, npm, etc.)."""
        tools = task.tools_required
        commands = task.task_data.get("commands", [])
        
        if not SWEREX_AVAILABLE:
            return {
                "commands_executed": commands,
                "tools_used": tools,
                "success": True
            }
        
        results = []
        for cmd in commands:
            try:
                # Execute command in the runtime environment
                action = BashAction(
                    command=cmd,
                    session=self.agent_id,
                    timeout=60.0
                )
                
                observation = await self.runtime.run_in_session(action)
                results.append({
                    "command": cmd,
                    "output": observation.output,
                    "exit_code": observation.exit_code,
                    "success": observation.exit_code == 0
                })
                
            except Exception as e:
                results.append({
                    "command": cmd,
                    "error": str(e),
                    "success": False
                })
        
        overall_success = all(r.get("success", False) for r in results)
        
        return {
            "results": results,
            "tools_used": tools,
            "success": overall_success
        }
    
    async def _execute_domain_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a domain-specific task (web, ML, systems, etc.)."""
        domain = task.domain
        task_type = task.task_data.get("type", "general")
        
        # Domain-specific execution logic
        if domain == "web":
            return await self._execute_web_task(task)
        elif domain == "ml":
            return await self._execute_ml_task(task)
        elif domain == "systems":
            return await self._execute_systems_task(task)
        else:
            return await self._execute_general_task(task)
    
    async def _execute_general_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a general code generation task."""
        if not SWEREX_AVAILABLE:
            return {
                "result": "Mock general task execution",
                "success": True
            }
        
        # Use HRM model for general reasoning
        if self.model and TORCH_AVAILABLE:
            input_data = self._prepare_hrm_input(task)
            with torch.no_grad():
                output = self.model(input_data)
                result = self._decode_hrm_output(output, "general")
        else:
            result = "General task completed"
        
        return {
            "result": result,
            "success": True
        }
    
    def _prepare_hrm_input(self, task: AgentTask) -> Optional[Any]:
        """Prepare input tensor for HRM model."""
        if not TORCH_AVAILABLE:
            return None
        # Mock implementation - in practice, this would encode the task
        # data into the appropriate tensor format for HRM
        return torch.randn(1, 100, 512)  # Mock tensor
    
    def _decode_hrm_output(self, output: Optional[Any], context: str) -> str:
        """Decode HRM model output to human-readable format."""
        # Mock implementation - in practice, this would decode the tensor
        # output into code or natural language
        if output is None or not TORCH_AVAILABLE:
            return f"# Mock generated output for {context} context\nprint('Hello from {context} agent!')"
        return f"Generated output for {context} context"
    
    async def _validate_generated_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate generated code by executing it in the runtime."""
        if not SWEREX_AVAILABLE:
            return {"valid": True, "message": "Mock validation"}
        
        try:
            # Create temporary file with the code
            temp_file = f"/tmp/generated_code.{self._get_file_extension(language)}"
            write_request = WriteFileRequest(content=code, path=temp_file)
            await self.runtime.write_file(write_request)
            
            # Run language-specific validation
            validation_cmd = self._get_validation_command(temp_file, language)
            if validation_cmd:
                command = Command(command=validation_cmd, timeout=30.0)
                result = await self.runtime.execute(command)
                
                return {
                    "valid": result.exit_code == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
            
            return {"valid": True, "message": "No validation available"}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language."""
        extensions = {
            "python": "py",
            "javascript": "js", 
            "rust": "rs",
            "go": "go",
            "java": "java",
            "cpp": "cpp"
        }
        return extensions.get(language.lower(), "txt")
    
    def _get_validation_command(self, filepath: str, language: str) -> Optional[str]:
        """Get validation command for a programming language."""
        commands = {
            "python": f"python -m py_compile {filepath}",
            "javascript": f"node --check {filepath}",
            "rust": f"rustc --crate-type lib {filepath}",
            "go": f"go build {filepath}",
            "java": f"javac {filepath}",
            "cpp": f"g++ -fsyntax-only {filepath}"
        }
        return commands.get(language.lower())
    
    async def _execute_web_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute web development specific task."""
        # Mock implementation for web tasks
        return {
            "html_generated": True,
            "css_generated": True,
            "js_generated": True,
            "success": True
        }
    
    async def _execute_ml_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute machine learning specific task."""
        # Mock implementation for ML tasks
        return {
            "model_created": True,
            "training_completed": True,
            "accuracy": 0.85,
            "success": True
        }
    
    async def _execute_systems_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute systems programming specific task."""
        # Mock implementation for systems tasks
        return {
            "system_component_implemented": True,
            "performance_optimized": True,
            "success": True
        }
    
    def _update_performance_metrics(self, task: AgentTask, success: bool):
        """Update performance tracking metrics."""
        execution_time = (task.completed_at or time.time()) - (task.started_at or time.time())
        
        self.performance_history.append({
            "task_id": task.task_id,
            "success": success,
            "execution_time": execution_time,
            "complexity": task.complexity_score,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update capabilities based on performance
        self._adapt_capabilities()
    
    def _adapt_capabilities(self):
        """Adapt agent capabilities based on performance history."""
        if len(self.performance_history) < 10:
            return
        
        recent_tasks = self.performance_history[-10:]
        success_rate = sum(1 for t in recent_tasks if t["success"]) / len(recent_tasks)
        avg_time = sum(t["execution_time"] for t in recent_tasks) / len(recent_tasks)
        
        # Update capabilities
        self.capabilities.success_rate = success_rate
        self.capabilities.avg_processing_time = avg_time
        
        # Adjust specialization score based on performance
        if success_rate > 0.9:
            self.capabilities.specialization_score *= 1.05
        elif success_rate < 0.7:
            self.capabilities.specialization_score *= 0.95
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent."""
        self.last_heartbeat = time.time()
        
        health_status = {
            "agent_id": self.agent_id,
            "is_active": self.is_active,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.capabilities.success_rate,
            "avg_processing_time": self.capabilities.avg_processing_time,
            "last_heartbeat": self.last_heartbeat,
            "runtime_alive": True,  # Always true in mock mode
            "runtime_message": "Mock runtime" if not SWEREX_AVAILABLE else "SWE-ReX runtime"
        }
        
        return health_status
    
    async def shutdown(self):
        """Shutdown the agent runtime."""
        try:
            self.is_active = False
            
            # Complete any running tasks
            for task in self.current_tasks[:]:
                task.status = "cancelled"
                self.current_tasks.remove(task)
            
            # Shutdown SWE-ReX deployment
            if self.deployment:
                await self.deployment.stop()
                self.deployment = None
                self.runtime = None
            
            self.logger.info(f"Agent {self.agent_id} shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during agent shutdown: {e}")


class HRMTaskDistributor:
    """Distributes tasks across available agents based on capabilities and load."""
    
    def __init__(self):
        self.task_queue: List[AgentTask] = []
        self.agent_assignments: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self.logger = logging.getLogger("hrm-task-distributor")
    
    def add_task(self, task: AgentTask):
        """Add a task to the distribution queue."""
        self.task_queue.append(task)
        self.logger.info(f"Added task {task.task_id} to queue")
    
    def find_best_agent(
        self, 
        task: AgentTask, 
        available_agents: List[HRMAgentRuntime]
    ) -> Optional[HRMAgentRuntime]:
        """Find the best agent for a given task."""
        if not available_agents:
            return None
        
        # Score agents based on suitability for the task
        agent_scores = []
        
        for agent in available_agents:
            if not agent.is_active:
                continue
            
            if len(agent.current_tasks) >= agent.capabilities.max_concurrent_tasks:
                continue
            
            score = self._calculate_agent_score(agent, task)
            agent_scores.append((agent, score))
        
        if not agent_scores:
            return None
        
        # Sort by score (higher is better) and return best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores[0][0]
    
    def _calculate_agent_score(self, agent: HRMAgentRuntime, task: AgentTask) -> float:
        """Calculate how well an agent matches a task."""
        score = 0.0
        
        # Base specialization score
        score += agent.capabilities.specialization_score
        
        # Language match
        if task.language and task.language in agent.capabilities.languages:
            score += 10.0
        
        # Tool match
        tool_matches = len(set(task.tools_required) & set(agent.capabilities.tools))
        score += tool_matches * 5.0
        
        # Domain match
        if task.domain and task.domain in agent.capabilities.domains:
            score += 8.0
        
        # Agent type match
        if self._agent_type_matches_task(agent.agent_type, task):
            score += 15.0
        
        # Performance factors
        score += agent.capabilities.success_rate * 5.0
        score -= (agent.capabilities.avg_processing_time / 10.0)  # Prefer faster agents
        
        # Load balancing - prefer less busy agents
        current_load = len(agent.current_tasks) / agent.capabilities.max_concurrent_tasks
        score -= current_load * 3.0
        
        # Complexity match
        complexity_diff = abs(agent.capabilities.complexity_threshold - task.complexity_score)
        score -= complexity_diff * 2.0
        
        return score
    
    def _agent_type_matches_task(self, agent_type: AgentType, task: AgentTask) -> bool:
        """Check if agent type is suitable for the task."""
        # Language agents
        if agent_type.value.startswith("language_"):
            return task.language is not None
        
        # Tool agents
        if agent_type.value.startswith("tool_"):
            tool_name = agent_type.value.split("_")[1]
            return tool_name in task.tools_required
        
        # Domain agents
        if agent_type.value.startswith("domain_"):
            domain_name = agent_type.value.split("_")[1]
            return task.domain == domain_name
        
        # Meta agents can handle any task
        if agent_type.value.startswith("meta_"):
            return True
        
        return False
    
    async def distribute_tasks(self, available_agents: List[HRMAgentRuntime]) -> List[Tuple[AgentTask, HRMAgentRuntime]]:
        """Distribute pending tasks to available agents."""
        assignments = []
        
        # Sort tasks by priority
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        for task in self.task_queue[:]:
            agent = self.find_best_agent(task, available_agents)
            if agent:
                assignments.append((task, agent))
                self.task_queue.remove(task)
                
                # Track assignment
                if agent.agent_id not in self.agent_assignments:
                    self.agent_assignments[agent.agent_id] = []
                self.agent_assignments[agent.agent_id].append(task.task_id)
        
        return assignments


class HRMAgentPool:
    """Manages the lifecycle of multiple HRM agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.agents: Dict[str, HRMAgentRuntime] = {}
        self.config_path = config_path
        self.logger = logging.getLogger("hrm-agent-pool")
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent pool configuration."""
        default_config = {
            "max_agents": 32,
            "agent_types": {
                "language_python": {"count": 4, "model_path": None},
                "language_javascript": {"count": 2, "model_path": None},
                "language_rust": {"count": 2, "model_path": None},
                "language_go": {"count": 2, "model_path": None},
                "tool_git": {"count": 2, "model_path": None},
                "tool_docker": {"count": 2, "model_path": None},
                "tool_testing": {"count": 3, "model_path": None},
                "domain_web": {"count": 3, "model_path": None},
                "domain_ml": {"count": 2, "model_path": None},
                "meta_coordinator": {"count": 1, "model_path": None}
            },
            "deployment": {
                "type": "local",
                "local": {},
                "docker": {
                    "image": "python:3.10-slim",
                    "memory_limit": "2g"
                }
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    async def initialize_agents(self) -> bool:
        """Initialize all agents based on configuration."""
        try:
            agent_types_config = self.config.get("agent_types", {})
            deployment_config = self.config.get("deployment", {})
            
            for agent_type_str, type_config in agent_types_config.items():
                count = type_config.get("count", 1)
                model_path = type_config.get("model_path")
                
                try:
                    agent_type = AgentType(agent_type_str)
                except ValueError:
                    self.logger.warning(f"Unknown agent type: {agent_type_str}")
                    continue
                
                for i in range(count):
                    agent_id = f"{agent_type_str}_{i}"
                    capabilities = self._create_capabilities(agent_type)
                    
                    agent = HRMAgentRuntime(
                        agent_id=agent_id,
                        agent_type=agent_type,
                        capabilities=capabilities,
                        deployment_config=deployment_config,
                        model_path=model_path
                    )
                    
                    if await agent.initialize():
                        self.agents[agent_id] = agent
                        self.logger.info(f"Initialized agent {agent_id}")
                    else:
                        self.logger.error(f"Failed to initialize agent {agent_id}")
            
            self.logger.info(f"Initialized {len(self.agents)} agents")
            return len(self.agents) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            return False
    
    def _create_capabilities(self, agent_type: AgentType) -> AgentCapabilities:
        """Create capabilities for a specific agent type."""
        base_capabilities = AgentCapabilities(agent_type=agent_type)
        
        # Customize based on agent type
        if agent_type.value.startswith("language_"):
            lang = agent_type.value.split("_")[1]
            base_capabilities.languages = [lang]
            base_capabilities.max_concurrent_tasks = 3
            base_capabilities.complexity_threshold = 0.8
        
        elif agent_type.value.startswith("tool_"):
            tool = agent_type.value.split("_")[1]
            base_capabilities.tools = [tool]
            base_capabilities.max_concurrent_tasks = 5
            base_capabilities.complexity_threshold = 0.6
        
        elif agent_type.value.startswith("domain_"):
            domain = agent_type.value.split("_")[1]
            base_capabilities.domains = [domain]
            base_capabilities.max_concurrent_tasks = 2
            base_capabilities.complexity_threshold = 0.9
        
        elif agent_type.value.startswith("meta_"):
            base_capabilities.languages = ["python", "javascript", "rust", "go"]
            base_capabilities.tools = ["git", "docker", "testing"]
            base_capabilities.domains = ["web", "ml", "systems"]
            base_capabilities.max_concurrent_tasks = 1
            base_capabilities.complexity_threshold = 1.0
            base_capabilities.specialization_score = 2.0
        
        return base_capabilities
    
    def get_available_agents(self) -> List[HRMAgentRuntime]:
        """Get list of active agents."""
        return [agent for agent in self.agents.values() if agent.is_active]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[HRMAgentRuntime]:
        """Get specific agent by ID."""
        return self.agents.get(agent_id)
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all agents."""
        health_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                health_results[agent_id] = await agent.health_check()
            except Exception as e:
                health_results[agent_id] = {
                    "agent_id": agent_id,
                    "error": str(e),
                    "is_active": False
                }
        
        return health_results
    
    async def shutdown_all(self):
        """Shutdown all agents."""
        shutdown_tasks = []
        for agent in self.agents.values():
            shutdown_tasks.append(agent.shutdown())
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.agents.clear()
        self.logger.info("All agents shut down")


class HRMMultiAgentOrchestrator:
    """Main orchestrator for the HRM multi-agent system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.agent_pool = HRMAgentPool(config_path)
        self.task_distributor = HRMTaskDistributor()
        self.logger = logging.getLogger("hrm-orchestrator")
        
        # System state
        self.is_running = False
        self.task_counter = 0
        
        # Performance tracking
        self.system_metrics = {
            "total_tasks_processed": 0,
            "total_tasks_successful": 0,
            "total_execution_time": 0.0,
            "average_task_time": 0.0,
            "system_throughput": 0.0  # tasks per minute
        }
        
        # Event loop for continuous operation
        self._orchestrator_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize the multi-agent orchestrator."""
        try:
            success = await self.agent_pool.initialize_agents()
            if success:
                self.logger.info("Multi-agent orchestrator initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize agent pool")
                return False
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def start(self):
        """Start the orchestrator's main processing loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self._orchestrator_task = asyncio.create_task(self._orchestrator_loop())
        self.logger.info("Multi-agent orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and shutdown all agents."""
        self.is_running = False
        
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
        
        await self.agent_pool.shutdown_all()
        self.logger.info("Multi-agent orchestrator stopped")
    
    async def _orchestrator_loop(self):
        """Main orchestrator processing loop."""
        while self.is_running:
            try:
                # Distribute pending tasks
                available_agents = self.agent_pool.get_available_agents()
                assignments = await self.task_distributor.distribute_tasks(available_agents)
                
                # Execute assigned tasks
                if assignments:
                    execution_tasks = []
                    for task, agent in assignments:
                        execution_task = asyncio.create_task(
                            self._execute_task_with_tracking(task, agent)
                        )
                        execution_tasks.append(execution_task)
                    
                    # Don't wait for completion, let tasks run concurrently
                    self.logger.info(f"Dispatched {len(assignments)} tasks to agents")
                
                # Health monitoring
                if len(available_agents) == 0:
                    self.logger.warning("No active agents available")
                
                # Wait before next cycle
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in orchestrator loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _execute_task_with_tracking(self, task: AgentTask, agent: HRMAgentRuntime):
        """Execute a task and track performance metrics."""
        start_time = time.time()
        
        try:
            result = await agent.execute_task(task)
            execution_time = time.time() - start_time
            
            # Update system metrics
            self.system_metrics["total_tasks_processed"] += 1
            self.system_metrics["total_execution_time"] += execution_time
            
            if task.status == "completed":
                self.system_metrics["total_tasks_successful"] += 1
            
            # Calculate running averages
            total_tasks = self.system_metrics["total_tasks_processed"]
            self.system_metrics["average_task_time"] = (
                self.system_metrics["total_execution_time"] / total_tasks
            )
            
            self.logger.info(
                f"Task {task.task_id} completed by {agent.agent_id} in {execution_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute task {task.task_id}: {e}")
    
    def submit_task(
        self,
        task_data: Dict[str, Any],
        agent_type: Optional[AgentType] = None,
        priority: int = 1,
        **kwargs
    ) -> str:
        """Submit a new task to the system."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{int(time.time())}"
        
        # Determine agent type if not specified
        if agent_type is None:
            agent_type = self._infer_agent_type(task_data)
        
        task = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            task_data=task_data,
            priority=priority,
            **kwargs
        )
        
        self.task_distributor.add_task(task)
        self.logger.info(f"Submitted task {task_id} (type: {agent_type.value})")
        
        return task_id
    
    def _infer_agent_type(self, task_data: Dict[str, Any]) -> AgentType:
        """Infer the best agent type for a task based on its data."""
        # Check for language hints
        if "language" in task_data:
            lang = task_data["language"].lower()
            try:
                return AgentType(f"language_{lang}")
            except ValueError:
                pass
        
        # Check for tool requirements
        if "tools_required" in task_data:
            tools = task_data["tools_required"]
            if tools:
                tool = tools[0].lower()
                try:
                    return AgentType(f"tool_{tool}")
                except ValueError:
                    pass
        
        # Check for domain hints
        if "domain" in task_data:
            domain = task_data["domain"].lower()
            try:
                return AgentType(f"domain_{domain}")
            except ValueError:
                pass
        
        # Default to meta coordinator
        return AgentType.META_COORDINATOR
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        available_agents = self.agent_pool.get_available_agents()
        
        status = {
            "is_running": self.is_running,
            "total_agents": len(self.agent_pool.agents),
            "active_agents": len(available_agents),
            "pending_tasks": len(self.task_distributor.task_queue),
            "metrics": self.system_metrics.copy()
        }
        
        # Calculate throughput (tasks per minute)
        if self.system_metrics["total_execution_time"] > 0:
            throughput = (
                self.system_metrics["total_tasks_processed"] * 60.0 /
                self.system_metrics["total_execution_time"]
            )
            status["metrics"]["system_throughput"] = throughput
        
        # Agent breakdown
        agent_breakdown = {}
        for agent in available_agents:
            agent_type = agent.agent_type.value
            if agent_type not in agent_breakdown:
                agent_breakdown[agent_type] = {
                    "count": 0,
                    "active_tasks": 0,
                    "total_completed": 0
                }
            
            agent_breakdown[agent_type]["count"] += 1
            agent_breakdown[agent_type]["active_tasks"] += len(agent.current_tasks)
            agent_breakdown[agent_type]["total_completed"] += agent.completed_tasks
        
        status["agent_breakdown"] = agent_breakdown
        
        return status
    
    async def wait_for_task_completion(self, task_id: str, timeout: float = 300.0) -> Optional[Dict[str, Any]]:
        """Wait for a specific task to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check all agents for the task
            for agent in self.agent_pool.agents.values():
                for task in agent.current_tasks:
                    if task.task_id == task_id:
                        if task.status in ["completed", "failed"]:
                            return {
                                "task_id": task_id,
                                "status": task.status,
                                "result": task.result,
                                "error": task.error,
                                "execution_time": (task.completed_at or 0) - (task.started_at or 0)
                            }
            
            await asyncio.sleep(0.5)
        
        return None  # Timeout


# Example usage and configuration
EXAMPLE_CONFIG = {
    "max_agents": 16,
    "agent_types": {
        "language_python": {"count": 3, "model_path": "checkpoints/hrm_python.pt"},
        "language_javascript": {"count": 2, "model_path": "checkpoints/hrm_js.pt"},
        "language_rust": {"count": 1, "model_path": "checkpoints/hrm_rust.pt"},
        "tool_git": {"count": 2, "model_path": "checkpoints/hrm_git.pt"},
        "tool_testing": {"count": 2, "model_path": "checkpoints/hrm_test.pt"},
        "domain_web": {"count": 2, "model_path": "checkpoints/hrm_web.pt"},
        "meta_coordinator": {"count": 1, "model_path": "checkpoints/hrm_meta.pt"}
    },
    "deployment": {
        "type": "local",
        "local": {
            "max_memory": "16GB",
            "max_cpu_cores": 8
        }
    }
}


async def main():
    """Example usage of the HRM multi-agent system."""
    # Save example config
    with open("agent_config.yaml", "w") as f:
        yaml.dump(EXAMPLE_CONFIG, f)
    
    # Initialize orchestrator
    orchestrator = HRMMultiAgentOrchestrator("agent_config.yaml")
    
    try:
        # Initialize the system
        if not await orchestrator.initialize():
            print("Failed to initialize orchestrator")
            return
        
        # Start the orchestrator
        await orchestrator.start()
        
        # Submit some example tasks
        task_ids = []
        
        # Python coding task
        task_id = orchestrator.submit_task({
            "problem_statement": "Implement a binary search algorithm",
            "language": "python",
            "complexity_score": 0.6
        })
        task_ids.append(task_id)
        
        # Git task
        task_id = orchestrator.submit_task({
            "commands": ["git status", "git add .", "git commit -m 'Update code'"],
            "tools_required": ["git"]
        })
        task_ids.append(task_id)
        
        # Web development task
        task_id = orchestrator.submit_task({
            "type": "create_component",
            "framework": "react",
            "domain": "web",
            "complexity_score": 0.8
        })
        task_ids.append(task_id)
        
        print(f"Submitted {len(task_ids)} tasks")
        
        # Monitor system for a while
        for i in range(30):  # 30 seconds
            status = orchestrator.get_system_status()
            print(f"System Status: {status['active_agents']} active agents, "
                  f"{status['pending_tasks']} pending tasks, "
                  f"{status['metrics']['total_tasks_processed']} total processed")
            
            await asyncio.sleep(1)
        
        # Wait for tasks to complete
        for task_id in task_ids:
            result = await orchestrator.wait_for_task_completion(task_id)
            if result:
                print(f"Task {task_id}: {result['status']} in {result['execution_time']:.2f}s")
            else:
                print(f"Task {task_id}: Timeout")
    
    finally:
        # Shutdown the system
        await orchestrator.stop()
        print("System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())