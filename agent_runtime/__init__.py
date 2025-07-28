"""
HRM Agent Runtime System

This package provides the runtime infrastructure for executing multiple specialized HRM agents
in parallel using SWE-ReX's sandboxed execution environments.

Main Components:
- swe_rex_integration: Core multi-agent orchestration system
- agent_configs: Configuration templates for different agent types
- monitoring: Performance and health monitoring utilities
"""

from .swe_rex_integration import (
    HRMMultiAgentOrchestrator,
    HRMAgentRuntime,
    HRMAgentPool,
    HRMTaskDistributor,
    AgentType,
    AgentCapabilities,
    AgentTask
)

__version__ = "0.1.0"

__all__ = [
    "HRMMultiAgentOrchestrator",
    "HRMAgentRuntime", 
    "HRMAgentPool",
    "HRMTaskDistributor",
    "AgentType",
    "AgentCapabilities",
    "AgentTask"
]