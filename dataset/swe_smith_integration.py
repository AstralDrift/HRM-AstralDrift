"""
SWE-smith Task Registry Integration for HRM Multi-Agent Training

This module implements the integration between SWE-smith's task registry system
and HRM's hierarchical reasoning architecture, enabling the processing of 52K+
software engineering tasks for multi-agent training.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache

import torch
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

# SWE-smith imports (with fallback for testing)
try:
    from swesmith.profiles import registry as swe_registry
    from swesmith.profiles.base import RepoProfile
    SWE_SMITH_AVAILABLE = True
except ImportError:
    logging.warning("SWE-smith not available, using mock implementation")
    SWE_SMITH_AVAILABLE = False
    swe_registry = None
    RepoProfile = None

# HRM imports  
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from dataset.common import PuzzleDatasetMetadata


logger = logging.getLogger(__name__)


@dataclass
class TaskAnalysis:
    """Analysis results for a SWE-smith task"""
    instance_id: str
    languages: List[str] = field(default_factory=list)
    language_complexity: Dict[str, float] = field(default_factory=dict)
    language_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    language_duration: Dict[str, float] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    tool_complexity: Dict[str, float] = field(default_factory=dict)
    tool_workflows: Dict[str, List[str]] = field(default_factory=dict)
    tool_duration: Dict[str, float] = field(default_factory=dict)
    coordination_needs: Dict[str, Any] = field(default_factory=dict)
    primary_domain: str = "general_programming"
    domain_complexity: float = 0.5
    domain_requirements: List[str] = field(default_factory=list)
    overall_complexity: float = 0.5
    estimated_agents_needed: int = 1
    hierarchical_depth: int = 2


@dataclass
class AgentAssignment:
    """Assignment of a task to a specific agent"""
    agent_type: str
    priority: float
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HRMTrainingInstance:
    """Complete HRM training instance from SWE-smith task"""
    instance_id: str
    raw_task: Dict[str, Any]
    task_analysis: TaskAnalysis
    agent_assignments: List[AgentAssignment]
    hierarchical_data: Dict[str, Any]
    profile: Optional[Any] = None  # RepoProfile if available
    is_valid: bool = True
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageDetector:
    """Detect programming languages in task patches and problem statements"""
    
    def __init__(self):
        self.language_patterns = {
            'python': ['.py', 'python', 'import ', 'def ', 'class ', '__init__'],
            'javascript': ['.js', '.ts', '.jsx', '.tsx', 'javascript', 'function', 'const ', 'let ', 'var '],
            'rust': ['.rs', 'rust', 'fn ', 'let mut', 'impl ', 'use ', 'mod '],
            'go': ['.go', 'golang', 'func ', 'package ', 'import ', 'type '],
            'java': ['.java', 'java', 'public class', 'private ', 'public ', 'static '],
            'cpp': ['.cpp', '.cc', '.cxx', '.h', '.hpp', 'c++', '#include', 'std::', 'namespace']
        }
    
    def detect_languages(self, patch_content: str, profile: Optional[Any] = None) -> List[str]:
        """Detect programming languages in the patch content"""
        
        detected = set()
        patch_lower = patch_content.lower()
        
        # Use profile information if available
        if profile and hasattr(profile, 'exts'):
            for ext in profile.exts:
                for lang, patterns in self.language_patterns.items():
                    if ext in patterns:
                        detected.add(lang)
        
        # Pattern-based detection
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if pattern in patch_lower:
                    detected.add(lang)
                    break
        
        # Default to Python if no language detected (common in SWE-smith)
        if not detected:
            detected.add('python')
        
        return list(detected)


class ComplexityAnalyzer:
    """Analyze task complexity for different aspects"""
    
    def analyze_language_complexity(
        self, 
        patch_content: str, 
        language: str, 
        problem_statement: str
    ) -> float:
        """Analyze complexity of language-specific code"""
        
        complexity_indicators = {
            'lines_changed': len([line for line in patch_content.split('\n') if line.strip()]),
            'files_modified': len([line for line in patch_content.split('\n') if line.startswith('diff --git')]),
            'imports_added': patch_content.count('import ') + patch_content.count('from '),
            'functions_added': patch_content.count('def ') + patch_content.count('function '),
            'classes_added': patch_content.count('class '),
            'async_complexity': patch_content.count('async ') + patch_content.count('await '),
            'error_handling': patch_content.count('try:') + patch_content.count('catch(') + patch_content.count('except'),
            'generic_types': patch_content.count('<') + patch_content.count('>'),
        }
        
        # Normalize and weight indicators
        weights = {
            'lines_changed': 0.001,  # Per line
            'files_modified': 0.2,   # Per file
            'imports_added': 0.05,   # Per import
            'functions_added': 0.1,  # Per function
            'classes_added': 0.15,   # Per class
            'async_complexity': 0.1, # Async complexity
            'error_handling': 0.05,  # Error handling
            'generic_types': 0.02,   # Generic complexity
        }
        
        complexity_score = sum(
            complexity_indicators[key] * weights[key] 
            for key in complexity_indicators
        )
        
        # Problem statement complexity
        statement_complexity = (
            len(problem_statement.split()) * 0.001 +  # Word count
            problem_statement.count('performance') * 0.1 +  # Performance requirements
            problem_statement.count('thread') * 0.1 +  # Threading complexity
            problem_statement.count('database') * 0.05 +  # Database complexity
            problem_statement.count('api') * 0.05  # API complexity
        )
        
        total_complexity = complexity_score + statement_complexity
        
        # Normalize to 0-1 range
        return min(max(total_complexity, 0.0), 1.0)
    
    def analyze_tool_complexity(
        self, 
        patch_content: str, 
        tool: str, 
        problem_statement: str
    ) -> float:
        """Analyze complexity of tool usage"""
        
        tool_complexity_indicators = {
            'git': {
                'merge_conflicts': patch_content.count('<<<<<<') + patch_content.count('>>>>>>'),
                'branch_operations': problem_statement.lower().count('branch') + problem_statement.lower().count('merge'),
                'commit_complexity': problem_statement.lower().count('commit') + problem_statement.lower().count('revert')
            },
            'docker': {
                'dockerfile_changes': patch_content.count('FROM ') + patch_content.count('RUN '),
                'container_orchestration': problem_statement.lower().count('container') + problem_statement.lower().count('docker'),
                'networking': problem_statement.lower().count('port') + problem_statement.lower().count('network')
            },
            'build': {
                'build_config_changes': patch_content.count('Makefile') + patch_content.count('CMakeLists'),
                'dependency_changes': patch_content.count('requirements.txt') + patch_content.count('package.json'),
                'compiler_flags': patch_content.count('-O') + patch_content.count('-g')
            }
        }
        
        if tool in tool_complexity_indicators:
            indicators = tool_complexity_indicators[tool]
            complexity = sum(indicators.values()) * 0.1
            return min(max(complexity, 0.1), 1.0)
        
        return 0.3  # Default moderate complexity


class DependencyAnalyzer:
    """Analyze dependencies between different components"""
    
    def analyze_language_dependencies(
        self, 
        patch_content: str, 
        language: str, 
        profile: Optional[Any] = None
    ) -> List[str]:
        """Analyze dependencies for a specific language"""
        
        dependencies = []
        
        # Extract import statements
        lines = patch_content.split('\n')
        
        if language == 'python':
            for line in lines:
                if 'import ' in line:
                    # Extract module names
                    if line.strip().startswith('import '):
                        module = line.strip().replace('import ', '').split(' as ')[0].split('.')[0]
                        dependencies.append(module)
                    elif line.strip().startswith('from '):
                        module = line.strip().split('from ')[1].split(' import')[0].split('.')[0]
                        dependencies.append(module)
        
        elif language == 'javascript':
            for line in lines:
                if 'require(' in line or 'import ' in line:
                    # Extract package names
                    if 'require(' in line:
                        start = line.find("require('") + 9
                        if start > 8:
                            end = line.find("'", start)
                            if end > start:
                                dependencies.append(line[start:end])
                    elif 'import ' in line and 'from ' in line:
                        start = line.find("from '") + 6
                        if start > 5:
                            end = line.find("'", start)
                            if end > start:
                                dependencies.append(line[start:end])
        
        # Remove duplicates and filter out standard library
        unique_deps = list(set(dependencies))
        filtered_deps = [dep for dep in unique_deps if len(dep) > 1 and dep not in ['os', 'sys', 'time']]
        
        return filtered_deps


class ToolDetector:
    """Detect required development tools from task content"""
    
    def __init__(self):
        self.tool_indicators = {
            'git': ['git', 'merge', 'branch', 'commit', 'pull request', 'pr', 'diff', 'patch'],
            'docker': ['docker', 'container', 'dockerfile', 'image', 'compose'],
            'npm': ['npm', 'package.json', 'node_modules', 'yarn', 'pnpm'],
            'pip': ['pip', 'requirements.txt', 'setup.py', 'pyproject.toml'],
            'cargo': ['cargo', 'Cargo.toml', 'rust', 'crate'],
            'maven': ['maven', 'pom.xml', 'mvn'],
            'gradle': ['gradle', 'build.gradle', 'gradlew'],
            'make': ['make', 'Makefile', 'cmake', 'CMakeLists'],
            'test': ['test', 'pytest', 'jest', 'mocha', 'junit', 'unittest'],
            'debug': ['debug', 'gdb', 'lldb', 'pdb', 'debugger'],
            'lint': ['lint', 'flake8', 'eslint', 'pylint', 'rustfmt'],
            'format': ['format', 'prettier', 'black', 'rustfmt']
        }
    
    def detect_required_tools(
        self, 
        patch_content: str, 
        problem_statement: str, 
        profile: Optional[Any] = None
    ) -> List[str]:
        """Detect development tools required for the task"""
        
        detected_tools = set()
        combined_text = (patch_content + " " + problem_statement).lower()
        
        for tool, indicators in self.tool_indicators.items():
            for indicator in indicators:
                if indicator in combined_text:
                    detected_tools.add(tool)
                    break
        
        # Add language-specific default tools
        if 'python' in combined_text or '.py' in patch_content:
            detected_tools.update(['pip', 'test'])
        if 'javascript' in combined_text or '.js' in patch_content:
            detected_tools.update(['npm', 'test'])
        if 'rust' in combined_text or '.rs' in patch_content:
            detected_tools.update(['cargo', 'test'])
        if 'java' in combined_text or '.java' in patch_content:
            detected_tools.update(['maven', 'test'])
        
        # Always include git for version control
        detected_tools.add('git')
        
        return list(detected_tools)


class CoordinationAnalyzer:
    """Analyze multi-agent coordination requirements"""
    
    def analyze_coordination_needs(
        self,
        languages: List[str],
        required_tools: List[str],
        patch_content: str,
        test_cases: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Analyze coordination requirements between agents"""
        
        coordination_needs = {
            'parallel_execution': len(languages) > 1 or len(required_tools) > 3,
            'sequential_dependencies': self._analyze_sequential_deps(patch_content),
            'shared_resources': self._analyze_shared_resources(patch_content),
            'communication_overhead': self._estimate_communication_overhead(languages, required_tools),
            'synchronization_points': self._identify_sync_points(patch_content, test_cases),
            'failure_recovery': self._assess_failure_recovery_needs(patch_content, test_cases)
        }
        
        # Calculate coordination complexity score
        complexity_factors = {
            'parallel_execution': 0.3 if coordination_needs['parallel_execution'] else 0.0,
            'sequential_dependencies': len(coordination_needs['sequential_dependencies']) * 0.1,
            'shared_resources': len(coordination_needs['shared_resources']) * 0.1,
            'communication_overhead': coordination_needs['communication_overhead'],
            'synchronization_points': len(coordination_needs['synchronization_points']) * 0.05,
            'failure_recovery': 0.2 if coordination_needs['failure_recovery'] else 0.0
        }
        
        coordination_needs['complexity_score'] = min(sum(complexity_factors.values()), 1.0)
        
        return coordination_needs
    
    def _analyze_sequential_deps(self, patch_content: str) -> List[str]:
        """Identify sequential dependencies in the task"""
        
        sequential_deps = []
        
        # Look for build dependencies
        if 'requirements.txt' in patch_content or 'package.json' in patch_content:
            sequential_deps.append('dependency_installation')
        
        # Look for compilation steps
        if any(ext in patch_content for ext in ['.c', '.cpp', '.rs', '.java']):
            sequential_deps.append('compilation')
        
        # Look for database migrations
        if any(term in patch_content.lower() for term in ['migration', 'schema', 'database']):
            sequential_deps.append('database_setup')
        
        return sequential_deps
    
    def _analyze_shared_resources(self, patch_content: str) -> List[str]:
        """Identify shared resources that need coordination"""
        
        shared_resources = []
        
        # File system resources
        if any(term in patch_content for term in ['file', 'directory', 'path']):
            shared_resources.append('filesystem')
        
        # Network resources
        if any(term in patch_content.lower() for term in ['port', 'socket', 'network', 'api']):
            shared_resources.append('network')
        
        # Database resources
        if any(term in patch_content.lower() for term in ['database', 'db', 'sql', 'table']):
            shared_resources.append('database')
        
        # Memory resources
        if any(term in patch_content.lower() for term in ['memory', 'cache', 'buffer']):
            shared_resources.append('memory')
        
        return shared_resources
    
    def _estimate_communication_overhead(self, languages: List[str], tools: List[str]) -> float:
        """Estimate communication overhead between agents"""
        
        # Base overhead for multi-language projects
        language_overhead = max(0, (len(languages) - 1) * 0.1)
        
        # Tool coordination overhead
        tool_overhead = max(0, (len(tools) - 2) * 0.05)
        
        # Cross-language communication penalty
        cross_lang_penalty = 0.1 if len(languages) > 2 else 0.0
        
        total_overhead = language_overhead + tool_overhead + cross_lang_penalty
        return min(total_overhead, 0.5)  # Cap at 50% overhead
    
    def _identify_sync_points(
        self, 
        patch_content: str, 
        test_cases: Dict[str, List[str]]
    ) -> List[str]:
        """Identify synchronization points in the workflow"""
        
        sync_points = []
        
        # Test execution synchronization
        if test_cases.get('FAIL_TO_PASS') or test_cases.get('PASS_TO_PASS'):
            sync_points.append('test_execution')
        
        # Build synchronization
        if any(term in patch_content for term in ['build', 'compile', 'make']):
            sync_points.append('build_completion')
        
        # Integration points
        if 'integration' in patch_content.lower():
            sync_points.append('integration_testing')
        
        return sync_points
    
    def _assess_failure_recovery_needs(
        self, 
        patch_content: str, 
        test_cases: Dict[str, List[str]]
    ) -> bool:
        """Assess if failure recovery mechanisms are needed"""
        
        # High failure risk indicators
        risk_indicators = [
            'performance' in patch_content.lower(),
            'thread' in patch_content.lower(),
            'concurrent' in patch_content.lower(),
            'async' in patch_content.lower(),
            len(test_cases.get('FAIL_TO_PASS', [])) > 3,
            'error' in patch_content.lower(),
            'exception' in patch_content.lower()
        ]
        
        return sum(risk_indicators) >= 2


class HRMTaskAnalyzer:
    """Main task analyzer for HRM-specific requirements"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.tool_detector = ToolDetector()
        self.coordination_analyzer = CoordinationAnalyzer()
    
    def analyze_task(self, task: Dict[str, Any], profile: Optional[Any] = None) -> TaskAnalysis:
        """Comprehensive analysis of SWE-smith task for HRM training"""
        
        # Extract core information
        instance_id = task['instance_id']
        patch_content = task.get('patch', '')
        problem_statement = task.get('problem_statement', '')
        test_cases = {
            'FAIL_TO_PASS': task.get('FAIL_TO_PASS', []),
            'PASS_TO_PASS': task.get('PASS_TO_PASS', [])
        }
        
        # Language analysis
        languages = self.language_detector.detect_languages(patch_content, profile)
        language_complexity = {}
        language_dependencies = {}
        language_duration = {}
        
        for lang in languages:
            complexity = self.complexity_analyzer.analyze_language_complexity(
                patch_content, lang, problem_statement
            )
            dependencies = self.dependency_analyzer.analyze_language_dependencies(
                patch_content, lang, profile
            )
            duration = self._estimate_language_duration(complexity, len(dependencies))
            
            language_complexity[lang] = complexity
            language_dependencies[lang] = dependencies
            language_duration[lang] = duration
        
        # Tool analysis
        required_tools = self.tool_detector.detect_required_tools(
            patch_content, problem_statement, profile
        )
        tool_complexity = {}
        tool_workflows = {}
        tool_duration = {}
        
        for tool in required_tools:
            complexity = self.complexity_analyzer.analyze_tool_complexity(
                patch_content, tool, problem_statement
            )
            workflow = self._generate_tool_workflow(tool, patch_content)
            duration = self._estimate_tool_duration(complexity)
            
            tool_complexity[tool] = complexity
            tool_workflows[tool] = workflow
            tool_duration[tool] = duration
        
        # Coordination analysis
        coordination_needs = self.coordination_analyzer.analyze_coordination_needs(
            languages, required_tools, patch_content, test_cases
        )
        
        # Domain classification
        primary_domain = self._classify_primary_domain(problem_statement, patch_content)
        domain_complexity = self._calculate_domain_complexity(primary_domain, patch_content)
        domain_requirements = self._extract_domain_requirements(primary_domain)
        
        # Overall metrics
        overall_complexity = self._calculate_overall_complexity(
            language_complexity, tool_complexity, coordination_needs, domain_complexity
        )
        estimated_agents = self._estimate_agents_needed(languages, required_tools, coordination_needs)
        hierarchical_depth = self._estimate_hierarchical_depth(overall_complexity)
        
        return TaskAnalysis(
            instance_id=instance_id,
            languages=languages,
            language_complexity=language_complexity,
            language_dependencies=language_dependencies,
            language_duration=language_duration,
            required_tools=required_tools,
            tool_complexity=tool_complexity,
            tool_workflows=tool_workflows,
            tool_duration=tool_duration,
            coordination_needs=coordination_needs,
            primary_domain=primary_domain,
            domain_complexity=domain_complexity,
            domain_requirements=domain_requirements,
            overall_complexity=overall_complexity,
            estimated_agents_needed=estimated_agents,
            hierarchical_depth=hierarchical_depth
        )
    
    def _estimate_language_duration(self, complexity: float, dependency_count: int) -> float:
        """Estimate duration for language-specific work"""
        base_duration = 30.0  # 30 seconds base
        complexity_factor = complexity * 60.0  # Up to 60 seconds for complexity
        dependency_factor = dependency_count * 5.0  # 5 seconds per dependency
        return base_duration + complexity_factor + dependency_factor
    
    def _generate_tool_workflow(self, tool: str, patch_content: str) -> List[str]:
        """Generate workflow steps for tool usage"""
        
        workflows = {
            'git': ['clone_repo', 'checkout_branch', 'apply_changes', 'commit', 'push'],
            'docker': ['build_image', 'run_container', 'execute_tests', 'cleanup'],
            'npm': ['install_dependencies', 'run_build', 'run_tests'],
            'pip': ['create_venv', 'install_requirements', 'run_tests'],
            'cargo': ['check_dependencies', 'build', 'test', 'format'],
            'test': ['setup_test_env', 'run_tests', 'collect_results', 'cleanup'],
            'debug': ['attach_debugger', 'set_breakpoints', 'step_through', 'analyze_state'],
            'lint': ['run_linter', 'fix_issues', 'verify_compliance'],
            'format': ['format_code', 'verify_formatting']
        }
        
        return workflows.get(tool, ['setup', 'execute', 'cleanup'])
    
    def _estimate_tool_duration(self, complexity: float) -> float:
        """Estimate duration for tool usage"""
        base_duration = 15.0  # 15 seconds base
        complexity_factor = complexity * 45.0  # Up to 45 seconds for complexity
        return base_duration + complexity_factor
    
    def _classify_primary_domain(self, problem_statement: str, patch_content: str) -> str:
        """Classify task into primary software engineering domain"""
        
        domain_indicators = {
            'web_development': ['http', 'api', 'web', 'server', 'client', 'html', 'css', 'rest'],
            'data_science': ['pandas', 'numpy', 'sklearn', 'data', 'analysis', 'ml', 'model', 'dataframe'],
            'systems_programming': ['memory', 'performance', 'optimization', 'low-level', 'kernel', 'buffer'],
            'devops': ['docker', 'deployment', 'ci', 'cd', 'infrastructure', 'monitoring', 'kubernetes'],
            'backend_development': ['database', 'sql', 'orm', 'backend', 'microservice', 'crud'],
            'frontend_development': ['ui', 'ux', 'component', 'frontend', 'react', 'vue', 'angular'],
            'mobile_development': ['mobile', 'ios', 'android', 'app', 'native', 'flutter'],
            'game_development': ['game', 'graphics', 'animation', 'engine', 'physics', 'unity'],
            'security': ['security', 'auth', 'encryption', 'vulnerability', 'ssl', 'certificate', 'oauth']
        }
        
        combined_text = f"{problem_statement} {patch_content}".lower()
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            domain_scores[domain] = score
        
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])
            return primary_domain[0] if primary_domain[1] > 0 else 'general_programming'
        
        return 'general_programming'
    
    def _calculate_domain_complexity(self, domain: str, patch_content: str) -> float:
        """Calculate domain-specific complexity"""
        
        domain_complexity_factors = {
            'web_development': 0.6,
            'data_science': 0.7,
            'systems_programming': 0.9,
            'devops': 0.8,
            'backend_development': 0.6,
            'frontend_development': 0.5,
            'mobile_development': 0.7,
            'game_development': 0.8,
            'security': 0.9,
            'general_programming': 0.5
        }
        
        base_complexity = domain_complexity_factors.get(domain, 0.5)
        
        # Adjust based on patch size and content
        patch_size_factor = min(len(patch_content) / 1000.0, 0.3)  # Up to 0.3 boost for large patches
        
        return min(base_complexity + patch_size_factor, 1.0)
    
    def _extract_domain_requirements(self, domain: str) -> List[str]:
        """Extract domain-specific requirements"""
        
        domain_requirements = {
            'web_development': ['http_handling', 'api_design', 'web_security', 'performance'],
            'data_science': ['data_processing', 'statistical_analysis', 'visualization', 'model_evaluation'],
            'systems_programming': ['memory_management', 'performance_optimization', 'concurrency', 'system_calls'],
            'devops': ['containerization', 'orchestration', 'monitoring', 'automation'],
            'backend_development': ['database_design', 'api_development', 'scalability', 'data_validation'],
            'frontend_development': ['user_interface', 'user_experience', 'responsive_design', 'accessibility'],
            'mobile_development': ['platform_specific', 'mobile_ui', 'app_lifecycle', 'device_integration'],
            'game_development': ['graphics_programming', 'physics_simulation', 'game_logic', 'optimization'],
            'security': ['threat_modeling', 'encryption', 'authentication', 'vulnerability_assessment'],
            'general_programming': ['code_quality', 'testing', 'documentation', 'maintainability']
        }
        
        return domain_requirements.get(domain, ['code_quality', 'testing'])
    
    def _calculate_overall_complexity(
        self,
        language_complexity: Dict[str, float],
        tool_complexity: Dict[str, float],
        coordination_needs: Dict[str, Any],
        domain_complexity: float
    ) -> float:
        """Calculate overall task complexity"""
        
        # Language complexity (weighted average)
        lang_complexity = sum(language_complexity.values()) / len(language_complexity) if language_complexity else 0.5
        
        # Tool complexity (weighted average)
        tool_complexity_avg = sum(tool_complexity.values()) / len(tool_complexity) if tool_complexity else 0.3
        
        # Coordination complexity
        coordination_complexity = coordination_needs.get('complexity_score', 0.3)
        
        # Weighted combination
        overall = (
            lang_complexity * 0.4 +
            tool_complexity_avg * 0.25 +
            coordination_complexity * 0.2 +
            domain_complexity * 0.15
        )
        
        return min(max(overall, 0.1), 1.0)
    
    def _estimate_agents_needed(
        self, 
        languages: List[str], 
        tools: List[str], 
        coordination_needs: Dict[str, Any]
    ) -> int:
        """Estimate number of agents needed for the task"""
        
        # Base agents needed
        base_agents = 1
        
        # Language-specific agents
        language_agents = len(languages) if len(languages) > 1 else 0
        
        # Tool-specific agents (for complex tools)
        complex_tools = ['docker', 'debug', 'performance']
        tool_agents = len([tool for tool in tools if tool in complex_tools])
        
        # Coordination overhead
        coordination_agents = 1 if coordination_needs.get('parallel_execution', False) else 0
        
        total_agents = base_agents + language_agents + tool_agents + coordination_agents
        
        return min(max(total_agents, 1), 10)  # Cap at 10 agents
    
    def _estimate_hierarchical_depth(self, complexity: float) -> int:
        """Estimate required hierarchical reasoning depth"""
        
        if complexity < 0.3:
            return 2  # Simple H_level + L_level
        elif complexity < 0.6:
            return 3  # H_level + intermediate + L_level
        elif complexity < 0.8:
            return 4  # Deep hierarchical reasoning
        else:
            return 5  # Maximum depth for very complex tasks


class SWESmithTaskRegistry:
    """Registry for loading and managing SWE-smith tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.task_analyzer = HRMTaskAnalyzer()
        self.cache_dir = Path(self.config.get('cache_dir', 'cache/swe_smith'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.processing_stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize registry
        self.swe_registry = None
        if SWE_SMITH_AVAILABLE:
            try:
                self.swe_registry = swe_registry
                logger.info("SWE-smith registry initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize SWE-smith registry: {e}")
        else:
            logger.warning("SWE-smith not available, using fallback implementation")
    
    @lru_cache(maxsize=1000)
    def get_profile_for_task(self, repo_name: str) -> Optional[Any]:
        """Get repository profile for a task (cached)"""
        
        if not self.swe_registry:
            return None
        
        try:
            # Create a mock task instance for profile lookup
            mock_instance = {'repo': repo_name, 'instance_id': f"{repo_name}.mock"}
            profile = self.swe_registry.get_from_inst(mock_instance)
            return profile
        except Exception as e:
            logger.debug(f"Could not get profile for {repo_name}: {e}")
            return None
    
    async def load_dataset_sample(
        self, 
        num_samples: int = 1000,
        split: str = 'train'
    ) -> List[Dict[str, Any]]:
        """Load sample of tasks from SWE-smith dataset"""
        
        try:
            # Load dataset from HuggingFace
            logger.info(f"Loading {num_samples} samples from SWE-smith dataset...")
            
            dataset = load_dataset(
                'SWE-bench/SWE-smith', 
                split=f'{split}[:{num_samples}]',
                trust_remote_code=True
            )
            
            tasks = [dict(item) for item in dataset]
            logger.info(f"Successfully loaded {len(tasks)} tasks")
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to load SWE-smith dataset: {e}")
            
            # Fallback to mock data for testing
            logger.info("Using mock data for testing")
            return self._generate_mock_tasks(num_samples)
    
    def _generate_mock_tasks(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate mock tasks for testing when dataset is unavailable"""
        
        mock_tasks = []
        
        for i in range(min(num_samples, 10)):  # Limit mock data
            task = {
                'instance_id': f'mock__repo.{i:06d}.pr_{1000+i}',
                'repo': f'mock/repo-{i % 3}',
                'patch': f'''diff --git a/src/module_{i}.py b/src/module_{i}.py
index abc123..def456 100644
--- a/src/module_{i}.py
+++ b/src/module_{i}.py
@@ -1,5 +1,8 @@
 def example_function():
+    # Added error handling
+    try:
         result = process_data()
         return result
+    except Exception as e:
+        return None
''',
                'problem_statement': f'''# Issue {1000+i}: Improve error handling in module_{i}
                
The current implementation of module_{i} doesn't handle errors properly. 
We need to add proper exception handling to prevent crashes.

## Expected behavior
The function should return None when an error occurs instead of crashing.

## Current behavior  
The function crashes with unhandled exceptions.
''',
                'FAIL_TO_PASS': [f'tests/test_module_{i}.py::test_error_handling'],
                'PASS_TO_PASS': [f'tests/test_module_{i}.py::test_normal_operation'],
                'image_name': f'swesmith.x86_64.mock-repo-{i}'
            }
            mock_tasks.append(task)
        
        return mock_tasks
    
    async def process_task_batch(
        self, 
        tasks: List[Dict[str, Any]], 
        batch_size: int = 32
    ) -> List[HRMTrainingInstance]:
        """Process batch of raw tasks into HRM training instances"""
        
        processed_instances = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            # Use thread pool for CPU-bound processing
            with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
                futures = [
                    executor.submit(self._process_single_task, task)
                    for task in batch
                ]
                
                # Collect results
                for future in futures:
                    try:
                        instance = future.result(timeout=30)  # 30 second timeout
                        if instance and instance.is_valid:
                            processed_instances.append(instance)
                    except Exception as e:
                        logger.error(f"Failed to process task: {e}")
                        self.processing_stats['tasks_failed'] += 1
            
            # Progress update
            logger.info(f"Processed {min(i + batch_size, len(tasks))}/{len(tasks)} tasks")
        
        logger.info(f"Successfully processed {len(processed_instances)}/{len(tasks)} tasks")
        return processed_instances
    
    def _process_single_task(self, raw_task: Dict[str, Any]) -> Optional[HRMTrainingInstance]:
        """Process single task into HRM training instance"""
        
        start_time = time.time()
        
        try:
            # Extract basic information
            instance_id = raw_task['instance_id']
            repo_name = raw_task.get('repo', '')
            
            # Check cache first
            cache_key = f"{instance_id}.pkl"
            cache_path = self.cache_dir / cache_key
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_instance = pickle.load(f)
                    self.processing_stats['cache_hits'] += 1
                    return cached_instance
                except Exception as e:
                    logger.debug(f"Cache read failed for {instance_id}: {e}")
            
            self.processing_stats['cache_misses'] += 1
            
            # Get repository profile
            profile = self.get_profile_for_task(repo_name)
            
            # Analyze task
            task_analysis = self.task_analyzer.analyze_task(raw_task, profile)
            
            # Generate agent assignments
            agent_assignments = self._generate_agent_assignments(task_analysis, raw_task)
            
            # Generate hierarchical training data structure
            hierarchical_data = self._generate_hierarchical_data(task_analysis, raw_task, agent_assignments)
            
            # Create training instance
            instance = HRMTrainingInstance(
                instance_id=instance_id,
                raw_task=raw_task,
                task_analysis=task_analysis,
                agent_assignments=agent_assignments,
                hierarchical_data=hierarchical_data,
                profile=profile,
                is_valid=True,
                processing_metadata={
                    'processing_time': time.time() - start_time,
                    'cache_hit': False
                }
            )
            
            # Cache the result
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(instance, f)
            except Exception as e:
                logger.debug(f"Cache write failed for {instance_id}: {e}")
            
            # Update stats
            self.processing_stats['tasks_processed'] += 1
            processing_time = time.time() - start_time
            self.processing_stats['average_processing_time'] = (
                (self.processing_stats['average_processing_time'] * (self.processing_stats['tasks_processed'] - 1) + processing_time) /
                self.processing_stats['tasks_processed']
            )
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to process task {raw_task.get('instance_id', 'unknown')}: {e}")
            self.processing_stats['tasks_failed'] += 1
            return None
    
    def _generate_agent_assignments(
        self, 
        task_analysis: TaskAnalysis, 
        raw_task: Dict[str, Any]
    ) -> List[AgentAssignment]:
        """Generate optimal agent assignments for the task"""
        
        assignments = []
        
        # Language-specific agents
        for language in task_analysis.languages:
            assignment = AgentAssignment(
                agent_type=f'language_{language}',
                priority=task_analysis.language_complexity[language],
                estimated_duration=task_analysis.language_duration[language],
                dependencies=task_analysis.language_dependencies[language],
                context={
                    'language': language,
                    'complexity': task_analysis.language_complexity[language],
                    'code_context': self._extract_language_context(raw_task, language)
                }
            )
            assignments.append(assignment)
        
        # Tool-specific agents
        for tool in task_analysis.required_tools:
            assignment = AgentAssignment(
                agent_type=f'tool_{tool}',
                priority=task_analysis.tool_complexity[tool],
                estimated_duration=task_analysis.tool_duration[tool],
                dependencies=[],
                context={
                    'tool': tool,
                    'workflow': task_analysis.tool_workflows[tool],
                    'complexity': task_analysis.tool_complexity[tool]
                }
            )
            assignments.append(assignment)
        
        # Domain-specific agent if needed
        if task_analysis.primary_domain != 'general_programming':
            assignment = AgentAssignment(
                agent_type=f'domain_{task_analysis.primary_domain}',
                priority=task_analysis.domain_complexity,
                estimated_duration=60.0,  # Domain agents typically take longer
                dependencies=[f'language_{lang}' for lang in task_analysis.languages],
                context={
                    'domain': task_analysis.primary_domain,
                    'requirements': task_analysis.domain_requirements,
                    'complexity': task_analysis.domain_complexity
                }
            )
            assignments.append(assignment)
        
        # Sort by priority (highest first)
        assignments.sort(key=lambda x: x.priority, reverse=True)
        
        return assignments
    
    def _extract_language_context(self, raw_task: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Extract language-specific context from the task"""
        
        patch_content = raw_task.get('patch', '')
        problem_statement = raw_task.get('problem_statement', '')
        
        # Extract relevant code sections
        code_sections = []
        lines = patch_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('@@'):  # Diff header
                # Extract surrounding context
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 10)
                context = '\n'.join(lines[context_start:context_end])
                code_sections.append(context)
        
        return {
            'language': language,
            'code_sections': code_sections,
            'problem_context': problem_statement,
            'test_requirements': raw_task.get('FAIL_TO_PASS', []),
            'existing_tests': raw_task.get('PASS_TO_PASS', [])
        }
    
    def _generate_hierarchical_data(
        self, 
        task_analysis: TaskAnalysis, 
        raw_task: Dict[str, Any],
        agent_assignments: List[AgentAssignment]
    ) -> Dict[str, Any]:
        """Generate hierarchical training data structure for HRM"""
        
        return {
            'meta_agent_data': {
                'strategic_planning': {
                    'problem_analysis': raw_task.get('problem_statement', ''),
                    'complexity_assessment': task_analysis.overall_complexity,
                    'approach_selection': task_analysis.primary_domain,
                    'resource_requirements': task_analysis.estimated_agents_needed
                },
                'agent_coordination': {
                    'agent_assignments': [
                        {
                            'agent_type': assignment.agent_type,
                            'priority': assignment.priority,
                            'dependencies': assignment.dependencies
                        }
                        for assignment in agent_assignments
                    ],
                    'coordination_strategy': task_analysis.coordination_needs,
                    'synchronization_points': task_analysis.coordination_needs.get('synchronization_points', [])
                }
            },
            'specialized_agent_data': {
                'language_agents': {
                    lang: {
                        'complexity': task_analysis.language_complexity[lang],
                        'dependencies': task_analysis.language_dependencies[lang],
                        'estimated_duration': task_analysis.language_duration[lang]
                    }
                    for lang in task_analysis.languages
                },
                'tool_agents': {
                    tool: {
                        'complexity': task_analysis.tool_complexity[tool],
                        'workflow': task_analysis.tool_workflows[tool],
                        'estimated_duration': task_analysis.tool_duration[tool]
                    }
                    for tool in task_analysis.required_tools
                },
                'domain_agent': {
                    'domain': task_analysis.primary_domain,
                    'complexity': task_analysis.domain_complexity,
                    'requirements': task_analysis.domain_requirements
                }
            },
            'coordination_data': {
                'parallel_execution': task_analysis.coordination_needs.get('parallel_execution', False),
                'sequential_dependencies': task_analysis.coordination_needs.get('sequential_dependencies', []),
                'shared_resources': task_analysis.coordination_needs.get('shared_resources', []),
                'failure_recovery': task_analysis.coordination_needs.get('failure_recovery', False)
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def clear_cache(self):
        """Clear the task processing cache"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Task processing cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Integration with HRM puzzle dataset format
class HRMSWESmithDataset:
    """Convert SWE-smith data to HRM puzzle dataset format"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = SWESmithTaskRegistry(config)
        
    async def create_hrm_dataset(
        self, 
        num_samples: int = 1000,
        output_dir: str = 'data/swe_smith_hrm'
    ) -> PuzzleDatasetMetadata:
        """Create HRM-compatible dataset from SWE-smith tasks"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process tasks
        logger.info(f"Creating HRM dataset with {num_samples} SWE-smith tasks...")
        
        raw_tasks = await self.registry.load_dataset_sample(num_samples)
        processed_instances = await self.registry.process_task_batch(raw_tasks)
        
        # Convert to HRM format
        hrm_data = []
        puzzle_identifiers = []
        
        for i, instance in enumerate(processed_instances):
            # Create puzzle identifier
            puzzle_id = i
            puzzle_identifiers.append(puzzle_id)
            
            # Convert to HRM input format
            hrm_instance = self._convert_to_hrm_format(instance, puzzle_id)
            hrm_data.append(hrm_instance)
        
        # Save dataset
        dataset_file = output_path / 'instances.json'
        with open(dataset_file, 'w') as f:
            json.dump(hrm_data, f, indent=2)
        
        # Create metadata with all required fields
        metadata = PuzzleDatasetMetadata(
            pad_id=0,  # Standard padding token
            ignore_label_id=-100,  # Standard ignore token for loss calculation
            blank_identifier_id=0,  # Standard blank puzzle identifier
            vocab_size=50000,  # Approximate for code tokenization
            seq_len=2048,  # Maximum sequence length
            num_puzzle_identifiers=len(set(puzzle_identifiers)),  # Number of unique puzzle types
            total_groups=1,  # Single group for SWE-smith tasks
            mean_puzzle_examples=float(len(hrm_data)),  # Average examples per puzzle type
            sets=['train', 'test']  # Available dataset splits
        )
        
        # Save metadata
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)
        
        logger.info(f"Created HRM dataset with {len(hrm_data)} instances at {output_path}")
        
        return metadata
    
    def _convert_to_hrm_format(
        self, 
        instance: HRMTrainingInstance, 
        puzzle_id: int
    ) -> Dict[str, Any]:
        """Convert SWE-smith instance to HRM puzzle format"""
        
        # Create input sequence (problem statement + context)
        input_text = f"PROBLEM: {instance.raw_task.get('problem_statement', '')}\n"
        input_text += f"REPO: {instance.raw_task.get('repo', '')}\n"
        input_text += f"LANGUAGES: {', '.join(instance.task_analysis.languages)}\n"
        input_text += f"TOOLS: {', '.join(instance.task_analysis.required_tools)}\n"
        input_text += f"DOMAIN: {instance.task_analysis.primary_domain}\n"
        
        # Create target sequence (patch content)
        target_text = instance.raw_task.get('patch', '')
        
        # Create agent coordination target
        coordination_target = {
            'agent_assignments': [
                {
                    'type': assignment.agent_type,
                    'priority': assignment.priority,
                    'duration': assignment.estimated_duration
                }
                for assignment in instance.agent_assignments
            ],
            'coordination_complexity': instance.task_analysis.coordination_needs.get('complexity_score', 0.3)
        }
        
        return {
            'puzzle_id': puzzle_id,
            'instance_id': instance.instance_id,
            'input_text': input_text,
            'target_text': target_text,
            'coordination_target': coordination_target,
            'metadata': {
                'languages': instance.task_analysis.languages,
                'tools': instance.task_analysis.required_tools,
                'domain': instance.task_analysis.primary_domain,
                'complexity': instance.task_analysis.overall_complexity,
                'agents_needed': instance.task_analysis.estimated_agents_needed,
                'hierarchical_depth': instance.task_analysis.hierarchical_depth
            }
        }


# Example usage and testing
async def main():
    """Example usage of SWE-smith integration"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create registry
    config = {
        'cache_dir': 'cache/swe_smith_test',
        'max_workers': 4
    }
    
    registry = SWESmithTaskRegistry(config)
    
    # Load sample tasks
    print("Loading SWE-smith tasks...")
    tasks = await registry.load_dataset_sample(num_samples=50)
    print(f"Loaded {len(tasks)} tasks")
    
    # Process tasks
    print("Processing tasks...")
    instances = await registry.process_task_batch(tasks, batch_size=10)
    print(f"Processed {len(instances)} instances")
    
    # Show processing stats
    stats = registry.get_processing_stats()
    print(f"Processing stats: {stats}")
    
    # Show sample instance
    if instances:
        sample = instances[0]
        print(f"\nSample instance: {sample.instance_id}")
        print(f"Languages: {sample.task_analysis.languages}")
        print(f"Tools: {sample.task_analysis.required_tools}")
        print(f"Domain: {sample.task_analysis.primary_domain}")
        print(f"Complexity: {sample.task_analysis.overall_complexity:.3f}")
        print(f"Agents needed: {sample.task_analysis.estimated_agents_needed}")
        print(f"Agent assignments: {len(sample.agent_assignments)}")
    
    # Create HRM dataset
    print("\nCreating HRM dataset...")
    dataset_creator = HRMSWESmithDataset(config)
    metadata = await dataset_creator.create_hrm_dataset(
        num_samples=len(instances),
        output_dir='data/swe_smith_hrm_test'
    )
    print(f"Created dataset with metadata: {metadata}")


if __name__ == "__main__":
    asyncio.run(main())