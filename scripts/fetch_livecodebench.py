#!/usr/bin/env python3
"""
Fetch and integrate real LiveCodeBench dataset
Replaces mock dataset with actual competition problems from LeetCode, AtCoder, CodeForces
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_livecodebench_dataset(version: str = "release_v2", max_problems: int = 50) -> List[Dict]:
    """
    Fetch real LiveCodeBench dataset from HuggingFace
    
    Args:
        version: Dataset version (release_v1 to release_v5)
        max_problems: Maximum number of problems to fetch
    
    Returns:
        List of problems in HRM format
    """
    logger.info(f"🔥 Fetching LiveCodeBench {version} from HuggingFace...")
    
    try:
        # Load dataset from HuggingFace
        lcb_dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version)
        
        logger.info(f"✅ Dataset loaded: {len(lcb_dataset['test'])} problems available")
        
        # Convert to HRM format
        hrm_problems = []
        
        for i, problem in enumerate(lcb_dataset['test']):
            if i >= max_problems:
                break
                
            # Extract problem details
            problem_id = f"lcb_{problem.get('question_id', f'{i+1:03d}')}"
            
            # Convert to HRM format matching our training pipeline
            hrm_problem = {
                "instance_id": problem_id,
                "domain": determine_domain(problem),
                "language": "python",  # Default to Python for now
                "complexity": determine_complexity(problem),
                "input_text": format_problem_description(problem),
                "target_text": problem.get('canonical_solution', ''),
                "metadata": {
                    "source": "livecodebench",
                    "difficulty": problem.get('difficulty', 'unknown'),
                    "contest": problem.get('platform', 'unknown'),
                    "tags": problem.get('topics', []),
                    "original_id": problem.get('question_id'),
                    "title": problem.get('question_title', ''),
                    "date_created": problem.get('date', ''),
                    "starter_code": problem.get('starter_code', ''),
                    "test_cases": problem.get('public_test_cases', []),
                    "memory_limit": problem.get('memory_limit', ''),
                    "time_limit": problem.get('time_limit', '')
                }
            }
            
            hrm_problems.append(hrm_problem)
            
        logger.info(f"✅ Converted {len(hrm_problems)} problems to HRM format")
        return hrm_problems
        
    except Exception as e:
        logger.error(f"❌ Error fetching LiveCodeBench: {e}")
        logger.warning("🔄 Falling back to enhanced mock dataset...")
        return create_enhanced_mock_dataset()

def determine_domain(problem: Dict) -> str:
    """Determine problem domain based on tags/topics"""
    topics = problem.get('topics', [])
    
    if any(topic in ['array', 'hash-table', 'two-pointers'] for topic in topics):
        return 'algorithms'
    elif any(topic in ['stack', 'queue', 'linked-list', 'tree'] for topic in topics):
        return 'data_structures'
    elif any(topic in ['graph', 'bfs', 'dfs', 'topological-sort'] for topic in topics):
        return 'graph_algorithms'
    elif any(topic in ['string', 'string-matching'] for topic in topics):
        return 'strings'
    elif any(topic in ['dynamic-programming', 'greedy'] for topic in topics):
        return 'dynamic_programming'
    else:
        return 'algorithms'

def determine_complexity(problem: Dict) -> float:
    """Determine problem complexity based on difficulty"""
    difficulty = problem.get('difficulty', 'Easy').lower()
    
    if difficulty == 'easy':
        return 0.3
    elif difficulty == 'medium':
        return 0.6
    elif difficulty == 'hard':
        return 0.9
    else:
        return 0.5

def format_problem_description(problem: Dict) -> str:
    """Format problem description for HRM training"""
    title = problem.get('question_title', 'Unknown Problem')
    content = problem.get('question_content', '')
    
    # Clean up HTML/markdown if present
    import re
    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
    content = content.replace('\n\n', '\n').strip()
    
    formatted = f"Problem: {title}\n{content}"
    
    # Add constraints if available
    if 'time_limit' in problem:
        formatted += f"\n\nTime Limit: {problem['time_limit']}"
    if 'memory_limit' in problem:
        formatted += f"\nMemory Limit: {problem['memory_limit']}"
        
    return formatted

def create_enhanced_mock_dataset() -> List[Dict]:
    """Enhanced mock dataset with more diverse problems if real fetch fails"""
    return [
        {
            "instance_id": "lcb_001_mock",
            "domain": "algorithms",
            "language": "python",
            "complexity": 0.4,
            "input_text": "Problem: Two Sum\nGiven an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\n\nConstraints: 2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9",
            "target_text": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
            "metadata": {
                "source": "mock_enhanced",
                "difficulty": "easy",
                "contest": "leetcode",
                "tags": ["array", "hash_table"]
            }
        },
        {
            "instance_id": "lcb_002_mock",
            "domain": "data_structures",
            "language": "python",
            "complexity": 0.7,
            "input_text": "Problem: Binary Tree Level Order Traversal\nGiven the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).",
            "target_text": "def level_order(root):\n    if not root:\n        return []\n    \n    result = []\n    queue = [root]\n    \n    while queue:\n        level = []\n        for _ in range(len(queue)):\n            node = queue.pop(0)\n            level.append(node.val)\n            if node.left:\n                queue.append(node.left)\n            if node.right:\n                queue.append(node.right)\n        result.append(level)\n    \n    return result",
            "metadata": {
                "source": "mock_enhanced",
                "difficulty": "medium",
                "contest": "leetcode",
                "tags": ["tree", "breadth-first-search"]
            }
        },
        {
            "instance_id": "lcb_003_mock",
            "domain": "graph_algorithms",
            "language": "python",
            "complexity": 0.8,
            "input_text": "Problem: Course Schedule II\nThere are numCourses courses labeled from 0 to numCourses - 1. Given prerequisites array, return the ordering of courses you should take to finish all courses.\n\nConstraints: 1 <= numCourses <= 2000, 0 <= prerequisites.length <= numCourses * (numCourses - 1)",
            "target_text": "def find_order(numCourses, prerequisites):\n    graph = [[] for _ in range(numCourses)]\n    indegree = [0] * numCourses\n    \n    for course, prereq in prerequisites:\n        graph[prereq].append(course)\n        indegree[course] += 1\n    \n    queue = [i for i in range(numCourses) if indegree[i] == 0]\n    result = []\n    \n    while queue:\n        course = queue.pop(0)\n        result.append(course)\n        \n        for next_course in graph[course]:\n            indegree[next_course] -= 1\n            if indegree[next_course] == 0:\n                queue.append(next_course)\n    \n    return result if len(result) == numCourses else []",
            "metadata": {
                "source": "mock_enhanced",
                "difficulty": "medium",
                "contest": "leetcode",
                "tags": ["graph", "topological_sort"]
            }
        },
        {
            "instance_id": "lcb_004_mock",
            "domain": "dynamic_programming",
            "language": "python",
            "complexity": 0.9,
            "input_text": "Problem: Edit Distance\nGiven two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.\n\nYou have the following three operations permitted on a word:\n- Insert a character\n- Delete a character\n- Replace a character",
            "target_text": "def min_distance(word1, word2):\n    m, n = len(word1), len(word2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    \n    for i in range(m + 1):\n        dp[i][0] = i\n    for j in range(n + 1):\n        dp[0][j] = j\n    \n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if word1[i-1] == word2[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    \n    return dp[m][n]",
            "metadata": {
                "source": "mock_enhanced",
                "difficulty": "hard",
                "contest": "leetcode",
                "tags": ["string", "dynamic_programming"]
            }
        },
        {
            "instance_id": "lcb_005_mock",
            "domain": "algorithms",
            "language": "python",
            "complexity": 0.6,
            "input_text": "Problem: Merge k Sorted Lists\nYou are given an array of k linked-lists lists, each linked-list is sorted in ascending order.\n\nMerge all the linked-lists into one sorted linked-list and return it.",
            "target_text": "def merge_k_lists(lists):\n    import heapq\n    \n    heap = []\n    for i, lst in enumerate(lists):\n        if lst:\n            heapq.heappush(heap, (lst.val, i, lst))\n    \n    dummy = ListNode(0)\n    curr = dummy\n    \n    while heap:\n        val, i, node = heapq.heappop(heap)\n        curr.next = node\n        curr = curr.next\n        \n        if node.next:\n            heapq.heappush(heap, (node.next.val, i, node.next))\n    \n    return dummy.next",
            "metadata": {
                "source": "mock_enhanced",
                "difficulty": "hard",
                "contest": "leetcode",
                "tags": ["linked_list", "divide_and_conquer", "heap", "merge_sort"]
            }
        }
    ]

def save_dataset(problems: List[Dict], output_path: str):
    """Save dataset to JSON file"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(problems, f, indent=2)
    
    logger.info(f"💾 Dataset saved to {output_path}")

def main():
    """Main function to fetch and save LiveCodeBench dataset"""
    logger.info("🚀 Starting LiveCodeBench dataset fetch...")
    
    # Fetch dataset
    problems = fetch_livecodebench_dataset(version="release_v2", max_problems=100)
    
    # Save to data directory
    output_path = "data/livecodebench_real/livecodebench_real.json"
    save_dataset(problems, output_path)
    
    # Print summary
    logger.info(f"📊 Dataset Summary:")
    logger.info(f"   Total problems: {len(problems)}")
    
    domains = {}
    complexities = []
    for p in problems:
        domain = p['domain']
        domains[domain] = domains.get(domain, 0) + 1
        complexities.append(p['complexity'])
    
    logger.info(f"   Domains: {domains}")
    logger.info(f"   Avg complexity: {sum(complexities)/len(complexities):.2f}")
    
    logger.info("✅ LiveCodeBench dataset integration complete!")

if __name__ == "__main__":
    main()