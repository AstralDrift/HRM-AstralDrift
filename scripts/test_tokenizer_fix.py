#!/usr/bin/env python3
"""
Comprehensive test script for tokenizer fix validation
"""

import torch
import ast
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from models.losses import ACTSWESearchLossHead
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config

def test_tokenizer_integration():
    """Test tokenizer integration with loss head"""
    print("🧪 Testing Tokenizer Integration...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy model config
    model_config = HierarchicalReasoningModel_ACTV1Config(
        vocab_size=tokenizer.vocab_size,
        batch_size=2,
        seq_len=128,
        num_puzzle_identifiers=100,
        h_dim=256,
        l_dim=256,
        max_seq_len=128,
        H_layers=2,
        L_layers=2,
        H_cycles=1,
        L_cycles=1,
        causal=False,
        head_size=32,
        hidden_size=256,
        expansion=2,
        num_heads=4,
        pos_encodings="rotary",
        act_threshold=0.99,
        halt_max_steps=3,
        halt_exploration_prob=0.4,
        puzzle_emb_vocab_size=1000,
        puzzle_emb_dim=256
    )
    
    # Create model and loss head (pass config as dict, not object)
    model = HierarchicalReasoningModel_ACTV1(model_config.dict())
    loss_head = ACTSWESearchLossHead(
        model,
        loss_type="softmax_cross_entropy",
        tokenizer=tokenizer
    )
    
    print("✅ Loss head created with tokenizer")
    return loss_head, tokenizer

def test_token_decoding():
    """Test token-to-string conversion with various code samples"""
    print("\n🔤 Testing Token Decoding...")
    
    loss_head, tokenizer = test_tokenizer_integration()
    
    # Test code samples
    test_codes = [
        "def add(a, b):\n    return a + b",
        "class MyClass:\n    def __init__(self):\n        self.value = 0",
        "for i in range(10):\n    print(i)",
        "if x > 0:\n    result = 'positive'\nelse:\n    result = 'negative'",
        "import numpy as np\narray = np.zeros(10)"
    ]
    
    results = []
    for i, code in enumerate(test_codes):
        print(f"\n📝 Test {i+1}: {code[:30]}...")
        
        # Encode then decode
        tokens = tokenizer.encode(code, return_tensors="pt").squeeze()
        decoded = loss_head._tokens_to_string(tokens)
        
        # Test syntax validity
        try:
            ast.parse(decoded)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        # Test compilation
        try:
            compile(decoded, '<string>', 'exec')
            compiles = True
        except:
            compiles = False
        
        result = {
            "original": code,
            "decoded": decoded,
            "syntax_valid": syntax_valid,
            "compiles": compiles,
            "match": code.strip() == decoded.strip()
        }
        results.append(result)
        
        print(f"   Original: {code}")
        print(f"   Decoded:  {decoded}")
        print(f"   Syntax:   {'✅' if syntax_valid else '❌'}")
        print(f"   Compile:  {'✅' if compiles else '❌'}")
        print(f"   Match:    {'✅' if result['match'] else '❌'}")
    
    return results

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔍 Testing Edge Cases...")
    
    loss_head, tokenizer = test_tokenizer_integration()
    
    edge_cases = [
        torch.tensor([]),  # Empty tensor
        torch.tensor([0, 1, 2]),  # Short sequence
        torch.tensor([tokenizer.vocab_size + 100]),  # Invalid token ID
        torch.tensor([[1, 2], [3, 4]]),  # Multi-dimensional
        None,  # None input
    ]
    
    for i, case in enumerate(edge_cases):
        print(f"\n🧪 Edge case {i+1}: {type(case)} {getattr(case, 'shape', 'N/A')}")
        try:
            decoded = loss_head._tokens_to_string(case)
            print(f"   Result: '{decoded[:50]}...' {'✅' if len(decoded) > 0 else '❌'}")
        except Exception as e:
            print(f"   Error: {e} ❌")

def test_code_metrics():
    """Test complete code metrics computation"""
    print("\n📊 Testing Code Metrics Computation...")
    
    loss_head, tokenizer = test_tokenizer_integration()
    
    # Create sample batch data
    batch_size = 2
    seq_len = 50
    vocab_size = tokenizer.vocab_size
    
    # Mock logits and labels for code generation
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create mock current_data
    current_data = {
        "input_text": ["def test1():", "def test2():"],
        "target_text": ["def test1(): pass", "def test2(): return 42"]
    }
    
    try:
        metrics = loss_head._compute_code_metrics(logits, labels, current_data)
        
        print("📈 Code Metrics Results:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"   {key}: {value}")
        
        # Check if metrics are non-zero
        syntax_valid = metrics.get("syntax_validity", 0)
        compilation = metrics.get("compilation_success", 0)
        
        if syntax_valid > 0:
            print("✅ Syntax validity is working!")
        else:
            print("❌ Syntax validity still at 0")
            
        if compilation > 0:
            print("✅ Compilation success is working!")
        else:
            print("❌ Compilation success still at 0")
            
    except Exception as e:
        print(f"❌ Error in code metrics: {e}")

def save_test_results(results):
    """Save test results for analysis"""
    output_path = "evaluations/tokenizer_fix_test_results.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {output_path}")

def main():
    """Run comprehensive tokenizer fix tests"""
    print("🚀 Comprehensive Tokenizer Fix Validation")
    print("=" * 50)
    
    try:
        # Run all tests
        test_tokenizer_integration()
        decode_results = test_token_decoding()
        test_edge_cases()
        test_code_metrics()
        
        # Calculate success rates
        syntax_success = sum(1 for r in decode_results if r["syntax_valid"])
        compile_success = sum(1 for r in decode_results if r["compiles"])
        total_tests = len(decode_results)
        
        print(f"\n📊 Summary:")
        print(f"   Syntax Success:    {syntax_success}/{total_tests} ({syntax_success/total_tests*100:.1f}%)")
        print(f"   Compile Success:   {compile_success}/{total_tests} ({compile_success/total_tests*100:.1f}%)")
        
        # Save results
        save_test_results({
            "decode_results": decode_results,
            "syntax_success_rate": syntax_success / total_tests,
            "compile_success_rate": compile_success / total_tests,
            "timestamp": str(torch.tensor([1.0]))  # Dummy timestamp
        })
        
        if syntax_success > 0:
            print("🎉 Tokenizer fix is working! Code metrics should now be non-zero.")
        else:
            print("⚠️ Issues detected. Check debug logs for details.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()