#!/usr/bin/env python3
"""
Debug tool for token decoding analysis
"""

import torch
import ast
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

def analyze_token_decoding():
    """Analyze how different codes are tokenized and decoded"""
    print("🔍 Token Decoding Analysis")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📚 Tokenizer: {tokenizer.__class__.__name__}")
    print(f"📊 Vocab size: {tokenizer.vocab_size}")
    print(f"🔤 Special tokens: {tokenizer.special_tokens_map}")
    
    # Test code samples
    test_codes = [
        "def hello():\n    print('world')",
        "x = [1, 2, 3]\nfor i in x:\n    print(i)",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
        "import numpy as np\narray = np.zeros((10, 10))",
        "if condition:\n    result = True\nelse:\n    result = False"
    ]
    
    for i, code in enumerate(test_codes):
        print(f"\n🧪 Test {i+1}:")
        print(f"Original code:\n{code}")
        
        # Tokenize
        tokens = tokenizer.encode(code, return_tensors="pt").squeeze()
        token_list = tokens.tolist()
        
        print(f"Tokens ({len(token_list)}): {token_list[:20]}{'...' if len(token_list) > 20 else ''}")
        
        # Decode
        decoded = tokenizer.decode(token_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"Decoded:\n{decoded}")
        
        # Check syntax
        try:
            ast.parse(decoded)
            syntax_ok = "✅"
        except SyntaxError as e:
            syntax_ok = f"❌ {e.msg}"
        
        # Check compilation
        try:
            compile(decoded, '<string>', 'exec')
            compile_ok = "✅"
        except Exception as e:
            compile_ok = f"❌ {str(e)[:50]}"
        
        print(f"Syntax: {syntax_ok}")
        print(f"Compile: {compile_ok}")
        print(f"Match original: {'✅' if code.strip() == decoded.strip() else '❌'}")
        
        # Show token-to-word mapping for debugging
        if len(token_list) <= 30:  # Only for short sequences
            print("Token details:")
            for j, token_id in enumerate(token_list[:10]):  # First 10 tokens
                token_str = tokenizer.decode([token_id])
                print(f"  {token_id:5d} -> '{token_str}'")

def test_edge_cases():
    """Test edge cases in token decoding"""
    print("\n\n🔍 Edge Case Analysis")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    edge_cases = [
        [],  # Empty
        [0],  # Start token only
        [tokenizer.pad_token_id] * 5,  # All padding
        [tokenizer.eos_token_id],  # End token only
        [tokenizer.vocab_size + 100],  # Invalid token (too high)
        [-1],  # Invalid token (negative)
        list(range(10)),  # Sequential tokens
        [tokenizer.encode("def")[-1]],  # Single 'def' token
    ]
    
    for i, tokens in enumerate(edge_cases):
        print(f"\n🧪 Edge case {i+1}: {tokens}")
        
        try:
            # Test with various decoding options
            decoded_basic = tokenizer.decode(tokens)
            decoded_clean = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            print(f"  Basic decode: '{decoded_basic}'")
            print(f"  Clean decode: '{decoded_clean}'")
            
            # Test syntax
            if decoded_clean.strip():
                try:
                    ast.parse(decoded_clean)
                    print("  Syntax: ✅")
                except:
                    print("  Syntax: ❌")
            else:
                print("  Syntax: ⚠️ Empty")
                
        except Exception as e:
            print(f"  Error: {e}")

def benchmark_decoding_speed():
    """Benchmark decoding performance"""
    print("\n\n⚡ Decoding Speed Benchmark")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test data
    test_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    tokens = tokenizer.encode(test_code)
    
    import time
    
    # Benchmark single decoding
    start_time = time.time()
    for _ in range(1000):
        _ = tokenizer.decode(tokens, skip_special_tokens=True)
    single_time = time.time() - start_time
    
    # Benchmark batch decoding
    batch_tokens = [tokens] * 100
    start_time = time.time()
    for _ in range(10):
        _ = tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)
    batch_time = time.time() - start_time
    
    print(f"Single decode (1000x): {single_time:.3f}s ({single_time/1000*1e6:.1f}μs each)")
    print(f"Batch decode (10x100): {batch_time:.3f}s ({batch_time/1000*1e6:.1f}μs each)")
    print(f"Speedup: {single_time/batch_time:.1f}x faster with batching")

def main():
    """Run all debug analyses"""
    try:
        analyze_token_decoding()
        test_edge_cases()
        benchmark_decoding_speed()
        
        print("\n\n✅ Debug analysis complete!")
        print("📝 Key findings:")
        print("   - Check if decoded strings match original code")
        print("   - Verify syntax validity after decoding")
        print("   - Ensure special tokens are handled properly")
        print("   - Consider batch decoding for performance")
        
    except Exception as e:
        print(f"❌ Debug analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()