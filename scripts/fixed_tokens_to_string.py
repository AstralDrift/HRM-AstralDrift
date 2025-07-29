
def _tokens_to_string(self, tokens):
    """Convert token tensor to string with proper decoding"""
    # Skip if we don't have a tokenizer
    if not hasattr(self, 'tokenizer'):
        # Log warning and return dummy for debugging
        if not hasattr(self, '_tokenizer_warned'):
            print("[DEBUG] No tokenizer available for code metrics - returning dummy strings")
            self._tokenizer_warned = True
        return "def dummy_function(): pass"  # Valid Python for testing
    
    try:
        # Convert tokens to list of integers
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        else:
            token_list = list(tokens)
        
        # Decode using tokenizer
        decoded = self.tokenizer.decode(token_list, skip_special_tokens=True)
        return decoded
    except Exception as e:
        print(f"[DEBUG] Token decoding error: {e}")
        return "def error(): pass"
