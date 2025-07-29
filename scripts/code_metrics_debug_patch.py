
# Add this to ACTSWESearchLossHead.__init__:
self.tokenizer = None  # Initialize as None
self._code_metrics_debug = True

# Add this method to help debug:
def set_tokenizer(self, tokenizer):
    """Set tokenizer for code metrics computation"""
    self.tokenizer = tokenizer
    print(f"[DEBUG] Tokenizer set: {type(tokenizer)}")

# Update _compute_code_metrics to add debug logging:
def _compute_code_metrics_debug(self, logits, labels, current_data):
    """Debug version with extensive logging"""
    device = logits.device
    batch_size = logits.size(0)
    
    print(f"[DEBUG] Computing code metrics for batch size: {batch_size}")
    print(f"[DEBUG] Logits shape: {logits.shape}")
    print(f"[DEBUG] Labels shape: {labels.shape}")
    
    # Force some non-zero values for testing
    metrics = {
        "syntax_validity": torch.tensor(0.5),  # Force 50% for debugging
        "compilation_success": torch.tensor(0.3),  # Force 30% for debugging
        "edit_distance": torch.tensor(0.7),
        "code_bleu": torch.tensor(0.6),
        "code_rouge": torch.tensor(0.65),
        "tiered_accuracy": torch.tensor(0.75),
        "syntax_accuracy": torch.tensor(0.8),
        "logical_accuracy": torch.tensor(0.6),
        "exact_match": torch.tensor(0.1)
    }
    
    print("[DEBUG] Returning forced non-zero metrics for debugging")
    return metrics
