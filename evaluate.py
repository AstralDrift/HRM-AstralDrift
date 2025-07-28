from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader
from swe_search_evaluator import create_swe_search_evaluator


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, assign=True)
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    
    # Standard evaluation
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print("Standard Evaluation Results:")
        print(metrics)
    
    # SWE-Search enhanced evaluation (if enabled)
    if getattr(config.arch, 'enable_swe_search', False) and RANK == 0:
        print("\nStarting SWE-Search Enhanced Evaluation")
        
        # Create SWE-Search evaluator
        swe_evaluator = create_swe_search_evaluator(train_state.model, config.arch, device='cuda')
        
        # Run comprehensive evaluation
        swe_results = swe_evaluator.evaluate_with_swe_search(
            eval_loader,
            compare_baseline=True,
            save_detailed_results=True
        )
        
        # Print performance summary
        performance_summary = swe_evaluator.get_performance_summary()
        if performance_summary:
            print(f"\nSWE-Search Performance Summary:")
            for key, value in performance_summary.items():
                print(f"  {key}: {value}")
    
    elif getattr(config.arch, 'enable_swe_search', False):
        print("SWE-Search evaluation skipped on non-zero rank")


if __name__ == "__main__":
    launch()
