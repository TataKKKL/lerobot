#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    print("DEBUG: Entering update_policy function")
    start_time = time.perf_counter()
    
    print("DEBUG: Getting device from policy parameters")
    device = get_device_from_parameters(policy)
    print(f"DEBUG: Device: {device}")
    
    print("DEBUG: Setting policy to train mode")
    policy.train()
    
    print("DEBUG: About to do forward pass")
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        print(f"DEBUG: Forward pass completed, loss: {loss.item():.6f}")
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    
    print("DEBUG: About to scale loss and backward")
    grad_scaler.scale(loss).backward()
    print("DEBUG: Backward pass completed")

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    print("DEBUG: About to unscale gradients")
    grad_scaler.unscale_(optimizer)
    print("DEBUG: Gradients unscaled")

    print("DEBUG: About to clip gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )
    print(f"DEBUG: Gradient clipping completed, grad_norm: {grad_norm.item():.6f}")

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    print("DEBUG: About to step optimizer")
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    print("DEBUG: Optimizer step completed")
    
    # Updates the scale for next iteration.
    print("DEBUG: About to update grad_scaler")
    grad_scaler.update()
    print("DEBUG: Grad scaler updated")
    
    print("DEBUG: About to zero gradients")
    optimizer.zero_grad()
    print("DEBUG: Gradients zeroed")

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        print("DEBUG: About to step learning rate scheduler")
        lr_scheduler.step()
        print("DEBUG: LR scheduler stepped")

    if has_method(policy, "update"):
        print("DEBUG: About to call policy.update()")
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()
        print("DEBUG: Policy update completed")

    print("DEBUG: About to update metrics")
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    print(f"DEBUG: Metrics updated, update_s: {train_metrics.update_s}")
    
    print("DEBUG: Exiting update_policy function")
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    print("DEBUG: Starting train function")
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    print("DEBUG: Setting up WandB logger")
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
        print("DEBUG: WandB logger created")
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
        print("DEBUG: Using local logging")

    if cfg.seed is not None:
        print(f"DEBUG: Setting seed: {cfg.seed}")
        set_seed(cfg.seed)

    # Check device is available
    print("DEBUG: Getting safe torch device")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    print(f"DEBUG: Device set to: {device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("DEBUG: Creating dataset")
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    print(f"DEBUG: Dataset created with {dataset.num_frames} frames and {dataset.num_episodes} episodes")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        print("DEBUG: Creating evaluation environment")
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
        print("DEBUG: Evaluation environment created")

    print("DEBUG: Creating policy")
    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    print("DEBUG: Policy created")

    print("DEBUG: Creating optimizer and scheduler")
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    print("DEBUG: Optimizer, scheduler, and grad_scaler created")

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        print("DEBUG: Resuming from checkpoint")
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        print(f"DEBUG: Resumed from step {step}")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    print("DEBUG: Creating dataloader")
    if hasattr(cfg.policy, "drop_n_last_frames"):
        print("DEBUG: Using EpisodeAwareSampler")
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        print("DEBUG: Using regular sampling with shuffle")
        shuffle = True
        sampler = None

    print(f"DEBUG: DataLoader config - num_workers: {cfg.num_workers}, batch_size: {cfg.batch_size}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    print("DEBUG: DataLoader created, creating cycle iterator")
    dl_iter = cycle(dataloader)
    print("DEBUG: Cycle iterator created")

    print("DEBUG: Setting policy to train mode")
    policy.train()

    print("DEBUG: Creating metrics")
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    print("DEBUG: Metrics and tracker created")

    print("DEBUG: About to start main training loop")
    logging.info("Start offline training on a fixed dataset")
    
    for iteration in range(step, cfg.steps):
        print(f"\n=== DEBUG: Starting training iteration {iteration} (step will be {step+1}) ===")
        
        print("DEBUG: Recording start time for data loading")
        start_time = time.perf_counter()
        
        print("DEBUG: About to get next batch from dataloader - THIS IS A COMMON HANG POINT")
        try:
            batch = next(dl_iter)
            print("DEBUG: Successfully got batch from dataloader")
            print(f"DEBUG: Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else type(batch)}")
        except Exception as e:
            print(f"DEBUG: ERROR getting batch from dataloader: {e}")
            raise
        
        print("DEBUG: Recording data loading time")
        train_tracker.dataloading_s = time.perf_counter() - start_time
        print(f"DEBUG: Data loading took {train_tracker.dataloading_s} seconds")

        print("DEBUG: Moving batch tensors to device")
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        print("DEBUG: Batch moved to device")

        print("DEBUG: About to call update_policy")
        try:
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )
            print("DEBUG: update_policy completed successfully")
        except Exception as e:
            print(f"DEBUG: ERROR in update_policy: {e}")
            raise

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        print(f"DEBUG: Incremented step to {step}")
        
        print("DEBUG: About to call train_tracker.step()")
        step += 1
        train_tracker.step()
        print("DEBUG: train_tracker.step() completed")
        
        print("DEBUG: Checking conditions for logging, saving, and evaluation")
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        print(f"DEBUG: is_log_step: {is_log_step}, is_saving_step: {is_saving_step}, is_eval_step: {is_eval_step}")
    

        if is_log_step:
            print("DEBUG: About to log metrics - THIS CAN HANG")
            try:
                logging.info(train_tracker)
                print("DEBUG: Basic logging completed")
                
                if wandb_logger:
                    print("DEBUG: About to log to WandB")
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                    print("DEBUG: WandB logging completed")
                
                print("DEBUG: About to reset averages")
                train_tracker.reset_averages()
                print("DEBUG: Averages reset")
            except Exception as e:
                print(f"DEBUG: ERROR during logging: {e}")
                raise

        if cfg.save_checkpoint and is_saving_step:
            print("DEBUG: About to save checkpoint - THIS CAN HANG")
            try:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                print(f"DEBUG: Checkpoint directory: {checkpoint_dir}")
                
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                print("DEBUG: save_checkpoint completed")
                
                update_last_checkpoint(checkpoint_dir)
                print("DEBUG: update_last_checkpoint completed")
                
                if wandb_logger:
                    print("DEBUG: About to log policy to WandB")
                    wandb_logger.log_policy(checkpoint_dir)
                    print("DEBUG: WandB policy logging completed")
            except Exception as e:
                print(f"DEBUG: ERROR during checkpoint saving: {e}")
                raise

        if cfg.env and is_eval_step:
            print("DEBUG: About to evaluate policy - THIS CAN HANG")
            try:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                
                print("DEBUG: About to enter evaluation context")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    print("DEBUG: About to call eval_policy")
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )
                    print("DEBUG: eval_policy completed")

                print("DEBUG: Creating evaluation metrics")
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                
                print("DEBUG: Updating evaluation tracker")
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                
                print("DEBUG: About to log evaluation results")
                logging.info(eval_tracker)
                
                if wandb_logger:
                    print("DEBUG: About to log evaluation to WandB")
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
                    print("DEBUG: WandB evaluation logging completed")
            except Exception as e:
                print(f"DEBUG: ERROR during evaluation: {e}")
                raise
        
        print(f"DEBUG: Completed iteration {iteration} (step {step})")
        print("DEBUG: About to start next iteration\n")

    print("DEBUG: Training loop completed")
    if eval_env:
        print("DEBUG: Closing evaluation environment")
        eval_env.close()
    logging.info("End of training")
    print("DEBUG: train function completed")


if __name__ == "__main__":
    print("DEBUG: Starting main execution")
    init_logging()
    print("DEBUG: Logging initialized, about to call train()")
    train()
    print("DEBUG: train() completed")