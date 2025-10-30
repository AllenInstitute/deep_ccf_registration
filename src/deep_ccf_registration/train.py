from pathlib import Path
from typing import Optional, ContextManager
from contextlib import nullcontext
import os
import math
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

# https://github.com/karpathy/nanoGPT/blob/master/train.py
def _get_lr(
        iteration: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr
):
    """
    Calculate learning rate with linear warmup and cosine decay.

    Parameters
    ----------
    iteration: Current training iteration
    warmup_iters: Number of warmup iterations
    learning_rate: Maximum learning rate after warmup
    lr_decay_iters: Total iterations for learning rate decay
    min_lr: Minimum learning rate after decay

    Returns
    -------
    Learning rate for current iteration
    """
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train(
        train_dataloader,
        val_dataloader,
        model: nn.Module,
        optimizer,
        n_epochs: int,
        model_weights_out_dir: str,
        learning_rate: float = 0.001,
        decay_learning_rate: bool = True,
        warmup_iters: int = 1000,
        loss_eval_interval: int = 500,
        eval_iters: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        autocast_context: ContextManager = nullcontext(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train a model using MSE loss with early stopping.

    Parameters
    ----------
    train_dataloader: DataLoader for training data
    val_dataloader: DataLoader for validation data
    model: Neural network model to train
    optimizer: Optimizer for training
    n_epochs: Number of epochs to train
    model_weights_out_dir: Directory to save model checkpoints
    learning_rate: Initial learning rate
    decay_learning_rate: Whether to decay learning rate during training
    warmup_iters: Number of warmup iterations for learning rate
    loss_eval_interval: Evaluate loss every N iterations
    eval_iters: Number of iterations to average for evaluation
    patience: Number of evaluations without improvement before stopping
    min_delta: Minimum change in validation loss to be considered improvement
    autocast_context: Context manager for mixed precision training
    device: Device to train on

    Returns
    -------
    Best validation loss achieved during training
    """
    os.makedirs(model_weights_out_dir, exist_ok=True)

    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    lr_decay_iters = len(train_dataloader) * n_epochs
    min_lr = learning_rate / 10 # should be ~= learning_rate/10 per Chinchilla

    model.to(device)

    logger.info(f"Starting training for {n_epochs} epochs")
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
    logger.info(f"Device: {device}")

    for epoch in range(1, n_epochs + 1):
        # Set epoch for distributed training
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch=epoch)

        model.train()
        train_losses = []

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Learning rate decay
            if decay_learning_rate:
                lr = _get_lr(
                    iteration=global_step,
                    warmup_iters=warmup_iters,
                    learning_rate=learning_rate,
                    lr_decay_iters=lr_decay_iters,
                    min_lr=min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Forward pass
            optimizer.zero_grad()
            with autocast_context:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            # Periodic evaluation
            if global_step % loss_eval_interval == 0:
                train_loss = evaluate_loss(
                    model=model,
                    dataloader=train_dataloader,
                    criterion=criterion,
                    eval_iters=eval_iters,
                    device=device,
                    autocast_context=autocast_context,
                )
                val_loss = evaluate_loss(
                    model=model,
                    dataloader=val_dataloader,
                    criterion=criterion,
                    eval_iters=eval_iters,
                    device=device,
                    autocast_context=autocast_context,
                )

                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.6e}"
                )

                # Check for improvement
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = Path(model_weights_out_dir) / "model.pt"
                    torch.save(
                        obj={
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        },
                        f=checkpoint_path,
                    )
                    logger.info(f"New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered after {global_step} steps")
                    logger.info(f"Best validation loss: {best_val_loss:.6f}")
                    return best_val_loss

                model.train()

        # End of epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{n_epochs} completed | Avg Train Loss: {avg_train_loss:.6f}")
        logger.info(f"{'=' * 60}\n")

        # Save epoch checkpoint
        checkpoint_path = Path(model_weights_out_dir) / f"epoch_{epoch}.pt"
        torch.save(
            obj={
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            },
            f=checkpoint_path,
        )

    logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    return best_val_loss


def evaluate_loss(
        model: nn.Module,
        dataloader,
        criterion,
        eval_iters: int,
        device: str,
        autocast_context: ContextManager = nullcontext(),
) -> float:
    """
    Evaluate average loss over eval_iters batches.

    Parameters
    ----------
    model: Neural network model to evaluate
    dataloader: DataLoader for evaluation data
    criterion: Loss function
    eval_iters: Number of batches to evaluate
    device: Device to evaluate on
    autocast_context: Context manager for mixed precision evaluation

    Returns
    -------
    Average loss over evaluated batches
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= eval_iters:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast_context:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float('inf')