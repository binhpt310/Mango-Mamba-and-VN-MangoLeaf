#!/usr/bin/env python3
# MangoMamba Training Script
import argparse
import time
import yaml
import os
import logging
import json
from collections import OrderedDict
from contextlib import suppress
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import model_parameters
from timm import utils
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

from mango_mamba import mango_mamba_tiny
import torch.nn.functional as F
import shutil

# Add window functions for token pruning visualization
def window_partition(x, window_size):
    """Partition feature map into windows"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partitioning"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


# Logging setup
torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# Config parser
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='MangoMamba Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('--data_dir', metavar='DIR', default='combined_dataset',
                    help='path to dataset')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='val',
                    help='dataset validation split (default: val)')
group.add_argument('--test-split', metavar='NAME', default='test',
                    help='dataset test split (default: test)')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='mango_mamba_tiny', type=str, metavar='MODEL',
                    help='Name of model to train')
group.add_argument('--num-classes', type=int, default=8, metavar='N',
                    help='number of disease classes')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input image dimensions (d h w)')
group.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N', #8 for only MangoLeafBD
                    help='Input batch size for training')
group.add_argument('--grad-accum-steps', type=int, default=16, metavar='N',
                    help='Gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)')
group.add_argument('--mixer-mode', type=str, default='hybrid', choices=['hybrid', 'pure_mamba', 'pure_lka'],
                    help='Mixer mode: hybrid (default), pure_mamba, or pure_lka')
group.add_argument('--use-eca', action='store_true', default=False,
                    help='Use Efficient Channel Attention (ECA)')
group.add_argument('--use-grn', action='store_true', default=False,
                    help='Use Global Response Normalization (GRN)')
group.add_argument('--use-swiglu', action='store_true', default=False,
                    help='Use SwiGLU activation in MLP (default: standard GELU MLP)')
group.add_argument('--window-size', type=int, nargs=4, default=None,
                    metavar='N N N N', help='Window sizes for 4 stages (default: [8,8,14,7])')
group.add_argument('--depths', type=int, nargs=4, default=None,
                    metavar='N N N N', help='Number of mixer blocks in the 4 stages (default: model setting)')
group.add_argument('--inter-window-mamba-stages', type=int, nargs='*', default=None,
                    metavar='STAGE', help='Hybrid stage indices receiving pooled inter-window Mamba')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer')
group.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate')
group.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='AdamW momentum (default: 0.9)')
group.add_argument('--eps', type=float, default=1e-8, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
group.add_argument('--betas', default=None, type=float, nargs=2, metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                    help='Clip gradient norm')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler')
group.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')
group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers')
group.add_argument('--patience-epochs', type=int, default=20, metavar='N',
                    help='patience epochs for early stopping (default: 20)')
group.add_argument('--early-stopping-monitor', type=str, default='loss', choices=['loss', 'accuracy'],
                    help='metric to monitor for early stopping (default: loss)')

# Loss parameters
group = parser.add_argument_group('Loss parameters')
group.add_argument('--smoothing', type=float, default=0.01,
                    help='Label smoothing')

# Augmentation parameters
group = parser.add_argument_group('Augmentation parameters')
group.add_argument('--drop-path-rate', type=float, default=0.01, metavar='PCT',
                    help='Drop path rate')
group.add_argument('--re-prob', type=float, default=0.05,
                    help='Random-erasing probability')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal-flip probability')
group.add_argument('--vflip', type=float, default=0.2,
                    help='Vertical-flip probability')
group.add_argument('--color-jitter', type=float, default=0.05,
                    help='Color-jitter strength')

# Misc parameters
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use')
group.add_argument('--amp', action='store_true', default=True,
                    help='use Native AMP for mixed precision training')
group.add_argument('--output', default='./output', type=str, metavar='PATH',
                    help='path to output folder')
group.add_argument('--experiment', default='mango_mamba', type=str, metavar='NAME',
                    help='name of train experiment')
group.add_argument('--train-from-checkpoint', action='store_true', default=False,
                    help='train from a pretrained checkpoint')
group.add_argument('--checkpoint-path', default='', type=str, metavar='PATH',
                    help='path to checkpoint file to load (e.g., ./output/model_best.pth.tar)')
group.add_argument('--resume', action='store_true', default=False,
                    help='restore optimizer state and epoch counter from --checkpoint-path')

class CustomModelTarget:
    """Custom target class for MangoMamba models"""
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # Handle different output types from custom models
        if isinstance(model_output, dict):
            logits = model_output.get('logits')
        elif hasattr(model_output, 'logits'):
            logits = model_output.logits
        else:
            logits = model_output
        
        if logits is None:
            return torch.tensor(0.0, requires_grad=True)
        
        # Handle different tensor shapes
        if len(logits.shape) == 1:
            return logits[self.category]
        else:
            return logits[:, self.category]

class ModelWrapper(nn.Module):
    """Wrapper for custom models to ensure tensor outputs for Grad-CAM"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        
        # Extract logits and ensure it's a tensor
        if isinstance(output, dict):
            logits = output.get('logits')
        elif hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        if logits is None:
            # Fallback: create dummy tensor
            batch_size = x.shape[0]
            num_classes = 8
            return torch.zeros(batch_size, num_classes, device=x.device, requires_grad=True)
        
        return logits

def find_target_layer(model):
    """Find the appropriate target layer for Grad-CAM in custom Mamba models"""
    target_layer = None
    
    # Look for the last normalization layer before classification
    if hasattr(model, 'norm'):
        target_layer = model.norm
    else:
        # Fallback: look for any BatchNorm2d layer in the last stage
        if hasattr(model, 'stages') and len(model.stages) > 0:
            last_stage = model.stages[-1]
            for name, module in last_stage.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    target_layer = module
    
    # Final fallback: look for any normalization layer in the model
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)) and 'head' not in name:
                target_layer = module
    
    return target_layer

def generate_gradcam_visualization(model, data_loader, output_dir, class_names):
    """Generate Grad-CAM visualizations for sample images - one per class"""
    print("Generating Grad-CAM visualizations...")
    
    try:
        model.eval()
        
        # Find appropriate target layer
        target_layer = find_target_layer(model)
        
        if target_layer is None:
            print("Warning: Could not find suitable target layer for Grad-CAM. Skipping Grad-CAM visualization.")
            return
        
        print(f"Using target layer: {target_layer}")
        target_layers = [target_layer]
        
        try:
            # Wrap the model to ensure tensor outputs
            wrapped_model = ModelWrapper(model)
            wrapped_model.eval()
            
            # Initialize Grad-CAM with wrapped model
            cam = GradCAM(model=wrapped_model, target_layers=target_layers)
            print("Grad-CAM initialized successfully with model wrapper")
        except Exception as e:
            print(f"Error initializing Grad-CAM: {e}")
            print("Skipping Grad-CAM visualization due to initialization error.")
            return
        
        # Collect samples from each class
        class_samples = {i: None for i in range(len(class_names))}
        
        # Iterate through the data loader to find one sample from each class
        for batch_idx, (images, labels) in enumerate(data_loader):
            for i in range(len(images)):
                label = labels[i].item()
                if class_samples[label] is None:
                    class_samples[label] = (images[i:i+1], label)
                    
                    # Check if we have samples from all classes
                    if all(sample is not None for sample in class_samples.values()):
                        break
            if all(sample is not None for sample in class_samples.values()):
                break
        
        # Generate Grad-CAM for each class
        for class_idx, (input_tensor, true_label) in class_samples.items():
            if input_tensor is None:
                print(f"No sample found for class {class_names[class_idx]}, skipping...")
                continue
                
            try:
                input_tensor = input_tensor.cuda()
                
                # Generate CAM
                targets = [CustomModelTarget(true_label)]
                print(f"Generating Grad-CAM for class {class_names[class_idx]} (class {true_label})")
                
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                
                # Handle different possible shapes
                if len(grayscale_cam.shape) == 3:
                    grayscale_cam = grayscale_cam[0, :]
                elif len(grayscale_cam.shape) == 2:
                    pass  # Already 2D
                else:
                    print(f"Unexpected Grad-CAM shape: {grayscale_cam.shape}")
                    continue
                
                # Convert input tensor to RGB image
                img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
                img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img = np.clip(img, 0, 1)
                
                # Create visualization
                try:
                    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                except Exception as overlay_error:
                    print(f"Error creating overlay: {overlay_error}")
                    # Create a simple visualization without overlay
                    visualization = img
                
                # Save visualization
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title(f'Original Image\nTrue: {class_names[true_label]}')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(grayscale_cam, cmap='jet')
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(visualization)
                plt.title('Grad-CAM Overlay')
                plt.axis('off')
                
                plt.tight_layout()
                gradcam_path = os.path.join(output_dir, f"gradcam_{class_names[true_label].replace(' ', '_')}.png")
                plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Generated Grad-CAM visualization for class {class_names[true_label]}")
                
            except Exception as e:
                print(f"Error generating Grad-CAM for class {class_names[class_idx]}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print("Grad-CAM visualizations completed.")
        
    except Exception as e:
        print(f"Error in Grad-CAM visualization: {e}")
        print("Skipping Grad-CAM visualization.")

def visualize_stage_tokens(stage, x, stage_name):
    """Visualize how tokens are pruned in a specific stage"""
    try:
        # Get the original spatial dimensions
        B, C, H, W = x.shape
        
        # Apply window partitioning and token pruning like in the stage
        window_size = stage.window_size
        
        # Window partitioning
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            x_padded = F.pad(x, (0, pad_r, 0, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
        else:
            x_padded = x
            Hp, Wp = H, W
        
        # Window partition
        windows = window_partition(x_padded, window_size)
        
        # Create a visualization mask showing which tokens are kept/pruned
        B_win, T, C_win = windows.shape
        S = int(T ** 0.5)  # window_size
        
        # Create a mask showing token importance
        with torch.no_grad():
            scores = windows.norm(p=2, dim=-1)  # (B_win, T)
            
            # Create a spatial mask for visualization
            vis_mask = torch.zeros(B_win, S, S, device=x.device)
            
            # Apply token pruning logic
            if stage.prune_keep_ratio < 1.0 and T > 4:
                k = max(4, int(T * stage.prune_keep_ratio))
                topk_idx = scores.topk(k, dim=1).indices  # (B_win, k)
                
                # Create binary mask for kept tokens
                for b in range(B_win):
                    for i in range(k):
                        token_idx = topk_idx[b, i]
                        row = token_idx // S
                        col = token_idx % S
                        vis_mask[b, row, col] = 1.0
            else:
                # If no pruning, all tokens are kept
                vis_mask = torch.ones(B_win, S, S, device=x.device)
        
        # Convert to full spatial resolution
        vis_full = torch.zeros(B, Hp, Wp, device=x.device)
        
        # Place each window's mask back to its original position
        num_win_cols = Wp // window_size
        for b in range(B_win):
            # Calculate window position in original image (grid indices)
            grid_row = b // num_win_cols
            grid_col = b % num_win_cols
            win_row = grid_row * window_size
            win_col = grid_col * window_size
            # Place the window mask
            vis_full[0, win_row:win_row+window_size, win_col:win_col+window_size] = vis_mask[b]
        
        # Crop to original size
        if pad_r > 0 or pad_b > 0:
            vis_full = vis_full[:, :H, :W]
        
        # Convert to numpy and normalize
        vis_tensor = vis_full[0].detach().cpu().numpy()
        
        # Create a more meaningful visualization by overlaying with original features
        # Take the first channel of input features for context
        feature_channel = x[0, 0].detach().cpu().numpy()
        
        # Normalize feature channel
        if feature_channel.max() > feature_channel.min():
            feature_channel = (feature_channel - feature_channel.min()) / (feature_channel.max() - feature_channel.min())
        
        # Combine: use feature channel as base, highlight kept tokens
        combined_vis = feature_channel * 0.3 + vis_tensor * 0.7
        
        return combined_vis
        
    except Exception as e:
        print(f"Error in visualize_stage_tokens for {stage_name}: {e}")
        import traceback
        traceback.print_exc()
        # Return a blank image if there's an error
        return np.zeros((H, W))

class ClassificationLoss(nn.Module):
    """Classification loss"""
    
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.cls_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.soft_target_loss = SoftTargetCrossEntropy()
    
    def forward(self, outputs, targets):
        logits = outputs['logits']
        
        if targets.dtype == torch.float32 and targets.dim() > 1:
            loss = self.soft_target_loss(logits, targets)
        else:
            # Hard targets - ensure they are long type
            targets = targets.long()
            loss = self.cls_loss(logits, targets)
        
        return loss, {'loss': loss.item()}

class NoHardLinkCheckpointSaver(utils.CheckpointSaver):
    """CheckpointSaver wrapper that uses copy instead of hard link (for Google Drive compatibility)"""
    
    def save_checkpoint(self, epoch, metric=None):
        """Override save_checkpoint to use copy instead of hard link"""
        # Temporarily replace os.link with shutil.copy2 to avoid hard link issues
        original_link = os.link
        def safe_link(src, dst):
            try:
                original_link(src, dst)
            except (OSError, PermissionError):
                # Fallback to copy if hard link fails (e.g., on Google Drive)
                shutil.copy2(src, dst)
        
        # Monkey patch os.link temporarily
        os.link = safe_link
        try:
            result = super().save_checkpoint(epoch, metric)
        finally:
            # Restore original os.link
            os.link = original_link
        
        return result

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def create_model_fn(model_name, **kwargs):
    """Create MangoMamba model"""
    if model_name == 'mango_mamba_tiny':
        return mango_mamba_tiny(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training and validation curves with detailed metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add final values as text
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    ax1.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy', color='blue', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add final values as text
    final_train_acc = train_accs[-1] if train_accs else 0
    final_val_acc = val_accs[-1] if val_accs else 0
    ax2.text(0.02, 0.02, f'Final Train Acc: {final_train_acc:.2f}%\nFinal Val Acc: {final_val_acc:.2f}%', 
             transform=ax2.transAxes, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_confusion_matrix(all_targets, all_predictions, class_names, save_path):
    """Plot confusion matrix with counts and percentages"""
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Calculate percentages (row-wise normalization)
    row_sums = cm.sum(axis=1)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(class_names)):
        if row_sums[i] > 0:
            cm_percent[i, :] = cm[i, :].astype('float') / row_sums[i] * 100
        else:
            cm_percent[i, :] = 0.0
    
    # Create annotation array with both count and percentage
    annotations = []
    for i in range(len(class_names)):
        row = []
        for j in range(len(class_names)):
            count = int(cm[i, j])
            percent = cm_percent[i, j]
            if count > 0:
                row.append(f'{count}\n({percent:.1f}%)')
            else:
                row.append('0\n(0.0%)')
        annotations.append(row)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def count_parameters(model):
    """Count and print model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print()

def save_lightweight_checkpoint(model, checkpoint_path):
    """Save lightweight checkpoint with only model state_dict for inference"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.__class__.__name__,
    }, checkpoint_path)
    print(f"Lightweight checkpoint saved to: {checkpoint_path}")

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, amp_autocast=suppress, loss_scaler=None):
    
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    
    model.train()
    
    # Clear GPU cache at the start of each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    end = time.time()
    last_idx = len(loader) - 1
    accumulation_steps = max(args.grad_accum_steps, 1)
    updates_per_epoch = (len(loader) + accumulation_steps - 1) // accumulation_steps
    num_updates = epoch * updates_per_epoch
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        # Forward pass with mixed precision
        with amp_autocast('cuda'):
            output = model(input)
            loss, loss_dict = loss_fn(output, target)
        
        # Update metrics
        losses_m.update(loss_dict['loss'], input.size(0))
        
        # Calculate accuracy
        acc1, _ = utils.accuracy(output['logits'], target, topk=(1, 5))
        top1_m.update(acc1.item(), input.size(0))
        
        # Average the gradients in each accumulation window. The final window
        # may contain fewer batches, so it needs its own divisor.
        window_start = batch_idx - batch_idx % accumulation_steps
        window_size = min(accumulation_steps, len(loader) - window_start)
        loss_for_backward = loss / window_size
        should_update = (batch_idx + 1) % accumulation_steps == 0 or last_batch

        # Backward pass and optimization
        if loss_scaler is not None:
            loss_scaler(
                loss_for_backward, optimizer,
                clip_grad=args.clip_grad, clip_mode='norm',
                parameters=model_parameters(model),
                create_graph=False, need_update=should_update,
            )
        else:
            loss_for_backward.backward()
            if should_update:
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        if should_update:
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates)
            num_updates += 1
        
        # Delete intermediate variables to free memory
        del input, target, output, loss, loss_for_backward
        batch_time_m.update(time.time() - end)
        
        # Clear cache periodically to avoid OOM (every 10 batches)
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Logging
        if batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            # Print GPU memory usage for debugging
            if torch.cuda.is_available() and batch_idx == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                _logger.info(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')
            
            _logger.info(
                f'Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)] '
                f'Loss: {losses_m.val:#.4g} ({losses_m.avg:#.3g}) '
                f'Acc@1: {top1_m.val:#.3g} ({top1_m.avg:#.3g}) '
                f'Time: {batch_time_m.val:.3f}s '
                f'LR: {lr:.3e} '
                f'Data: {data_time_m.val:.3f}s'
            )
        
        end = time.time()
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    
    model.eval()
    
    # Clear GPU cache before validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            with amp_autocast('cuda'):
                output = model(input)
                logits = output['logits']
                loss = loss_fn(logits, target)
            
            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
            
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
            
            # Delete intermediate variables to free memory
            del input, target, output, logits, loss
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # Clear cache periodically
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_idx % args.log_interval == 0:
                _logger.info(
                    f'Test: [{batch_idx:>4d}/{last_idx}] '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f}) '
                    f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f}) '
                    f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f})'
                )
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    metrics = OrderedDict([
        ('loss', losses_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg)
    ])
    
    return metrics

def generate_confusion_matrix_data(model, loader, amp_autocast=suppress):
    """Generate targets and predictions for confusion matrix"""
    all_targets = []
    all_predictions = []
    
    model.eval()
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            with amp_autocast('cuda'):
                output = model(input)
                logits = output['logits']
            
            predictions = logits.argmax(dim=1)
            
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            # Delete intermediate variables and clear cache periodically
            del input, target, output, logits, predictions
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_targets, all_predictions

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    
    # Track training start time
    training_start_time = time.time()
    
    # Setup device and distributed training
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0
    args.distributed = False
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.device = f'cuda:{args.local_rank}'
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
    
    # Set random seed
    utils.random_seed(args.seed, args.rank)
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _logger.info(f'GPU Memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Check GPU memory and batch size
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.batch_size > 128 and gpu_memory_gb < 16:
            _logger.warning(f'Large batch size ({args.batch_size}) detected on GPU with {gpu_memory_gb:.2f} GB memory. '
                          f'Consider reducing batch size or using gradient accumulation to avoid OOM errors.')
        if args.batch_size > 256:
            _logger.warning(f'Very large batch size ({args.batch_size}) detected. This may cause CUDA OOM errors.')
    
    # Create model
    model_kwargs = {
        'num_classes': args.num_classes,
        'drop_path_rate': args.drop_path_rate,
        'mixer_mode': args.mixer_mode,
        'use_eca': args.use_eca,
        'use_grn': args.use_grn,
        'use_swiglu': args.use_swiglu
    }
    if args.window_size is not None:
        model_kwargs['window_size'] = args.window_size
    if args.depths is not None:
        if any(depth < 1 for depth in args.depths):
            raise ValueError('--depths must contain four positive integers')
        model_kwargs['depths'] = args.depths
    if args.inter_window_mamba_stages is not None:
        model_kwargs['inter_window_mamba_stages'] = args.inter_window_mamba_stages
    
    model = create_model_fn(args.model, **model_kwargs)
    
    model.cuda()
    
    # Clear cache after model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load checkpoint weights. Full optimizer-state restoration for --resume
    # happens after the optimizer and scheduler are initialized.
    start_epoch = 0
    resume_checkpoint = None
    if args.resume and not (args.train_from_checkpoint and args.checkpoint_path):
        raise ValueError('--resume requires --train-from-checkpoint and --checkpoint-path')
    if args.train_from_checkpoint and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location='cuda', weights_only=False)
            
            # Check available keys in checkpoint
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try to load the state dict
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                print(f"Checkpoint loaded successfully.")
            except RuntimeError as e:
                if "size mismatch" in str(e) and ("head.weight" in str(e) or "head.bias" in str(e)):
                    print(f"Size mismatch detected in head layer (likely different number of classes).")
                    print(f"Current model classes: {args.num_classes}")
                    
                    # Extract checkpoint info
                    checkpoint_state = checkpoint['state_dict'].copy()
                    if 'head.weight' in checkpoint_state:
                        checkpoint_classes = checkpoint_state['head.weight'].shape[0]
                        print(f"Checkpoint classes: {checkpoint_classes}")
                    
                    # Remove incompatible head layers from checkpoint
                    keys_to_remove = []
                    for key in checkpoint_state.keys():
                        if key.startswith('head.') and ('weight' in key or 'bias' in key):
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        print(f"Removing incompatible key: {key} (shape: {checkpoint_state[key].shape})")
                        del checkpoint_state[key]
                    
                    # Load with the filtered state dict
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
                    
                    print(f"Loaded checkpoint with partial weights:")
                    if missing_keys:
                        print(f"  Missing keys (will be randomly initialized): {missing_keys}")
                    if unexpected_keys:
                        print(f"  Unexpected keys (will be ignored): {unexpected_keys}")
                    
                    print(f"Note: The classification head layer will be randomly initialized for {args.num_classes} classes.")
                    print(f"This is normal when transferring between datasets with different numbers of classes.")
                else:
                    print(f"Error loading checkpoint: {e}")
                    print("Training from scratch instead.")
        else:
            print(f"Warning: Checkpoint path {args.checkpoint_path} does not exist. Training from scratch.")

    if args.resume:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f'resume checkpoint does not exist: {args.checkpoint_path}')
        resume_checkpoint = torch.load(args.checkpoint_path, map_location='cuda', weights_only=False)
        if 'optimizer' not in resume_checkpoint:
            raise RuntimeError('resume checkpoint is missing optimizer state')
        model.load_state_dict(resume_checkpoint['state_dict'], strict=True)
        start_epoch = int(resume_checkpoint.get('epoch', -1)) + 1
        print(f"Checkpoint loaded for resume; next epoch is {start_epoch + 1}.")
    
    # Print model parameters
    count_parameters(model)

    # Setup mixed precision
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP for mixed precision training.')
    
    # Create optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    
    # Create datasets and data loaders
    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)
    
    dataset_train = create_dataset(
        '', root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size
    )
    
    dataset_eval = create_dataset(
        '', root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.batch_size
    )
    
    dataset_test = create_dataset(
        '', root=args.data_dir, split=args.test_split, is_training=False,
        batch_size=args.batch_size
    )
    
    
    # Create data loaders
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        re_prob=args.re_prob,
        re_mode='pixel',
        re_count=1,
        scale=[0.8, 1.0],
        ratio=[3./4., 4./3.],
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=True,
    )
    
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
    )
    
    loader_test = create_loader(
        dataset_test,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
    )
    
    # Setup loss function
    train_loss_fn = ClassificationLoss(smoothing=args.smoothing).cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    # Setup learning rate scheduler
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        if loss_scaler is not None and resume_checkpoint.get('amp_scaler') is not None:
            loss_scaler.load_state_dict(resume_checkpoint['amp_scaler'])
        if start_epoch >= num_epochs:
            raise ValueError(
                f'resume checkpoint is already at epoch {start_epoch}; --epochs must be greater than {start_epoch}'
            )
        if lr_scheduler is not None:
            # The historical checkpoint has no scheduler state. This aligns the
            # new 200-epoch cosine schedule with the resumed epoch counter.
            lr_scheduler.step(start_epoch)
        _logger.info(
            f'Resuming from epoch {start_epoch + 1}/{num_epochs}; optimizer restored and scheduler advanced to epoch {start_epoch}.'
        )
    
    # Setup output directory
    exp_name = args.experiment
    if os.path.isfile(os.path.join(args.output, exp_name, 'training_results.json')):
        raise FileExistsError('refusing to overwrite a completed experiment; use a new --output')
    output_dir = utils.get_outdir(args.output, exp_name)
    
    # Setup checkpoint saver (use custom saver to avoid hard link issues on Google Drive)
    checkpoint_uses_loss = args.early_stopping_monitor == 'loss'
    saver = NoHardLinkCheckpointSaver(
        model=model, optimizer=optimizer, args=args,
        amp_scaler=loss_scaler, checkpoint_dir=output_dir,
        recovery_dir=output_dir, decreasing=checkpoint_uses_loss, max_history=3
    )
    
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    
    # Training history for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    _logger.info(f'Starting training at epoch {start_epoch + 1} of {num_epochs}...')
    
    best_metric = float(resume_checkpoint.get('metric', 0.0)) if resume_checkpoint is not None else 0.0
    best_epoch = start_epoch - 1 if resume_checkpoint is not None else 0
    patience_counter = 0
    best_val_loss = float('inf')
    best_val_acc = best_metric if resume_checkpoint is not None and args.early_stopping_monitor == 'accuracy' else 0.0
    
    for epoch in range(start_epoch, num_epochs):
        if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)
        
        # Training
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, train_loss_fn, args,
            lr_scheduler=lr_scheduler, amp_autocast=amp_autocast,
            loss_scaler=loss_scaler
        )
        
        # Validation
        eval_metrics = validate(
            model, loader_eval, validate_loss_fn, args,
            amp_autocast=amp_autocast
        )
        
        # Store metrics for plotting
        train_losses.append(train_metrics['loss'])
        val_losses.append(eval_metrics['loss'])
        train_accs.append(train_metrics.get('top1', 0))
        val_accs.append(eval_metrics['top1'])
        
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics.get('top1', 0))
        
        # Early stopping check
        monitor_value = (
            eval_metrics['loss']
            if args.early_stopping_monitor == 'loss'
            else eval_metrics['top1']
        )
        improved = False
        if args.early_stopping_monitor == 'loss':
            if eval_metrics['loss'] < best_val_loss:
                best_val_loss = eval_metrics['loss']
                improved = True
        else:  # monitor accuracy
            if eval_metrics['top1'] > best_val_acc:
                best_val_acc = eval_metrics['top1']
                improved = True
        
        if improved:
            best_metric = eval_metrics['top1']
            best_epoch = epoch
            patience_counter = 0
            saver.save_checkpoint(epoch, metric=monitor_value)
            # Save lightweight best checkpoint
            save_lightweight_checkpoint(model, os.path.join(output_dir, 'model_best_inference.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience_epochs:
                monitor_metric = 'validation loss' if args.early_stopping_monitor == 'loss' else 'validation accuracy'
                _logger.info(f"Early stopping triggered at epoch {epoch+1}. No improvement in {monitor_metric} for {args.patience_epochs} epochs.")
                break
        
        # Logging
        best_monitor_value = best_val_loss if args.early_stopping_monitor == 'loss' else best_val_acc
        monitor_name = 'Val Loss' if args.early_stopping_monitor == 'loss' else 'Val Acc'
        
        _logger.info(
            f'Epoch {epoch+1}/{num_epochs} - '
            f'Train Loss: {train_metrics["loss"]:.4f} - '
            f'Val Acc: {eval_metrics["top1"]:.4f} - '
            f'Best Acc: {best_metric:.4f} (epoch {best_epoch+1}) - '
            f'{monitor_name}: {monitor_value:.4f} (best: {best_monitor_value:.4f}) - '
            f'Patience: {patience_counter}/{args.patience_epochs}'
        )
    
    # Save lightweight last checkpoint
    save_lightweight_checkpoint(model, os.path.join(output_dir, 'model_last_inference.pth'))
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(output_dir, 'training_curves.png')
    )
    
    # Test evaluation
    _logger.info('Evaluating on test set...')
    last_epoch_test_metrics = validate(
        model, loader_test, validate_loss_fn, args,
        amp_autocast=amp_autocast
    )
    
    # Load the best model for final evaluation
    best_checkpoint_path = os.path.join(output_dir, 'model_best.pth.tar')
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best model from epoch {best_epoch+1} for final evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location='cuda', weights_only=False)
        
        # Try to load the best model state dict (should work since it was saved during this training)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            print(f"Best model loaded successfully. Best accuracy: {best_metric:.4f}")
        except RuntimeError as e:
            print(f"Warning: Error loading best model checkpoint: {e}")
            print("Using current model for evaluation instead.")
            model.eval()
    else:
        print("Warning: Best model checkpoint not found. Using current model for evaluation.")

    best_test_eval_metrics = validate(
        model, loader_test, validate_loss_fn, args,
        amp_autocast=amp_autocast
    )
    
    # Generate confusion matrix on validation set using best model
    all_targets, all_predictions = generate_confusion_matrix_data(
        model, loader_eval, amp_autocast
    )
    
    # Generate confusion matrix on test set using best model
    test_targets, test_predictions = generate_confusion_matrix_data(
        model, loader_test, amp_autocast
    )
    
    # Define class names based on dataset - adjust for your specific dataset
    if args.num_classes == 7:
        # Assuming 7-class mango leaf disease dataset (adjust as needed)
        class_names = ['Anthracnose', 'Dieback', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Red Rust', 'Sooty Mold']
    elif args.num_classes == 8:
        class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
                       'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    else:
        # Generic class names for other numbers of classes
        class_names = [f'Class_{i}' for i in range(args.num_classes)]
        print(f"Using generic class names for {args.num_classes} classes: {class_names}")
        print("Consider updating the class_names definition in the script for your specific dataset.")
    
    # Calculate detailed metrics for validation set using best model
    val_accuracy = accuracy_score(all_targets, all_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics for validation set
    val_precision_per_class, val_recall_per_class, val_f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0, labels=range(len(class_names))
    )
    
    # Calculate detailed metrics for test set
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_targets, test_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics for test set
    test_precision_per_class, test_recall_per_class, test_f1_per_class, _ = precision_recall_fscore_support(
        test_targets, test_predictions, average=None, zero_division=0, labels=range(len(class_names))
    )
    
    # Plot confusion matrices
    plot_confusion_matrix(
        all_targets, all_predictions, class_names,
        os.path.join(output_dir, 'confusion_matrix_val.png')
    )
    
    plot_confusion_matrix(
        test_targets, test_predictions, class_names,
        os.path.join(output_dir, 'confusion_matrix_test.png')
    )
    
    # Generate Grad-CAM visualizations using best model
    try:
        generate_gradcam_visualization(model, loader_eval, output_dir, class_names)
    except Exception as gradcam_error:
        print(f"Grad-CAM failed: {gradcam_error}")
    

    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Get model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create training results summary
    training_results = {
        'training_time_seconds': total_training_time,
        'training_time_formatted': f"{total_training_time//3600:.0f}h {(total_training_time%3600)//60:.0f}m {total_training_time%60:.0f}s",
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'model_size_mb': total_params * 4 / 1024 / 1024,  # float32
        'best_validation_accuracy': float(best_metric),
        'early_stopping': {
            'patience_epochs': args.patience_epochs,
            'monitor_metric': args.early_stopping_monitor,
            'early_stopping_triggered': start_epoch + len(train_losses) < num_epochs,
            'epochs_completed_this_run': len(train_losses),
            'epochs_completed_total': start_epoch + len(train_losses),
            'best_epoch': int(best_epoch + 1)
        },
        'resume': {
            'enabled': resume_checkpoint is not None,
            'checkpoint_path': args.checkpoint_path if resume_checkpoint is not None else None,
            'start_epoch': int(start_epoch),
            'target_epochs': int(num_epochs),
        },
        'final_metrics': {
            'train_loss': float(train_losses[-1]) if train_losses else 0.0,
            'val_loss': float(val_losses[-1]) if val_losses else 0.0,
            'train_acc': float(train_accs[-1]) if train_accs else 0.0,
            'val_acc': float(best_metric),  # Use best accuracy instead of last epoch
            'test_loss': float(last_epoch_test_metrics['loss']),
            'test_acc': float(last_epoch_test_metrics['top1'])
        },
        'validation_metrics': {
            'accuracy': float(val_accuracy),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1_score': float(val_f1)
        },
        'test_metrics': {
            'loss': float(best_test_eval_metrics['loss']),
            'accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1)
        },
        'validation_per_class_metrics': {
            'precision': val_precision_per_class.tolist(),
            'recall': val_recall_per_class.tolist(),
            'f1_score': val_f1_per_class.tolist(),
            'class_names': class_names
        },
        'test_per_class_metrics': {
            'precision': test_precision_per_class.tolist(),
            'recall': test_recall_per_class.tolist(),
            'f1_score': test_f1_per_class.tolist(),
            'class_names': class_names
        },
        'epochs_completed': start_epoch + len(train_losses),
        'best_epoch': int(best_epoch + 1)
    }
    
    # Save training results
    results_file = os.path.join(output_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    _logger.info(f'Training completed! Best validation accuracy: {best_metric:.4f} (epoch {best_epoch+1})')
    _logger.info(f'Total training time: {training_results["training_time_formatted"]}')
    _logger.info(f'Model parameters: {trainable_params:,} trainable / {total_params:,} total')
    _logger.info(f'Early stopping: {"Triggered" if training_results["early_stopping"]["early_stopping_triggered"] else "Not triggered"} (patience: {args.patience_epochs} epochs, monitor: {args.early_stopping_monitor})')
    _logger.info(f'Last-epoch metrics - Train Acc: {training_results["final_metrics"]["train_acc"]:.2f}% | Val Acc: {best_metric:.2f}% | Test Acc: {training_results["final_metrics"]["test_acc"]:.2f}%')
    _logger.info(f'Last-epoch metrics - Train Loss: {training_results["final_metrics"]["train_loss"]:.4f} | Val Loss: {training_results["final_metrics"]["val_loss"]:.4f} | Test Loss: {training_results["final_metrics"]["test_loss"]:.4f}')
    _logger.info(f'Validation metrics - Accuracy: {training_results["validation_metrics"]["accuracy"]:.4f} | Precision: {training_results["validation_metrics"]["precision"]:.4f} | Recall: {training_results["validation_metrics"]["recall"]:.4f} | F1: {training_results["validation_metrics"]["f1_score"]:.4f}')
    _logger.info(f'Best-checkpoint test metrics - Loss: {training_results["test_metrics"]["loss"]:.4f} | Accuracy: {training_results["test_metrics"]["accuracy"]:.4f} | Precision: {training_results["test_metrics"]["precision"]:.4f} | Recall: {training_results["test_metrics"]["recall"]:.4f} | F1: {training_results["test_metrics"]["f1_score"]:.4f}')
    _logger.info(f'Training results saved to: {results_file}')
    _logger.info(f'Checkpoints saved: Full (model_best.pth.tar, last.pth.tar) | Lightweight (model_best_inference.pth, model_last_inference.pth)')
    _logger.info(f'Visualizations generated: Grad-CAM heatmaps, Token pruning progressive sparsification')

if __name__ == '__main__':
    main()
