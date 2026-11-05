"""Checkpoint state needed to resume this trainer at an epoch boundary."""

from pathlib import Path
import random
import shutil

import numpy as np
import torch
from timm import utils


class TrainingCheckpointSaver(utils.CheckpointSaver):
    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, task=self, **kwargs)
        self.can_hardlink = False
        self.scheduler = scheduler
        self.training_state = {}

    def get_checkpoint_state(self, *, ema=False, unwrap_fn=None):
        if ema:
            return {}
        return {
            'state_dict': utils.get_state_dict(self.model, unwrap_fn),
            'training_state': self.training_state,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'rng_state': {
                'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if any(p.is_cuda for p in self.model.parameters()) else None,
            },
        }

    def restore(self, checkpoint, checkpoint_path):
        state = checkpoint.get('training_state')
        if not state or state.get('version') != 1 or 'rng_state' not in checkpoint:
            raise ValueError('Legacy checkpoint lacks full resume state; use it only to initialize a new run.')
        previous_args = checkpoint.get('args')
        for key in ('epochs', 'early_stopping_monitor'):
            if previous_args is not None and getattr(previous_args, key, None) != getattr(self.args, key, None):
                raise ValueError(f'Resume must keep the original {key}; use a new run to change it.')
        if state['best_epoch'] < 0 or state['best_epoch'] > checkpoint['epoch']:
            raise ValueError('Invalid best epoch in resume checkpoint')
        if any(len(state[key]) != checkpoint['epoch'] + 1
               for key in ('train_losses', 'val_losses', 'train_accs', 'val_accs')):
            raise ValueError('Resume history length does not match the saved epoch')

        best_path = Path(self.checkpoint_dir) / 'model_best.pth.tar'
        # Recover a save interrupted after committing last but before copying the new best.
        if state['best_epoch'] == checkpoint['epoch']:
            source = Path(checkpoint_path)
            if source.resolve() != best_path.resolve():
                shutil.copy2(source, best_path)
        if not best_path.is_file():
            raise FileNotFoundError('Resume requires model_best.pth.tar beside the last checkpoint')
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        monitor = state['best_val_loss'] if self.decreasing else state['best_val_acc']
        if best['epoch'] != state['best_epoch'] or best['metric'] != monitor:
            raise ValueError('The saved best checkpoint does not match the resume state')

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.amp_scaler is not None:
            self.amp_scaler.load_state_dict(checkpoint[self.amp_scaler.state_dict_key])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_metric = monitor
        self.best_epoch = state['best_epoch']
        self.training_state = state
        for path in Path(self.checkpoint_dir).glob(f'{self.save_prefix}-*{self.extension}'):
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['epoch'] <= checkpoint['epoch']:
                self.checkpoint_files.append((str(path), saved['metric']))
        self.checkpoint_files.sort(key=lambda item: item[1], reverse=not self.decreasing)

        rng = checkpoint['rng_state']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
        return state
