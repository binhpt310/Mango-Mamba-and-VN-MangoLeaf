"""Resume must preserve the best checkpoint and the next optimizer/RNG step."""

import hashlib
import inspect
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from timm.utils import CheckpointSaver

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from training_checkpoint import TrainingCheckpointSaver


@unittest.skipUnless('task' in inspect.signature(CheckpointSaver).parameters, 'Run with the pinned timm 1.0.28 environment')
class ResumeTests(unittest.TestCase):
    def make(self, directory):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        saver = TrainingCheckpointSaver(model, optimizer, args=SimpleNamespace(model='linear', epochs=4,
            early_stopping_monitor='loss'), checkpoint_dir=directory, decreasing=True, scheduler=scheduler)
        return model, optimizer, scheduler, saver

    def step(self, objects, epoch):
        model, optimizer, scheduler, saver = objects
        optimizer.zero_grad()
        x = torch.randn(3, 1) * (random.random() + np.random.random())
        loss = (model(x) - torch.randn(3, 1)).square().mean()
        loss.backward(); optimizer.step(); scheduler.step()
        metric = (.1, .5, .8, .05)[epoch]
        previous = saver.training_state
        better = not previous or metric < previous['best_val_loss']
        saver.training_state = {
            'version': 1, 'best_accuracy': 90.,
            'best_epoch': epoch if better else previous['best_epoch'],
            'best_val_loss': metric if better else previous['best_val_loss'], 'best_val_acc': 90.,
            'patience_counter': 0 if better else previous['patience_counter'] + 1,
            'train_losses': previous.get('train_losses', []) + [loss.item()],
            'val_losses': previous.get('val_losses', []) + [metric],
            'train_accs': previous.get('train_accs', []) + [90.],
            'val_accs': previous.get('val_accs', []) + [90.], 'elapsed_seconds': float(epoch + 1),
        }
        saver.save_checkpoint(epoch, metric)

    def seed(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)

    def test_resume_preserves_best_last_rng_optimizer_and_history(self):
        with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as split_dir:
            self.seed(); full = self.make(full_dir)
            for epoch in range(4): self.step(full, epoch)
            self.seed(); split = self.make(split_dir)
            for epoch in range(2): self.step(split, epoch)
            best_path = Path(split_dir) / 'model_best.pth.tar'
            digest = hashlib.sha256(best_path.read_bytes()).hexdigest()
            last = Path(split_dir) / 'last.pth.tar'
            saved = torch.load(last, weights_only=False)
            self.assertEqual(saved['epoch'], 1)
            self.assertEqual(saved['metric'], .5)
            resumed = self.make(split_dir)
            resumed[0].load_state_dict(saved['state_dict'])
            state = resumed[3].restore(saved, last)
            self.assertEqual(state['patience_counter'], 1)
            self.assertEqual(state['val_losses'], [.1, .5])
            self.step(resumed, 2)
            self.assertEqual(hashlib.sha256(best_path.read_bytes()).hexdigest(), digest)
            self.step(resumed, 3)
            for a, b in zip(full[0].parameters(), resumed[0].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(full[2].state_dict(), resumed[2].state_dict())
            self.assertEqual(full[3].training_state, resumed[3].training_state)

    def test_legacy_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = self.make(directory)[3]
            with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                saver.restore({'epoch': 0}, Path(directory) / 'last.pth.tar')


if __name__ == '__main__':
    unittest.main()
