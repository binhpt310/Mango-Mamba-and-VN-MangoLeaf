"""Exercise trainer.main through an interrupted run using a small CPU model."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
HAS_STACK = importlib.util.find_spec('mamba_ssm') is not None
if HAS_STACK:
    import train_mango_mamba as trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 8)

    def forward(self, x):
        return {'logits': self.head(x.mean((-2, -1)))}


@unittest.skipUnless(HAS_STACK, 'Run with the pinned training environment')
class TrainerResumeTests(unittest.TestCase):
    def test_main_continuous_and_resumed_runs_match(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text('amp: false\nwarmup_epochs: 0\n')
            generator = torch.Generator().manual_seed(7)
            batches = [(torch.randn(4, 3, 8, 8, generator=generator), torch.arange(4) + (i % 2) * 4)
                       for i in range(4)]

            def loader(*args, **kwargs):
                self.assertFalse(kwargs['persistent_workers'])
                return batches

            def run(name, interrupt=False, resume=False):
                argv = ['trainer', '-c', str(config), '--output', str(root), '--experiment', name,
                        '--epochs', '4', '--batch-size', '4', '--grad-accum-steps', '2', '--workers', '0']
                if resume:
                    argv += ['--resume', '--train-from-checkpoint', '--checkpoint-path', str(root / name / 'last.pth.tar')]
                original = trainer.train_one_epoch

                def epoch(index, *args, **kwargs):
                    if interrupt and index == 2:
                        raise InterruptedError('controlled test interruption')
                    return original(index, *args, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(patch.object(sys, 'argv', argv))
                    stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
                    stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda value, *a, **kw: value))
                    stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value, *a, **kw: value))
                    stack.enter_context(patch.object(trainer, 'create_model_fn', side_effect=lambda *a, **kw: TinyModel()))
                    stack.enter_context(patch.object(trainer, 'create_dataset', return_value=SimpleNamespace()))
                    stack.enter_context(patch.object(trainer, 'create_loader', side_effect=loader))
                    stack.enter_context(patch.object(trainer, 'generate_gradcam_visualization'))
                    stack.enter_context(patch.object(trainer, 'train_one_epoch', side_effect=epoch))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    trainer.main()

            run('continuous')
            with self.assertRaises(InterruptedError):
                run('resumed', interrupt=True)
            run('resumed', resume=True)
            a = torch.load(root / 'continuous/last.pth.tar', weights_only=False)
            b = torch.load(root / 'resumed/last.pth.tar', weights_only=False)
            self.assertEqual(b['epoch'], 3)
            self.assertEqual(len(b['training_state']['train_losses']), 4)
            for key in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)
            result = json.loads((root / 'resumed/training_results.json').read_text())
            self.assertEqual(result['epochs_completed'], 4)
            self.assertEqual(result['early_stopping']['epochs_completed_this_run'], 2)


if __name__ == '__main__':
    unittest.main()
