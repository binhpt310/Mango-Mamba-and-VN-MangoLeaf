"""Check experiment contracts without importing the GPU training stack."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('run_curriculum', Path(__file__).resolve().parents[1] / 'code/run_curriculum.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_ablation_disables_inter_window_and_uses_224(self):
        for case, features in runner.ABLATIONS.items():
            with self.subTest(case=case):
                args = runner.command('case', Path('data'), 8, Path('out'), 42,
                                      resolution=224, features=features)
                self.assertEqual(args[args.index('--input-size') + 1:args.index('--input-size') + 4], ['3', '224', '224'])
                self.assertEqual(args[args.index('--window-size') + 1:args.index('--window-size') + 5], ['8', '8', '14', '7'])
                self.assertEqual(args[-1], '--inter-window-mamba-stages')
                self.assertEqual([x for x in args if x.startswith('--use-')], [f'--use-{x}' for x in features])

    def test_curriculum_and_direct_keep_seed_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            phase1 = output / '01_phase1_mangoleafbd_320_seed_123/model_best.pth.tar'
            phase1.parent.mkdir(); phase1.touch()
            combined = output / '03_phase2_combined_320_seed_123/model_best.pth.tar'
            with patch.object(runner, 'train', side_effect=[combined, Path('phase3')]) as train:
                runner.run_task('curriculum', 123, output, Path('data'))
                self.assertEqual(train.call_args_list[0].args[-1], phase1)
                self.assertEqual(train.call_args_list[1].args[-1], combined)
                self.assertEqual(train.call_args_list[1].args[0], '04_phase3_vn_mangoleaf_320_seed_123')
            with patch.object(runner, 'train') as train:
                runner.run_task('direct-vn', 123, output, Path('data'))
                self.assertEqual(train.call_args.args[-1], phase1)
                self.assertEqual(train.call_args.args[0], '05_direct_vn_from_phase1_320_seed_123')

    def test_missing_checkpoint_does_not_fall_back_to_scratch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'train') as train:
            with self.assertRaises(FileNotFoundError):
                runner.run_task('direct-vn', 42, Path(tmp), Path('data'))
            train.assert_not_called()

    def test_completed_run_requires_weights_and_is_never_retrained(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run') as launch:
            output = Path(tmp); run = output / 'case'; run.mkdir()
            (run / 'training_results.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):
                runner.train('case', Path('missing_data'), 8, output, 42)
            (run / 'model_best.pth.tar').touch()
            self.assertEqual(runner.train('case', Path('missing_data'), 8, output, 42), run / 'model_best.pth.tar')
            launch.assert_not_called()

    def test_resume_uses_last_checkpoint_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'trainer')) as launch:
            output = Path(tmp); run = output / 'case'; run.mkdir()
            last = run / 'last.pth.tar'; last.touch()
            data = output / 'data'
            for split in ('train', 'val', 'test'): (data / split).mkdir(parents=True)
            with self.assertRaises(subprocess.CalledProcessError):
                runner.train('case', data, 8, output, 42)
            args = launch.call_args.args[0]
            self.assertIn('--resume', args)
            self.assertEqual(args[args.index('--checkpoint-path') + 1], str(last))


if __name__ == '__main__':
    unittest.main()
