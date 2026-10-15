#!/usr/bin/env python3
"""Run the thesis cases using one flat directory per case and seed."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024, 3333)
ABLATIONS = {
    'full': ('eca', 'grn', 'swiglu'),
    'no_eca': ('grn', 'swiglu'),
    'no_grn': ('eca', 'swiglu'),
    'no_swiglu': ('eca', 'grn'),
    'hybrid_minimal': (),
}


def command(experiment, dataset, classes, output, seed, checkpoint=None, *,
            resolution=320, features=('eca', 'grn', 'swiglu'), resume=False):
    window = (8, 8, 5, 5) if resolution == 320 else (8, 8, 14, 7)
    args = [
        sys.executable, str(ROOT / 'code/train_mango_mamba.py'),
        '--data_dir', str(dataset), '--num-classes', str(classes),
        '--model', 'mango_mamba_tiny', '--input-size', '3', str(resolution), str(resolution),
        '--window-size', *map(str, window),
        '--epochs', '100', '--batch-size', '64', '--grad-accum-steps', '4', '--workers', '4',
        '--opt', 'adamw', '--lr', '0.0005', '--weight-decay', '0.01', '--sched', 'cosine',
        '--warmup-epochs', '5', '--min-lr', '0.000005', '--smoothing', '0.01',
        '--patience-epochs', '20', '--early-stopping-monitor', 'loss', '--drop-path-rate', '0.01',
        '--re-prob', '0.05', '--color-jitter', '0.05', '--hflip', '0.5', '--vflip', '0.2',
        '--mixer-mode', 'hybrid', *[f'--use-{feature}' for feature in features],
        '--seed', str(seed), '--output', str(output), '--experiment', experiment,
        '--log-interval', '10', '--inter-window-mamba-stages',
    ]
    if resolution == 320:
        args.append('0')
    if checkpoint is not None:
        args.extend(('--train-from-checkpoint', '--checkpoint-path', str(checkpoint)))
    if resume:
        if checkpoint is None:
            raise ValueError('resume requires a checkpoint')
        args.append('--resume')
    return args


def require_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f'missing checkpoint: {path}')
    return path


def train(experiment, dataset, classes, output, seed, checkpoint=None, **settings):
    run = output / experiment
    best = run / 'model_best.pth.tar'
    if (run / 'training_results.json').is_file():
        require_checkpoint(best)
        print(f'SKIP completed {experiment}', flush=True)
        return best
    last = run / 'last.pth.tar'
    if checkpoint is not None:
        require_checkpoint(checkpoint)
    for split in ('train', 'val', 'test'):
        if not (dataset / split).is_dir():
            raise FileNotFoundError(f'missing dataset split: {dataset / split}')
    if run.exists() and any(run.iterdir()) and not last.is_file():
        raise FileExistsError(f'refusing to overwrite incomplete evidence without last.pth.tar: {run}')
    args = command(experiment, dataset, classes, output, seed,
                   last if last.is_file() else checkpoint, resume=last.is_file(), **settings)
    run.mkdir(parents=True, exist_ok=True)
    # Keep each invocation when resuming so the original launch remains traceable.
    with (run / 'invocations.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': args}) + '\n')
    print(f'RUN {experiment}; log: {run / "train.log"}', flush=True)
    with (run / 'train.log').open('a', encoding='utf-8') as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (run / 'training_results.json').is_file():
        raise RuntimeError(f'{experiment} exited without training_results.json')
    return require_checkpoint(best)


def run_task(task, seed, output, data_root, case=None):
    phase1 = output / f'01_phase1_mangoleafbd_320_seed_{seed}/model_best.pth.tar'
    if task == 'phase1':
        return train(phase1.parent.name, data_root / 'mangoleafBD', 8, output, seed)
    if task == 'ablation':
        return train(f'02_ablation_{case}_224_seed_{seed}', data_root / 'mangoleafBD', 8,
                     output, seed, resolution=224, features=ABLATIONS[case])
    if task == 'direct-vn':
        return train(f'05_direct_vn_from_phase1_320_seed_{seed}', data_root / 'vn_mangoleaf',
                     7, output, seed, require_checkpoint(phase1))
    if task != 'curriculum':
        raise ValueError(f'unknown task: {task}')
    combined = train(f'03_phase2_combined_320_seed_{seed}', data_root / 'combined_dataset',
                     8, output, seed, require_checkpoint(phase1))
    return train(f'04_phase3_vn_mangoleaf_320_seed_{seed}', data_root / 'vn_mangoleaf',
                 7, output, seed, combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('phase1', 'ablation', 'curriculum', 'direct-vn'), required=True)
    parser.add_argument('--seed', choices=SEEDS, type=int, required=True)
    parser.add_argument('--case', choices=tuple(ABLATIONS))
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    if (args.task == 'ablation') != (args.case is not None):
        parser.error('--case is required only for --task ablation')
    run_task(args.task, args.seed, args.output.resolve(), args.data_root.resolve(), args.case)


if __name__ == '__main__':
    main()
