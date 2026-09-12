#!/usr/bin/env python3
"""Idempotently finish one H3 follow-up task for one seed."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H3_ROOT = ROOT / 'results' / 'test new mangomamba baseline to surpase 99.75 acc' / 'phase2_window_8_8_5_5'
DEFAULT_OUTPUT = ROOT / 'results' / 'h3_320_aligned_followup'
SEEDS = (42, 123, 2024, 3333)


def newest(paths: list[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f'no {description} found')
    return max(paths, key=lambda path: path.stat().st_mtime)


def h3_checkpoint(seed: int) -> Path:
    matches = list(H3_ROOT.glob(f'phase2_window_8_8_5_5_seed_{seed}-*/model_best.pth.tar'))
    return newest(matches, f'H3 checkpoint for seed {seed}')


def completed_checkpoint(output: Path, experiment: str) -> Path | None:
    results = list(output.glob(f'{experiment}-*/training_results.json'))
    if not results:
        return None
    return newest(results, f'completed {experiment} result').parent / 'model_best.pth.tar'


def interrupted_checkpoint(output: Path, experiment: str) -> Path | None:
    checkpoints = [
        checkpoint for checkpoint in output.glob(f'{experiment}-*/last.pth.tar')
        if not (checkpoint.parent / 'training_results.json').exists()
    ]
    return newest(checkpoints, f'interrupted {experiment} checkpoint') if checkpoints else None


def train(
    experiment: str,
    dataset: Path,
    classes: int,
    output: Path,
    seed: int,
    initial_checkpoint: Path | None,
    *,
    inter_window: bool,
) -> Path:
    complete = completed_checkpoint(output, experiment)
    if complete is not None:
        print(f'SKIP completed {experiment}: {complete}', flush=True)
        return complete

    resume_checkpoint = interrupted_checkpoint(output, experiment)
    command = [
        sys.executable, 'train_mango_mamba.py',
        '--data_dir', str(dataset), '--num-classes', str(classes),
        '--model', 'mango_mamba_tiny', '--input-size', '3', '320', '320',
        '--window-size', '8', '8', '5', '5',
        '--epochs', '100', '--batch-size', '64', '--grad-accum-steps', '4', '--workers', '4',
        '--opt', 'adamw', '--lr', '0.0005', '--weight-decay', '0.01', '--sched', 'cosine',
        '--warmup-epochs', '5', '--min-lr', '0.000005', '--smoothing', '0.01',
        '--patience-epochs', '20', '--early-stopping-monitor', 'loss', '--drop-path-rate', '0.01',
        '--re-prob', '0.05', '--color-jitter', '0.05', '--hflip', '0.5', '--vflip', '0.2',
        '--mixer-mode', 'hybrid', '--use-eca', '--use-grn', '--use-swiglu',
        '--seed', str(seed), '--output', str(output), '--experiment', experiment, '--log-interval', '10',
    ]
    if inter_window:
        command.extend(('--inter-window-mamba-stages', '0'))
    checkpoint = resume_checkpoint or initial_checkpoint
    if checkpoint is not None:
        command.extend(('--train-from-checkpoint', '--checkpoint-path', str(checkpoint)))
    if resume_checkpoint is not None:
        command.append('--resume')
        print(f'RESUME {experiment} from {resume_checkpoint}', flush=True)
    else:
        print(f'RUN {experiment}', flush=True)

    subprocess.run(command, cwd=ROOT, check=True)
    complete = completed_checkpoint(output, experiment)
    if complete is None:
        raise RuntimeError(f'{experiment} exited without training_results.json')
    return complete


def run_task(task: str, seed: int, output: Path) -> None:
    mango = ROOT / 'data' / 'mangoleafBD'
    combined = ROOT / 'data' / 'combined_dataset'
    vn = ROOT / 'data' / 'vn_mangoleaf'
    if task == 'no-inter':
        train(f'h3_no_inter_mangoleafbd_seed_{seed}', mango, 8, output / 'no_inter_mangoleafbd', seed, None, inter_window=False)
    elif task == 'direct-vn':
        train(f'h3_no_curriculum_vn_seed_{seed}', vn, 7, output / 'no_curriculum_vn', seed, h3_checkpoint(seed), inter_window=True)
    else:
        combined_best = train(f'h3_combined_seed_{seed}', combined, 8, output / 'curriculum' / f'seed_{seed}' / 'combined', seed, h3_checkpoint(seed), inter_window=True)
        train(f'h3_vn_curriculum_seed_{seed}', vn, 7, output / 'curriculum' / f'seed_{seed}' / 'vn_mangoleaf', seed, combined_best, inter_window=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('no-inter', 'direct-vn', 'curriculum'), required=True)
    parser.add_argument('--seed', choices=SEEDS, type=int, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for dataset in (ROOT / 'data' / 'mangoleafBD', ROOT / 'data' / 'combined_dataset', ROOT / 'data' / 'vn_mangoleaf'):
        if not all((dataset / split).is_dir() for split in ('train', 'val', 'test')):
            raise RuntimeError(f'missing train/val/test split under {dataset}')
    run_task(args.task, args.seed, args.output.resolve())


if __name__ == '__main__':
    main()
