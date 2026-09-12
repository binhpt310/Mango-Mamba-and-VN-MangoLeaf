#!/usr/bin/env python3
"""Sequential VN-MangoLeaf transfer experiments from inter-window Mamba weights."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'inter_window_mamba_100' / 'mangoleafbd' / 'fixed_window_inter_mamba'
DEFAULT_OUTPUT = ROOT / 'results' / 'fixed_window_inter_mamba_vn_transfer_100'


def source_checkpoint(seed: int) -> Path:
    matches = sorted(SOURCE.glob(f'fixed_window_inter_mamba_seed_{seed}-*/model_best.pth.tar'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one MangoLeafBD best checkpoint for seed {seed}, found {matches}')
    return matches[0]


def newest_best(output: Path) -> Path:
    matches = sorted(output.glob('*/model_best.pth.tar'), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f'no best checkpoint written under {output}')
    return matches[-1]


def train(label: str, dataset: Path, classes: int, output: Path, seed: int, checkpoint: Path) -> None:
    command = [
        sys.executable, 'train_mango_mamba.py',
        '--data_dir', str(dataset), '--num-classes', str(classes),
        '--epochs', '100', '--batch-size', '64', '--grad-accum-steps', '4', '--workers', '4',
        '--opt', 'adamw', '--lr', '0.0005', '--weight-decay', '0.01', '--sched', 'cosine',
        '--warmup-epochs', '5', '--min-lr', '0.000005', '--smoothing', '0.01',
        '--patience-epochs', '20', '--early-stopping-monitor', 'loss', '--drop-path-rate', '0.01',
        '--seed', str(seed), '--output', str(output), '--experiment', label, '--log-interval', '10',
        '--mixer-mode', 'hybrid', '--use-eca', '--use-grn', '--use-swiglu',
        '--inter-window-mamba-stages', '0', '--window-size', '7', '7', '7', '7',
        '--train-from-checkpoint', '--checkpoint-path', str(checkpoint),
    ]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_manifest(output: Path, seeds: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'seeds': seeds,
        'model': 'mango_mamba_tiny',
        'inter_window_mamba_stages': [0],
        'source_checkpoints': {str(seed): str(source_checkpoint(seed)) for seed in seeds},
        'protocols': {
            'no_curriculum_vn': 'MangoLeafBD model_best -> VN-MangoLeaf',
            'curriculum': 'MangoLeafBD model_best -> combined_dataset -> VN-MangoLeaf',
        },
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024, 3333])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    write_manifest(output, args.seeds)

    for seed in args.seeds:
        train(
            f'no_curriculum_vn_seed_{seed}', ROOT / 'data' / 'vn_mangoleaf', 7,
            output / 'no_curriculum_vn', seed, source_checkpoint(seed),
        )

    for seed in args.seeds:
        phase1 = output / 'curriculum' / f'seed_{seed}' / 'phase1_combined'
        train(
            f'phase1_combined_seed_{seed}', ROOT / 'data' / 'combined_dataset', 8,
            phase1, seed, source_checkpoint(seed),
        )
        train(
            f'phase2_vn_seed_{seed}', ROOT / 'data' / 'vn_mangoleaf', 7,
            output / 'curriculum' / f'seed_{seed}' / 'phase2_vn_mangoleaf',
            seed, newest_best(phase1),
        )


if __name__ == '__main__':
    main()
