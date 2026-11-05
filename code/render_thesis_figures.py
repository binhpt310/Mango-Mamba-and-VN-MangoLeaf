#!/usr/bin/env python3
"""Assemble thesis figures from the retained results and original CAM panels."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'docs/figures'


def stack_panels(sources, destination):
    if not sources or not all(path.is_file() for path in sources):
        raise FileNotFoundError(f'missing figure panels: {sources}')
    panels = []
    for source in sources:
        with Image.open(source) as image:
            panel = image.convert('RGB')
            panels.append(panel.resize((1800, round(panel.height * 1800 / panel.width)), Image.Resampling.LANCZOS))
    canvas = Image.new('RGB', (1800, sum(panel.height for panel in panels)), 'white')
    offset = 0
    for panel in panels:
        canvas.paste(panel, (0, offset))
        offset += panel.height
    canvas.save(destination)


def render(experiments, figures, error_panels=None):
    experiments = experiments.resolve()
    figures = figures.resolve()
    vn = experiments / '04_phase3_vn_mangoleaf_320_seed_42'
    error_panels = error_panels.resolve() if error_panels is not None else vn
    if not error_panels.is_dir():
        raise FileNotFoundError(error_panels)
    figures.mkdir(parents=True, exist_ok=True)
    seeds = (42, 123, 2024, 3333)
    groups = [('01_phase1_mangoleafbd', 'MangoLeafBD'), ('03_phase2_combined', 'Combined'), ('04_phase3_vn_mangoleaf', 'VN-MangoLeaf')]
    sources = []
    accuracies = []
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, (case, label) in enumerate(groups):
        values = []
        for seed in seeds:
            path = experiments / f'{case}_320_seed_{seed}/training_results.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            values.append(result['test_metrics']['accuracy'] * 100)
            sources.append(path)
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            raise ValueError(f'Invalid accuracy values: {values}')
        accuracies.extend(values)
        axis.bar(np.arange(4) + (index - 1) * .25, values, width=.25, label=label)
    ymin = 98 if min(accuracies) >= 98 else max(0, math.floor(min(accuracies)) - 1)
    axis.set(xticks=np.arange(4), xticklabels=seeds, ylim=(ymin, 100.2),
             xlabel='Training seed', ylabel='Held-out test accuracy (%)',
             title='Test accuracy by seed across the three training phases')
    axis.legend(loc='lower right')
    fig.text(.01, .01, f'Source: best-checkpoint test_metrics; four seeds per phase. Y-axis starts at {ymin}%.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    output = figures / 'fig_4_06_accuracy_by_seed.png'
    fig.savefig(output, dpi=200)
    plt.close(fig)
    manifest = {output.name: sources}
    gradcam = [vn / f'gradcam_{name}.png' for name in (
        'Healthy', 'Anthracnose', 'Gall_Midge', 'Powdery_Mildew', 'Sooty_Mold', 'Dieback', 'Red_Rust')]
    output = figures / 'fig_4_07_gradcam_vn_seed42.png'
    stack_panels(gradcam, output)
    manifest[output.name] = gradcam
    errors = sorted(error_panels.glob('error_*_cam.png'))[:3]
    output = figures / 'fig_4_08_errors_vn_seed42.png'
    if errors:
        stack_panels(errors, output)
    else:
        summary = next((p for p in (error_panels / 'summary.json', error_panels / 'error_analysis_summary.json')
                        if p.is_file()), None)
        if summary is None or json.loads(summary.read_text())['error_analysis']['error_count'] != 0:
            raise ValueError('Run analyze_vn_errors.py first and pass its output with --error-panels.')
        fig, axis = plt.subplots(figsize=(9, 2))
        axis.text(.5, .5, 'No misclassified test images', ha='center', va='center')
        axis.axis('off')
        fig.savefig(output, dpi=200, bbox_inches='tight')
        plt.close(fig)
        errors = [summary]
    manifest[output.name] = errors
    metadata = {}
    for name, paths in manifest.items():
        metadata[name] = [{'path':path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    (figures / 'derived_figures_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('Rendered figures 4.6, 4.7 and 4.8 from retained evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--output', type=Path, default=FIGURES)
    parser.add_argument('--error-panels', type=Path, help='Output directory from analyze_vn_errors.py')
    args = parser.parse_args()
    render(args.experiments, args.output, args.error_panels)


if __name__ == '__main__':
    main()
