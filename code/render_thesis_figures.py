#!/usr/bin/env python3
"""Assemble thesis figures from the retained results and original CAM panels."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'docs/figures'
VN = ROOT / 'experiments/04_phase3_vn_mangoleaf_320_seed_42'


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


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    seeds = (42, 123, 2024, 3333)
    groups = [('01_phase1_mangoleafbd', 'MangoLeafBD'), ('03_phase2_combined', 'Combined'), ('04_phase3_vn_mangoleaf', 'VN-MangoLeaf')]
    sources = []
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, (case, label) in enumerate(groups):
        values = []
        for seed in seeds:
            path = ROOT / f'experiments/{case}_320_seed_{seed}/training_results.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            values.append(result['test_metrics']['accuracy'] * 100)
            sources.append(path)
        assert all(98 <= value <= 100 for value in values), values
        axis.bar(np.arange(4) + (index - 1) * .25, values, width=.25, label=label)
    axis.set(xticks=np.arange(4), xticklabels=seeds, ylim=(98, 100.2),
             xlabel='Training seed', ylabel='Held-out test accuracy (%)',
             title='Test accuracy by seed across the three training phases')
    axis.legend(loc='lower right')
    fig.text(.01, .01, 'Source: best-checkpoint test_metrics; four seeds per phase. Y-axis starts at 98%.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    output = FIGURES / 'fig_4_06_accuracy_by_seed.png'
    fig.savefig(output, dpi=200)
    plt.close(fig)
    manifest = {output.name: sources}
    gradcam = [VN / f'gradcam_{name}.png' for name in (
        'Healthy', 'Anthracnose', 'Gall_Midge', 'Powdery_Mildew', 'Sooty_Mold', 'Dieback', 'Red_Rust')]
    output = FIGURES / 'fig_4_07_gradcam_vn_seed42.png'
    stack_panels(gradcam, output)
    manifest[output.name] = gradcam
    errors = []
    for index in (1, 2, 3):
        matches = sorted(VN.glob(f'error_{index:03d}_*_cam.png'))
        if len(matches) != 1:
            raise ValueError(f'expected exactly one panel for error {index}: {matches}')
        errors.extend(matches)
    output = FIGURES / 'fig_4_08_errors_vn_seed42.png'
    stack_panels(errors, output)
    manifest[output.name] = errors
    metadata = {}
    for name, paths in manifest.items():
        metadata[name] = [{'path':path.relative_to(ROOT).as_posix(),
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    (FIGURES / 'derived_figures_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('Rendered figures 4.6, 4.7 and 4.8 from retained evidence.')


if __name__ == '__main__':
    main()
