#!/usr/bin/env python3
"""Aggregate the nine thesis cases from four completed result JSONs per case."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import sys

SEEDS = (42, 123, 2024, 3333)
CASES = (
    '01_phase1_mangoleafbd_320', '02_ablation_full_224', '02_ablation_no_eca_224',
    '02_ablation_no_grn_224', '02_ablation_no_swiglu_224', '02_ablation_hybrid_minimal_224',
    '03_phase2_combined_320', '04_phase3_vn_mangoleaf_320', '05_direct_vn_from_phase1_320',
)


def summarize(root):
    rows = []
    for case in CASES:
        results = [json.loads((root / f'{case}_seed_{seed}/training_results.json').read_text(encoding='utf-8'))
                   for seed in SEEDS]
        row = {'case': case, 'n': len(results)}
        for key in ('accuracy', 'precision', 'recall', 'f1_score'):
            values = [result['test_metrics'][key] * 100 for result in results]
            row[f'test_{key}_mean_pct'] = mean(values)
            row[f'test_{key}_sample_sd_pct'] = stdev(values)
        for key, values in (
            ('best_epoch', [r['best_epoch'] for r in results]),
            ('last_epoch_val_loss', [r['final_metrics']['val_loss'] for r in results]),
        ):
            row[f'{key}_mean'] = mean(values)
            row[f'{key}_sample_sd'] = stdev(values)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=Path(__file__).resolve().parents[1] / 'experiments')
    args = parser.parse_args()
    rows = summarize(args.experiments)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
