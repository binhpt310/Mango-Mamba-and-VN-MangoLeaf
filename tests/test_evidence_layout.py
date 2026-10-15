"""Verify the thesis evidence inventory after moving files."""

import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


class EvidenceLayoutTests(unittest.TestCase):
    def test_exactly_36_flat_runs(self):
        cases = ('02_ablation_full_224', '02_ablation_no_eca_224', '02_ablation_no_grn_224',
                 '02_ablation_no_swiglu_224', '02_ablation_hybrid_minimal_224',
                 '01_phase1_mangoleafbd_320', '03_phase2_combined_320',
                 '04_phase3_vn_mangoleaf_320', '05_direct_vn_from_phase1_320')
        expected = {f'{case}_seed_{seed}' for case in cases for seed in (42, 123, 2024, 3333)}
        self.assertEqual({p.name for p in (ROOT / 'experiments').iterdir()}, expected)
        for name in expected:
            run = ROOT / 'experiments' / name
            self.assertTrue((run / 'args.yaml').is_file(), name)
            self.assertTrue((run / 'training_results.json').is_file(), name)
            self.assertFalse(any(p.is_dir() for p in run.iterdir()), name)

    def test_original_evidence_bytes_unchanged(self):
        manifest = json.loads((ROOT / 'docs/REORGANIZATION_MANIFEST.json').read_text(encoding='utf-8-sig'))
        checked = 0
        for entry in manifest['files']:
            for name in entry['new_paths']:
                if not name.startswith(('experiments/', 'android/')):
                    continue
                with (ROOT / name).open('rb') as handle:
                    self.assertEqual(hashlib.file_digest(handle, 'sha256').hexdigest(), entry['sha256'], name)
                checked += 1
        self.assertGreater(checked, 400)

    def test_checkpoint_inventory(self):
        entries = json.loads((ROOT / 'docs/CHECKPOINT_SYNC_MANIFEST.json').read_text())['entries']
        self.assertEqual(sum(e['status'] == 'verified' for e in entries), 16)
        self.assertEqual(sum(e['status'] == 'not_found_on_host' for e in entries), 20)
        for entry in entries:
            path = ROOT / entry['local_path']
            self.assertEqual(path.is_file(), entry['status'] == 'verified', str(path))
            if path.is_file():
                self.assertEqual(path.stat().st_size, entry['bytes'])

    def test_five_test_errors_and_seven_validation_gradcams(self):
        run = ROOT / 'experiments/04_phase3_vn_mangoleaf_320_seed_42'
        self.assertEqual(len(list(run.glob('error_*_cam.png'))), 5)
        self.assertEqual(len(list(run.glob('error_*.jpg'))), 5)
        self.assertEqual(len(list(run.glob('gradcam_*.png'))), 7)
        self.assertEqual(json.loads((run / 'error_analysis_summary.json').read_text())['error_analysis']['error_count'], 5)
        sources = json.loads((ROOT / 'docs/figures/derived_figures_manifest.json').read_text())
        self.assertEqual(len(sources['fig_4_06_accuracy_by_seed.png']), 12)
        self.assertEqual(len(sources['fig_4_07_gradcam_vn_seed42.png']), 7)
        self.assertEqual(len(sources['fig_4_08_errors_vn_seed42.png']), 3)
        for name, inputs in sources.items():
            self.assertTrue((ROOT / 'docs/figures' / name).is_file())
            for item in inputs:
                self.assertEqual(hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest(), item['sha256'])

    def test_current_android_assets(self):
        assets = ROOT / 'android/app/src/main/assets'
        self.assertEqual(len(list(assets.glob('*.onnx'))), 1)
        model = assets / 'mango_mamba_v3_phase3_320_cam.onnx'
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), 'f312db7a8046e8481063752393450dc985d85ef2ac59aec3131cbbd8a081e3a4')
        manifest = json.loads((assets / 'mango_mamba_v3_phase3_320_cam_manifest.json').read_text())
        self.assertEqual(manifest['input']['shape'], [1, 3, 320, 320])
        self.assertEqual(manifest['checkpoint_sha256'], '208b07937526b35260767afa62da43616d8a0245ba1fd7680102990bc34ae8ae')

    def test_no_legacy_names_or_broken_document_links(self):
        self.assertEqual({p.name for p in (ROOT / 'docs').glob('*.md')},
                         {'EVIDENCE_INDEX.md', 'EXPERIMENT_PROTOCOL.md'})
        for folder in ('code', 'experiments', 'android', 'tests'):
            for path in (ROOT / folder).rglob('*'):
                self.assertNotIn('h3', path.name.lower())
        for path in [ROOT / 'README.md', ROOT / 'android/README.md', *(ROOT / 'docs').glob('*.md')]:
            text = path.read_text(encoding='utf-8-sig')
            for link in re.findall(r'\]\((<[^>]+>|[^)]+)\)', text):
                link = unquote(link.strip('<>').split('#')[0])
                if not link or '://' in link:
                    continue
                self.assertTrue((path.parent / link).exists(), f'{path.name}: {link}')


if __name__ == '__main__':
    unittest.main()
