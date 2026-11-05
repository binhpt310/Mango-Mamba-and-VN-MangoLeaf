"""New runs can have lower accuracy and zero, one, or two test errors."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

spec = importlib.util.spec_from_file_location('figures', Path(__file__).resolve().parents[1] / 'code/render_thesis_figures.py')
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


class RendererTests(unittest.TestCase):
    def test_new_run_accuracy_and_external_error_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / 'experiments'
            for group in ('01_phase1_mangoleafbd', '03_phase2_combined', '04_phase3_vn_mangoleaf'):
                for seed in (42, 123, 2024, 3333):
                    run = experiments / f'{group}_320_seed_{seed}'
                    run.mkdir(parents=True)
                    (run / 'training_results.json').write_text(json.dumps({'test_metrics': {'accuracy': .92}}))
            vn = experiments / '04_phase3_vn_mangoleaf_320_seed_42'
            for name in ('Healthy', 'Anthracnose', 'Gall_Midge', 'Powdery_Mildew', 'Sooty_Mold', 'Dieback', 'Red_Rust'):
                Image.new('RGB', (20, 10), 'white').save(vn / f'gradcam_{name}.png')
            for errors in (0, 1, 2):
                with self.subTest(errors=errors):
                    panels = root / f'errors_{errors}'
                    panels.mkdir()
                    (panels / 'summary.json').write_text(json.dumps({'error_analysis': {'error_count': errors}}))
                    for i in range(errors):
                        Image.new('RGB', (20, 10), 'white').save(panels / f'error_{i + 1:03d}_example_cam.png')
                    output = root / f'figures_{errors}'
                    figures.render(experiments, output, panels)
                    manifest = json.loads((output / 'derived_figures_manifest.json').read_text())
                    self.assertEqual(len(manifest['fig_4_08_errors_vn_seed42.png']), max(errors, 1))
                    self.assertTrue((output / 'fig_4_06_accuracy_by_seed.png').is_file())


if __name__ == '__main__':
    unittest.main()
