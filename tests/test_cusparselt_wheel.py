"""Retag only the confirmed ARM64 wheel; never relabel x86 native code."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

from wheel.wheelfile import WheelFile

spec = importlib.util.spec_from_file_location('fix_wheel', Path(__file__).resolve().parents[1] / 'environment/fix_cusparselt_wheel.py')
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


class WheelTests(unittest.TestCase):
    def make(self, directory, machine):
        path = Path(directory) / 'nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl'
        native = b'\x7fELF\x02\x01' + b'\x00' * 12 + machine.to_bytes(2, 'little') + b'payload'
        with WheelFile(path, 'w') as wheel:
            wheel.writestr('nvidia_cusparselt_cu13-0.8.1.dist-info/WHEEL',
                'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n')
            wheel.writestr('nvidia/cusparselt/lib/libcusparseLt.so.0', native)
        return path, native

    def test_retag_preserves_native_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path, native = self.make(directory, 183)
            fixed = fix.corrected_wheel(path)
            with zipfile.ZipFile(fixed) as wheel:
                self.assertEqual(wheel.read('nvidia/cusparselt/lib/libcusparseLt.so.0'), native)
                self.assertIn(b'Tag: py3-none-manylinux2014_aarch64', wheel.read('nvidia_cusparselt_cu13-0.8.1.dist-info/WHEEL'))
            self.assertIn('-1-', fixed.name)

    def test_x86_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make(directory, 62)
            with self.assertRaisesRegex(RuntimeError, 'non-AArch64'):
                fix.corrected_wheel(path)


if __name__ == '__main__':
    unittest.main()
