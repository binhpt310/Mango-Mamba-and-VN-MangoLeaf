"""Correct the known SBSA packaging tag in NVIDIA's ARM64 cuSPARSELt 0.8.1 wheel."""

from importlib.metadata import distribution
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


def corrected_wheel(path):
    with zipfile.ZipFile(path) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith('.dist-info/WHEEL'))
        tags = [line[5:] for line in archive.read(metadata).decode().splitlines() if line.startswith('Tag: ')]
        if tags == ['py3-none-manylinux2014_aarch64']:
            return path
        if tags != ['py3-none-manylinux2014_sbsa']:
            raise RuntimeError(f'Unexpected cuSPARSELt wheel tags: {tags}')
        libraries = [n for n in archive.namelist() if '.so' in Path(n).name]
        if not libraries:
            raise RuntimeError('The cuSPARSELt wheel contains no native library')
        for name in libraries:
            with archive.open(name) as handle:
                header = handle.read(20)
            if header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20], 'little') != 183:
                raise RuntimeError(f'Refusing to retag a non-AArch64 ELF library: {name}')
    # wheel tags requires the source filename to agree with its existing WHEEL metadata.
    source = path.with_name(path.name.replace('manylinux2014_aarch64', 'manylinux2014_sbsa'))
    if source != path:
        path.rename(source)
    name = subprocess.check_output([sys.executable, '-m', 'wheel', 'tags',
        '--platform-tag=manylinux2014_aarch64', '--build=1', str(source)], text=True).strip()
    return source.with_name(name)


def main():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        return
    installed = distribution('nvidia-cusparselt-cu13')
    if 'Tag: py3-none-manylinux2014_sbsa' not in installed.read_text('WHEEL'):
        return
    if installed.version != '0.8.1':
        raise RuntimeError('The SBSA tag correction is limited to cuSPARSELt 0.8.1')
    with tempfile.TemporaryDirectory(prefix='cusparselt-wheel-') as directory:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
            '--index-url', 'https://pypi.org/simple', '--dest', directory, 'nvidia-cusparselt-cu13==0.8.1'], check=True)
        path, = Path(directory).glob('*.whl')
        fixed = corrected_wheel(path)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(fixed)], check=True)
    print('Corrected cuSPARSELt 0.8.1 ARM64 wheel metadata; native library contents are unchanged.')


if __name__ == '__main__':
    main()
