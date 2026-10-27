#!/usr/bin/env python3
"""Simplified compile script: incrementally compile root and kernels CUDA/C++ sources."""

import os
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

# 必须在导入 torch CUDA extension 之前固定工具链，
# 否则 PyTorch 可能会缓存 PATH 中的其他 CUDA 版本。
CUDA_HOME = "/public/app/cuda/12.6"
HOST_CC = "/usr/bin/gcc"
HOST_CXX = "/usr/bin/g++"

os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDACXX"] = f"{CUDA_HOME}/bin/nvcc"
os.environ["CC"] = HOST_CC
os.environ["CXX"] = HOST_CXX
os.environ["CUDAHOSTCXX"] = HOST_CXX
os.environ["PATH"] = f"{CUDA_HOME}/bin{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"{CUDA_HOME}/lib64{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch.utils.cpp_extension as cpp_ext


def find_sources() -> list[str]:
    root = Path('.')
    kernels_dir = Path('kernels')

    root_sources = [str(p) for p in root.glob('*.cu')] + [str(p) for p in root.glob('*.cpp')]
    kernel_sources = []
    if kernels_dir.is_dir():
        kernel_sources = [str(p) for p in kernels_dir.glob('*.cu')] + [str(p) for p in kernels_dir.glob('*.cpp')]

    return sorted(set(root_sources + kernel_sources))


def build_fingerprint(sources, cuda_flags, libraries, arch_flags):
    """Python model changes do not affect the native extension cache key."""
    digest = hashlib.sha256()
    inputs = sorted(set(Path(p) for p in sources) | {
        p for p in Path('.').rglob('*')
        if p.suffix in {'.h', '.hpp', '.cuh'} and 'build' not in p.parts
    } | {Path(__file__)})
    for path in inputs:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    settings = {
        'cuda_flags': cuda_flags, 'libraries': libraries, 'arch_flags': arch_flags,
        'torch': cpp_ext.torch.__version__, 'python': sys.version,
        'env': {key: os.environ.get(key) for key in (
            'TORCH_CUDA_ARCH_LIST', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CUDA_AGENT_FAST_MATH')},
        'toolchain': [(name, Path(name).stat().st_mtime_ns, Path(name).stat().st_size)
                      for name in (HOST_CXX, f'{CUDA_HOME}/bin/nvcc')],
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def compile_kernels() -> int:
    build_dir = Path('build/forced_compile')
    output_so = Path('cuda_extension.so')
    sources = find_sources()

    if not sources:
        print('Error: no source files found (*.cu, *.cpp in root or kernels/)')
        return 1

    print(f'Compiling {len(sources)} files: {", ".join(sources)}')

    build_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = build_dir / 'cache_manifest.json'
    built_so = build_dir / 'cuda_extension.so'

    # Correctness-first builds do not enable approximate math by default.
    cuda_flags = ['-O3', f'-ccbin={HOST_CXX}']
    if os.environ.get('CUDA_AGENT_FAST_MATH') == '1':
        cuda_flags.append('--use_fast_math')
    source_text = '\n'.join(Path(source).read_text(encoding='utf-8') for source in sources)
    libraries = []
    source_text = re.sub(r'/\*.*?\*/|//[^\n]*', '', source_text, flags=re.S)
    if re.search(r'#\s*include\s*[<"]cublas|\bcublas[A-Z]\w*\s*\(', source_text):
        libraries.append('-lcublas')
    if re.search(r'#\s*include\s*[<"]cudnn|\bcudnn[A-Z]\w*\s*\(', source_text):
        libraries.append('-lcudnn')

    try:
        arch_flags = cpp_ext._get_cuda_arch_flags()
        fingerprint = build_fingerprint(sources, cuda_flags, libraries, arch_flags)
        try:
            cached = json.loads(cache_manifest.read_text())
        except (OSError, ValueError):
            cached = {}
        if (cached.get('fingerprint') == fingerprint and built_so.exists()
                and cached.get('binary_sha256') == hashlib.sha256(built_so.read_bytes()).hexdigest()):
            shutil.copy2(built_so, output_so)
            print('Compile cache hit: native sources unchanged; reused cuda_extension.so')
            return 0
        # Never expose a previous binary after a failed build.
        output_so.unlink(missing_ok=True)
        cache_manifest.unlink(missing_ok=True)
        print('Compile cache miss: incremental Ninja build')
        cpp_ext.load(
            name='cuda_extension',
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=['-O3', '-std=c++20'],
            extra_cuda_cflags=cuda_flags + arch_flags,
            extra_ldflags=libraries,
        )
    except Exception as exc:
        output_so.unlink(missing_ok=True)
        print('Compilation failed.')
        print(str(exc))
        return 1

    if built_so.exists():
        shutil.copy2(built_so, output_so)
        cache_manifest.write_text(json.dumps({'fingerprint': fingerprint,
            'binary_sha256': hashlib.sha256(built_so.read_bytes()).hexdigest()}))
        print(f'Compile success: {output_so}')
        return 0

    print('Compilation finished but cuda_extension.so was not generated.')
    return 1


def main() -> int:
    return compile_kernels()


if __name__ == '__main__':
    sys.exit(main())
