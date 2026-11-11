"""Discover native development dependencies without importing torch or using a GPU."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import tempfile


def needs_cudnn(source):
    """Reference constructor/call inspection, including common import aliases."""
    try:
        tree = ast.parse(source)
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = (node.module or '') + '.' + alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                root = name.split('.')[0]
                name = aliases.get(root, root) + name[len(root):]
                leaf = name.split('.')[-1]
                if name.startswith('torch.') and (leaf.startswith(('Conv', 'conv')) and any(c.isdigit() for c in leaf)):
                    return True
    except SyntaxError:
        pass
    return False


def _linkable(library, env):
    compiler = env.get('CXX') or shutil.which('c++', path=env.get('PATH'))
    if not compiler:
        return False, 'host C++ linker unavailable'
    try:
        with tempfile.TemporaryDirectory(prefix='cuda-agent-cudnn-') as temp:
            result = subprocess.run([compiler, '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed',
                                     str(library), '-o', str(Path(temp) / 'probe.so')], input='int probe(){return 0;}\n',
                                    text=True, capture_output=True, timeout=15, env=env)
            return result.returncode == 0, result.stderr[-2000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def discover_cudnn(env=None, roots=None):
    env = dict(os.environ if env is None else env)
    prefixes = [Path(env[k]) for k in ('CUDNN_HOME', 'CUDNN_ROOT', 'CUDA_HOME', 'CUDA_PATH') if env.get(k)]
    includes = [Path(env[k]) for k in ('CUDNN_INCLUDE_DIR', 'CUDNN_INCLUDE_PATH') if env.get(k)]
    libraries = [Path(env[k]) for k in ('CUDNN_LIBRARY_DIR', 'CUDNN_LIBRARY_PATH') if env.get(k)]
    includes += [Path(p) for key in ('CPATH', 'CPLUS_INCLUDE_PATH') for p in env.get(key, '').split(os.pathsep) if p]
    libraries += [Path(p) for key in ('LIBRARY_PATH', 'LD_LIBRARY_PATH') for p in env.get(key, '').split(os.pathsep) if p]
    if roots is None:
        prefixes += [Path('/usr'), Path('/usr/local'), Path('/usr/local/cuda')]
        for path in site.getsitepackages() + [site.getusersitepackages()]:
            prefixes.extend([Path(path) / 'nvidia/cudnn', Path(path) / 'torch'])
        spec = importlib.util.find_spec('torch')
        if spec and spec.origin:
            prefixes.append(Path(spec.origin).parent)
    else:
        prefixes += [Path(p) for p in roots]
    includes = [p.parent if p.is_file() else p for p in includes]
    includes += [p / 'include' for p in prefixes]
    libraries += [p / suffix for p in prefixes for suffix in ('lib', 'lib64', 'lib/x86_64-linux-gnu', 'lib/aarch64-linux-gnu')]
    headers = list(dict.fromkeys(p / 'cudnn.h' for p in includes if (p / 'cudnn.h').is_file()))
    if env.get('CUDNN_LIBRARY'):
        libraries.insert(0, Path(env['CUDNN_LIBRARY']))
    libs = list(dict.fromkeys(p.resolve() for folder in libraries for p in ([folder] if folder.is_file() else sorted(folder.glob('libcudnn.so*'))) if p.is_file()))
    reason = 'cudnn development dependency missing: ' + ('header' if not headers else 'linkable library')
    for header in headers:
        for library in libs:
            ok, detail = _linkable(library, env)
            if ok:
                files = [header] + sorted(header.parent.glob('cudnn*.h')) + [library]
                identity = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in dict.fromkeys(files)]
                return {'status': 'passed', 'include_dirs': [str(header.parent)], 'library_dirs': [str(library.parent)],
                        'libraries': [str(library)], 'fingerprint': hashlib.sha256(json.dumps(identity).encode()).hexdigest()}
            reason = 'cudnn development dependency is not linkable: ' + detail
    return {'status': 'failed', 'failure_type': 'environment', 'headers': [str(p) for p in headers],
            'libraries_found': [str(p) for p in libs], 'reason': reason}


def configure_dependency(dependency, env):
    if dependency.get('status') != 'passed':
        return
    for key, values in (('CPATH', dependency['include_dirs']), ('LIBRARY_PATH', dependency['library_dirs']),
                        ('LD_LIBRARY_PATH', dependency['library_dirs'])):
        env[key] = os.pathsep.join(dict.fromkeys(values + [p for p in env.get(key, '').split(os.pathsep) if p]))
    env['CUDA_AGENT_DEPENDENCY_FINGERPRINT'] = dependency['fingerprint']
