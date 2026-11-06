"""Conservative source checks: definite interface errors block, heuristics warn."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
import re

from agent_workdir.utils.operator_contract import RETURN_CONSTRUCTORS, contract_call_errors, source_binding_checks


def native_structure_errors(name: str, text: str) -> list[str]:
    """Check the ownership boundary without trying to parse arbitrary C++."""
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    errors = []
    if name.endswith('.cu'):
        if re.search(r'#\s*include\s*[<"](?:torch/|ATen/|c10/|pybind11/|[^>"\n]*binding_registry)', text) or re.search(
            r'\b(?:torch|at|c10|pybind11|py)::|\b(?:REGISTER_BINDING|PYBIND11_MODULE|TORCH_CHECK)\s*\(', code
        ):
            errors.append(f'{name}: CUDA source must contain only kernels and raw launchers; move tensor wrappers and pybind registration to kernels/*_binding.cpp')
    elif name.endswith('.cpp') and re.search(r'\b__(?:global|device)__\b|<<<', code):
        errors.append(f'{name}: CUDA kernels and launch syntax belong in kernels/custom_ops.cu')
    return errors


@dataclass
class SourceChecks:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def as_dict(self):
        return {'status': 'failed' if self.errors else 'passed', 'errors': self.errors, 'warnings': self.warnings, 'pending': self.pending}


def check_sources(workdir: Path, contract=None) -> SourceChecks:
    result = SourceChecks()
    cuda_sources = list((workdir / 'kernels').rglob('*.cu'))
    if len(cuda_sources) != 1 or cuda_sources[0].relative_to(workdir).as_posix() != 'kernels/custom_ops.cu':
        result.errors.append('kernels must contain exactly one CUDA source: kernels/custom_ops.cu')
    path = workdir / 'model_new.py'
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, SyntaxError) as exc:
        result.errors.append(f'model_new.py: {exc}')
        return result
    model = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelNew'), None)
    if model is None:
        result.errors.append('model_new.py must define ModelNew')
        return result
    forward = next((n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None)
    if forward is None:
        result.errors.append('ModelNew.forward is missing')
        return result
    aliases = {'cuda_extension'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names if a.name == 'cuda_extension')
    sources = list((workdir / 'kernels').glob('*.cu')) + list((workdir / 'kernels').glob('*.cpp'))
    registrations = set()
    for source in sources:
        text = source.read_text(encoding='utf-8')
        result.errors.extend(native_structure_errors(source.name, text))
        registrations.update(re.findall(r'\b\w+\s*\.\s*def\s*\(\s*"([^"]+)"', text))
        result.warnings.extend(f'{source.name}: {warning}' for warning in coverage_warnings(text))
        if '__half' in text and 'cuda_fp16.h' not in text:
            result.warnings.append(f'{source.name}: __half used without a direct cuda_fp16.h include')
        if 'tanh' in text and 'torch.tan(' in (workdir / 'model.py').read_text(encoding='utf-8'):
            result.warnings.append(f'{source.name}: reference uses tan, confirm tanh is not substituting for it')
        if re.search(r'\btorch::(?:nn::functional::|(?:matmul|conv\w*|relu|softmax|sum|mean)\s*\()', text):
            result.errors.append(f'{source.name}: prohibited torch computation in generated C++')
    allowed = {'empty', 'empty_like', 'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like', 'tensor', 'arange'}
    for method in (n for n in model.body if isinstance(n, ast.FunctionDef) and n.name != '__init__'):
        for call in (n for n in ast.walk(method) if isinstance(n, ast.Call)):
            func = call.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in aliases:
                if func.attr not in registrations:
                    result.errors.append(f'{method.name}: cuda_extension.{func.attr} has no visible pybind registration')
            name = ast.unparse(func)
            if name.startswith('torch.ops.'):
                result.errors.append(f'{method.name}: {name} is incompatible with the pybind cuda_extension module')
            if name.startswith(('F.', 'torch.nn.functional.')):
                result.errors.append(f'{method.name}: prohibited functional computation {name}')
            if name.startswith('torch.') and name.split('.')[1] not in allowed and not name.startswith('torch.ops.') and name not in RETURN_CONSTRUCTORS:
                result.errors.append(f'{method.name}: prohibited torch computation {name}')
    if contract is not None:
        result.errors.extend(contract_call_errors(path.read_text(encoding='utf-8'), contract))
        binding = source_binding_checks('\n'.join(p.read_text() for p in sources if p.suffix == '.cpp'), contract)
        result.errors.extend(binding['errors'])
        result.pending.extend(binding['pending'])
    # Source heuristics cannot prove correct math or memory safety. Actual imports,
    # strict state loading and normal-forward tests remain authoritative.
    return result


def _body(text, start):
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return text[start:]


def coverage_warnings(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    warnings = []
    kernels = {m.group(1): _body(text, m.end()) for m in re.finditer(
        r'__global__\s+(?:void|[\w:<>]+)\s+(\w+)\s*\([^{}]*\)\s*\{', text)}
    launches = list(re.finditer(r'\b(\w+)\s*<<<([^>]+)>>>\s*\(([^;]*)\)', text))
    for launch in launches:
        name = launch.group(1)
        body = kernels.get(name, '')
        # Restrict grid configuration to the containing launcher body.
        start = text.rfind('{', 0, launch.start())
        local = text[start + 1:launch.start()]
        config = launch.group(2).split(',')
        if len(config) < 2:
            continue
        def constant(expression):
            expression = expression.strip()
            if expression.isdigit():
                return int(expression)
            values = re.findall(r'\b' + re.escape(expression) + r'\s*=\s*(\d+)\b', local)
            return int(values[-1]) if values else None
        grid, block = constant(config[0]), constant(config[1])
        cap = re.search(r'\b' + re.escape(config[0].strip()) + r'\s*(?:>|=\s*(?:std::)?min\s*\([^,]+,)\s*(\d+)', local)
        if cap:
            grid = int(cap.group(1))
        # Recognize the increment in this kernel's loop, not any unrelated loop.
        products = r'(?:blockDim\.x\s*\*\s*gridDim\.x|gridDim\.x\s*\*\s*blockDim\.x)'
        stride_names = re.findall(r'\b(\w+)\s*=\s*' + products, body)
        increments = products + (r'|\b(?:' + '|'.join(map(re.escape, stride_names)) + r')\b' if stride_names else '')
        stride = bool(re.search(r'\bfor\s*\([^;]*;[^;]*;[^)]*\+=\s*(?:' + increments + r')\s*\)', body))
        if not stride:
            for loop in re.finditer(r'\bwhile\s*\(\s*(\w+)\s*<[^)]*\)\s*\{', body):
                loop_body = _body(body, loop.end())
                if re.search(r'\b' + re.escape(loop.group(1)) + r'\s*\+=\s*(?:' + increments + r')\s*;', loop_body):
                    stride = True
        if stride:
            continue
        # A numeric extent is evidence only when the bounded kernel parameter
        # can be mapped to its actual launcher argument.
        signature = re.search(r'__global__\s+\w+\s+' + re.escape(name) + r'\s*\(([^{}]*)\)\s*\{', text)
        numel = None
        if signature:
            parameters = signature.group(1).split(',')
            arguments = launch.group(3).split(',')
            for parameter, argument in zip(parameters, arguments):
                identifiers = re.findall(r'\b\w+\b', parameter)
                bound = identifiers[-1] if identifiers else ''
                if bound in {'n', 'count', 'numel'} and re.search(r'<\s*' + bound + r'\b', body):
                    numel = constant(argument)
                    break
        if grid is not None and block is not None and numel is not None and numel > grid * block:
            warnings.append(f'{name}: numel={numel} > grid={grid} × block={block}; no recognized grid-stride loop')
        elif cap or grid == 65535:
            warnings.append(f'{name}: possible incomplete coverage: grid={grid}, block={block}; no recognized grid-stride loop')
    if not launches and re.search(r'grid\w*\s*>\s*65535', text):
        warnings.append('possible incomplete coverage: grid capped at 65535; launcher mapping unavailable')
    return warnings
