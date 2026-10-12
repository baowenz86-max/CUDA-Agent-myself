"""Conservative source checks: definite interface errors block, heuristics warn."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
import re

from agent_workdir.utils.operator_contract import RETURN_CONSTRUCTORS, contract_call_errors


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

    def as_dict(self):
        return {'status': 'failed' if self.errors else 'passed', 'errors': self.errors, 'warnings': self.warnings}


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
        if re.search(r'grid\w*\s*>\s*65535', text) and not re.search(r'\b(?:for|while)\s*\(', text):
            result.warnings.append(f'{source.name}: grid is capped at 65535 without a visible loop; check complete output coverage')
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
    # Source heuristics cannot prove correct math or memory safety. Actual imports,
    # strict state loading and normal-forward tests remain authoritative.
    return result
