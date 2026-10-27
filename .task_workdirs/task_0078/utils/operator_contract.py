"""Small, JSON-only operator contracts shared by generation and preflight.

Contracts describe task-specific interfaces; they do not establish correctness.
Unknown dynamic calls and undocumented native signatures remain runtime checks.
"""

import ast
import re


RETURN_CONSTRUCTORS = {'torch.return_types.max', 'torch.return_types.min',
                       'torch.return_types.median', 'torch.return_types.sort',
                       'torch.return_types.topk', 'torch.return_types.kthvalue'}


def validate_contract(contract, reference=None):
    if not isinstance(contract, dict) or not isinstance(contract.get('operators'), list):
        raise ValueError('operator_contract must contain an operators list')
    names = set()
    for op in contract['operators']:
        if not isinstance(op, dict) or not isinstance(op.get('name'), str) or not op['name'].isidentifier():
            raise ValueError('Each operator needs an identifier name')
        if op['name'] in names:
            raise ValueError('Duplicate operator: ' + op['name'])
        names.add(op['name'])
        if not isinstance(op.get('parameters'), list):
            raise ValueError(op['name'] + ': parameters must be a list')
        parameters = set()
        default_seen = False
        for parameter in op['parameters']:
            if not isinstance(parameter, dict) or not isinstance(parameter.get('name'), str) or not parameter['name'].isidentifier():
                raise ValueError(op['name'] + ': parameter needs an identifier name')
            if parameter['name'] in parameters:
                raise ValueError(op['name'] + ': duplicate parameter name')
            parameters.add(parameter['name'])
            if parameter.get('kind') not in {'tensor', 'int', 'float', 'bool'}:
                raise ValueError(op['name'] + ': parameter kind must be tensor/int/float/bool')
            if not isinstance(parameter.get('optional', False), bool) or not isinstance(parameter.get('nullable', False), bool):
                raise ValueError(op['name'] + ': optional and nullable must be booleans')
            if parameter.get('optional'):
                if 'default' not in parameter:
                    raise ValueError(op['name'] + ': optional parameters need an explicit default')
                default = parameter['default']
                if default is None and not parameter.get('nullable', False):
                    raise ValueError(op['name'] + ': None default requires nullable=true')
                if default is not None:
                    allowed = {'int': (int,), 'float': (float, int), 'bool': (bool,), 'tensor': ()}
                    if type(default) not in allowed[parameter['kind']]:
                        raise ValueError(op['name'] + ': default must match parameter kind')
                default_seen = True
            elif default_seen:
                raise ValueError(op['name'] + ': required parameters must precede optional parameters')
            if parameter['kind'] == 'tensor':
                for key in ('dtype', 'device', 'shape', 'layout'):
                    if not isinstance(parameter.get(key), str) or not parameter[key].strip():
                        raise ValueError(op['name'] + ': tensor parameter needs ' + key)
        result = op.get('returns')
        if not isinstance(result, dict) or not isinstance(result.get('type'), str) or not result['type'].strip():
            raise ValueError(op['name'] + ': returns must describe the exact container/type')
        if result['type'] == 'tensor':
            for key in ('dtype', 'device', 'shape'):
                if not isinstance(result.get(key), str) or not result[key].strip():
                    raise ValueError(op['name'] + ': tensor return needs ' + key)
        if not isinstance(op.get('semantics'), list) or not op['semantics'] or not all(isinstance(x, str) and x.strip() for x in op['semantics']):
            raise ValueError(op['name'] + ': semantics needs nonempty evidence-backed statements')
    steps = contract.get('reference_steps')
    if not isinstance(steps, list):
        raise ValueError('operator_contract must contain reference_steps')
    if reference and reference.get('trace_available'):
        expected = [(x['id'], x['source']) for x in reference['checkpoints']]
        observed = [(x.get('checkpoint'), x.get('source')) for x in steps if isinstance(x, dict)]
        if observed != expected:
            raise ValueError('reference_steps must preserve every reference checkpoint and exact source')
    for step in steps:
        if (not isinstance(step, dict) or not isinstance(step.get('semantics'), list) or not step['semantics']
                or not all(isinstance(x, str) and x.strip() for x in step['semantics'])):
            raise ValueError('Each reference step needs semantic notes: actual bias, axes, broadcasting, order and output type')
    return contract


def extension_calls(source):
    tree = ast.parse(source)
    aliases = {'cuda_extension'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names if a.name == 'cuda_extension')
    model = next((x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ModelNew'), None)
    if model is None:
        raise ValueError('model_new.py must define ModelNew')
    return [(method.name, call) for method in model.body if isinstance(method, ast.FunctionDef)
            and method.name != '__init__' for call in ast.walk(method) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id in aliases]


def contract_call_errors(source, contract):
    operators = {x['name']: x for x in contract['operators']}
    errors = []
    for method, call in extension_calls(source):
        name = call.func.attr
        if name not in operators:
            errors.append(f'{method}: cuda_extension.{name} is missing from operator_contract')
            continue
        # *args/**kwargs cannot be proven statically.
        if any(isinstance(x, ast.Starred) for x in call.args) or any(x.arg is None for x in call.keywords):
            continue
        params = operators[name]['parameters']
        names = [x['name'] for x in params]
        keywords = [x.arg for x in call.keywords]
        supplied = set(names[:len(call.args)]) | set(keywords)
        required = {x['name'] for x in params if not x.get('optional', False)}
        if (len(call.args) > len(params) or set(keywords) - set(names)
                or set(keywords) & set(names[:len(call.args)]) or len(keywords) != len(set(keywords))
                or required - supplied):
            errors.append(f'{method}: cuda_extension.{name} call {ast.unparse(call)} does not match parameters {names}')
            continue
        for parameter, value in [(params[i], x) for i, x in enumerate(call.args)] + [
                (params[names.index(x.arg)], x.value) for x in call.keywords]:
            if not isinstance(value, ast.Constant):
                continue
            allowed = {'int': (int,), 'float': (float, int), 'bool': (bool,)}
            if value.value is None and parameter.get('nullable', False):
                continue
            kind = parameter['kind']
            if (kind == 'tensor' or type(value.value) not in allowed[kind]):
                errors.append(f'{method}: {name}.{parameter["name"]} literal {value.value!r} violates {kind}/nullable contract')
    return errors


def _split_parameters(text):
    """Split native doc signatures while retaining nested defaults/types."""
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [x for x in parts if x]


def binding_contract_errors(extension, contract):
    """Check explicit pybind doc signatures only; never guess arbitrary C++."""
    errors = []
    for op in contract['operators']:
        fn = getattr(extension, op['name'], None)
        if not callable(fn):
            errors.append(op['name'] + ': compiled export is missing')
            continue
        signatures = []
        for line in (fn.__doc__ or '').splitlines():
            match = re.match(r'\s*(?:\d+\.\s*)?' + re.escape(op['name']) + r'\((.*)\)\s*->', line)
            if match:
                signatures.append(_split_parameters(match.group(1)))
        if not signatures:
            continue
        def compatible(parameters):
            if len(parameters) != len(op['parameters']):
                return False
            for native, parameter in zip(parameters, op['parameters']):
                if ':' not in native:
                    return False
                name, annotation = native.split(':', 1)
                name = name.strip()
                if not re.fullmatch(r'arg\d+', name) and name != parameter['name']:
                    return False
                if ('=' in annotation) != parameter.get('optional', False):
                    return False
                if '=' in annotation:
                    try:
                        if ast.literal_eval(annotation.split('=', 1)[1].strip()) != parameter['default']:
                            return False
                    except (ValueError, SyntaxError):
                        pass  # undocumented/complex defaults need a real invocation
                kind = parameter['kind']
                annotation = annotation.split('=')[0]
                known = ('Tensor' in annotation or any(x in annotation for x in ('SupportsInt', 'SupportsIndex', 'int', 'float', 'bool', 'SupportsFloat')))
                if not known:
                    continue
                checks = {'tensor': 'Tensor' in annotation,
                          'int': any(x in annotation for x in ('SupportsInt', 'SupportsIndex', 'int')),
                          'float': any(x in annotation for x in ('SupportsFloat', 'float', 'double')),
                          'bool': 'bool' in annotation}
                if kind != 'tensor' and re.search(r'\b(?:tuple|list|Tuple|List|Sequence)\b', annotation):
                    return False
                if not checks[kind]:
                    return False
                if parameter.get('nullable') and not any(x in annotation for x in ('None', 'Optional')):
                    return False
            return True
        if not any(compatible(x) for x in signatures):
            errors.append(op['name'] + ': compiled signatures differ from operator_contract: ' + repr(signatures))
    return errors
