"""Conservative correspondence checks, not a proof of native mathematical correctness."""
import ast
import copy
import re
import textwrap


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _expression(statement):
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return)):
        return statement.value
    raise ValueError('unsupported computation: ' + type(statement).__name__)


def _expand(node, methods, active=()):
    node = copy.deepcopy(node)
    class Expand(ast.NodeTransformer):
        def visit_Call(self, call):
            call = self.generic_visit(call)
            fn = call.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'self' and fn.attr in methods:
                if fn.attr in active:
                    raise ValueError('recursive helper')
                method = methods[fn.attr]
                body = [s for s in method.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                # Only pure single-return helpers can be safely substituted.
                if len(body) != 1 or not isinstance(body[0], ast.Return) or method.args.vararg or method.args.kwarg:
                    raise ValueError('helper cannot be safely expanded')
                params = [a.arg for a in method.args.args][1:]
                bindings = dict(zip(params, call.args))
                bindings.update({k.arg: k.value for k in call.keywords if k.arg})
                if len(bindings) != len(params):
                    raise ValueError('helper argument mapping unavailable')
                class Substitute(ast.NodeTransformer):
                    def visit_Name(self, n):
                        return copy.deepcopy(bindings.get(n.id, n))
                value = Substitute().visit(copy.deepcopy(body[0].value))
                return _expand(value, methods, active + (fn.attr,))
            return call
    return Expand().visit(node)


def _operation(node, modules=None):
    if isinstance(node, ast.BinOp):
        return {ast.Div: 'div', ast.Mult: 'mul', ast.Add: 'add', ast.Sub: 'sub'}.get(type(node.op))
    if not isinstance(node, ast.Call):
        return None
    name = ast.unparse(node.func).split('.')[-1]
    module = (modules or {}).get(name, {})
    if module:
        name = module.get('type', name)
    key = re.sub('[^a-z0-9]', '', name.lower())
    aliases = {'groupnorm': 'group_norm', 'batchnorm': 'batch_norm', 'convtranspose': 'conv_transpose',
               'avgpool': 'avg_pool', 'maxpool': 'max_pool', 'linear': 'linear', 'sigmoid': 'sigmoid',
               'lstm': 'lstm', 'flatten': 'flatten', 'unsqueeze': 'unsqueeze', 'squeeze': 'squeeze',
               'permute': 'permute', 'sqrt': 'sqrt', 'round': 'round', 'digamma': 'digamma',
               'relu': 'relu', 'softmax': 'softmax', 'tanh': 'tanh', 'rsqrt': 'rsqrt',
               'scalardiv': 'div', 'div': 'div', 'mul': 'mul', 'add': 'add', 'sub': 'sub'}
    # Unknown/fused names cannot prove a one-reference-step correspondence.
    if 'fused' in key:
        return None
    hits = [value for token, value in aliases.items() if key.startswith(token)]
    return hits[0] if hits else None


def native_correspondence(name, operation, source):
    """Follow visible wrapper/launcher/kernel calls and require operation evidence.

    This rejects labels, declarations and identity stubs. It deliberately does
    not certify arbitrary C++ control flow, numerical accuracy or memory safety.
    """
    source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    source = re.sub(r'"(?:\\.|[^"\\])*"', '""', source)
    functions = {}
    for match in re.finditer(r'\b(?!(?:if|for|while|switch)\b)(\w+)\s*\([^;{}]*\)\s*\{', source):
        if match.group(1) in {'if', 'for', 'while', 'switch'}:
            continue
        depth = 1
        end = match.end()
        while end < len(source) and depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        functions[match.group(1)] = source[match.end():end-1]
    visited, pending = set(), [name]
    while pending:
        fn = pending.pop()
        if fn in visited or fn not in functions:
            continue
        visited.add(fn)
        body = functions[fn]
        pending.extend(re.findall(r'\b(\w+)\s*(?:<<<[^>]+>>>\s*)?\(', body))
    bodies = '\n'.join(functions[fn] for fn in sorted(visited))
    patterns = {
        'sigmoid': r'\b(?:expf?|__expf)\s*\(',
        'group_norm': r'\b(?:rsqrtf?|sqrtf?|cudnnNormalizationForwardInference|cudnnBatchNormalizationForwardInference)\s*\(',
        'batch_norm': r'\b(?:rsqrtf?|sqrtf?|cudnnBatchNormalizationForwardInference)\s*\(',
        'permute': r'\.permute\s*\(|\[[^]]+\]\s*=\s*\w+\[[^]]+\]',
        'flatten': r'cudaMemcpy\w*\s*\(|\.view\s*\(|\.reshape\s*\(|\[[^]]+\]\s*=\s*\w+\[[^]]+\]',
        'avg_pool': r'\b(?:sum|avg|val|value)\s*(?:\+=|=)|cudnnPoolingForward',
        'max_pool': r'\b(?:fmaxf?|max|cudnnPoolingForward)\s*\(',
        'linear': r'\b(?:cublas\w*gemm|cublas\w*Gemm\w*)\s*\(|\+=.*\*',
        'conv_transpose': r'cudnnConvolutionBackwardData\s*\(',
        'lstm': r'\b(?:cudnnRNNForward\w*|tanhf?)\s*\(',
        'sqrt': r'\bsqrtf?\s*\(', 'round': r'\b(?:roundf?|nearbyintf?)\s*\(',
        'rsqrt': r'\brsqrtf?\s*\(', 'digamma': r'\b(?:logf?|digamma)\s*\(',
        'relu': r'\b(?:fmaxf?|max)\s*\(|>\s*0', 'tanh': r'\btanhf?\s*\(',
        'softmax': r'\bexpf?\s*\(', 'div': r'/', 'mul': r'\*', 'add': r'\+', 'sub': r'-',
        'unsqueeze': r'\.unsqueeze\s*\(|cudaMemcpy\w*\s*\(',
        'squeeze': r'\.squeeze\s*\(|cudaMemcpy\w*\s*\(',
    }
    supported = operation in patterns and bool(re.search(patterns[operation], bodies))
    if operation == 'sqrt' and re.search(r'\b(?:roundf?|nearbyintf?)\s*\(', bodies):
        supported = False  # visible sqrt/round fusion crosses a reference checkpoint
    return {'supported': supported, 'reachable_functions': sorted(visited),
            'reason': 'visible native operation evidence' if supported else 'native operation evidence unavailable'}


def checkpoint_mapping(source, reference, contract=None, native_source=''):
    """Return per-step structural trust; later invalid steps preserve earlier prefix.

    Native source is required for extension calls. Matching operation identities
    establish correspondence only; numerical verification remains authoritative.
    """
    stages = reference.get('checkpoints', [])
    result = []
    try:
        tree = ast.parse(textwrap.dedent(source))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelNew')
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        forward, debug = methods['forward'], methods['debug_forward']
        helpers = {k: v for k, v in methods.items() if k not in {'forward', 'debug_forward', '__init__'}}
        def safe_expand(node):
            try:
                return _expand(node, helpers)
            except ValueError as exc:
                fallback = copy.deepcopy(node)
                fallback.trust_error = str(exc)
                return fallback
        normal = [safe_expand(s) for s in forward.body
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        pending, events = [], []
        for statement in debug.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Yield):
                value = statement.value.value
                if not isinstance(value, ast.Tuple) or len(value.elts) != 2 or not isinstance(value.elts[0], ast.Constant):
                    events.append((None, ast.Constant(value=None), pending))
                    pending = []
                    continue
                events.append((value.elts[0].value, safe_expand(value.elts[1]), pending))
                pending = []
            elif isinstance(statement, ast.Return) and statement.value is None:
                continue
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            else:
                pending.append(safe_expand(statement))
        cursor = 0
        prefix = True
        contract_steps = {s.get('checkpoint'): s for s in (contract or {}).get('reference_steps', [])}
        for index, stage in enumerate(stages):
            reason = 'normal/debug computation and reference operation correspond'
            expr = None
            native_checks = []
            valid = prefix
            if index >= len(events):
                valid, reason = False, 'missing checkpoint'
            else:
                checkpoint, yielded, statements = events[index]
                expr = ast.unparse(yielded)
                ref = ast.parse(stage['source']).body[0]
                is_return = isinstance(ref, ast.Return)
                if checkpoint != stage['id']:
                    valid, reason = False, 'checkpoint ID/order differs'
                elif cursor >= len(normal):
                    valid, reason = False, 'debug has extra computation'
                elif getattr(normal[cursor], 'trust_error', None) or any(getattr(s, 'trust_error', None) for s in statements) or getattr(yielded, 'trust_error', None):
                    valid, reason = False, 'local helper cannot be safely expanded'
                elif is_return:
                    if statements or not isinstance(normal[cursor], ast.Return) or _dump(normal[cursor].value) != _dump(yielded):
                        valid, reason = False, 'debug return differs from normal forward'
                    cursor += 1
                elif len(statements) != 1 or _dump(statements[0]) != _dump(normal[cursor]):
                    valid, reason = False, 'debug computation differs from normal forward or skips steps'
                else:
                    computation = statements[0]
                    expression = _expression(computation)
                    expr = ast.unparse(expression)
                    targets = computation.targets[-1] if isinstance(computation, ast.Assign) else computation.target
                    if _dump(targets).replace('Store()', 'Load()') != _dump(yielded):
                        valid, reason = False, 'yield does not expose the computed value/container'
                    op = _operation(expression)
                    if op is None or op != _operation(_expression(ref), reference.get('modules')):
                        valid, reason = False, 'reference operation mapping unavailable or fused across steps'
                    calls = [n for n in ast.walk(expression) if isinstance(n, ast.Call)]
                    for call in calls:
                        name = ast.unparse(call.func)
                        evidence = native_correspondence(name.split('.')[-1], op, native_source)
                        native_checks.append(dict(call=name, **evidence))
                        if not name.startswith('cuda_extension.') or not evidence['supported']:
                            valid, reason = False, 'native implementation correspondence unavailable'
                    cursor += 1
                if contract and (stage['id'] not in contract_steps or contract_steps[stage['id']].get('source') != stage['source']):
                    valid, reason = False, 'contract/reference source mapping differs'
            if not prefix:
                valid, reason = False, 'earlier checkpoint correspondence is unreliable'
            prefix = prefix and valid
            result.append({'id': stage['id'], 'trustworthy': bool(valid), 'reason': reason,
                           'reference_source': stage['source'], 'reference_semantics': contract_steps.get(stage['id'], {}).get('semantics', []),
                           'candidate_expression': expr, 'native_evidence': native_checks})
        if len(events) != len(stages) and result:
            # Missing suffix does not invalidate an independently checked prefix.
            result[-1]['reason'] += '; checkpoint count differs'
    except (SyntaxError, ValueError, KeyError, StopIteration, AttributeError) as exc:
        result = [{'id': s['id'], 'trustworthy': False, 'reason': str(exc),
                   'reference_source': s['source'], 'candidate_expression': None} for s in stages]
    return result


def metadata_compatible(actual, expected):
    """Container, shape and dtype from the current input, never first-run shapes."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if actual.get('type') != expected.get('type'):
        return False
    if expected.get('type') == 'tensor':
        return actual.get('shape') == expected.get('shape') and actual.get('dtype') == expected.get('dtype')
    if 'items' in expected:
        a, b = actual.get('items'), expected['items']
        if isinstance(b, dict):
            return isinstance(a, dict) and a.keys() == b.keys() and all(metadata_compatible(a[k], b[k]) for k in b)
        return isinstance(a, list) and len(a) == len(b) and all(metadata_compatible(x, y) for x, y in zip(a, b))
    return True
