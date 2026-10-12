"""Trusted reference tracing and reproducible correctness diagnostics.

Torch is imported lazily so reference instrumentation can be tested without CUDA.
"""

import ast
import inspect
import json
import math
import random
import textwrap
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def seed_all(seed):
    import torch
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transform_tree(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(transform_tree(v, fn) for v in value)
    if isinstance(value, list):
        return [transform_tree(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: transform_tree(v, fn) for k, v in value.items()}
    return value


def _target_value(target):
    if isinstance(target, ast.Name):
        return ast.Name(id=target.id, ctx=ast.Load())
    if isinstance(target, (ast.Tuple, ast.List)):
        return type(target)(elts=[_target_value(t) for t in target.elts], ctx=ast.Load())
    raise ValueError("Reference trace only supports local variable assignments")


def trace_function(source):
    """Return an instrumented generator AST and stable statement checkpoints.

    Unsupported control flow is rejected instead of silently changing semantics.
    """
    tree = ast.parse(textwrap.dedent(source))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        raise ValueError("Reference forward function was not found")
    if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)) for n in ast.walk(fn)):
        raise ValueError("Generator/async reference forwards cannot be traced")
    stages, body = [], []
    for statement in fn.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            body.append(statement)
            continue
        if isinstance(statement, ast.Assign):
            body.append(statement)
            value = _target_value(statement.targets[-1])
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            body.append(statement)
            value = _target_value(statement.target)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            value = statement.value
        else:
            raise ValueError(f"Unsupported reference statement: {type(statement).__name__}")
        checkpoint = f"step_{len(stages) + 1:03d}"
        stages.append({"id": checkpoint, "source": ast.unparse(statement), "line": statement.lineno})
        body.append(ast.Expr(value=ast.Yield(value=ast.Tuple(
            elts=[ast.Constant(value=checkpoint), value], ctx=ast.Load()))))
        if isinstance(statement, ast.Return):
            body.append(ast.Return(value=None))
            break
    if not stages or not isinstance(statement, ast.Return):
        raise ValueError("Reference trace requires an explicit final return")
    # Do not re-evaluate defaults or annotations while instrumenting the function.
    fn.args.defaults = [ast.Constant(value=None) for _ in fn.args.defaults]
    fn.args.kw_defaults = [ast.Constant(value=None) if d is not None else None for d in fn.args.kw_defaults]
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    if fn.args.vararg:
        fn.args.vararg.annotation = None
    if fn.args.kwarg:
        fn.args.kwarg.annotation = None
    fn.returns = None
    fn.name = "_reference_debug_forward"
    fn.decorator_list = []
    fn.body = body
    return ast.fix_missing_locations(tree), stages


def reference_trace(model):
    forward = type(model).forward
    if forward.__closure__:
        raise ValueError("Reference forward closures cannot be safely instrumented")
    tree, stages = trace_function(inspect.getsource(forward))
    namespace = dict(forward.__globals__)
    exec(compile(tree, "<reference-checkpoints>", "exec"), namespace)
    traced = namespace["_reference_debug_forward"]
    traced.__defaults__ = forward.__defaults__
    traced.__kwdefaults__ = forward.__kwdefaults__
    return traced.__get__(model), stages


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device), "stride": list(value.stride()), "numel": value.numel()}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [describe(v) for v in value]}
    if isinstance(value, dict):
        return {"type": "dict", "items": {str(k): describe(v) for k, v in value.items()}}
    return {"type": type(value).__name__, "value": repr(value)[:200]}


def api_signatures(model):
    """Resolve APIs in the installed runtime rather than trusting variable names."""
    forward = type(model).forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(forward)))
    signatures = {}
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        expression = ast.unparse(call.func)
        try:
            def resolve(node):
                if isinstance(node, ast.Name):
                    return model if node.id == "self" else forward.__globals__[node.id]
                if isinstance(node, ast.Attribute):
                    return getattr(resolve(node.value), node.attr)
                raise ValueError("dynamic callable")
            obj = resolve(call.func)
            target = obj.forward if hasattr(obj, "forward") else obj
            try:
                signature = str(inspect.signature(target))
            except (TypeError, ValueError):
                signature = "builtin signature unavailable; preserve exact reference arguments"
            signatures[expression] = {"signature": signature, "documentation": (inspect.getdoc(target) or inspect.getdoc(obj) or "")[:2400]}
        except (KeyError, AttributeError, TypeError, ValueError):
            signatures[expression] = "unresolved; retain the reference call's exact arguments"
    return signatures


class ComparisonFailure(AssertionError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(json.dumps(detail, ensure_ascii=False, allow_nan=False))


def _finite_json(value):
    return value if math.isfinite(value) else str(value)


def tensor_error(actual, expected):
    """Calculate failure metrics in chunks; do not allocate full float64 copies."""
    import torch
    detail = {"actual": describe(actual), "expected": describe(expected)}
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        detail["kind"] = "tensor_structure"
        return detail
    detail.update(kind="tensor_values", mismatched=0, numel=expected.numel(),
                  actual_nan=0, expected_nan=0, actual_inf=0, expected_inf=0)
    a, b = actual.reshape(-1), expected.reshape(-1)
    max_abs, max_rel = 0.0, 0.0
    samples = []
    for start in range(0, a.numel(), 1_048_576):
        av, bv = a[start:start + 1_048_576], b[start:start + 1_048_576]
        close = torch.isclose(av, bv, atol=1e-2, rtol=1e-2, equal_nan=False)
        detail["mismatched"] += int((~close).sum().item())
        for label, chunk in (("actual", av), ("expected", bv)):
            detail[label + "_nan"] += int(torch.isnan(chunk).sum().item())
            detail[label + "_inf"] += int(torch.isinf(chunk).sum().item())
        if (~close).any().item():
            bad = (~close).nonzero().flatten()
            for offset in bad[:max(0, 5 - len(samples))].tolist():
                flat = start + offset
                index = []
                for dimension in reversed(actual.shape):
                    index.append(flat % dimension)
                    flat //= dimension
                samples.append({"index": list(reversed(index)),
                                "actual": repr(av[offset].item()), "expected": repr(bv[offset].item())})
            # Complex values retain their imaginary parts; integer differences cannot overflow.
            dtype = torch.complex128 if av.is_complex() else torch.float64
            delta = (av.to(dtype) - bv.to(dtype)).abs()
            finite = torch.isfinite(delta)
            if finite.any().item():
                max_abs = max(max_abs, float(delta[finite].max().item()))
                relative = delta[finite] / bv.to(dtype).abs()[finite].clamp_min(1e-30)
                max_rel = max(max_rel, float(relative.max().item()))
            if ((~close) & (~finite)).any().item():
                max_abs = max_rel = math.inf
    detail.update(mismatch_ratio=detail["mismatched"] / max(1, detail["numel"]),
                  max_absolute_error=_finite_json(max_abs), max_relative_error=_finite_json(max_rel), samples=samples)
    return detail


def check_equal(actual, expected, path="output"):
    import torch
    if type(actual) is not type(expected):
        raise ComparisonFailure({"kind": "type", "path": path, "actual": describe(actual), "expected": describe(expected)})
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise ComparisonFailure({"kind": "length", "path": path, "actual": len(actual), "expected": len(expected)})
        for i, (x, y) in enumerate(zip(actual, expected)):
            check_equal(x, y, f"{path}[{i}]")
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ComparisonFailure({"kind": "keys", "path": path, "actual": list(map(str, actual)), "expected": list(map(str, expected))})
        for key in expected:
            check_equal(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(actual, torch.Tensor):
        try:
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        except AssertionError as exc:
            detail = tensor_error(actual, expected)
            detail.update(path=path, message=str(exc))
            raise ComparisonFailure(detail) from exc
    elif isinstance(actual, (str, float, int)):
        if actual != expected:
            raise ComparisonFailure({"kind": "scalar", "path": path, "actual": repr(actual), "expected": repr(expected)})
    else:
        raise ComparisonFailure({"kind": "unsupported_type", "path": path, "type": type(actual).__name__})
