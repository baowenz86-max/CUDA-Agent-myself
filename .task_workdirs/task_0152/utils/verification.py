#!/usr/bin/env python3
"""Reproducible final-forward verification; debug results never grant a pass."""

import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F

from .diagnostics import (
    ComparisonFailure, check_equal, reference_trace,
    seed_all, transform_tree, write_json, boundary_context,
)


@contextmanager
def block_torch_functional(excludes=None):
    excludes = set() if excludes is None else excludes
    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(f'Function torch.nn.functional.{__name} is not allowed in this context.')
            setattr(F, name, wrapper)
    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def fixture_directory():
    return Path(os.environ.get('CUDA_AGENT_RUN_DIR', '.correctness')).resolve()


def make_reference_fixture(directory, seed, device):
    """Create once per run, independently of any generated model."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'reference_fixture.pt'
    if not path.exists():
        reference = importlib.import_module('model')
        seed_all(seed)
        init = reference.get_init_inputs()
        init = init if isinstance(init, (tuple, list)) else [init]
        model = reference.Model(*init).eval().to(device)
        torch.save({'seed': seed, 'init_inputs': transform_tree(init, lambda t: t.detach().cpu()),
                    'state_dict': transform_tree(model.state_dict(), lambda t: t.detach().cpu())}, path)
    fixture = torch.load(path, map_location='cpu', weights_only=False)
    if fixture['seed'] != seed:
        raise ValueError('Saved reference fixture seed differs from requested seed; use its replay seed')
    return fixture


def initialize_models(directory, seed, device):
    reference = importlib.import_module('model')
    fixture = make_reference_fixture(directory, seed, device)
    seed_all(seed)
    torch_model = reference.Model(*fixture['init_inputs']).eval().to(device)
    generated = importlib.import_module('model_new')
    seed_all(seed)
    cuda_model = generated.ModelNew(*fixture['init_inputs']).eval().to(device)
    torch_model.load_state_dict(fixture['state_dict'], strict=True)
    cuda_model.load_state_dict(fixture['state_dict'], strict=True)
    return torch_model, cuda_model, fixture


def build_inputs(seed):
    reference = importlib.import_module('model')
    seed_all(seed)
    inputs = reference.get_inputs()
    inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    # Retain untouched CPU fixtures even when the forward modifies its inputs.
    return transform_tree(inputs, lambda t: t.detach().cpu().clone())


def device_inputs(inputs, device):
    return transform_tree(inputs, lambda t: t.to(device).clone())


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def runtime_metadata(device):
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': str(device), 'gpu': torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def rng_state():
    import random
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state['numpy'] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng(state):
    import random
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def diagnose(torch_model, cuda_model, fixture, inputs, input_seed, device):
    if not callable(getattr(cuda_model, 'debug_forward', None)):
        return {'status': 'unavailable', 'reason': 'ModelNew.debug_forward is missing'}
    matched = 0
    stage = None
    try:
        ref_fn, stages = reference_trace(torch_model)
        torch_model.load_state_dict(fixture['state_dict'], strict=True)
        cuda_model.load_state_dict(fixture['state_dict'], strict=True)
        seed_all(input_seed)
        initial_rng = rng_state()
        reference = ref_fn(*device_inputs(inputs, device))
        with block_torch_functional():
            actual = iter(cuda_model.debug_forward(*device_inputs(inputs, device)))
        ref_rng = initial_rng
        actual_rng = rng_state()
        matched = 0
        previous = []
        sensitive = ('sign', 'median', 'ne(', 'rsqrt', 'softmax')
        for stage_index, stage in enumerate(stages):
            restore_rng(ref_rng)
            expected_id, expected = next(reference)
            ref_rng = rng_state()
            restore_rng(actual_rng)
            with block_torch_functional():
                checkpoint, value = next(actual)
                synchronize(device)
            actual_rng = rng_state()
            if checkpoint != expected_id:
                return {'status': 'failed', 'checkpoint': stage, 'kind': 'checkpoint_contract',
                        'actual_id': str(checkpoint), 'matched_checkpoints': matched, 'input_seed': input_seed}
            try:
                check_equal(value, expected, expected_id)
            except ComparisonFailure as exc:
                diagnostic = {'status': 'failed', 'checkpoint': stage, 'error': exc.detail,
                              'matched_checkpoints': matched, 'input_seed': input_seed}
                if any(token in stage['source'] for token in sensitive):
                    diagnostic['boundary_context'] = previous
                return diagnostic
            matched += 1
            # Record small summaries before subsequent in-place operations can
            # change the tensors. No whole-tensor clones or retained references.
            upcoming = stages[stage_index + 1:stage_index + 3]
            if any(token in item['source'] for item in upcoming for token in sensitive):
                previous.append({'checkpoint': expected_id, 'actual': boundary_context(value),
                                 'expected': boundary_context(expected)})
            previous = previous[-2:]
            del expected, value
        sentinel = object()
        with block_torch_functional():
            if next(actual, sentinel) is not sentinel:
                return {'status': 'failed', 'kind': 'extra_checkpoint', 'matched_checkpoints': matched,
                        'input_seed': input_seed}
        return {'status': 'matched', 'matched_checkpoints': matched, 'input_seed': input_seed,
                'note': 'Debug checkpoints match; normal forward still failed and remains rejected'}
    except Exception as exc:
        return {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}',
                'input_seed': input_seed, 'checkpoint': stage, 'matched_checkpoints': matched,
                'traceback': traceback.format_exc()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=int(os.environ.get('CUDA_AGENT_VERIFICATION_SEED', '2026')))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--replay', type=Path, help='Replay a trusted, locally saved failure_inputs.pt')
    parser.add_argument('--diagnose', action='store_true', help='Diagnose the saved failure in a fresh process')
    parser.add_argument('--diagnostic-replay', type=Path,
                        help='Diagnose an earlier candidate input, writing evidence to the current run only')
    args = parser.parse_args(argv)
    if args.diagnostic_replay and (not args.diagnose or args.replay):
        parser.error('--diagnostic-replay requires --diagnose and cannot be combined with --replay')
    directory = args.replay.resolve().parent if args.replay else fixture_directory()
    report_path = directory / 'verification.json'
    result = {'status': 'failed', 'seed': args.seed, 'num_checks': 5, 'checks': [],
              'passed_checks': 0, 'stage': 'initialization'}
    try:
        result['environment'] = runtime_metadata(args.device)
        torch_model, cuda_model, fixture = initialize_models(directory, args.seed, args.device)
        replay = torch.load(args.replay, map_location='cpu', weights_only=False) if args.replay else None
        if args.diagnose:
            failure = (torch.load(args.diagnostic_replay, map_location='cpu', weights_only=False)
                       if args.diagnostic_replay else replay or
                       torch.load(directory / 'failure_inputs.pt', map_location='cpu', weights_only=False))
            with torch.no_grad():
                diagnostic = diagnose(torch_model, cuda_model, fixture, failure['inputs'], failure['seed'], args.device)
            write_json(directory / 'diagnostic.json', diagnostic)
            print('[DIAGNOSTIC] ' + str(diagnostic))
            return 0
        indices = [replay['check_index']] if replay else range(5)
        first_failure_saved = False
        with torch.no_grad():
            for i in indices:
                input_seed = replay['seed'] if replay else args.seed + 1 + i
                inputs = replay['inputs'] if replay else build_inputs(input_seed)
                row = {'index': i, 'seed': input_seed, 'status': 'failed'}
                actual = expected = None
                try:
                    torch_model.load_state_dict(fixture['state_dict'], strict=True)
                    cuda_model.load_state_dict(fixture['state_dict'], strict=True)
                    seed_all(input_seed)
                    result['stage'] = 'reference'
                    expected = torch_model(*device_inputs(inputs, args.device))
                    synchronize(args.device)
                    seed_all(input_seed)
                    result['stage'] = 'forward'
                    with block_torch_functional():
                        actual = cuda_model(*device_inputs(inputs, args.device))
                        synchronize(args.device)
                    result['stage'] = 'comparison'
                    check_equal(actual, expected)
                    row['status'] = 'passed'
                    result['passed_checks'] += 1
                    print(f'[PASS] check {i + 1}/5')
                except Exception as exc:
                    row.update(stage=result['stage'], error=exc.detail if isinstance(exc, ComparisonFailure)
                               else {'kind': 'runtime', 'message': f'{type(exc).__name__}: {exc}'},
                               traceback=traceback.format_exc())
                    print(row['traceback'])
                    if not first_failure_saved:
                        result['first_failure'] = row
                        # Never overwrite a replay fixture while reading it.
                        if not args.replay:
                            temporary = directory / 'failure_inputs.tmp.pt'
                            torch.save({'inputs': inputs, 'seed': input_seed, 'check_index': i}, temporary)
                            temporary.replace(directory / 'failure_inputs.pt')
                        first_failure_saved = True
                finally:
                    result['checks'].append(row)
                    write_json(report_path if not args.replay else directory / 'replay_verification.json', result)
                    del actual, expected, inputs
                if row.get('stage') in ('reference', 'forward') and row['status'] == 'failed':
                    # A failed CUDA context cannot be reused safely; diagnose in a new subprocess.
                    break
        result['status'] = 'passed' if result['checks'] and all(r['status'] == 'passed' for r in result['checks']) and (replay or len(result['checks']) == 5) else 'failed'
        result['stage'] = 'verification'
    except Exception as exc:
        result.update(error={'kind': 'initialization', 'message': f'{type(exc).__name__}: {exc}'},
                      traceback=traceback.format_exc())
        print(result['traceback'])
    if args.diagnose:
        write_json(directory / 'diagnostic.json', {'status': 'unavailable', 'reason': result.get('error'), 'traceback': result.get('traceback')})
        return 1
    result['replay_command'] = f'{sys.executable} -m utils.verification --seed {args.seed} --device {args.device} --replay {directory / "failure_inputs.pt"}'
    write_json(report_path if not args.replay else directory / 'replay_verification.json', result)
    if result['status'] == 'passed':
        print('[PASS] verify success')
        return 0
    print('[FAIL] Structured report: ' + str(report_path))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
