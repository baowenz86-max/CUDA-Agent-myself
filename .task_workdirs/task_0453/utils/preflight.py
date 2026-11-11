"""Collect reference truth before generation; check compiled interfaces afterward."""

import argparse
import ast
import importlib
import json
import os
import inspect
import re
import traceback
import sys
import types
from pathlib import Path

import torch

from .diagnostics import api_signatures, describe, reference_calls, reference_trace, seed_all, write_json
from .verification import build_inputs, device_inputs, fixture_directory, initialize_models, make_reference_fixture, synchronize
from .operator_contract import binding_contract_errors, validate_contract, model_interface_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--interface-only', action='store_true')
    parser.add_argument('--seed', type=int, default=int(os.environ.get('CUDA_AGENT_VERIFICATION_SEED', '2026')))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    directory = fixture_directory()
    name = 'interface_precompile.json' if args.interface_only else 'interface.json' if args.compiled else 'task_contract.json'
    report = {'status': 'failed', 'seed': args.seed}
    try:
        if args.compiled or args.interface_only:
            if args.interface_only:
                class InterfaceStub(types.ModuleType):
                    def __getattr__(self, name):
                        if name.startswith('__'):
                            raise AttributeError(name)
                        def forbidden(*args, **kwargs):
                            raise ValueError('Native execution during interface-only preflight: ' + name)
                        return forbidden
                sys.modules['cuda_extension'] = InterfaceStub('cuda_extension')
            reference, generated, fixture = initialize_models(directory, args.seed, args.device)
            errors = model_interface_errors(reference, generated)
            if errors:
                raise ValueError('\n'.join(errors))
            contract = json.loads((directory / 'task_contract.json').read_text())
            if contract.get('trace_available') and not callable(getattr(generated, 'debug_forward', None)):
                raise ValueError('ModelNew.debug_forward is required for the reference checkpoints')
            generated_module = importlib.import_module('model_new')
            extension = importlib.import_module('cuda_extension')
            aliases = {name for name, value in vars(generated_module).items() if value is extension}
            tree = ast.parse(inspect.getsource(generated_module))
            exports = set()
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                func = call.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in aliases:
                    if not callable(getattr(extension, func.attr, None)):
                        raise ValueError(f'Compiled cuda_extension.{func.attr} is missing or not callable')
                    exports.add(func.attr)
            report['compiled_exports'] = sorted(exports)
            operator_path = directory / 'custom_ops_contract.json'
            if operator_path.exists() and not args.interface_only:
                operators = validate_contract(json.loads(operator_path.read_text()), contract)
                report['compiled_signatures'] = {op['name']: (getattr(extension, op['name']).__doc__ or '')[:4000]
                                                 for op in operators['operators'] if hasattr(extension, op['name'])}
                report['binding_signature_pending'] = [op['name'] for op in operators['operators']
                    if not re.search(r'\b' + re.escape(op['name']) + r'\(', report['compiled_signatures'].get(op['name'], ''))]
                report['binding_signature_status'] = 'pending' if report['binding_signature_pending'] else 'checked'
                errors = binding_contract_errors(extension, operators)
                if errors:
                    raise ValueError('\n'.join(errors))
            report.update(status='passed', state_dict_keys=list(fixture['state_dict']))
        else:
            from .dependencies import needs_cudnn, discover_cudnn, configure_dependency
            source = Path('model.py').read_text()
            if needs_cudnn(source):
                dependency = discover_cudnn()
                write_json(directory / 'dependencies.json', dependency)
                if dependency['status'] != 'passed':
                    report['failure_type'] = 'environment'
                    raise RuntimeError(dependency['reason'])
                configure_dependency(dependency, os.environ)
            reference = importlib.import_module('model')
            fixture = make_reference_fixture(directory, args.seed, args.device)
            seed_all(args.seed)
            model = reference.Model(*fixture['init_inputs']).eval().to(args.device)
            model.load_state_dict(fixture['state_dict'], strict=True)
            inputs = build_inputs(args.seed + 1)
            report.update(forward_signature=str(inspect.signature(type(model).forward)),
                          constructor_signature=str(inspect.signature(reference.Model)),
                          init_inputs=describe(fixture['init_inputs']), inputs=describe(inputs), execution_device=args.device,
                          state_dict={k: describe(v) for k, v in fixture['state_dict'].items()},
                          api_signatures=api_signatures(model),
                          reference_calls=reference_calls(model),
                          modules={name: {'type': type(module).__name__, 'configuration': module.extra_repr(), 'training': module.training}
                                   for name, module in model.named_modules()}, input_seeds=[args.seed + 1 + i for i in range(5)])
            with torch.no_grad():
                try:
                    tracer, stages = reference_trace(model)
                except (ValueError, OSError, TypeError) as exc:
                    report.update(trace_available=False, trace_unavailable_reason=str(exc), checkpoints=[])
                    output = model(*device_inputs(inputs, args.device))
                    synchronize(args.device)
                    report['output'] = describe(output)
                else:
                    # A runtime reference failure is not an instrumentation fallback.
                    metadata = []
                    seed_all(args.seed + 1)
                    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                        synchronize(args.device)
                        metadata.append({**stage, 'value': describe(value)})
                        report['output'] = describe(value)
                        del value
                    report.update(trace_available=True, checkpoints=metadata)
            report.update(status='passed', rules=[
                'Use the reference code, not the dataset ops labels, as semantic truth.',
                'Preserve positional argument semantics, normalization axes and dtype promotion.',
                'Preserve constructor, forward signature and all state_dict keys; do not call torch modules in forward.',
                'Implement reference computational statements separately before attempting fusion.',
                'debug_forward must yield (checkpoint_id, value) for each listed checkpoint, including the final return.',
                'Debug values must match tuple/list/dict structure exactly; debug execution uses the same operator restrictions.',
                'Grid caps require grid-stride loops; every output element must be written.',
                'Only implement dtypes/devices exercised by this task unless extra tests are supplied.',
            ])
    except Exception as exc:
        text = str(exc).lower()
        category = 'resource' if any(x in text for x in ('out of memory', 'cannot allocate memory')) else (
            'environment' if any(x in text for x in ('cuda driver', 'no cuda gpus', 'cuda is not available', 'cudnn development dependency')) else
            'interface' if args.compiled or args.interface_only else 'reference')
        report.update(failure_type=report.get('failure_type', category), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        print(report['traceback'])
    write_json(directory / name, report)
    print(f'[{report["status"].upper()}] {directory / name}')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
