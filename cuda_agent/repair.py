"""Version-bound, category-specific repair evidence."""

import json
import re


def failure_category(record):
    stage = record['stage']
    if stage in {'sources', 'compile', 'interface'}:
        return stage
    verification = record.get('verification') or {}
    error = verification.get('first_failure', {}).get('error', {})
    if not error:
        error = next((x.get('error', {}) for x in verification.get('checks', []) if x.get('status') == 'failed'), {})
    text = error.get('message', record.get('error', '')).lower()
    if any(x in text for x in ('illegal memory', 'launch failure', 'out of bounds', 'misaligned address')):
        return 'memory'
    if error.get('kind') == 'runtime':
        return 'runtime'
    if error.get('kind') in {'type', 'length', 'keys', 'tensor_structure'}:
        return 'structure'
    if error.get('kind') in {'tensor_values', 'scalar'}:
        return 'numerical'
    return 'verification'


def error_fingerprint(record):
    verification = record.get('verification') or {}
    error = verification.get('first_failure', {}).get('error', {})
    if not error:
        error = next((x.get('error', {}) for x in verification.get('checks', []) if x.get('status') == 'failed'), {})
    diagnostic = record.get('diagnostic') or {}
    return json.dumps({'category': failure_category(record), 'kind': error.get('kind'),
                       'path': error.get('path'), 'checkpoint': diagnostic.get('checkpoint'),
                       'message': error.get('message', record.get('error', '')).split('\n')[0][:500]}, sort_keys=True)


def concise_log(log):
    """Surface root errors and signatures; leave full command logs on disk."""
    lines = log.splitlines()
    selected = []
    for i, line in enumerate(lines):
        if re.search(r'error:|undefined (?:symbol|reference)|Error:|Missing key|Unexpected key|incompatible function|no visible|prohibited|contract|AssertionError|ERROR SUMMARY|Invalid (?:__global__|read|write)', line, re.I):
            selected.extend(lines[i:i + 4])
    return '\n'.join(dict.fromkeys(selected))[:4000] if selected else log[-2000:]
