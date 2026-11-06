"""Versioned offline correctness and repair metrics; missing old evidence stays unknown."""
from collections import Counter
from .candidates import progress, trusted_advancement
from .repair import failure_category

BASELINE_81_140_EXCLUDED = {101, 105, 107, 115, 116}
BASELINE_81_140_EVALUABLE = [i for i in range(81, 141) if i not in BASELINE_81_140_EXCLUDED]


def metric_category(category):
    return 'interface' if category in {'binding', 'contract'} else category


def _untrusted_count(diagnostic):
    static, observed = diagnostic.get('mapping', []), diagnostic.get('checkpoints', [])
    count = sum((i < len(static) and static[i].get('trustworthy') is False) or
                (i < len(observed) and observed[i].get('trustworthy') is False)
                for i in range(max(len(static), len(observed))))
    return count + (_untrusted_count(diagnostic['comparison']) if diagnostic.get('comparison') else 0)


def candidate_metrics(records, effects=None, complete=False):
    modern = complete or bool(records) and all(r.get('schema_version', 0) >= 2 for r in records)
    effects = effects if effects is not None else [r['repair'] for r in records if r.get('repair')]
    repairs = [r for r in records if r.get('repair') or r.get('api_stage') == 'repair']
    success = any(progress(r)['correctness'] for r in records)
    categories = Counter(metric_category(failure_category(r)) for r in records if not r.get('success'))
    fixed = sum(e.get('outcome') == 'fixed' for e in effects)
    advanced = sum(e.get('outcome') in {'fixed', 'progressed'} for e in effects)
    # Independent audit: do not derive this count from the rejection reason.
    by_id = {r['id']: r for r in records}
    useful_rejected = sum(r.get('decision') == 'rejected' and r.get('previous_best') in by_id
                          and trusted_advancement(r, by_id[r['previous_best']]) for r in records) if modern else None
    return {'schema_version': 2, 'first_pass': bool(records and progress(records[0])['correctness']),
            'correctness_success': success, 'entered_repair': bool(repairs or effects),
            'repair_recovered': bool(success and records and not progress(records[0])['correctness'] and repairs),
            'candidate_failures': dict(categories), 'repair_attempts': len(repairs),
            'progression_accepted': sum(r.get('decision') == 'best' and r.get('previous_best') is not None for r in records) if modern else None,
            'same_failure': sum(e.get('outcome') == 'same_failure' for e in effects) if modern or effects else None,
            'repair_not_applied': sum(bool(e.get('repair_not_applied')) for e in effects) if modern or effects else None,
            'untrusted_checkpoints': sum(_untrusted_count(r.get('diagnostic') or {}) for r in records) if modern else None,
            'useful_candidate_rollback_count': useful_rejected,
            'correctness_finishing_repairs': fixed if modern or effects else None,
            'successful_repair_rate': fixed / len(repairs) if modern and repairs else None,
            'progression_effective_rate': advanced / len(repairs) if modern and repairs else None}
