"""Prospective frozen final-test plans and group-randomized marginal reports.

Only plan creation re-audits validation sources. Runtime consumes the sealed,
self-contained snapshot; test outcomes never update the selected policy.
"""
from __future__ import annotations
import copy
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import digest, read_json, stage_seed
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.interventions.single import _check_live_protocol
from cave_mem_v3.policy.frozen import BASELINES, load_frozen_policy, validate_bundle
from cave_mem_v3.policy.selection import (_common, _resolve, _paths_disjoint,
    _protect_training_inputs, _task_groups, load_selection_lock)
from cave_mem_v3.policy.conservative_selection import load_conservative_selection_lock
from cave_mem_v3.target_data.collection import _close, _exclusive_json


@dataclass(frozen=True)
class FinalEvaluationConfig:
    probabilities: dict
    allocation_seed: int = 42
    bootstrap_seed: int = 43
    bootstrap_samples: int = 400
    minimum_groups: int = 2
    alpha: float = .1
    estimator: str = 'horvitz_thompson_equal_task'
    stratification: tuple = ('family', 'mechanism', 'template_id')

    def __post_init__(self):
        if not isinstance(self.probabilities, dict) or len(self.probabilities) < 2:
            raise ValueError('final evaluation requires selected policy plus fixed baseline probabilities')
        for key, value in self.probabilities.items():
            if not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError('every final policy needs a known nonzero assignment probability')
        if not math.isclose(math.fsum(self.probabilities.values()), 1, abs_tol=1e-12, rel_tol=0):
            raise ValueError('final assignment probabilities must sum to one')
        for key, minimum in (('allocation_seed', 0), ('bootstrap_seed', 0), ('bootstrap_samples', 2), ('minimum_groups', 2)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError('invalid final evaluation ' + key)
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (float, int)) or not 0 < self.alpha < 1:
            raise ValueError('invalid final evaluation alpha')
        if self.estimator != 'horvitz_thompson_equal_task' or tuple(self.stratification) != ('family', 'mechanism', 'template_id'):
            raise ValueError('unsupported final estimator or metadata stratification')


def _test_tasks(bundle):
    return sorted((copy.deepcopy(t) for t in bundle['manifest']['tasks'] if t['role'] == 'test'), key=lambda t: t['task_id'])


def _allocation(bundle, config):
    tasks = _test_tasks(bundle)
    if not tasks:
        raise ValueError('final plan requires held-out test tasks')
    groups = _task_groups(bundle['manifest'])
    records = []
    for group in sorted({groups[t['task_id']] for t in tasks}):
        seed = stage_seed(config.allocation_seed, group, 0, 'final-test-group-assignment')
        draw = random.Random(seed).random()
        cumulative, assigned = 0., None
        for arm, probability in sorted(config.probabilities.items()):
            cumulative += probability
            if draw < cumulative:
                assigned = arm
                break
        if assigned is None:
            assigned = sorted(config.probabilities)[-1]  # roundoff only
        records.append({'group_id': group, 'task_ids': sorted(t['task_id'] for t in tasks if groups[t['task_id']] == group),
            'policy_id': assigned, 'propensity': config.probabilities[assigned], 'probabilities': dict(config.probabilities),
            'seed': seed, 'draw': draw})
    order = [t['task_id'] for t in tasks]
    random.Random(config.allocation_seed).shuffle(order)
    return records, order


def _lock_snapshot(value):
    schemas = {'policy-selection-lock.v1', 'conservative-policy-selection-lock.v1'}
    if value.get('schema_version') not in schemas:
        raise ValueError('unsupported final selection-lock protocol')
    checked(value, schema=value['schema_version'])
    if value['status'] != 'selected' or value['registered_plan_complete'] is not True or not value.get('winner'):
        raise ValueError('final evaluation requires a complete qualified selected lock')
    winner = value['winner']
    bundle = validate_bundle(winner['bundle'])
    reports = [c for c in value['candidates'] if c['candidate_id'] == winner['candidate_id']]
    if (len(reports) != 1 or reports[0]['feasible'] is not True or reports[0]['reasons']
            or winner['bundle_hash'] != bundle['record_hash'] or reports[0]['bundle_hash'] != bundle['record_hash']
            or any(c.get('served_identity', {}).get('status') != 'reported' for c in value['candidates'])):
        raise ValueError('final selection winner or observed actor identity is unqualified')
    return bundle, reports[0]['served_identity']


def create_final_evaluation_plan(selection_lock, baselines, *, config, output, run_dir, report_path,
                                 env_factory, actor_factory):
    header = read_json(selection_lock)
    loader = {'policy-selection-lock.v1': load_selection_lock,
              'conservative-policy-selection-lock.v1': load_conservative_selection_lock}.get(header.get('schema_version'))
    if loader is None:
        raise ValueError('unsupported selection-lock schema')
    lock = loader(selection_lock)  # Only this creation boundary can read validation outcomes.
    chosen, served = _lock_snapshot(lock)
    if not isinstance(config, FinalEvaluationConfig) or not isinstance(baselines, list) or not baselines:
        raise ValueError('explicit final config and fixed baselines are required')
    entries = [{'policy_id': 'selected', 'source': 'validation_selected', 'bundle': chosen, 'bundle_hash': chosen['record_hash']}]
    inputs = [_resolve(selection_lock)]
    identifiers = {'selected'}
    bundle_hashes = {chosen['record_hash']}
    for item in baselines:
        if set(item) != {'policy_id', 'bundle'} or not isinstance(item['policy_id'], str) or not item['policy_id'] or item['policy_id'] in identifiers:
            raise ValueError('fixed baselines require unique policy IDs and a bundle path')
        bundle = load_frozen_policy(item['bundle'])
        if bundle['policy_config']['kind'] not in BASELINES:
            raise ValueError('additional final arms must be frozen component baselines, not new learned candidates')
        if _common(bundle) != _common(chosen):
            raise ValueError('final baseline conditions differ from selected policy')
        if bundle['record_hash'] in bundle_hashes:
            raise ValueError('duplicate final policy bundle; select a distinct fixed baseline')
        bundle_hashes.add(bundle['record_hash'])
        inputs.append(_resolve(item['bundle']))
        identifiers.add(item['policy_id'])
        entries.append({'policy_id': item['policy_id'], 'source': 'fixed_component_baseline', 'bundle': bundle, 'bundle_hash': bundle['record_hash']})
    if set(config.probabilities) != identifiers:
        raise ValueError('final probabilities do not match registered policy inventory')
    output, run_dir, report_path = map(_resolve, (output, run_dir, report_path))
    outputs = [output, run_dir, report_path]
    _paths_disjoint(outputs + inputs)
    _protect_training_inputs(outputs, [e['bundle'] for e in entries])
    protected = sorted(set(lock['raw_sources'] + [lock['plan_path'], _resolve(selection_lock)]))
    for source in protected:
        _paths_disjoint(outputs + [_resolve(source)])
    if any(Path(p).exists() for p in outputs):
        raise ValueError('final plan must precede every reserved test output')
    validate_manifest(chosen['manifest'], verify_assets=True)
    assignments, order = _allocation(chosen, config)
    for task in _test_tasks(chosen):
        if any(task.get(k) is not None and not isinstance(task[k], str) for k in config.stratification):
            raise ValueError('final metadata strata must be strings or unknown')
    environments, capabilities = {}, {}
    actor = actor_factory()
    try:
        if actor.configuration() != chosen['actor_config']:
            raise ValueError('final actor differs from selected frozen actor')
        actor_caps = actor.capabilities()
        for task in _test_tasks(chosen):
            env = env_factory(copy.deepcopy(task))
            try:
                _check_live_protocol(env, actor)
                environments[task['task_id']], capabilities[task['task_id']] = env.configuration(), env.capabilities()
            finally:
                _close(env)
    finally:
        _close(actor)
    plan = seal({'schema_version': 'final-evaluation-plan.v1', 'role': 'test', 'validation_role': None,
        'protocol': 'group_randomized_frozen_policy_final_test.v1', 'plan_path': output, 'run_dir': run_dir,
        'report_path': report_path, 'config': asdict(config), 'selection_lock': lock,
        'selection_lock_hash': lock['record_hash'], 'selection_protocol': lock['schema_version'],
        'policies': sorted(entries, key=lambda e: e['policy_id']), 'conditions': _common(chosen),
        'served_identity': served, 'environment_configs': environments, 'environment_capabilities': capabilities,
        'actor_capabilities': actor_caps, 'group_assignments': assignments, 'task_order': order,
        'execution_policy_order': sorted(identifiers),
        'execution_task_order': [task_id for arm in sorted(identifiers) for task_id in order
            if next(g for g in assignments if task_id in g['task_ids'])['policy_id'] == arm],
        'execution_order_scope': 'batched by sorted policy ID, then filtered preregistered task order',
        'protected_validation_roots': protected,
        'report_contract': 'equal-task HT marginal means and joint group-bootstrap contrasts; assigned-task descriptive means; no individual H/rescue',
        'checkpoint_coverage_report': 'frozen_v_rho_e_descriptive_only.v1',
        'chronology': 'validation raw evidence re-audited only at creation; output paths absent; local hashes are provenance, not trusted signatures'})
    _exclusive_json(Path(output), plan)
    return plan


def validate_final_evaluation_plan(plan):
    checked(plan, schema='final-evaluation-plan.v1')
    config = FinalEvaluationConfig(**plan['config'])
    chosen, served = _lock_snapshot(plan['selection_lock'])
    if (plan['role'] != 'test' or plan.get('validation_role') is not None
            or plan['protocol'] != 'group_randomized_frozen_policy_final_test.v1'
            or plan['selection_protocol'] != plan['selection_lock']['schema_version']
            or plan['selection_lock_hash'] != plan['selection_lock']['record_hash']
            or plan['served_identity'] != served or plan['conditions'] != _common(chosen)):
        raise ValueError('final plan frozen selection/role/protocol mismatch')
    if plan.get('checkpoint_coverage_report') not in (None, 'frozen_v_rho_e_descriptive_only.v1'):
        raise ValueError('unsupported frozen checkpoint descriptive report')
    entries = plan['policies']
    if [e['policy_id'] for e in entries] != sorted(config.probabilities) or len(entries) != len(config.probabilities):
        raise ValueError('final policy inventory mismatch')
    if len({e['bundle_hash'] for e in entries}) != len(entries):
        raise ValueError('duplicate final policy bundle')
    for entry in entries:
        bundle = validate_bundle(entry['bundle'])
        if entry['bundle_hash'] != bundle['record_hash'] or _common(bundle) != plan['conditions']:
            raise ValueError('final bundle/conditions mismatch')
        if entry['policy_id'] == 'selected':
            if bundle != chosen or entry['source'] != 'validation_selected':
                raise ValueError('final selected policy replaced')
        elif bundle['policy_config']['kind'] not in BASELINES or entry['source'] != 'fixed_component_baseline':
            raise ValueError('unregistered final learned candidate')
    assignments, order = _allocation(chosen, config)
    if assignments != plan['group_assignments'] or order != plan['task_order']:
        raise ValueError('final test group assignment or task order was changed')
    expected_execution = [t['task_id'] for arm in sorted(config.probabilities) for t in final_policy_tasks(plan,arm)]
    if (plan.get('execution_policy_order') != sorted(config.probabilities)
            or plan.get('execution_task_order') != expected_execution
            or plan.get('execution_order_scope') != 'batched by sorted policy ID, then filtered preregistered task order'):
        raise ValueError('final preregistered execution order changed')
    for key in ('environment_configs', 'environment_capabilities'):
        if set(plan[key]) != set(order) or any(not isinstance(v, dict) or not v for v in plan[key].values()):
            raise ValueError('final environment inventory mismatch')
    if plan['actor_capabilities'].get('stateless') is not True:
        raise ValueError('final actor must be stateless')
    outputs = [plan[k] for k in ('plan_path', 'run_dir', 'report_path')]
    if any(_resolve(p) != p for p in outputs):
        raise ValueError('final output path alias changed')
    _paths_disjoint(outputs)
    _protect_training_inputs(outputs, [e['bundle'] for e in entries])
    for source in plan['protected_validation_roots']:
        _paths_disjoint(outputs + [_resolve(source)])
    return plan


def load_final_evaluation_plan(path):
    plan = validate_final_evaluation_plan(read_json(path))
    if _resolve(path) != plan['plan_path']:
        raise ValueError('final plan path changed')
    return plan


def final_policy_tasks(plan, policy_id):
    if policy_id not in plan['config']['probabilities']:
        raise ValueError('unregistered final policy')
    ids = {t for group in plan['group_assignments'] if group['policy_id'] == policy_id for t in group['task_ids']}
    tasks = {t['task_id']: t for t in plan['conditions']['manifest']['tasks']}
    return [copy.deepcopy(tasks[t]) for t in plan['task_order'] if t in ids]


def _policy_directory(plan, policy_id):
    return Path(plan['run_dir']) / ('policy-' + digest(policy_id)[:24])


def _execution_header(plan):
    return seal({'schema_version': 'final-evaluation-execution.v1', 'plan_hash': plan['record_hash'],
        'role': 'test', 'policy_order': [e['policy_id'] for e in plan['policies']], 'task_order': plan['task_order'],
        'execution_task_order': plan['execution_task_order'],
        'execution_order_scope': plan['execution_order_scope']})


def collect_final_evaluation(plan_path, *, env_factory, actor_factory, encoder):
    from cave_mem_v3.policy.final_rollout import collect_final_policy
    plan = load_final_evaluation_plan(plan_path)
    root = Path(plan['run_dir'])
    if root.exists():
        result = summarize_final_evaluation(plan_path)
        if (root / 'closed.json').exists():
            return result
        raise ValueError('incomplete final evaluation preserved; no automatic retry or reassignment')
    if Path(plan['report_path']).exists():
        raise ValueError('final report exists; collection is closed')
    if encoder.identity != plan['conditions']['encoder_identity']:
        raise ValueError('final encoder changed')
    _exclusive_json(root / 'execution.json', _execution_header(plan))
    for entry in plan['policies']:
        report = collect_final_policy(plan, entry['policy_id'], env_factory=env_factory, actor_factory=actor_factory, encoder=encoder)
        if report['pending_tasks'] or report['attempted_tasks'] != report['planned_tasks'] or report['error_tasks']:
            return summarize_final_evaluation(plan_path)
    _exclusive_json(root / 'closed.json', seal({'schema_version': 'final-evaluation-closed.v1', 'plan_hash': plan['record_hash']}))
    return summarize_final_evaluation(plan_path)


def _ess(weights):
    denominator = math.fsum(w*w for w in weights)
    return math.fsum(weights) ** 2 / denominator if denominator > 0 else 0.


def _metric(rows, policy_id, key, config):
    assigned = [r for r in rows if r['policy_id'] == policy_id]
    valid = [r for r in assigned if r['metrics'][key] is not None]
    group_ids = sorted({r['randomization_group'] for r in assigned})
    result = {'planned_population_tasks': len(rows), 'assigned_tasks': len(assigned), 'known_tasks': len(valid),
        'unknown_tasks': len(assigned) - len(valid), 'assigned_groups': len(group_ids),
        'estimate': None, 'assigned_mean': math.fsum(r['metrics'][key] for r in valid) / len(valid) if valid else None,
        'assigned_mean_scope': 'descriptive known-only; not complete-population estimate when missing',
        'task_ess': _ess([1/r['propensity'] for r in valid]),
        'group_ess': _ess([math.fsum(1/r['propensity'] for r in valid if r['randomization_group'] == g) for g in group_ids]),
        'status': 'unavailable', 'reason': None, 'interval': None}
    if len(valid) != len(assigned):
        result['reason'] = 'assigned_metric_unknown'
    elif len(group_ids) < config.minimum_groups:
        result['reason'] = 'insufficient_assigned_groups'
    elif not rows or not assigned:
        result['reason'] = 'no_assigned_support'
    else:
        result.update(status='available', estimate=math.fsum(r['metrics'][key]/r['propensity'] for r in assigned)/len(rows))
    return result


def _statistical_report(rows, plan):
    config = FinalEvaluationConfig(**plan['config'])
    policies = sorted(config.probabilities)
    keys = sorted({key for row in rows for key in row['metrics']})
    reports = {arm: {key: _metric(rows, arm, key, config) for key in keys} for arm in policies}
    # Resample the original randomized groups jointly across arms, retaining
    # unassigned-to-a-particular-arm groups as zero HT contributions.
    groups = sorted({r['randomization_group'] for r in rows})
    grouped = {g: [r for r in rows if r['randomization_group'] == g] for g in groups}
    rng = random.Random(config.bootstrap_seed)
    draws = {(arm,key): [] for arm in policies for key in keys}
    contrast_draws = {(arm,key): [] for arm in policies if arm != 'selected' for key in keys}
    contrasts = {}
    for arm in policies:
        if arm == 'selected':
            continue
        contrasts[arm] = {}
        for key in keys:
            left, right = reports['selected'][key], reports[arm][key]
            valid = left['status'] == right['status'] == 'available'
            contrasts[arm][key] = {'status': 'available' if valid else 'unavailable',
                'estimate': left['estimate'] - right['estimate'] if valid else None,
                'interval': None, 'reason': None if valid else 'arm_metric_or_group_support_unavailable'}
    if groups:
        for _ in range(config.bootstrap_samples):
            selected = [groups[rng.randrange(len(groups))] for _ in groups]
            sample = [row for group in selected for row in grouped[group]]
            values = {}
            for arm in policies:
                for key in keys:
                    if reports[arm][key]['status'] == 'available':
                        value = math.fsum(r['metrics'][key]/r['propensity'] for r in sample if r['policy_id'] == arm)/len(sample)
                        draws[arm,key].append(value)
                        values[arm,key] = value
            for arm in contrasts:
                for key in keys:
                    if contrasts[arm][key]['status'] == 'available':
                        contrast_draws[arm,key].append(values['selected',key] - values[arm,key])
    def interval(samples):
        samples = sorted(samples)
        return {'lower': samples[max(0, math.floor(config.alpha/2*len(samples)))],
            'upper': samples[min(len(samples)-1, math.ceil((1-config.alpha/2)*len(samples))-1)],
            'alpha': config.alpha, 'method': 'joint_original_group_percentile_bootstrap_diagnostic',
            'groups': len(groups), 'samples': len(samples)}
    for (arm,key), values in draws.items():
        if values:
            reports[arm][key]['interval'] = interval(values)
    for (arm,key), values in contrast_draws.items():
        if values:
            contrasts[arm][key]['interval'] = interval(values)
    return {'policies': reports, 'selected_minus_baseline': contrasts}


def summarize_final_evaluation(plan_path):
    from cave_mem_v3.agents.runtime import empty_costs
    from cave_mem_v3.policy.dataset import _response_identities, _served_identity
    from cave_mem_v3.policy.final_rollout import summarize_final_policy
    plan = load_final_evaluation_plan(plan_path)
    root = Path(plan['run_dir'])
    if not (root / 'execution.json').exists() or read_json(root / 'execution.json') != _execution_header(plan):
        raise ValueError('final execution is absent or not bound to the frozen plan')
    allowed = {'execution.json', 'closed.json'} | {_policy_directory(plan,e['policy_id']).name for e in plan['policies']}
    if any(p.name not in allowed for p in root.iterdir()):
        raise ValueError('unregistered final evaluation output')
    assignments = {t:g for g in plan['group_assignments'] for t in g['task_ids']}
    tasks = {t['task_id']: t for t in plan['conditions']['manifest']['tasks']}
    rows, summaries, identities = [], [], []
    checkpoint_strata = {}
    stopped = False
    for entry in plan['policies']:
        arm = entry['policy_id']
        directory = _policy_directory(plan, arm)
        if stopped and directory.exists():
            raise ValueError('final evaluation continued after incomplete policy run')
        if (directory / 'run.json').exists():
            summary = summarize_final_policy(directory, plan, arm)
            summaries.append({'policy_id': arm, 'summary_hash': summary['record_hash'], 'run_hash': summary['run_hash']})
            if summary['planned_tasks']:
                identities.append(summary['served_identity'])
            arm_rows = summary['tasks']
            stopped = bool(summary['pending_tasks'] or summary['error_tasks'] or summary['attempted_tasks'] != summary['planned_tasks'])
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError('unbound final policy output')
            stopped = True
            arm_rows = [{'task_id': t['task_id'], 'status': 'unattempted', 'reason': 'policy_not_started',
                'success': None, 'utility': None, 'outcome_valid': False, 'utility_valid': False, 'costs': None,
                'decision': None, 'arm': None} for t in final_policy_tasks(plan, arm)]
        for index, row in enumerate(arm_rows):
            assignment = assignments[row['task_id']]
            known_decision = row['decision'] is not None or row['status'] == 'completed'
            coverage = None
            reminder_requested = None
            state = None
            if row.get('trace_id'):
                session = directory / 'sessions' / f"{index:06d}-{row['trace_id']}"
                if (session / 'input.json').exists():
                    item = read_json(session / 'input.json')
                    state = item.get('pre_treatment_state')
                    if state is not None:
                        coverage = float(state.get('memory_id') is not None)
                    elif row['status'] == 'completed':
                        coverage = 0.
                if row['status'] == 'completed':
                    events = [read_json(p)['event'] for p in (session / 'events').glob('*.json')]
                    reminder_requested = float(any(e['kind'] == 'actor.started' and e.get('stage') == 'target_remind' for e in events))
            checkpoint_strata[row['task_id']] = {k: (state['x'][k].get('value', '?') if k == 'e' else state['x'][k]) for k in ('v','rho','e')} if state else None
            metrics = {'pass_at_1': float(row['success']) if row['outcome_valid'] else None,
                'utility': row['utility'] if row['utility_valid'] else None,
                'checkpoint_coverage': float(row['decision'] is not None) if known_decision else None,
                'retrieved_memory_coverage': coverage,
                'reminder_assignment_rate': float(row['arm'] == 'remind') if known_decision else None,
                'reminder_request_rate': reminder_requested}
            for key in empty_costs():
                metrics['cost_' + key] = row['costs'].get(key) if row['status'] == 'completed' and row['costs'] else None
            # Operational time is reported separately from charged/utility costs.
            for key in ('wall_seconds', 'reset_wall_seconds', 'finalization_wall_seconds', 'finalization_calls'):
                phases = row.get('operational_costs', {})
                metrics['operational_' + key] = math.fsum(p.get(key, 0.) for p in phases.values()) if row['status'] == 'completed' and phases else None
            metadata = {key: tasks[row['task_id']].get(key) for key in ('family', 'mechanism', 'template_id')}
            rows.append({**row, 'policy_id': arm, 'randomization_group': assignment['group_id'],
                'propensity': assignment['propensity'], 'assignment_seed': assignment['seed'],
                'assignment_probabilities': assignment['probabilities'], 'metadata': metadata, 'metrics': metrics})
    if [r['task_id'] for r in sorted(rows,key=lambda r: plan['task_order'].index(r['task_id']))] != plan['task_order']:
        raise ValueError('final report task denominator differs from plan')
    identities_complete = bool(identities) and all(i['status'] in ('reported','not_required_no_actor_calls') for i in identities)
    called_identities = [i for i in identities if i['status'] != 'not_required_no_actor_calls']
    merged_identity = _served_identity(called_identities)
    if identities and not called_identities:
        merged_identity['status'] = 'not_required_no_actor_calls'
    merged_identity['zero_call_policy_exemptions'] = len(identities) - len(called_identities)
    identity_matches = all(i.get(k) in (None, plan['served_identity'][k]) for i in identities for k in ('model','fingerprint'))
    counts = Counter(r['status'] for r in rows)
    closed = root / 'closed.json'
    if closed.exists():
        expected = seal({'schema_version': 'final-evaluation-closed.v1', 'plan_hash': plan['record_hash']})
        if read_json(closed) != expected or stopped:
            raise ValueError('final completion marker conflicts with raw task completion')
    statistics = _statistical_report(rows, plan)
    strata = {}
    for field in plan['config']['stratification']:
        strata[field] = {value: _statistical_report([r for r in rows if (r['metadata'][field] or '<unknown>') == value], plan)
            for value in sorted({r['metadata'][field] or '<unknown>' for r in rows})}
    if not closed.exists() or not identities_complete or not identity_matches:
        reason = 'final_execution_incomplete' if not closed.exists() else 'actual_identity_unknown_or_mismatched'
        def suppress(report):
            for section in ('policies', 'selected_minus_baseline'):
                for values in report[section].values():
                    for metric in values.values():
                        metric.update(status='unavailable', estimate=None, interval=None, reason=reason)
        suppress(statistics)
        for fields in strata.values():
            for report in fields.values():
                suppress(report)
    extensions = {}
    if plan.get('checkpoint_coverage_report'):
        extensions['checkpoint_coverage'] = _checkpoint_coverage(rows, checkpoint_strata)
    return seal({'schema_version': 'final-evaluation-report.v2' if extensions else 'final-evaluation-report.v1', **extensions, 'plan_hash': plan['record_hash'], 'role': 'test',
        'selection_protocol': plan['selection_protocol'], 'selection_lock_hash': plan['selection_lock_hash'],
        'execution_complete': closed.exists(), 'identity_complete': identities_complete,
        'identity_matches_selection': identity_matches, 'served_identity': merged_identity,
        'evaluation_valid': closed.exists() and identities_complete and identity_matches,
        'planned_tasks': len(rows), 'attempted_tasks': len(rows)-counts['unattempted'],
        'observed_outcomes': sum(r['outcome_valid'] for r in rows), 'unknown_outcomes': sum(not r['outcome_valid'] for r in rows),
        'valid_utilities': sum(r['utility_valid'] for r in rows), 'pending_tasks': counts['pending'], 'error_tasks': counts['error'],
        'group_assignments': plan['group_assignments'], 'config': plan['config'], 'statistics': statistics,
        'execution_policy_order': plan['execution_policy_order'], 'execution_task_order': plan['execution_task_order'],
        'execution_order_scope': plan['execution_order_scope'],
        'strata': strata, 'policy_summaries': summaries, 'tasks': rows,
        'risk_evidence': {'scope': 'validation selected-checkpoint constraint evidence only, not test individual harm',
            'selection_protocol': plan['selection_protocol'], 'selected_validation_report': next(c for c in plan['selection_lock']['candidates'] if c['candidate_id'] == plan['selection_lock']['winner']['candidate_id'])},
        'claim_boundary': 'One randomly assigned policy per test task group; pass@1 is one complete task attempt. HT marginal estimates may exceed natural outcome bounds and are not clipped. Unknown assigned metrics block their full estimates; observed-only means are descriptive. Joint group-bootstrap intervals require independent groups and sufficient support, are not exact randomization inference or multiplicity-adjusted guarantees. No individual H, rescue or paired counterfactual outcome is observed. Policy batches run sequentially; identity freezing does not eliminate time/batch cost effects. Fixture outputs are software tests only.'})


def export_final_evaluation_report(plan_path):
    plan = load_final_evaluation_plan(plan_path)
    report = summarize_final_evaluation(plan_path)
    path = Path(plan['report_path'])
    if path.exists():
        if read_json(path) != report:
            raise ValueError('final report exists with different evidence; never overwrite')
    else:
        _exclusive_json(path, report)
    return report



def _checkpoint_coverage(rows, strata):
    """Selected-checkpoint descriptions, not randomized population subgroups."""
    result = {'scope': 'descriptive frozen checkpoint v/rho/e coverage only; not population subgroup causal effects',
        'planned_tasks': len(rows), 'frozen_checkpoints': sum(v is not None for v in strata.values()), 'policies': {}}
    for arm in sorted({r['policy_id'] for r in rows}):
        assigned = [r for r in rows if r['policy_id'] == arm]
        cells = {}
        for row in assigned:
            values = strata[row['task_id']]
            label = '|'.join(k+'='+str(values[k]) for k in ('v','rho','e')) if values else 'no_frozen_memory_features'
            cells.setdefault(label, []).append(row)
        result['policies'][arm] = {'assigned_tasks': len(assigned), 'cells': {label: {
            'tasks': len(cell), 'observed_outcomes': sum(r['outcome_valid'] for r in cell),
            'observed_successes': sum(r['success'] is True for r in cell),
            'unknown_outcomes': sum(not r['outcome_valid'] for r in cell),
            'reminder_assignments': sum(r.get('arm') == 'remind' for r in cell),
            'known_decision_tasks': sum(r.get('decision') is not None or r['status'] == 'completed' for r in cell),
            'unknown_decision_tasks': sum(r.get('decision') is None and r['status'] != 'completed' for r in cell),
            'known_utility_tasks': sum(r['utility_valid'] for r in cell),
            'known_only_mean_utility': math.fsum(r['utility'] for r in cell if r['utility_valid'])/sum(r['utility_valid'] for r in cell) if any(r['utility_valid'] for r in cell) else None
        } for label,cell in sorted(cells.items())}}
    return result
