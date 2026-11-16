"""Prospective software-fixture selection; no real research result."""
import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.selection import SelectionConfig, load_selection_plan, load_selection_lock
from cave_mem_v3.policy.conservative_selection import (create_conservative_selection_plan, load_conservative_selection_plan,
    collect_conservative_selection_plan, evaluate_conservative_selection_plan, select_conservative_policy, load_conservative_selection_lock)
from test_single_execution import NoncloneableFixture
from cave_mem_v3.target_data.sampling import RiskHazard
from cave_mem_v3.cli import main
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture
from test_validation_partitions import manifest


class ConservativeSelectionTests(unittest.TestCase):
    def fixture(self, root, *, tasks=3, policies=None, hazard=.99, risk=1, budget=5, config=None, study=None):
        m = manifest()
        extra = [{"task_id": "selection-" + str(i), "group_id": "selection-family-" + str(i),
            "role": "validation", "validation_role": "selection"} for i in range(tasks - 1)]
        m = build_manifest(m["tasks"] + extra, benchmark=m["benchmark"], revision=m["revision"])
        fixture = make_prepared_fixture(root / "inputs", manifest=m)
        candidates = []
        for identifier, kind in policies or [("silent", "silent"), ("remind", "always_remind")]:
            bundle_path = root / (identifier + ".bundle.json")
            freeze_policy(m, fixture["bank"], actor_config=ScriptedActor().configuration(), encoder_identity=EncoderFixture.identity,
                study=study or StudySpec(max_steps=4, max_model_calls=4), hazard=RiskHazard(low_risk_probability=hazard, high_risk_probability=hazard),
                policy_config={"kind": kind}, output=bundle_path)
            candidates.append({"candidate_id": identifier, "bundle": str(bundle_path), "rollout_dir": str(root / (identifier + "-slot"))})
        plan = create_conservative_selection_plan(candidates, lock_path=root / "lock.json",
            config=config or SelectionConfig(risk, budget, bootstrap_samples=30), output=root / "plan.json")
        return fixture, plan

    def collect(self, root, fixture):
        return collect_conservative_selection_plan(root / "plan.json", env_factory=lambda task: NoncloneableFixture(task_id=task['task_id']), actor_factory=ScriptedActor, encoder=EncoderFixture())


    def test_noncloneable_complete_plan_worst_case_audit_and_immutable_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            with patch('cave_mem_v3.interventions.collection.collect_target_pairs', side_effect=AssertionError('no paired targets')):
                self.assertEqual(self.collect(root, fixture)['status'], 'collected')
            result = select_conservative_policy(root / 'plan.json')
            self.assertEqual(result, load_conservative_selection_lock(root / 'lock.json'))
            self.assertEqual(result['winner']['candidate_id'], 'remind')
            reports = {c['candidate_id']: c for c in result['candidates']}
            self.assertEqual(reports['remind']['risk']['mean'], 1)
            self.assertEqual(reports['remind']['actor_calls']['mean'], 4)
            self.assertEqual(reports['silent']['risk']['upper'], 0)
            self.assertEqual(reports['silent']['actor_calls']['mean'], 1)
            self.assertNotIn('audit_dir', plan)
            for report in reports.values():
                self.assertEqual(report['full_task_denominator']['planned_tasks'], 3)
                self.assertEqual(report['audit_denominator']['usable'], 3)
                for row in report['checkpoint_audit']:
                    self.assertEqual(row['actor_call_contribution'], 1 + row['effective_probability'] * row['remaining_model_calls'])
                    self.assertNotIn('H', row)
            with self.assertRaisesRegex(ValueError, 'lock exists'):
                select_conservative_policy(root / 'plan.json')
            with self.assertRaises(ValueError):
                load_selection_plan(root / 'plan.json')
            with self.assertRaises(ValueError):
                load_selection_lock(root / 'lock.json')

    def test_strict_risk_and_cost_select_global_silent(self):
        for risk, budget in ((0, 5), (1, 1)):
            with self.subTest(risk=risk), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture, _ = self.fixture(root, risk=risk, budget=budget)
                self.collect(root, fixture)
                self.assertEqual(select_conservative_policy(root / 'plan.json')['winner']['candidate_id'], 'silent')

    def test_infeasible_or_insufficient_groups_yields_no_winner(self):
        for kwargs in ({'budget': .5}, {'tasks': 1}, {'hazard': 1e-9}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture, _ = self.fixture(root, **kwargs)
                self.collect(root, fixture)
                result = select_conservative_policy(root / 'plan.json')
                self.assertEqual(result['status'], 'no_selection')
                self.assertIsNone(result['winner'])
                if kwargs.get('hazard') == 1e-9:
                    self.assertTrue(all(c['full_task_denominator']['observed_outcomes'] == 3 for c in result['candidates']))
                    self.assertTrue(all(c['audit_denominator']['usable'] == 0 for c in result['candidates']))

    def test_contextual_sample_zero_not_global_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, policies=[('a-exposure', 'exposure'), ('z-silent', 'silent')], risk=0)
            self.collect(root, fixture)
            result = select_conservative_policy(root / 'plan.json')
            self.assertEqual(result['winner']['candidate_id'], 'z-silent')
            report = result['candidates'][0]
            self.assertEqual(report['risk']['mean'], 0)
            self.assertGreater(report['risk']['upper'], 0)

    def test_missing_registered_candidate_blocks_every_winner(self):
        import cave_mem_v3.policy.conservative_selection as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            original = module._check_slot
            missing = plan['candidates'][0]['rollout_dir']
            with patch.object(module, '_check_slot', side_effect=lambda path, *a: False if str(path) == missing else original(path, *a)):
                result = evaluate_conservative_selection_plan(root / 'plan.json')
            self.assertFalse(result['registered_plan_complete'])
            self.assertIsNone(result['winner'])
            self.assertTrue(all('registered_plan_incomplete' in c['reasons'] for c in result['candidates']))

    def test_unknown_fee_keeps_denominators_blocks_plan(self):
        class Unknown(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                return replace(response, usage={**response.usage, 'api_cost': None})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, study=StudySpec(max_steps=4, max_model_calls=4, inference_weights={'api_cost': 1}))
            collect_conservative_selection_plan(root / 'plan.json', env_factory=lambda task: NoncloneableFixture(task_id=task['task_id']), actor_factory=Unknown, encoder=EncoderFixture())
            result = select_conservative_policy(root / 'plan.json')
            self.assertFalse(result['registered_plan_complete'])
            self.assertIsNone(result['winner'])
            self.assertTrue(all(c['full_task_denominator']['observed_outcomes'] == 3 for c in result['candidates']))
            self.assertTrue(all(c['full_task_denominator']['valid_utilities'] == 0 for c in result['candidates']))

    def test_prospective_bindings_prevent_old_raw_run_adoption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            path = Path(plan['candidates'][0]['rollout_dir']) / 'run/run.json'
            run = read_json(path)
            run['config_identity'] = None
            write_json(path, seal(run))
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                evaluate_conservative_selection_plan(root / 'plan.json')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.collect(root, fixture)

    def test_cli_readonly_selection_and_prospective_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            with patch('cave_mem_v3.cli._actor', side_effect=AssertionError('read only')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['select-conservative-policy', '--plan', str(root / 'plan.json')]), 0)
            entry = plan['candidates'][0]
            candidate = {'candidate_id': 'x', 'bundle': entry['bundle_path'], 'rollout_dir': entry['rollout_dir']}
            with self.assertRaisesRegex(ValueError, 'before any reserved'):
                create_conservative_selection_plan([candidate], lock_path=root / 'new.lock', config=SelectionConfig(1, 5), output=root / 'new.plan')
            candidate['rollout_dir'] = str(root / 'overlap')
            with self.assertRaisesRegex(ValueError, 'contain'):
                create_conservative_selection_plan([candidate], lock_path=root / 'overlap/lock', config=SelectionConfig(1, 5), output=root / 'new.plan')

    def test_missing_actual_identity_in_one_response_blocks_whole_plan(self):
        class PartialIdentity(ScriptedActor):
            def _propose(self, *args, **kwargs):
                response = super()._propose(*args, **kwargs)
                return replace(response, fingerprint=None) if response.action['type'] == 'operate' else response
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            collect_conservative_selection_plan(root / 'plan.json', env_factory=lambda task: NoncloneableFixture(task_id=task['task_id']), actor_factory=PartialIdentity, encoder=EncoderFixture())
            result = select_conservative_policy(root / 'plan.json')
            self.assertIsNone(result['winner'])
            self.assertFalse(result['registered_plan_complete'])
            self.assertTrue(any(c['served_identity']['fingerprint'] is not None and c['served_identity']['status'] == 'unknown' for c in result['candidates']))
            self.assertTrue(all('registered_plan_incomplete' in c['reasons'] for c in result['candidates']))

    def test_one_candidate_unknown_identity_cannot_borrow_other_candidate_identity(self):
        class Unknown(ScriptedActor):
            def _propose(self, *args, **kwargs):
                return replace(super()._propose(*args, **kwargs), fingerprint=None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            created = []
            def actor_factory():
                actor = Unknown() if len(created) < 4 else ScriptedActor()
                created.append(actor)
                return actor
            collect_conservative_selection_plan(root / 'plan.json', env_factory=lambda task: NoncloneableFixture(task_id=task['task_id']), actor_factory=actor_factory, encoder=EncoderFixture())
            result = select_conservative_policy(root / 'plan.json')
            self.assertEqual({c['served_identity']['status'] for c in result['candidates']}, {'unknown', 'reported'})
            self.assertIsNone(result['winner'])
            self.assertFalse(result['registered_plan_complete'])
