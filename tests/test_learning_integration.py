"""Whole-chain fitted software acceptance and genuine ordinary actor controls."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.agents.providers import ScriptedActor
from test_target_prepare import EncoderFixture
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.rollout import collect_policy_validation
from cave_mem_v3.policy.final_evaluation import collect_final_evaluation, export_final_evaluation_report, _checkpoint_coverage
from cave_mem_v3.target_data.sampling import RiskHazard
import test_policy_rollout as rollout_support
import test_final_evaluation as final_support

spec=importlib.util.spec_from_file_location('four_phase_learning_demo',Path(__file__).resolve().parents[1]/'examples/four_phase_learning_demo.py')
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class LearningClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)/'closure'
        cls.summary=demo.run_demo(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_fitted_heads_gate_both_bandits_and_final_inference(self):
        result=self.summary
        self.assertTrue(all(status=='fitted' for status in result['model_heads'].values()))
        self.assertEqual(result['gate_status'],'fitted')
        self.assertEqual(result['paired_bandit_status'],'candidate')
        self.assertEqual(result['single_bandit_status'],'candidate')
        self.assertEqual(result['dr_summary']['valid_rows'],24)
        self.assertIn(result['selected_kind'],('supervised','bandit'))
        self.assertTrue(result['final_evaluation_valid'])
        self.assertEqual(result['final_observed_outcomes'],result['final_planned_tasks'])
        self.assertTrue(result['original_evidence_unchanged'])
        self.assertGreater(result['learned_final_decisions'],0)
        self.assertGreater(result['learned_final_reminder_assignments'],0)
        self.assertGreater(result['learned_final_effective_probability_range'][1],0)
        model=read_json(self.root/'model.json')
        lock=read_json(self.root/'selection.lock')
        self.assertEqual(lock['winner']['bundle']['artifacts']['model'],model)
        self.assertEqual(lock['winner']['bundle']['artifacts']['warm_gate'],read_json(self.root/'gate.json'))
        report=read_json(self.root/'final.report')
        self.assertEqual(report['schema_version'],'final-evaluation-report.v2')
        self.assertIn('v=1|rho=0|e=1',report['checkpoint_coverage']['policies']['selected']['cells'])
        self.assertIn('v=1|rho=1|e=1',report['checkpoint_coverage']['policies']['selected']['cells'])
        with self.assertRaisesRegex(ValueError,'new output'):
            demo.run_demo(self.root)

    def test_no_memory_validation_never_uses_encoder_retrieval_features_or_reminder(self):
        with patch('cave_mem_v3.interventions.single.retrieve_memory',side_effect=AssertionError('retrieval')),\
             patch('cave_mem_v3.interventions.single.build_pre_treatment_state',side_effect=AssertionError('features')),\
             patch.object(demo.LearningActor,'propose_target',side_effect=AssertionError('reminder or resampled continuation')):
            result=collect_policy_validation(self.root/'no-memory.bundle.json',env_factory=demo.environment,actor_factory=demo.LearningActor,
                encoder=None,output_dir=self.root/'ordinary-validation')
        self.assertEqual(result['observed_outcomes'],8)
        self.assertTrue(all(r['decision'] is None for r in result['tasks']))
        self.assertTrue(all(r['reason']=='ordinary_actor_no_memory' for r in result['tasks']))
        for path in (self.root/'ordinary-validation').glob('sessions/*/preparation.json'):
            record=read_json(path)
            self.assertEqual(record['selection']['status'],'disabled_no_memory')
            self.assertTrue(all('pre_treatment_state' not in step for step in record['steps']))
        no_memory=next(c for c in read_json(self.root/'selection.lock')['candidates'] if c['candidate_id']=='no-memory')
        self.assertEqual(no_memory['risk']['upper'],0)
        self.assertEqual(no_memory['actor_calls']['upper'],0)
        self.assertTrue(all(r['costs']['model_calls']>0 for r in result['tasks']))

    def test_final_no_memory_has_no_encoded_features_and_preserves_all_costs(self):
        report=read_json(self.root/'final.report')
        ordinary=[r for r in report['tasks'] if r['policy_id']=='no-memory']
        self.assertTrue(ordinary)
        self.assertTrue(all(r['decision'] is None and r['metrics']['retrieved_memory_coverage']==0 for r in ordinary))
        self.assertTrue(all(r['metrics']['reminder_request_rate']==0 for r in ordinary))
        self.assertTrue(all(r['costs']['model_calls']>0 for r in ordinary))
        coverage=report['checkpoint_coverage']['policies']['no-memory']['cells']['no_frozen_memory_features']
        self.assertEqual(coverage['tasks'],len(ordinary))
        self.assertEqual(coverage['unknown_decision_tasks'],0)
        self.assertNotIn('interval',coverage)


class OrdinaryActorTests(unittest.TestCase):
    def test_full_actor_budget_has_no_reserved_reminder_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            helper=rollout_support.PolicyRolloutTests()
            counts={}
            for kind in ('no_memory','silent'):
                path,bundle,_,_=helper.fixture(root/kind,kind=kind,calls=3,safe=True)
                def factory(task):
                    return demo.LearningFixture(task_id=task['task_id'],safe_intent=True,max_steps=4)
                report=collect_policy_validation(path,env_factory=factory,actor_factory=ScriptedActor,
                    encoder=None if kind=='no_memory' else EncoderFixture(),output_dir=root/(kind+'-run'))
                row=report['tasks'][0]
                counts[kind]=(row['costs']['model_calls'],row['success'])
            self.assertEqual(counts['no_memory'],(3,True))
            self.assertEqual(counts['silent'],(2,False))

    def test_pending_ordinary_actor_remains_unknown_and_no_retry(self):
        from cave_mem_v3.agents.providers import ScriptedActor
        class Timeout(ScriptedActor):
            def propose(self,*a,**k):
                raise TimeoutError('unknown original actor request')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,_,factory,_=rollout_support.PolicyRolloutTests().fixture(root,kind='no_memory')
            result=collect_policy_validation(path,env_factory=factory,actor_factory=Timeout,encoder=None,output_dir=root/'run')
            self.assertEqual(result['pending_tasks'],1)
            self.assertEqual(result['unknown_outcomes'],1)
            self.assertIsNone(result['tasks'][0]['success'])
            with self.assertRaisesRegex(ValueError,'incomplete'):
                collect_policy_validation(path,env_factory=None,actor_factory=None,encoder=None,output_dir=root/'run')

    def test_legacy_final_plan_keeps_original_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); helper=final_support.FinalEvaluationTests();plan=helper.fixture(root)
            plan.pop('checkpoint_coverage_report')
            write_json(root/'final.plan',seal(plan))
            result=helper.collect(root)
            self.assertEqual(result['schema_version'],'final-evaluation-report.v1')
            self.assertNotIn('checkpoint_coverage',result)
            self.assertEqual(result,export_final_evaluation_report(root/'final.plan'))

    def test_missing_checkpoint_decision_is_not_reported_as_known_silence(self):
        rows=[{'task_id':'pending','policy_id':'a','status':'pending','decision':None,'arm':None,'success':None,'utility':None,'outcome_valid':False,'utility_valid':False},
              {'task_id':'ordinary','policy_id':'a','status':'completed','decision':None,'arm':None,'success':False,'utility':-.01,'outcome_valid':True,'utility_valid':True}]
        result=_checkpoint_coverage(rows,{'pending':None,'ordinary':None})
        cell=result['policies']['a']['cells']['no_frozen_memory_features']
        self.assertEqual(cell['unknown_decision_tasks'],1)
        self.assertEqual(cell['known_decision_tasks'],1)
        self.assertEqual(cell['unknown_outcomes'],1)
