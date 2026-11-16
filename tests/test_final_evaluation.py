"""Offline final-test protocol acceptance. No real benchmark observations."""
import contextlib
import copy
import io
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor, ProviderError
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.selection import SelectionConfig, create_selection_plan, collect_selection_plan, select_policy
from cave_mem_v3.policy.conservative_selection import create_conservative_selection_plan, collect_conservative_selection_plan, select_conservative_policy
from cave_mem_v3.policy.final_evaluation import (FinalEvaluationConfig, create_final_evaluation_plan, load_final_evaluation_plan,
    collect_final_evaluation, summarize_final_evaluation, export_final_evaluation_report, _policy_directory, _statistical_report)
from cave_mem_v3.target_data.sampling import RiskHazard, OnlineCheckpointSampler
from cave_mem_v3.cli import main
from test_single_execution import NoncloneableFixture
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture
from test_validation_partitions import manifest


def env_factory(task):
    return NoncloneableFixture(task_id=task['task_id'])


class FinalEvaluationTests(unittest.TestCase):
    def fixture(self, root, *, route='single', study=None, hazard=.99, empty_bank=False, test_groups=12, final_env_factory=env_factory):
        m = manifest()
        extra = [{'task_id':'selection-'+str(i), 'group_id':'selection-extra-'+str(i), 'role':'validation', 'validation_role':'selection'} for i in range(2)]
        extra += [{'task_id':'test-'+str(i), 'group_id':'test-extra-'+str(i), 'role':'test', 'family':'even' if i%2 == 0 else 'odd'} for i in range(test_groups-1)]
        extra += [{'task_id':'test-linked', 'group_id':'test-extra-0', 'role':'test', 'family':'even'}]
        m = build_manifest(m['tasks']+extra, benchmark=m['benchmark'], revision=m['revision'])
        fixture = make_prepared_fixture(root/'inputs', manifest=m, empty_bank=empty_bank)
        entries = []
        for kind in ('silent','always_remind'):
            path = root/(kind+'.bundle.json')
            freeze_policy(m, fixture['bank'], actor_config=ScriptedActor().configuration(), encoder_identity=EncoderFixture.identity,
                study=study or StudySpec(max_steps=4,max_model_calls=4), hazard=RiskHazard(low_risk_probability=hazard,high_risk_probability=hazard),
                policy_config={'kind':kind}, output=path)
            entries.append({'candidate_id':kind,'bundle':str(path),'rollout_dir':str(root/(kind+'-validation'))})
        cfg = SelectionConfig(1,5,bootstrap_samples=20)
        if route == 'single':
            create_conservative_selection_plan(entries,lock_path=root/'selection.lock',config=cfg,output=root/'selection.plan')
            collect_conservative_selection_plan(root/'selection.plan',env_factory=env_factory,actor_factory=ScriptedActor,encoder=EncoderFixture())
            lock = select_conservative_policy(root/'selection.plan')
        else:
            create_selection_plan(entries,audit_dir=root/'audit',lock_path=root/'selection.lock',config=cfg,output=root/'selection.plan')
            collect_selection_plan(root/'selection.plan',env_factory=fixture['env_factory'],actor_factory=ScriptedActor,encoder=EncoderFixture())
            lock = select_policy(root/'selection.plan')
        self.assertEqual(lock['status'],'selected')
        baseline = 'silent' if lock['winner']['bundle']['policy_config']['kind'] != 'silent' else 'always_remind'
        config = FinalEvaluationConfig({'selected':.5,'baseline':.5},bootstrap_samples=30)
        plan = create_final_evaluation_plan(root/'selection.lock',[{'policy_id':'baseline','bundle':str(root/(baseline+'.bundle.json'))}],
            config=config,output=root/'final.plan',run_dir=root/'final-run',report_path=root/'final.report',env_factory=final_env_factory,actor_factory=ScriptedActor)
        return plan

    def collect(self, root, actor=ScriptedActor):
        return collect_final_evaluation(root/'final.plan',env_factory=env_factory,actor_factory=actor,encoder=EncoderFixture())

    def test_both_selection_protocols_reach_self_contained_test_and_readonly_report(self):
        for route in ('single','paired'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); plan=self.fixture(root,route=route)
                # Move all original inputs/validation artifacts away after plan creation.
                stash=root/'preserved-sources';stash.mkdir()
                for path in list(root.iterdir()):
                    if path not in (root/'final.plan',stash):
                        shutil.move(str(path),str(stash/path.name))
                with patch('cave_mem_v3.policy.final_evaluation.load_selection_lock',side_effect=AssertionError('validation reread')),patch('cave_mem_v3.policy.final_evaluation.load_conservative_selection_lock',side_effect=AssertionError('validation reread')):
                    result=self.collect(root)
                    self.assertTrue(result['execution_complete'])
                    self.assertTrue(result['evaluation_valid'])
                    self.assertEqual(result['role'],'test')
                    self.assertEqual(result['planned_tasks'],13)
                    self.assertEqual(result['observed_outcomes'],13)
                    self.assertEqual(result,export_final_evaluation_report(root/'final.plan'))
                    never=lambda *a,**k: (_ for _ in ()).throw(AssertionError('repeat provider'))
                    self.assertEqual(result,collect_final_evaluation(root/'final.plan',env_factory=never,actor_factory=never,encoder=None))
                assigned={r['task_id']:r for r in result['tasks']}
                self.assertEqual(assigned['test-0']['policy_id'],assigned['test-linked']['policy_id'])
                self.assertEqual(assigned['test-0']['randomization_group'],assigned['test-linked']['randomization_group'])
                self.assertTrue(all(r['propensity']==.5 for r in result['tasks']))
                self.assertTrue(all('H' not in r and 'rescue' not in r for r in result['tasks']))
                self.assertIn('family',result['strata'])
                for entry in plan['policies']:
                    raw=read_json(_policy_directory(plan,entry['policy_id'])/'run.json')
                    self.assertEqual(raw['role'],'test')
                    self.assertEqual(raw['schema_version'],'policy-test-run.v1')
                self.assertEqual(result,export_final_evaluation_report(root/'final.plan'))

    def test_allocation_is_recomputed_and_tampering_rejected_before_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.fixture(root)
            bad=copy.deepcopy(plan);bad['group_assignments'][0]['policy_id']='selected' if bad['group_assignments'][0]['policy_id']=='baseline' else 'baseline'
            write_json(root/'final.plan',seal(bad))
            with self.assertRaisesRegex(ValueError,'assignment'):
                self.collect(root)
            self.assertFalse((root/'final-run').exists())

    def test_old_entrypoints_still_reject_test_before_providers(self):
        from cave_mem_v3.policy.rollout import collect_policy_validation
        from cave_mem_v3.interventions.single import collect_single_target
        with self.assertRaisesRegex(ValueError,'test'):
            OnlineCheckpointSampler(role='test',task_id='x',trace_id='x',seed=1)
        never=lambda *a,**k: (_ for _ in ()).throw(AssertionError('provider'))
        with self.assertRaisesRegex(ValueError,'test'):
            collect_policy_validation('missing',role='test',env_factory=never,actor_factory=never,encoder=None,output_dir='not-created')
        with self.assertRaisesRegex(ValueError,'test'):
            collect_single_target({},bank={},role='test',spec=StudySpec(),env_factory=never,actor_factory=never,output_dir='not-created',encoder=None,assignment_probability=.5)

    def test_unknown_fee_preserves_successes_and_blocks_utility_estimate(self):
        class Unknown(ScriptedActor):
            def _propose(self,*a,**k):
                response=super()._propose(*a,**k)
                return replace(response,usage={**response.usage,'api_cost':None})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,study=StudySpec(max_steps=4,max_model_calls=4,inference_weights={'api_cost':1}))
            result=self.collect(root,Unknown)
            self.assertEqual(result['observed_outcomes'],13)
            self.assertEqual(result['valid_utilities'],0)
            for arm in result['statistics']['policies'].values():
                self.assertIsNone(arm['utility']['estimate'])
                self.assertEqual(arm['utility']['reason'],'assigned_metric_unknown')
                self.assertEqual(arm['pass_at_1']['status'],'available')

    def test_ht_manual_unequal_groups_and_missing_support(self):
        config=FinalEvaluationConfig({'selected':.25,'baseline':.75},bootstrap_samples=20)
        plan={'config':{**config.__dict__}}
        rows=[]
        for group,arm,values in [('a','selected',[1,1]),('b','selected',[0]),('c','baseline',[1]),('d','baseline',[0,0,0])]:
            rows += [{'policy_id':arm,'propensity':config.probabilities[arm],'randomization_group':group,'metrics':{'pass_at_1':float(v)}} for v in values]
        report=_statistical_report(rows,plan)
        self.assertAlmostEqual(report['policies']['selected']['pass_at_1']['estimate'],8/7)
        self.assertAlmostEqual(report['policies']['baseline']['pass_at_1']['estimate'],(4/3)/7)
        self.assertAlmostEqual(report['selected_minus_baseline']['baseline']['pass_at_1']['estimate'],(8-4/3)/7)
        self.assertAlmostEqual(report['policies']['selected']['pass_at_1']['group_ess'],(8+4)**2/(8**2+4**2))
        rows[0]['metrics']['pass_at_1']=None
        report=_statistical_report(rows,plan)
        self.assertIsNone(report['policies']['selected']['pass_at_1']['estimate'])
        self.assertIsNone(report['selected_minus_baseline']['baseline']['pass_at_1']['interval'])

    def test_unknown_identity_retains_observations_but_suppresses_formal_metrics(self):
        class Unknown(ScriptedActor):
            def _propose(self,*a,**k):
                response=super()._propose(*a,**k)
                return replace(response,fingerprint=None) if response.action['type']=='operate' else response
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            result=self.collect(root,Unknown)
            self.assertEqual(result['observed_outcomes'],13)
            self.assertFalse(result['evaluation_valid'])
            self.assertFalse(result['identity_complete'])
            for arm in result['statistics']['policies'].values():
                self.assertIsNone(arm['pass_at_1']['estimate'])
                self.assertIsNotNone(arm['pass_at_1']['assigned_mean'])

    def test_pending_is_preserved_with_full_denominator_and_no_retry(self):
        class Timeout(ScriptedActor):
            def propose_target(self,*a,**k):
                raise TimeoutError('unknown server response')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            result=self.collect(root,Timeout)
            self.assertEqual(result['planned_tasks'],13)
            self.assertEqual(result['pending_tasks'],1)
            self.assertFalse(result['execution_complete'])
            self.assertFalse(result['evaluation_valid'])
            self.assertTrue(any(r['status']=='unattempted' for r in result['tasks']))
            never=lambda *a,**k: (_ for _ in ()).throw(AssertionError('retry'))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                collect_final_evaluation(root/'final.plan',env_factory=never,actor_factory=never,encoder=None)
            for arm in result['statistics']['policies'].values():
                self.assertIsNone(arm['pass_at_1']['estimate'])

    def test_zero_remaining_budget_and_empty_memory_still_complete_tasks(self):
        for kwargs in ({'study':StudySpec(max_steps=4,max_model_calls=1)},{'empty_bank':True}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root,**kwargs)
                result=self.collect(root)
                self.assertTrue(result['evaluation_valid'])
                self.assertEqual(result['observed_outcomes'],13)
                self.assertTrue(all(r['arm']=='silent' for r in result['tasks']))
                self.assertTrue(all(r['decision']['effective_probability']==0 for r in result['tasks']))
                if 'study' in kwargs:
                    self.assertTrue(all(r['costs']['model_calls']==1 for r in result['tasks']))

    def test_no_checkpoint_preserves_whole_task_outcomes_and_zero_call_scope(self):
        class NoSteps(NoncloneableFixture):
            def configuration(self):
                return {**super().configuration(),'fixture_initial_steps':0}
            def reset(self,seed=0):
                return replace(super().reset(seed),remaining_steps=0)
        factory=lambda task:NoSteps(task_id=task['task_id'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,final_env_factory=factory)
            result=collect_final_evaluation(root/'final.plan',env_factory=factory,actor_factory=ScriptedActor,encoder=EncoderFixture())
            self.assertEqual(result['observed_outcomes'],13)
            self.assertTrue(result['evaluation_valid'])
            self.assertEqual(result['served_identity']['status'],'not_required_no_actor_calls')
            self.assertEqual(result['served_identity']['zero_call_policy_exemptions'],2)
            self.assertTrue(all(r['decision'] is None for r in result['tasks']))
            self.assertTrue(all(r['metrics']['checkpoint_coverage']==0 for r in result['tasks']))

    def test_changed_registered_environment_refused_before_any_actor_request(self):
        class Changed(NoncloneableFixture):
            def configuration(self):
                return {**super().configuration(),'revision':'changed'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch.object(ScriptedActor,'_propose',side_effect=AssertionError('actor request')):
                with self.assertRaisesRegex(ValueError,'changed after preregistration'):
                    collect_final_evaluation(root/'final.plan',env_factory=lambda t:Changed(task_id=t['task_id']),actor_factory=ScriptedActor,encoder=EncoderFixture())
            report=summarize_final_evaluation(root/'final.plan')
            self.assertEqual(report['attempted_tasks'],0)
            self.assertFalse(report['evaluation_valid'])

    def test_raw_test_role_and_assignment_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.fixture(root);self.collect(root)
            path=_policy_directory(plan,plan['policies'][0]['policy_id'])/'run.json'
            original=read_json(path);bad=copy.deepcopy(original);bad['role']='validation';write_json(path,seal(bad))
            with self.assertRaisesRegex(ValueError,'role'):
                summarize_final_evaluation(root/'final.plan')
            write_json(path,original)
            assignment=next(path.parent.glob('sessions/*/assignment.json'))
            bad=read_json(assignment);bad['arm']='remind' if bad['arm']=='silent' else 'silent';write_json(assignment,seal(bad))
            with self.assertRaisesRegex(ValueError,'decision was changed'):
                summarize_final_evaluation(root/'final.plan')

    def test_cli_report_is_readonly_and_repeated_collection_avoids_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);self.collect(root)
            with patch('cave_mem_v3.cli._actor',side_effect=AssertionError('provider')),patch('cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder',side_effect=AssertionError('encoder')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['report-final-evaluation','--plan',str(root/'final.plan')]),0)
                self.assertEqual(main(['collect-final-evaluation','--plan',str(root/'final.plan'),'--config','missing','--encoder-path','missing','--encoder-revision','none']),0)

    def test_cli_incomplete_collection_returns_nonzero(self):
        with patch('cave_mem_v3.cli._collect_final_evaluation',return_value={'evaluation_valid':False,'pending_tasks':1}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['collect-final-evaluation','--plan','unused','--config','unused','--encoder-path','unused','--encoder-revision','unused']),2)
        with patch('cave_mem_v3.cli._collect_conservative_selection',return_value={'status':'incomplete'}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['collect-conservative-selection','--plan','unused','--config','unused','--encoder-path','unused','--encoder-revision','unused']),2)

    def test_duplicate_bundle_and_unknown_lock_cannot_create_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.fixture(root)
            chosen=plan['selection_lock']['winner']['bundle']['policy_config']['kind']
            with self.assertRaisesRegex(ValueError,'duplicate'):
                create_final_evaluation_plan(root/'selection.lock',[{'policy_id':'baseline','bundle':str(root/(chosen+'.bundle.json'))}],
                    config=FinalEvaluationConfig({'selected':.5,'baseline':.5}),output=root/'duplicate.plan',run_dir=root/'duplicate.run',report_path=root/'duplicate.report',env_factory=env_factory,actor_factory=ScriptedActor)
            bad=copy.deepcopy(plan);bad['selection_lock']['status']='no_selection';bad['selection_lock']=seal(bad['selection_lock']);bad['selection_lock_hash']=bad['selection_lock']['record_hash'];write_json(root/'final.plan',seal(bad))
            with self.assertRaisesRegex(ValueError,'qualified selected'):
                load_final_evaluation_plan(root/'final.plan')
