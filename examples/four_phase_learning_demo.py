"""Four-phase fitted software closure. No external model or benchmark requests.

The two public toy conditions deliberately exercise both beneficial and harmful
reminders. Fixed illustrative settings below are software inputs, not research
thresholds chosen from these synthetic outcomes. Weights are always fitted from
collected, audited raw events; no model artifact or label is hand edited.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, canonical, digest, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment, EnvStep
from cave_mem_v3.memory_generation.collection import collect_history, CollectionConfig
from cave_mem_v3.memory_generation.candidates import create_candidate
from cave_mem_v3.memory_generation.replay import verify_candidate, ReplaySpec
from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.interventions.single import collect_single_target
from cave_mem_v3.policy.dataset import write_policy_dataset
from cave_mem_v3.policy.supervised import fit_supervised, SupervisedConfig
from cave_mem_v3.policy.calibration import calibrate_supervised, CalibrationConfig
from cave_mem_v3.policy.gate import fit_initial_gate, GateConfig
from cave_mem_v3.policy.bandit import train_bandit, BanditConfig
from cave_mem_v3.policy.dr import estimate_single_dr, DRConfig
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.conservative_selection import create_conservative_selection_plan, collect_conservative_selection_plan, select_conservative_policy, load_conservative_selection_lock
from cave_mem_v3.policy.selection import SelectionConfig
from cave_mem_v3.policy.final_evaluation import create_final_evaluation_plan, FinalEvaluationConfig, collect_final_evaluation, export_final_evaluation_report
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard


class LearningFixture(PreparationEnvironment):
    """A fixed hidden mode makes preparation helpful or harmful in this toy task."""
    def __init__(self, *, operation_mode='normal', **kwargs):
        super().__init__(**kwargs)
        if operation_mode not in ('normal','waiver_fragile'):
            raise ValueError('unsupported learning fixture operation mode')
        self.operation_mode = operation_mode

    def configuration(self):
        return {**super().configuration(),'learning_fixture_revision':'public-two-conditions.v2',
                'operation_mode':self.operation_mode}

    def step(self, action, seed=0):
        state=self._require()
        valid,_=self.validate_action(action)
        if (valid and self.operation_mode=='waiver_fragile' and self.requires_preparation
                and state['phase']=='ready' and not state['prepared']
                and action['type'] in ('operate','prepare')):
            state['steps']+=1
            if action['type']=='prepare' or state['steps']>=self.max_steps:
                state.update(phase='failed',terminal=True,success=False)
            else:
                state['phase']='operated'
            state['history'].append({'action':{'type':action['type']},
                                     'observation':f"phase={state['phase']}; prepared={state['prepared']}"})
            return EnvStep(self.observe(),state['terminal'],state['success'])
        return super().step(action,seed=seed)

    def finalize(self):
        if not self.terminal_status()[0]:
            self._state.update(phase='failed',terminal=True,success=False)
        return EnvStep(self.observe(),True,self._state['success'])


class LearningActor(ScriptedActor):
    """Toy failure mechanism: unnecessary reminders sometimes trigger early finish."""
    def configuration(self):
        return {**super().configuration(),'actor':'scripted-two-condition-learning.v2'}

    def _propose(self,context,**kwargs):
        response=super()._propose(context,**kwargs)
        action=response.action
        if kwargs.get('reminder') and context.facts.get('requires_preparation') is False:
            import random
            action={'type':'finish' if random.Random(kwargs.get('seed', 0)).random() < .5 else 'operate'}
        return replace(response,action=action,text=canonical(action),model='scripted-learning.v2',fingerprint='scripted-learning-seeded.v2')


class LearningEncoder:
    """Deterministic public-text fixture columns; never a learned semantic encoder."""
    identity={'kind':'offline-learning-fixture-encoder','revision':'public-text.v1','research_role':'fixture_only'}
    def encode_query(self,text):
        return [1.,float('preparation_required=True' in text),float('preparation_required=False' in text),float('"type":"prepare"' in text)]
    def encode_documents(self,texts):
        return [self.encode_query(t) for t in texts]


def environment(task):
    return LearningFixture(task_id=task['task_id'],requires_preparation=task.get('fixture_condition','benefit')=='benefit',
                           operation_mode=task.get('fixture_operation_mode','normal'),max_steps=4)


def _hashes(paths):
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for directory in paths for p in sorted(Path(directory).rglob('*.json'))}


def run_demo(output):
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('use a new output directory; failed and completed demo evidence is preserved')
    output.mkdir(parents=True)
    spec=StudySpec(study_id='four-phase-software-fixture',max_steps=4,max_model_calls=4,base_seed=42)
    hazard=RiskHazard(low_risk_probability=.999999,high_risk_probability=.999999)
    tasks=[{'task_id':'history-benefit','group_id':'history-only','role':'history','fixture_condition':'benefit'}]
    for role,subrole,n in [('target_train',None,24),('validation','calibration',12),('validation','selection',8),('test',None,12)]:
        for i in range(n):
            prefix=subrole or role
            # Names denote public conditions, not each task's observed reminder effect.
            condition='benefit' if i%2==0 else 'harm'
            task_id=prefix+'-'+str(i)
            operation_mode=('waiver_fragile' if condition=='benefit'
                            and int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:2],'big')%32==0
                            else 'normal')
            task={'task_id':prefix+'-'+str(i),'group_id':prefix+'-group-'+str(i),'role':role,
                  'family':condition,'mechanism':'software-two-condition','fixture_condition':condition,
                  'fixture_operation_mode':operation_mode}
            if subrole:
                task['validation_role']=subrole
            tasks.append(task)
    manifest=build_manifest(tasks,benchmark='four-phase-software-fixture',revision='offline-learning.v1')
    write_json(output/'manifest.json',manifest)
    collect_history(manifest,CollectionConfig(max_runs=1),spec,environment,LearningActor,output/'history')
    trace=read_json(next((output/'history/traces').glob('*.json')))
    candidate=create_candidate(trace,0,mechanism='Toy operation needs public preparation requirement.',correction={'type':'prepare'},
        positive_conditions=[{'field':'requires_preparation','equals':True}],exclusion_conditions=[{'field':'prepared','equals':True}],
        scope='Explicit software fixture only.',reviewer='offline-fixture-author')
    write_json(output/'candidate.json',candidate)
    evidence=verify_candidate(candidate,trace,lambda:environment(tasks[0]),LearningActor,study=spec,replay_spec=ReplaySpec(replay_pairs=2),output_dir=output/'verification')
    bank=freeze_bank([build_memory(candidate,trace,evidence)],output/'bank.json',source_traces=[trace])
    memory_id=bank['memories'][0]['memory_id']
    rule={'schema_version':'action-exposure-rules.v1','frozen':True,'rules':{memory_id:{
        'schema_version':'action-exposure-rule.v1','confidence':'high',
        'exposed_when':{'action':[{'field':'type','equals':'operate'}],'context':[]},
        'not_exposed_when':{'action':[{'field':'type','not_equals':'operate'}],'context':[]}}}}
    rule['rules_hash']=digest(rule)
    write_json(output/'exposure.json',rule)
    source_dirs=[output/'history',output/'verification']
    for label,role,subrole in [('train','target_train',None),('calibration','validation','calibration')]:
        prep,pairs=output/(label+'-prepared'),output/(label+'-paired')
        collect_target(manifest,role=role,validation_role=subrole,spec=spec,env_factory=environment,actor_factory=LearningActor,
            output_dir=prep,bank=bank,encoder=LearningEncoder(),hazard=hazard,exposure_rules=rule,prepare_before_action=True)
        collect_target_pairs(prep,bank,env_factory=environment,actor_factory=LearningActor,output_dir=pairs)
        write_policy_dataset([{'kind':'paired','run':str(pairs),'prepared':str(prep)}],output/(label+'.json'),role=role,validation_role=subrole)
        source_dirs.extend((prep,pairs))
    original_hashes=_hashes(source_dirs)
    model=fit_supervised(output/'train.json',config=SupervisedConfig(gamma=1.0,max_iterations=2000),output=output/'model.json')
    calibration=calibrate_supervised(output/'model.json',output/'calibration.json',config=CalibrationConfig(alpha=.2,minimum_groups=2,risk_bins=2),output=output/'bounds.json')
    gate=fit_initial_gate(output/'model.json',output/'bounds.json',output/'train.json',config=GateConfig(.9),output=output/'gate.json')
    paired_bandit=train_bandit(output/'train.json',output/'model.json',output/'bounds.json',output/'gate.json',
        config=BanditConfig('paired',1,5,iterations=40,bootstrap_samples=30),output=output/'paired-bandit.json')
    collect_single_target(manifest,bank=bank,spec=spec,env_factory=environment,actor_factory=LearningActor,output_dir=output/'single-train',
        encoder=LearningEncoder(),hazard=hazard,exposure_rules=rule,assignment_probability=.5)
    single_hashes=_hashes([output/'single-train'])
    write_policy_dataset([{'kind':'single','run':str(output/'single-train')}],output/'single.json')
    dr=estimate_single_dr(output/'single.json',config=DRConfig(.5),output=output/'dr.json')
    single_bandit=train_bandit(output/'single.json',output/'model.json',output/'bounds.json',output/'gate.json',
        config=BanditConfig('single',1,5,iterations=40,bootstrap_samples=30),dr_path=output/'dr.json',output=output/'single-bandit.json')
    if any(h['status']!='fitted' for h in model['heads'].values()) or gate['head']['status']!='fitted' or paired_bandit['status']!='candidate' or single_bandit['status']!='candidate':
        raise ValueError('fixture learned fit unavailable; preserve evidence rather than substitute hand-filled weights')
    candidates=[]
    for identifier,kind,artifact in [('supervised','supervised',None),('paired-bandit','bandit','paired-bandit.json'),
                                      ('single-bandit','bandit','single-bandit.json'),('no-memory','no_memory',None)]:
        learned=kind!='no_memory'
        path=output/(identifier+'.bundle.json')
        freeze_policy(manifest,bank,actor_config=LearningActor().configuration(),encoder_identity=LearningEncoder.identity,study=spec,hazard=hazard,
            exposure_rules=rule,policy_config={'kind':kind,**({'risk_threshold':.9,'minimum_advantage':0} if learned else {})},
            model_path=output/'model.json' if learned else None,calibration_path=output/'bounds.json' if learned else None,
            warm_gate_path=output/'gate.json' if learned else None,bandit_path=output/artifact if artifact else None,output=path)
        candidates.append({'candidate_id':identifier,'bundle':str(path),'rollout_dir':str(output/(identifier+'-selection'))})
    create_conservative_selection_plan(candidates,lock_path=output/'selection.lock',config=SelectionConfig(1,5,bootstrap_samples=30),output=output/'selection.plan')
    collect_conservative_selection_plan(output/'selection.plan',env_factory=environment,actor_factory=LearningActor,encoder=LearningEncoder())
    selected=select_conservative_policy(output/'selection.plan')
    if selected['status']!='selected':
        raise ValueError('fixture had no qualified selected policy; preserve evidence rather than changing thresholds')
    if selected['winner']['bundle']['policy_config']['kind']=='no_memory':
        raise ValueError('fixture selected ordinary actor; preserve outputs, do not force a learned winner')
    create_final_evaluation_plan(output/'selection.lock',[{'policy_id':'no-memory','bundle':str(output/'no-memory.bundle.json')}],
        config=FinalEvaluationConfig({'selected':.5,'no-memory':.5},bootstrap_samples=30),output=output/'final.plan',
        run_dir=output/'final-run',report_path=output/'final.report',env_factory=environment,actor_factory=LearningActor)
    final=collect_final_evaluation(output/'final.plan',env_factory=environment,actor_factory=LearningActor,encoder=LearningEncoder())
    if final!=export_final_evaluation_report(output/'final.plan') or selected!=load_conservative_selection_lock(output/'selection.lock'):
        raise AssertionError('fixture read-only reconstruction changed')
    if original_hashes!=_hashes(source_dirs) or single_hashes!=_hashes([output/'single-train']):
        raise AssertionError('original source or paired evidence changed')
    learned_decisions=[r['decision'] for r in final['tasks'] if r['policy_id']=='selected' and r.get('decision')]
    from collections import Counter
    summary={'fixture_only':True,'external_model_requests':0,'research_parameters_selected':False,
        'model_heads':{k:v['status'] for k,v in model['heads'].items()},'gate_status':gate['head']['status'],
        'paired_bandit_status':paired_bandit['status'],'single_bandit_status':single_bandit['status'],
        'dr_summary':dr['summary'],'selected_candidate':selected['winner']['candidate_id'],
        'selected_kind':selected['winner']['bundle']['policy_config']['kind'],'final_evaluation_valid':final['evaluation_valid'],
        'final_planned_tasks':final['planned_tasks'],'final_observed_outcomes':final['observed_outcomes'],
        'final_checkpoint_coverage':final.get('checkpoint_coverage'),'original_evidence_unchanged':True,
        'learned_final_decisions':len(learned_decisions),
        'learned_final_decision_reasons':dict(Counter(d['reason'] for d in learned_decisions)),
        'learned_final_effective_probability_range':[min(d['effective_probability'] for d in learned_decisions),max(d['effective_probability'] for d in learned_decisions)] if learned_decisions else None,
        'learned_final_reminder_assignments':sum(d['arm']=='remind' for d in learned_decisions),
        'claim_boundary':'Toy public conditions exercise software fitting and execution, never real research effectiveness.'}
    write_json(output/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/four-phase-learning-demo')
    args=parser.parse_args(argv)
    result=run_demo(args.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
