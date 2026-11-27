import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

BENCH = Path(__file__).resolve().parents[1] / 'benchmarks/long_context'
sys.path.insert(0, str(BENCH))
from check_inkling_draft_overlap import MANIFEST, overlap, required_overlap, validate_launch
from verify_batch_schedule import append_step
from benchmark import dspark_asset, inkling_draft_runtime, llama_revision
from weights import checkpoints


class TestDraftOverlap(unittest.TestCase):
    def event(self, new=(), work=(), proposals=()):
        return {'admissions': list(new), 'scheduled': list(work), 'draft_widths': list(proposals)}

    def test_mixed_and_joint_are_separate_target_forwards(self):
        rows = [self.event([('a','x')],[('a',3)]),
                self.event([('b','y')],[('a',6),('b',7)],[('a',5)]),
                self.event(work=[('a',4),('b',2)],proposals=[('a',3),('b',1)])]
        result = overlap(rows,['x','y'],[3,7],True)
        self.assertEqual(result,dict(mixed_steps=1,joint_steps=1,scheduled_draft_tokens=[8,1],
                                     joint_scheduled_token_counts={'6':1},joint_five_proposal_steps=0))
        rows.append(self.event(work=[('a',6),('b',6)],proposals=[('a',5),('b',5)]))
        wide=overlap(rows,['x','y'],[3,7],True)
        self.assertEqual(wide['joint_scheduled_token_counts'],{'6':1,'12':1})
        self.assertEqual(wide[required_overlap('joint',True)],1)
        self.assertEqual(result[required_overlap('joint',True)],0)
        self.assertEqual(result[required_overlap('joint',False)],1)
        with self.assertRaises(ValueError):
            overlap(rows,['x','y'],[3,7],False)
        rows[1]['draft_widths'].append(('b',1))
        with self.assertRaisesRegex(ValueError,'before'):
            overlap(rows,['x','y'],[3,7],True)

    def test_agreeing_outputs_alone_cannot_certify_joint_verification(self):
        rows = [self.event([('a','x'),('b','y')],[('a',3),('b',3)]),
                self.event(work=[('a',6)],proposals=[('a',5)]),self.event(work=[('b',6)],proposals=[('b',5)])]
        summary=overlap(rows,['x','y'],[3,3],True)
        self.assertEqual(summary['joint_steps'],0)
        self.assertEqual(summary['mixed_steps'],0)
        rows[-1]['draft_widths']=[('b',6)]
        with self.assertRaisesRegex(ValueError,'width'):
            overlap(rows,['x','y'],[3,3],True)

    def test_observer_keeps_actual_draft_widths_per_request(self):
        source=SimpleNamespace(num_scheduled_tokens={'left':6,'right':3},total_num_scheduled_tokens=9,
                               scheduled_new_reqs=[],scheduled_spec_decode_tokens={'left':[11]*5,'right':[17]*2})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'steps.jsonl';append_step(source,path)
            record=json.loads(path.read_bytes());self.assertEqual([r[1] for r in record['draft_widths']],[5,2])
            self.assertEqual([r[0] for r in record['draft_widths']],[r[0] for r in record['scheduled']])
            self.assertNotIn('left',path.read_text())

    def test_downloads_native_placements_and_checks_follow_target_drafter(self):
        target = {'family':'inkling', 'revision':'current', 'repo':'publisher/target', 'dspark':'publisher-head',
                  'files':{'wt.gguf':{'sha256':'1'*64}}, 'kai_dspark_batches':[1]}
        draft = {'repo':target['repo'], 'revision':'current', 'files':{'head.gguf':{'sha256':'2'*64}}}
        old_draft = {**draft, 'revision':'older', 'files':{'head.gguf':{'sha256':'3'*64}}}
        placement = {'model_revision':'current','hardware_id':'one-card','gpu':'Card','matrix_gpu':'Card',
                     'batch':1,'revision':'engine','source':'source'}
        with patch.dict(MANIFEST, models={'new-target':target}, drafters={'publisher-head':draft},
                        drafter_history={'publisher-head':{'older':old_draft}}):
            self.assertEqual(checkpoints(MANIFEST)['publisher-head'], draft)
            self.assertEqual(dspark_asset({**target, 'revision':'older'}), ('publisher-head',old_draft))
            self.assertEqual(dspark_asset({}), ('dspark', MANIFEST['dspark']))
            self.assertEqual(llama_revision({**target, 'llama_revision_override':'new-stock'}), 'new-stock')
            self.assertEqual(llama_revision(target), llama_revision('inkling'))
            for token in (None,'3'*64,'2'*64):
                target['kai_dspark_placements'] = [{**placement,'drafter_sha256':token}]
                pin = inkling_draft_runtime(target,1,gpu_names=['Card'],hardware_id='one-card')
                self.assertEqual(pin, ('engine','source') if token=='2'*64 else None)
                self.assertIsNone(inkling_draft_runtime(target,2,gpu_names=['Card'],hardware_id='one-card'))
            argv=['--max-num-seqs','2','--max-model-len','73728','--speculative-config','']
            identity={'arguments':argv,'humming_deterministic':'1','checkpoint_sha256':'1'*64}
            for token in ('2'*64,'3'*64):
                argv[-1]=json.dumps({'method':'dflash','model':'/cache/'+token+'-201024-v2','num_speculative_tokens':5})
                if token=='2'*64:
                    validate_launch('native', identity)
                else:
                    with self.assertRaisesRegex(ValueError,'pinned DSpark'):
                        validate_launch('native', identity)

    def test_final_duplicate_flag_controls_rejection_policy(self):
        digest=next(iter(MANIFEST['dspark']['files'].values()))['sha256']
        spec={'method':'dflash','model':'/cache/'+digest+'-v2','num_speculative_tokens':5}
        argv=['--max-num-seqs','2','--max-model-len','73728','--speculative-config',json.dumps(spec)]
        identity={'arguments':argv,'humming_deterministic':'1',
                  'checkpoint_sha256':next(iter(MANIFEST['models']['inkling-loop-iq4']['files'].values()))['sha256']}
        validate_launch('native',identity)
        forced=dict(spec,rejection_sample_method='synthetic',synthetic_acceptance_rates=[0.0]*5)
        argv.extend(['--speculative-config',json.dumps(forced)])
        validate_launch('forced',identity)
        with self.assertRaises(ValueError):validate_launch('native',identity)


if __name__ == '__main__':
    unittest.main()
