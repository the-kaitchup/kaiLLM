"""Prove independent B1 pins cannot authorize another source, board, donor, slot or numerical policy."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from test_long_context_inkling_native import graph_proof, head_proof, draft_graph_proof
from benchmark import MANIFEST,full_graph_runtime,inkling_draft_runtime,runtime_pin
from devices import physical_id
from report import verify_kai_native_record

class PlacementTest(unittest.TestCase):
    def test_plain_dondon_s_pin_is_limited_to_its_profiled_board_and_batch(self):
        boards = {
            '8732d77ace5c7eb426ac18161f3fe79c799a2bcf': 'NVIDIA GeForce RTX 5090',
            '8a0cf56f7571b094122f76e171efca6f59b0c137': 'NVIDIA GeForce RTX 5080',
        }
        for source, gpu in boards.items():
            with self.subTest(gpu=gpu):
                entry = next(pin for pin in MANIFEST['kaillm']['full_graph_runtimes']
                             if pin['revision'] == source and pin['models'] == ['inkling-dondon-s'])
                layout = dict(batch=1, gpu_names=[gpu], hardware_id=entry['hardware_id'])
                self.assertEqual(full_graph_runtime('inkling-dondon-s', 'plain', source, **layout),
                                 (source, entry['source']))
                self.assertIsNone(full_graph_runtime('inkling-dondon-s', 'dspark5', source, **layout))
                self.assertIsNone(full_graph_runtime('inkling-dondon-m', 'plain', source, **layout))
                for change in ({'batch': 2}, {'hardware_id': 'different'}, {'hardware_id': None}):
                    self.assertIsNone(full_graph_runtime('inkling-dondon-s', 'plain', source, **(layout | change)))

    def test_independent_donors_and_same_source_plain_and_native(self):
        head=MANIFEST['dspark']
        name,draft=next(iter(head['files'].items()))
        for key,short,gpu in (('inkling-udq2','5080','NVIDIA GeForce RTX 5080'),('inkling-loop-iq4','PRO 6000','NVIDIA RTX PRO 6000 Blackwell Workstation Edition')):
            model=MANIFEST['models'][key];board=physical_id([short])
            pin=('candidate-'+short,'lock-'+short)
            p=dict(revision=pin[0],source=pin[1],gpu=gpu,matrix_gpu=short,model_revision=model['revision'],hardware_id=board,batch=1,
                   kv_mib=3764,plain_kv_mib=3136,serial_verify_combine=True,verify_combine_fma4='scaled-fp32',
                   capture_sizes=[1,2,4,6,128,2048],plain_capture_sizes=[1,128,2048],qualification_sha256='e'*64)
            weights=[s['sha256'] for s in model['files'].values()]
            with patch.dict(model,kai_dspark_placements=[p]):
                for mode in ('plain','dspark5'):
                    def full(revision=pin[0],gpu_name=gpu,batch=1,hardware=board):
                        return full_graph_runtime(key,mode,revision,batch=batch,gpu_names=[gpu_name],hardware_id=hardware)
                    self.assertEqual(full(),pin)
                    self.assertIsNone(full(gpu_name='other'))
                    self.assertIsNone(full(batch=2))
                    self.assertIsNone(full(hardware='another board'))
                    self.assertIsNone(full(revision='wrong'))
                    self.assertIsNone(full_graph_runtime(key,mode,pin[0]))
                    self.assertIsNone(inkling_draft_runtime(model,1,gpu_names=[gpu],revision='other-source',hardware_id=board))
                    native=mode=='dspark5'
                    row=dict(engine='kaiLLM',checkpoint=key,model_revision=model['revision'],gpu=[gpu],batch=1,mode=mode,hardware_id=board,
                             kai_source=pin[0],engine_revision=pin[1],kai_humming_deterministic=True,llama_source=MANIFEST['llama_cpp']['inkling_revision'],
                             sha256=weights,measured_gguf_sha256=weights,kai_inkling_serial_verify_combine=True,kai_inkling_verify_combine_fma4='scaled-fp32',
                             kai_inkling_qualification_sha256='e'*64,kai_inkling_capture_sizes=p['capture_sizes' if native else 'plain_capture_sizes'],
                             kai_kv={'bytes_per_device':p['kv_mib' if native else 'plain_kv_mib']<<20},cuda_graphs=graph_proof(),target_head_cuda_graphs=head_proof())
                    if native:
                        row.update(measured_drafter_sha256=draft['sha256'],drafter_cuda_graphs=draft_graph_proof(),
                                   drafter=dict(repo=head['repo'],revision=head['revision'],filename=name,sha256=draft['sha256']),
                                   kai_dspark=dict(scope='full-wave',proposed=100,accepted=10,rejected=90))
                    verify_kai_native_record(row)
                    for change in (dict(hardware_id='other'),dict(kai_inkling_serial_verify_combine=1),dict(kai_inkling_verify_combine_fma4='unscaled-bf16'),
                                   dict(kai_inkling_qualification_sha256=None),dict(kai_humming_deterministic=False),dict(measured_gguf_sha256=[]),
                                   dict(kai_kv={'bytes_per_device':123}),dict(batch=2),dict(kai_inkling_capture_sizes=[])):
                        with self.subTest(key=key,mode=mode,change=change),self.assertRaises(ValueError):
                            verify_kai_native_record(dict(row,**change))
                if key=='inkling-loop-iq4':
                    historical=runtime_pin(model['kai_dspark_runtime'])
                    self.assertEqual(inkling_draft_runtime(model,1,gpu_names=[gpu],revision=historical[0],hardware_id=board),historical)
                    self.assertEqual(inkling_draft_runtime(model,1,gpu_names=['NVIDIA GeForce RTX 3090'],revision=historical[0]),historical)
                for historical_revision in MANIFEST.get('model_history',{}).get(key,{}):
                    self.assertIsNone(full_graph_runtime(key,'plain',pin[0],batch=1,gpu_names=[gpu],
                                                         hardware_id=board,model_revision=historical_revision))
                otherkey='inkling-loop-iq4' if key=='inkling-udq2' else 'inkling-udq2'
                self.assertIsNone(full_graph_runtime(otherkey,'dspark5',pin[0],batch=1,gpu_names=[gpu],hardware_id=board))
                old=deepcopy(model);old['revision']='changed-checkpoint'
                self.assertIsNone(inkling_draft_runtime(old,1,gpu_names=[gpu],revision=pin[0],hardware_id=board))
