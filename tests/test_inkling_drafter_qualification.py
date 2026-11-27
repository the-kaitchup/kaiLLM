import importlib.util
import copy
import sys
import unittest
from pathlib import Path


path = Path(__file__).resolve().parents[1] / 'benchmarks/long_context/verify_inkling_drafter.py'
sys.path.insert(0,str(path.parent))
spec = importlib.util.spec_from_file_location('verify_inkling', path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def capture(proposed, accepted, slots=None):
    row = {'prompt_hash': 'same prompts', 'output_ids': {'prose': list(range(48)), 'code': list(range(48))},
           'counts': {'draft_tokens': proposed, 'accepted_tokens': accepted}}
    if slots is not None:
        row['client_concurrency'] = slots
    return row


class QualificationTests(unittest.TestCase):
    def test_single_request_qualification(self):
        result = verify.compare(capture(0, 0, 1), capture(55, 27, 1), capture(100, 0, 1))
        self.assertEqual(result['client_concurrency'], 1)
        self.assertEqual(result['matched_tokens'], {'prose': 48, 'code': 48})

    def test_existing_two_request_captures(self):
        result = verify.compare(capture(0, 0), capture(55, 27, 2), capture(100, 0))
        self.assertEqual(result['matched_tokens'], {'prose': 48, 'code': 48})
        self.assertNotIn('client_concurrency', result)

    def test_wide_concurrency_requires_each_physical_target_and_verified_draft(self):
        from check_inkling_draft_overlap import overlap
        def wide(slots, native, accepted):
            outputs = {label:list(range(48)) for label in verify.request_labels(slots)}
            proof = {}
            for domain in ('prose', 'code'):
                hashes=[str(i)+domain for i in range(slots)]
                ids = [str(i) for i in range(slots)]
                events = [{'admissions':list(zip(ids,hashes)), 'scheduled':[(i,128) for i in ids]},
                          {'admissions':[], 'scheduled':[(i,6 if native else 1) for i in ids]}]
                if native:
                    events[1]['draft_widths'] = [(i,5) for i in ids]
                proof[domain] = {'prompt_sha256':hashes, 'prompt_length':128, 'events':events,
                                 'overlap':overlap(events,hashes,[128]*slots,native)}
            return {**capture(120 if native else 0,accepted,slots), 'output_ids':outputs,'physical_schedule':proof}
        for slots in (4,8,16,32):
            controls = [wide(slots,False,0), wide(slots,True,15), wide(slots,True,0)]
            result = verify.compare(*controls)
            self.assertEqual(result['client_concurrency'], slots)
            self.assertEqual(len(result['matched_tokens']), 2*slots)
            bad = copy.deepcopy(controls[1])
            case=bad['physical_schedule']['code']
            event=case['events'][1]
            event['scheduled'].pop()
            event['draft_widths'].pop()
            case['overlap']=overlap(case['events'], case['prompt_sha256'], [128]*slots, True)
            with self.assertRaisesRegex(ValueError,'requested decode width'):
                verify.compare(controls[0], bad, controls[2])
            short=copy.deepcopy(controls[1])
            short['output_ids'].pop('code-1')
            with self.assertRaisesRegex(ValueError,'48 output token IDs'):
                verify.compare(controls[0], short, controls[2])

    def test_shared_tokens_do_not_qualify_different_batch_sizes(self):
        with self.assertRaisesRegex(ValueError, 'same client concurrency'):
            verify.compare(capture(0, 0, 1), capture(55, 27, 2), capture(100, 0, 1))


if __name__ == '__main__':
    unittest.main()
