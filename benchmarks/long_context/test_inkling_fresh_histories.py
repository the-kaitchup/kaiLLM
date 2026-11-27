import unittest

from check_inkling_native_graphs import qualifying_histories

class FreshHistories(unittest.TestCase):
    def setUp(self):
        self.bank={'domains':{d:{'rows':[[i,0,2,3] for i in range(33)]} for d in ('prose','code')}}
    def test_both_have_separate_proofs_and_prompts(self):
        samples=list(qualifying_histories(self.bank,(1,4),1))
        self.assertEqual([s[0] for s in samples],[1,4,'1-row-1','4-row-1'])
        self.assertIs(samples[0][2],self.bank)
        for d in ('prose','code'):
            self.assertEqual(samples[-1][2]['domains'][d]['rows'][0],self.bank['domains'][d]['rows'][1])
        self.assertEqual(len(list(qualifying_histories(self.bank,(1,4)))),2)
    def test_duplicate_prefix_and_row_zero(self):
        for row in (0,33):
            with self.assertRaises(ValueError):list(qualifying_histories(self.bank,(1,4),row))
        self.bank['domains']['code']['rows'][3][0]=0
        with self.assertRaises(ValueError):list(qualifying_histories(self.bank,(1,4),3))
