import unittest
from decimal import Decimal as D
from global_agent.global_sim import run, ACCOUNTS

def frame(day, price='100', target='2000'):
    return {'at':f'2026-11-{day:02d}T15:00:00Z',
            'targets_generated_at':f'2026-11-{day:02d}T14:00:00Z',
            'synthetic_quotes':True,
            'quotes':{'fixture:ETF':{'bid':price,'ask':str(D(price)+1),'observed_at':f'2026-11-{day:02d}T15:00:00Z'}},
            'targets':{a:{'fixture:ETF':'45000' if a.startswith('E:') else target} for a in ACCOUNTS}}

def scenario(*frames):
    return {'synthetic_only':True,'real_trading':False,'commission_bps':'10',
            'slippage_bps':'10','minimum_fee_pln':'5','frames':list(frames)}

class GlobalSimTests(unittest.TestCase):
    def test_buy_sell_benchmark(self):
        r=run(scenario(frame(25),frame(26,target='0')))
        self.assertFalse(r['real_trading'])
        self.assertTrue(r['isolated_temporary_ledger'])
        self.assertEqual(len(r['executions']),9)
        for a in r['accounts'][:4]:
            self.assertEqual(a['positions'],{})
            self.assertEqual(a['fills'],2)
            self.assertLess(D(a['cash_pln']),D(50000))
        self.assertEqual(r['accounts'][4]['fills'],1)
        self.assertTrue(r['accounts'][4]['positions'])
        self.assertTrue(all(D(x['quantity'])%1==0 for x in r['executions']))
    def test_deterministic(self):
        s=scenario(frame(25),frame(26,price='101'))
        self.assertEqual(run(s),run(s))
    def test_invalid_chronology(self):
        with self.assertRaises(ValueError): run(scenario(frame(25),frame(25)))
        f=frame(25);f['targets_generated_at']=f['at']
        with self.assertRaises(ValueError): run(scenario(f))
    def test_stale_quote(self):
        f=frame(25);f['quotes']['fixture:ETF']['observed_at']='2026-11-25T14:00:00Z'
        with self.assertRaises(ValueError): run(scenario(f))
    def test_fixture_only(self):
        f=frame(25);f['synthetic_quotes']=False
        with self.assertRaises(ValueError): run(scenario(f))
    def test_target_caps(self):
        with self.assertRaises(ValueError): run(scenario(frame(25,target='5000')))
    def test_missing_held_quote(self):
        a,b=frame(25),frame(26)
        b['quotes']={};b['targets']={x:{} for x in ACCOUNTS}
        with self.assertRaisesRegex(ValueError,'Missing quote for held'): run(scenario(a,b))
    def test_risk_blocks_after_loss(self):
        a,b=frame(25),frame(26,price='10')
        r=run(scenario(a,b))
        self.assertTrue(any(x['reason']=='Risk limit' for x in r['blocked']))
