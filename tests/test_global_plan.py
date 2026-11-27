import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal as D
from global_agent.global_plan import plan
from global_agent.ledger import EXPOSURE

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

def fixtures(n=30):
    rows=[{'instrument_id':str(i),'kind':'etf' if i==0 else 'stock','mic':'XNAS','source':'fixture',
           'last_session':'2026-11-26','return_20_pct':10,'return_60_pct':20,
           'annualized_volatility_pct':10,'max_drawdown_60_pct':2,'score':12} for i in range(n)]
    ranking={'research_only':True,'real_trading':False,'eligible_for_orders':False,'currency':'PLN',
             'generated_utc':NOW.isoformat(),'cohorts':[{'mic':'XNAS','source':'fixture','last_session':'2026-11-26','instruments':rows}]}
    a={'research_only':True,'commission_bps':'10','spread_slippage_bps':'20','minimum_fee_pln':'5',
       'minimum_adv_pln':'100000','max_adv_fraction':'.01','benchmark_id':'0',
       'instruments':{str(i):{'research_eligible':True,'sector':'sector'+str(i%5),'adv_pln':'10000000','benchmark_eligible':i==0} for i in range(n)}}
    return ranking,a

class GlobalPlanTests(unittest.TestCase):
    def test_limits_and_balance(self):
        r,a=fixtures()
        out=plan(r,a,now=NOW)
        self.assertFalse(out['eligible_for_orders'])
        self.assertEqual(len(out['plans']),5)
        for p in out['plans'][:4]:
            self.assertEqual(D(p['cash_pln'])+D(p['allocated_pln'])+D(p['estimated_cost_pln']),D(50000))
            self.assertGreaterEqual(D(p['cash_pln']),D(10000))
            self.assertLessEqual(D(p['allocated_pln']),D(50000)*EXPOSURE[p['account'][0]])
            sectors={}
            for row in p['allocations']:
                v=D(row['notional_pln'])
                self.assertLessEqual(v,D(2250))
                sectors[row['sector']]=sectors.get(row['sector'],D(0))+v
            self.assertTrue(all(x<=10000 for x in sectors.values()))
        self.assertEqual(out['plans'][4]['allocations'][0]['instrument_id'],'0')
    def test_missing_and_low_liquidity(self):
        r,a=fixtures(3)
        del a['instruments']['1']
        a['instruments']['2']['adv_pln']='10'
        out=plan(r,a,now=NOW)
        self.assertEqual(len(out['excluded']),2)
        self.assertTrue(all(x['instrument_id']=='0' for p in out['plans'] for x in p['allocations']))
    def test_no_benchmark_stock(self):
        r,a=fixtures(3)
        a['benchmark_id']='1'
        a['instruments']['1']['benchmark_eligible']=True
        self.assertEqual(plan(r,a,now=NOW)['plans'][4]['allocations'],[])
    def test_no_positive_scores(self):
        r,a=fixtures(3)
        for x in r['cohorts'][0]['instruments']:
            x.update(return_20_pct=-10,return_60_pct=-10,score=-10)
        out=plan(r,a,now=NOW)
        self.assertTrue(all(not p['allocations'] for p in out['plans'][:4]))
    def test_stale_and_invalid(self):
        r,a=fixtures()
        with self.assertRaises(ValueError): plan(r,a,now=NOW+timedelta(days=2))
        a['commission_bps']='NaN'
        with self.assertRaises(ValueError): plan(r,a,now=NOW)
    def test_liquidity_budget(self):
        r,a=fixtures(1)
        a['instruments']['0']['adv_pln']='100000'
        for p in plan(r,a,now=NOW)['plans']:
            self.assertLessEqual(D(p['allocated_pln'])+D(p['estimated_cost_pln']),D(1000))
