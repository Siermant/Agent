import unittest
import tempfile
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from global_agent.global_paper_initial import open_initial,validate,ACCOUNTS
from global_agent.equity_quotes import SYMBOLS

NOW=datetime(2026,11,27,15,tzinfo=timezone.utc)
ID='alpaca:XNAS:WMT'

def plan():
    return {'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'initial_cash_only':True,'aligned_us_research':True,'state':'allocation_preview',
            'generated_utc':(NOW-timedelta(minutes=1)).isoformat(),
            'cost_scenario':{'commission_bps':'10','spread_slippage_bps':'10','minimum_fee_pln':'2','minimum_adv_pln':'100000','max_adv_fraction':'.001'},
            'instruments':[{'mic':'XNAS','ticker':'WMT','research_eligible':True,
                            'sector_assumed':'consumer_staples','iex_turnover_20_pln':'10000000'}],
            'plans':[dict(account=a,capital_pln='50000',cash_pln='47752' if a[0]!='E' else '50000',
                          allocated_pln='2243.51' if a[0]!='E' else '0',estimated_cost_pln='4.49' if a[0]!='E' else '0',
                          allocations=[dict(instrument_id=ID,sector='consumer_staples',notional_pln='2243.51',estimated_entry_cost_pln='4.49')] if a[0]!='E' else [])
                     for a in ACCOUNTS]}

def snapshot():
    return {'ready':True,'market_open':True,'research_only':True,'real_trading':False,
            'eligible_for_orders':False,'state':'qualified','source':'alpaca_iex_latest',
            'fx_source':'nbp_table_a_prior_day','fx_date':'2026-11-26',
            'checked_utc':NOW.isoformat(),'session_close_utc':(NOW+timedelta(hours=2)).isoformat(),
            'quotes':{s:{'bid_pln_reference':'400','ask_pln_reference':'401',
                         'observed_at':NOW.isoformat(),'ask_size_shares':100} for s in SYMBOLS}}

class GlobalInitialTests(unittest.TestCase):
    def test_atomic_open_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'global-paper.db'
            r=open_initial(tmp,plan(),snapshot(),lambda:NOW)
            self.assertEqual(r['state'],'opened')
            self.assertEqual(len(r['fills']),4)
            self.assertTrue(all(f['quantity']=='5' for f in r['fills']))
            self.assertFalse(r['eligible_for_orders'])
            self.assertEqual(r['accounts'][-1]['cash_pln'],'50000')
            self.assertTrue(p.is_file())
            original=p.read_bytes()
            later=open_initial(tmp,{}, {},lambda:NOW+timedelta(days=1))
            self.assertEqual(later['state'],'already_opened')
            self.assertEqual(p.read_bytes(),original)
            with closing(sqlite3.connect(p)) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM fills').fetchone()[0],4)
                self.assertEqual(db.execute('SELECT count(*) FROM initial_receipt').fetchone()[0],1)
    def test_rollback_if_second_fill_fails(self):
        from global_agent.ledger import Ledger
        original=Ledger.execute
        calls=[]
        def fail_second(self,*a,**kw):
            calls.append(1)
            if len(calls)==2: raise RuntimeError('fixture')
            return original(self,*a,**kw)
        with tempfile.TemporaryDirectory() as tmp,patch.object(Ledger,'execute',fail_second):
            with self.assertRaises(RuntimeError):open_initial(tmp,plan(),snapshot(),lambda:NOW)
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
    def test_stale_and_closed(self):
        for s in (dict(snapshot(),market_open=False),dict(snapshot(),checked_utc=(NOW-timedelta(seconds=16)).isoformat())):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):open_initial(tmp,plan(),s,lambda:NOW)
                self.assertFalse((Path(tmp)/'global-paper.db').exists())
    def test_displayed_size(self):
        s=snapshot();s['quotes']['WMT']['ask_size_shares']=2
        with tempfile.TemporaryDirectory() as tmp:
            r=open_initial(tmp,plan(),s,lambda:NOW)
            self.assertTrue(all(f['quantity']=='2' for f in r['fills']))
    def test_reject_budget_and_benchmark(self):
        p=plan();p['plans'][0]['allocations'][0]['notional_pln']='2500'
        with self.assertRaises(ValueError):validate(p,NOW)
        p=plan();p['plans'][-1]['allocations']=p['plans'][0]['allocations']
        with self.assertRaises(ValueError):validate(p,NOW)

    def test_only_selected_symbols_required(self):
        s=snapshot();s['quotes']={'WMT':s['quotes']['WMT']}
        with tempfile.TemporaryDirectory() as tmp:
            r=open_initial(tmp,plan(),s,lambda:NOW)
            self.assertEqual(len(r['fills']),4)
            self.assertEqual(r['selected_symbols'],['WMT'])
            self.assertEqual(r['unquoted_symbols'],[])
    def test_missing_selected_blocks_default(self):
        s=snapshot();del s['quotes']['WMT']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):open_initial(tmp,plan(),s,lambda:NOW)
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
    def test_explicit_cash_fallback_and_coverage(self):
        p=plan()
        symbols=['WMT','COST','KO','PEP']
        ids=['alpaca:XNAS:WMT','alpaca:XNAS:COST','alpaca:XNYS:KO','alpaca:XNAS:PEP']
        for sym,ident in zip(symbols[1:],ids[1:]):
            p['instruments'].append({'mic':ident.split(':')[1],'ticker':sym,
                'research_eligible':True,'sector_assumed':'consumer_staples','iex_turnover_20_pln':'10000000'})
        for row in p['plans'][:-1]:
            row['allocations']=[dict(row['allocations'][0],instrument_id=k) for k in ids]
            row.update(cash_pln='41008',allocated_pln='8974.04',estimated_cost_pln='17.96')
        s=snapshot()
        s.update(state='incomplete_quotes',ready=False,expected_instruments=4,valid_quotes=3,rejected_symbols=['PEP'])
        s['quotes']={k:s['quotes'][k] for k in symbols[:3]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):open_initial(tmp,p,s,lambda:NOW)
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
            r=open_initial(tmp,p,s,lambda:NOW,allow_unquoted_cash=True)
            self.assertEqual(len(r['fills']),12)
            self.assertEqual(r['unquoted_symbols'],['PEP'])
            self.assertEqual(r['quote_policy'],'selected_plan_cash_fallback')
            self.assertEqual(len(r['skipped']),4)
            self.assertTrue(all(v['reason']=='unqualified_quote_cash_retained' for v in r['skipped']))
        del s['quotes']['KO'];s.update(valid_quotes=2,rejected_symbols=['PEP','KO'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):open_initial(tmp,p,s,lambda:NOW,allow_unquoted_cash=True)
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
