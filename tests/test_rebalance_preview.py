import unittest
import tempfile
from pathlib import Path
from datetime import timedelta,date
from decimal import Decimal as D
from global_agent.rebalance_preview import propose,read_state,read_audit,event_blocks
from global_agent.global_paper_initial import open_initial
from global_agent.global_paper_mark import mark
from global_agent.corporate_audit import record
from global_agent.corporate_actions import scan
from global_agent.equity_quotes import SYMBOLS
from test_global_paper_initial import plan,snapshot,NOW,ID
from test_corporate_actions import event,payload

LATER=NOW+timedelta(days=8)
NEW='alpaca:XNYS:KO'
def current_snapshot(at):
    s=snapshot()
    s['checked_utc']=at.isoformat()
    s['session_close_utc']=(at+timedelta(hours=2)).isoformat()
    for q in s['quotes'].values():
        q['observed_at']=at.isoformat()
        q['bid_size_shares']=100
    return s

def new_plan(at):
    p=plan()
    p['generated_utc']=(at-timedelta(minutes=1)).isoformat()
    p['instruments'].append({'mic':'XNYS','ticker':'KO','research_eligible':True,
        'sector_assumed':'consumer_staples','iex_turnover_20_pln':'10000000'})
    for r in p['plans'][:-1]:
        r['allocations'][0]['instrument_id']=NEW
    return p

def prepared(tmp):
    open_initial(tmp,plan(),snapshot(),lambda:NOW)
    def fetch(symbols):
        s=current_snapshot(LATER)
        s['quotes']={k:s['quotes'][k] for k in symbols}
        return s
    mark(tmp,fetch,lambda:LATER)
    return read_state(Path(tmp)/'global-paper.db')

class RebalancePreviewTests(unittest.TestCase):
    def test_sells_before_buys_and_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt,holdings=prepared(tmp)
            path=Path(tmp)/'global-paper.db'
            original=path.read_bytes()
            r=propose(receipt,holdings,new_plan(LATER),current_snapshot(LATER),{},LATER)
            for a in r['accounts'][:-1]:
                self.assertEqual([x['side'] for x in a['proposed_changes']],['sell','buy'])
                self.assertEqual([x['ticker'] for x in a['proposed_changes']],['WMT','KO'])
                self.assertGreater(D(a['estimated_cost_pln']),0)
                self.assertLessEqual(D(a['turnover_pln']),D(a['equity_pln'])*D('.15'))
                self.assertGreaterEqual(D(a['projected_cash_pln']),D(a['projected_equity_pln'])*D('.20'))
            self.assertFalse(r['eligible_for_orders'])
            self.assertEqual(r['accounts'][-1]['state'],'benchmark_waiting')
            self.assertEqual(path.read_bytes(),original)
    def test_cooldown_and_held_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt,h=prepared(tmp)
            r=propose(receipt,h,new_plan(NOW),current_snapshot(NOW),{},NOW)
            self.assertTrue(all(a['state']=='cooldown' for a in r['accounts'][:-1]))
            r=propose(receipt,h,new_plan(LATER),current_snapshot(LATER),{'WMT':['unresolved_dividend']},LATER)
            self.assertTrue(all(a['state']=='held_corporate_event_review' and not a['proposed_changes'] for a in r['accounts'][:-1]))
    def test_stale_snapshot_and_risk_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt,h=prepared(tmp)
            with self.assertRaises(ValueError):
                propose(receipt,h,new_plan(LATER),current_snapshot(NOW),{},LATER)
            h['A:global']['risk']['peak']='100000'
            r=propose(receipt,h,new_plan(LATER),current_snapshot(LATER),{},LATER)
            self.assertEqual(r['accounts'][0]['state'],'risk_blocked')
    def test_retained_names_not_churned(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt,h=prepared(tmp)
            p=plan();p['generated_utc']=LATER.isoformat()
            r=propose(receipt,h,p,current_snapshot(LATER),{},LATER)
            self.assertTrue(all(not a['proposed_changes'] for a in r['accounts']))
    def test_audit_revision_and_old_entitlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=event(symbol='KO',ex_date='2026-12-01',payable_date='2026-12-15',process_date='2026-12-15',currency='')
            old=event(symbol='WMT',ex_date='2026-11-20',payable_date='2026-12-15',process_date='2026-12-15',currency='')
            def feed(rows):
                return scan(lambda _:payload({'cash_dividends':rows}),SYMBOLS,date(2026,11,1),date(2026,12,20))
            report=feed([e,old])
            record(tmp,report,LATER)
            a=read_audit(Path(tmp)/'corporate-audit.db',LATER)
            blocks=event_blocks(report,a,NOW.date(),LATER.date())
            self.assertIn('unresolved_dividend',blocks['KO'])
            self.assertNotIn('WMT',blocks)
            changed=feed([dict(e,rate='1.0'),old])
            record(tmp,changed,LATER+timedelta(minutes=1))
            a=read_audit(Path(tmp)/'corporate-audit.db',LATER+timedelta(minutes=1))
            self.assertIn('unresolved_event_history',event_blocks(changed,a,NOW.date(),LATER.date())['KO'])
            with self.assertRaises(ValueError):
                read_audit(Path(tmp)/'corporate-audit.db',LATER+timedelta(days=3))
