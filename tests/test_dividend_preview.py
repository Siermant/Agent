import unittest
import tempfile
from datetime import date,timedelta
from global_agent.dividend_preview import preview
from global_agent.global_paper_initial import open_initial
from global_agent.corporate_actions import scan
from test_global_paper_initial import plan,snapshot,NOW
from test_corporate_actions import event,payload

class DividendPreviewTests(unittest.TestCase):
    def test_entitlement_and_currency_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=open_initial(tmp,plan(),snapshot(),lambda:NOW)
            ordinary=dict(symbol='WMT',ex_date='2026-12-01',payable_date='2026-12-15',process_date='2026-12-15')
            report=scan(lambda _:payload({'cash_dividends':[
                event(**ordinary),event(**dict(ordinary,ex_date='2026-12-02'),currency=''),event(**dict(ordinary,ex_date='2026-12-03'),currency='',special=True),
                event(symbol='WMT',ex_date='2026-11-27',payable_date='2026-12-15',process_date='2026-12-15')
            ]}),['WMT'],date(2026,11,20),date(2026,12,20))
            r=preview(receipt,report,NOW)
            self.assertEqual(r['accounts']['A:global'],{'explicit_usd_gross':'1.25','assumed_usd_gross':'1.25'})
            self.assertEqual(r['accounts']['E:global'],{'explicit_usd_gross':'0','assumed_usd_gross':'0'})
            self.assertEqual(sum(e['state']=='not_entitled_opened_on_or_after_ex_date' for e in r['events']),4)
            self.assertEqual(sum(e['state']=='manual_review_required' for e in r['events']),4)
            self.assertFalse(r['accounting_enabled'])
    def test_split_blocks_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=open_initial(tmp,plan(),snapshot(),lambda:NOW)
            ordinary=dict(symbol='WMT',ex_date='2026-12-01',payable_date='2026-12-15',process_date='2026-12-15')
            report=scan(lambda _:payload({'cash_dividends':[event(**ordinary)],
                'forward_splits':[event(**ordinary,old_rate=1,new_rate=2)]}),['WMT'],date(2026,11,20),date(2026,12,20))
            r=preview(receipt,report,NOW)
            self.assertEqual(r['blocked_symbols'],['WMT'])
            self.assertEqual(r['accounts']['A:global']['explicit_usd_gross'],'0')

    def test_overlapping_provider_ids_not_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=open_initial(tmp,plan(),snapshot(),lambda:NOW)
            ordinary=dict(symbol='WMT',ex_date='2026-12-01',payable_date='2026-12-15',process_date='2026-12-15')
            report=scan(lambda _:payload({'cash_dividends':[event(**ordinary),event(**ordinary,currency='')]}),['WMT'],date(2026,11,20),date(2026,12,20))
            r=preview(receipt,report,NOW)
            self.assertEqual(r['overlapping_cash_events'],1)
            self.assertEqual(r['accounts']['A:global'],{'explicit_usd_gross':'0','assumed_usd_gross':'0'})
            self.assertTrue(all(e['state']=='overlapping_events_review_required' for e in r['events']))
