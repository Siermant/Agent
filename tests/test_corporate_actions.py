import unittest
from datetime import date,timedelta
from uuid import uuid4
from global_agent.corporate_actions import scan
START=date(2026,11,1)
END=date(2026,11,30)
def event(**kw):
    return dict({'id':str(uuid4()),'symbol':'AAPL','process_date':'2026-11-20',
                 'ex_date':'2026-11-18','payable_date':'2026-11-20',
                 'currency':'USD','foreign':False,'special':False,'rate':'0.25'},**kw)
def payload(groups,token=None):
    return {'corporate_actions':groups,'next_page_token':token}
class CorporateActionTests(unittest.TestCase):
    def test_candidates_and_manual_review(self):
        r=scan(lambda _:payload({'cash_dividends':[event(),event(currency=''),event(special=True)],
                    'forward_splits':[event(old_rate=1,new_rate=4)],
                    'reverse_splits':[event(old_rate=4,new_rate=1)],
                    'stock_dividends':[event()]}),['AAPL'],START,END)
        self.assertEqual(r['counts'],{'simple_dividend_candidate':1,'simple_split_candidate':2,'manual_review_required':3})
        self.assertFalse(r['accounting_enabled'])
        self.assertFalse(r['corporate_actions_verified'])
    def test_pagination(self):
        calls=[]
        def get(p):
            calls.append(p)
            return payload({'cash_dividends':[event()]},'next' if len(calls)==1 else None)
        r=scan(get,['AAPL'],START,END)
        self.assertEqual(r['events_count'],2)
        self.assertEqual(calls[1]['page_token'],'next')
        self.assertEqual(calls[0]['data_quality'],'all')
        self.assertNotIn('types',calls[0])
    def test_duplicates_and_unknown(self):
        e=event()
        for groups in ({'cash_dividends':[e,e]},{'unexpected':[e]},
                        {'cash_dividends':[event(symbol='FAKE')]},
                        {'cash_dividends':[event(process_date='2026-10-01')]}):
            with self.assertRaises(ValueError):
                scan(lambda _:payload(groups),['AAPL'],START,END)
    def test_empty_and_chunking(self):
        calls=[]
        def get(p):
            calls.append(p);return payload({})
        r=scan(get,['AAPL'],START,START+timedelta(days=90))
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['end'],'2026-12-30')
        self.assertEqual(calls[1]['start'],'2026-12-31')
        self.assertFalse(r['coverage_guaranteed'])
    def test_bad_token(self):
        with self.assertRaises(ValueError):
            scan(lambda _:payload({},'repeat'),['AAPL'],START,END)
    def test_incomplete(self):
        r=scan(lambda _:payload({'cash_dividends':[event(ex_date=None)]}),['AAPL'],START,END)
        self.assertEqual(r['events'][0]['state'],'manual_review_required')

    def test_diagnostics_preserve_ambiguous_currency(self):
        e=event(currency='')
        e['cusip']='PRIVATE-FIXTURE'
        r=scan(lambda _:payload({'cash_dividends':[e]}),['AAPL'],START,END)
        v=r['events'][0]
        self.assertEqual(v['currency_status'],'empty')
        self.assertEqual(v['reported_rate'],'0.25')
        self.assertNotIn('rate_usd',v)
        self.assertNotIn('cusip',v)
        self.assertEqual(r['review_reason_counts'],{'currency_not_explicit_usd':1})
        self.assertFalse(r['accounting_enabled'])
    def test_multiple_reasons(self):
        e=event(special=True,rate=None)
        del e['currency']
        r=scan(lambda _:payload({'cash_dividends':[e]}),['AAPL'],START,END)
        self.assertEqual(r['events'][0]['currency_status'],'missing')
        self.assertEqual(set(r['review_reason_counts']),{'currency_not_explicit_usd','special_not_explicit_false','invalid_rate'})
