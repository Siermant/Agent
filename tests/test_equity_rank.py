import unittest
from datetime import date, datetime, timedelta, timezone
from global_agent.equity_rank import rank, metrics, WINDOW

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

def series(ident, growth=.001, end=date(2026,11,26), mic='XNAS', n=WINDOW):
    dates=[]
    d=end
    while len(dates)<n:
        if d.weekday()<5:
            dates.append(d)
        d-=timedelta(days=1)
    result=[]
    for i,d in enumerate(reversed(dates)):
        result.append({'instrument_id':ident,'ticker':ident,'mic':mic,
                       'kind':'stock','source':'fixture','currency':'USD',
                       'session':str(d),'close_at':str(d)+'T21:00:00Z',
                       'retrieved_at':str(d)+'T22:00:00Z',
                       'adjustment':'split_dividend',
                       'adjusted_close_pln':str(100*(1+growth)**i)})
    return result

def dataset(*groups):
    return {'research_only':True,'real_trading':False,'instruments':len(groups),
            'bars':[row for g in groups for row in g]}

class EquityRankTests(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(metrics([100]*WINDOW)['score'],0)
    def test_ranking(self):
        r=rank(dataset(series('slow'),series('fast',.003)),NOW)
        self.assertEqual([x['ticker'] for x in r['cohorts'][0]['instruments']],['fast','slow'])
        self.assertFalse(r['eligible_for_orders'])
        self.assertFalse(r['liquidity_evaluated'])
    def test_risk(self):
        p=[100]*30+[60]*30+[100]
        self.assertAlmostEqual(metrics(p)['max_drawdown_60_pct'],40)
        self.assertLess(metrics(p)['score'],0)
    def test_incomparable_windows(self):
        r=rank(dataset(series('a'),series('b',end=date(2026,11,25)),series('c',mic='XNYS')),NOW)
        self.assertEqual(len(r['cohorts']),3)
    def test_exclusions(self):
        raw=series('raw')
        for row in raw:
            row['adjustment']='none'
            row['adjusted_close_pln']=None
        r=rank(dataset(raw,series('short',n=20),series('old',end=date(2026,11,18))),NOW)
        self.assertEqual({x['reason'] for x in r['excluded']},{'stale_history','insufficient_sessions','missing_total_return_adjustment'})
    def test_future_and_duplicate(self):
        s=series('a')
        s[-1]['retrieved_at']='2026-11-28T00:00:00Z'
        with self.assertRaises(ValueError): rank(dataset(s),NOW)
        s=series('a')
        with self.assertRaises(ValueError): rank(dataset(s+[s[0]]),NOW)
    def test_invalid_price(self):
        s=series('a')
        s[-1]['adjusted_close_pln']='NaN'
        with self.assertRaises(ValueError): rank(dataset(s),NOW)
