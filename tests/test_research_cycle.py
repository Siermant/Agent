import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime,date,timedelta,timezone
from pathlib import Path
from global_agent.equity_data import FIELDS,CAL_FIELDS,FX_FIELDS
from global_agent.research_cycle import research,publish,provider_cycle

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

class ResearchCycleTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.calendar=self.root/'calendar.csv'
        self.bars=self.root/'bars.csv'
        self.fx=self.root/'fx.csv'
        self.a={'research_only':True,'commission_bps':'10','spread_slippage_bps':'20',
                'minimum_fee_pln':'5','minimum_adv_pln':'100000','max_adv_fraction':'.01',
                'benchmark_id':'fixture:ETF','instruments':{
                    'fixture:ETF':{'research_eligible':True,'benchmark_eligible':True,
                                   'sector':'diversified','adv_pln':'10000000'},
                    'fixture:STOCK':{'research_eligible':True,'sector':'industry','adv_pln':'10000000'}}}
    def csv(self,path,fields,data):
        with path.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(fields)
            w.writerows(data)
    def fixtures(self,n=61,second_venue=False):
        days=[]
        d=date(2026,11,26)
        while len(days)<n:
            if d.weekday()<5:
                days.append(d)
            d-=timedelta(days=1)
        cal, bars=[],[]
        for i,d in enumerate(reversed(days)):
            close=str(d)+'T16:00:00Z'
            cal.append(['XWAR',str(d),close])
            if second_venue:
                cal.append(['XETR',str(d),close])
            for ticker,kind,venue in [('ETF','etf','XWAR'),('STOCK','stock','XETR' if second_venue else 'XWAR')]:
                bars.append(['fixture:'+ticker,ticker,venue,kind,'PLN','Europe/Warsaw',
                             str(d),close,str(d)+'T17:00:00Z',str(100+i),str(100+i),
                             'split_dividend','synthetic'])
        self.csv(self.calendar,CAL_FIELDS,cal)
        self.csv(self.bars,FIELDS,bars)
        self.csv(self.fx,FX_FIELDS,[])
    def run_cycle(self,cohort=None):
        return research(self.calendar,self.bars,self.fx,self.a,cohort,now=NOW)
    def test_end_to_end(self):
        self.fixtures()
        r=self.run_cycle()
        self.assertEqual(r['state'],'hypothetical_plan')
        self.assertFalse(r['eligible_for_orders'])
        self.assertEqual(len(r['allocation']['plans']),5)
        self.assertEqual(r['allocation']['plans'][-1]['allocations'][0]['instrument_id'],'fixture:ETF')
        p=self.root/'result.json'
        publish(r,p)
        self.assertEqual(json.loads(p.read_text()),r)
    def test_cohort_guard(self):
        self.fixtures(second_venue=True)
        r=self.run_cycle()
        self.assertEqual(r['state'],'cohort_selection_required')
        self.assertIsNone(r['allocation'])
        self.assertEqual(self.run_cycle(0)['state'],'hypothetical_plan')
    def test_short_history(self):
        self.fixtures(n=10)
        r=self.run_cycle()
        self.assertEqual(r['state'],'no_rankable_instruments')
        self.assertIsNone(r['allocation'])
    def test_invalid_input_does_not_publish(self):
        self.fixtures()
        p=self.root/'result.json'
        publish(self.run_cycle(),p)
        before=p.read_bytes()
        self.bars.write_text('invalid')
        with self.assertRaises(ValueError): self.run_cycle()
        self.assertEqual(p.read_bytes(),before)
    def test_atomic_serialization_failure(self):
        p=self.root/'result.json'
        publish({'ok':True},p)
        with self.assertRaises(ValueError): publish({'bad':float('nan')},p)
        self.assertEqual(json.loads(p.read_text()),{'ok':True})

    def test_provider_orchestration(self):
        self.fixtures()
        with self.bars.open(newline='') as f:
            data=list(csv.DictReader(f))
        paths=[]
        def fake_fx(start,end,output,**kwargs):
            paths.append(output)
            self.csv(output,FX_FIELDS,[])
        with patch('global_agent.research_cycle.alpaca_data.collect',return_value=data) as collect, patch('global_agent.research_cycle.fx_reference.export',side_effect=fake_fx) as fx, patch('global_agent.research_cycle.research',side_effect=lambda *args: research(*args,now=NOW)):
            result=provider_cycle(self.root/'manifest.csv',self.calendar,self.a,lambda _:None)
        self.assertEqual(result['state'],'hypothetical_plan')
        self.assertEqual(collect.call_count,1)
        self.assertEqual(fx.call_count,1)
        self.assertFalse(paths[0].exists())
