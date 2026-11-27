"""Publish only the fields used by the local dashboard, never credentials or raw data."""
import json
import os
import tempfile
from pathlib import Path

FLAGS=('research_only','real_trading','eligible_for_orders')
STATUS_FIELDS=FLAGS+('state','started_utc','finished_utc','ranking_generated_utc','instruments','cohorts')
RANK_FIELDS=FLAGS+('state','generated_utc','expected_instruments')
COHORT_FIELDS=('mic','source','last_session')
ROW_FIELDS=('instrument_id','ticker','mic','rank','return_20_pct','return_60_pct',
            'annualized_volatility_pct','max_drawdown_60_pct','score')


def select(obj,keys):
    return {k:obj[k] for k in keys if k in obj}


def export(directory,status,ranking=None):
    root=Path(directory)
    snapshot={'status':select(status,STATUS_FIELDS)}
    if status.get('state')=='ok':
        if ranking is None:
            raise ValueError('Successful snapshot requires ranking')
        snapshot['ranking']=dict(select(ranking,RANK_FIELDS),cohorts=[
            dict(select(c,COHORT_FIELDS),instruments=[select(row,ROW_FIELDS) for row in c['instruments']])
            for c in ranking['cohorts']])
    data=json.dumps(snapshot,allow_nan=False,ensure_ascii=True).encode()
    if len(data)>256*1024:
        raise ValueError('Oversized panel export')
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
            temp=Path(f.name)
            f.write(data)
            f.flush()
            os.fchmod(f.fileno(),0o644)
            os.fsync(f.fileno())
        os.replace(temp,root/'panel.json')
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
