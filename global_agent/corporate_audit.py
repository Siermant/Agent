"""Append-only observations of the normalized corporate action feed.
Separate research database. Never posts cash, adjusts holdings or places orders.
Absence from a later response is a review signal, not proof of cancellation.
"""
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime,date,timedelta,timezone
from pathlib import Path
from uuid import UUID
from .alpaca_data import NY
from .corporate_actions import scan,client,KNOWN,DATE_FIELDS
from .equity_quotes import SYMBOLS
from .equity_job import credentials
from .equity_data import moment

FLAGS={'research_only':True,'real_trading':False,'accounting_enabled':False,
       'eligible_for_orders':False,'coverage_guaranteed':False}
SCHEMA=(
    "CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, at TEXT NOT NULL UNIQUE, start TEXT NOT NULL, end TEXT NOT NULL, digest TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS revisions(event_id TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL, first_scan INTEGER NOT NULL REFERENCES scans(id), PRIMARY KEY(event_id,digest))",
    "CREATE TABLE IF NOT EXISTS observations(scan_id INTEGER NOT NULL REFERENCES scans(id), event_id TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(scan_id,event_id), FOREIGN KEY(event_id,digest) REFERENCES revisions(event_id,digest))",
)
FIELDS=('id','type','symbols','dates','state','currency_status','foreign_status',
        'special_status','sub_type_status','rate_valid','reported_rate','review_reasons',
        'rate_usd','old_rate','new_rate')

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def validate(report):
    if (report.get('state')!='feed_qualified' or report.get('date_basis')!='provider_process_date'
        or any(type(report.get(k)) is not type(v) or report[k]!=v for k,v in FLAGS.items())
        or report.get('symbols_checked')!=len(SYMBOLS)):
        raise ValueError('Expected full-universe qualified feed')
    start,end=(date.fromisoformat(report[k]) for k in ('process_start','process_end'))
    if not 0<=(end-start).days<=180:
        raise ValueError('Invalid scan window')
    events=report['events']
    if not isinstance(events,list) or len(events)!=report['events_count'] or len(events)>3000:
        raise ValueError('Invalid event count')
    out={}
    for raw in events:
        e={k:raw[k] for k in FIELDS if k in raw}
        ident=str(UUID(e['id']))
        if ident!=e['id'] or ident in out or e['type'] not in KNOWN:
            raise ValueError('Invalid event identity')
        if not e['symbols'] or not set(e['symbols'])<=set(SYMBOLS) or len(set(e['symbols']))!=len(e['symbols']):
            raise ValueError('Unexpected event symbols')
        if not isinstance(e['dates'],dict) or not set(e['dates'])<=set(DATE_FIELDS):
            raise ValueError('Invalid event dates')
        for d in e['dates'].values():
            if date.fromisoformat(d).isoformat()!=d:
                raise ValueError('Invalid date')
        p=e['dates'].get('process_date')
        if p and not start<=date.fromisoformat(p)<=end:
            raise ValueError('Event outside window')
        if e['state'] not in ('manual_review_required','simple_dividend_candidate','simple_split_candidate'):
            raise ValueError('Invalid classification')
        e['symbols']=sorted(e['symbols'])
        if 'review_reasons' in e:
            e['review_reasons']=sorted(set(e['review_reasons']))
        if len(canonical(e))>8192:
            raise ValueError('Oversized event')
        out[ident]=e
    return str(start),str(end),out

def record(root,report,now=None):
    root=Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Dedicated audit directory required')
    now=moment((now or datetime.now(timezone.utc)).isoformat())
    start,end,events=validate(report)
    today=now.astimezone(NY).date()
    if not date.fromisoformat(start)<=today<=date.fromisoformat(end):
        raise ValueError('Current date outside scan window')
    at=now.isoformat(timespec='microseconds')
    hashes={k:digest(v) for k,v in events.items()}
    fingerprint=digest({'start':start,'end':end,'events':hashes})
    dbpath=root/'corporate-audit.db'
    with closing(sqlite3.connect(dbpath,timeout=10,isolation_level=None)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        try:
            for sql in SCHEMA:db.execute(sql)
            previous=db.execute('SELECT id,at,digest FROM scans ORDER BY at DESC LIMIT 1').fetchone()
            if previous and previous[1]>at:
                raise ValueError('Clock moved backwards')
            if previous and previous[1]==at:
                if previous[2]!=fingerprint:
                    raise ValueError('Conflicting scan timestamp')
                db.rollback()
                return dict(FLAGS,state='already_recorded',checked_utc=at,events_count=len(events))
            last={}
            for ident,revision,body in db.execute(
                "SELECT o.event_id,o.digest,r.payload FROM observations o JOIN revisions r USING(event_id,digest) "
                "JOIN (SELECT event_id,max(scan_id) AS last_scan FROM observations GROUP BY event_id) x "
                "ON x.event_id=o.event_id AND x.last_scan=o.scan_id"):
                last[ident]=(revision,json.loads(body))
            scan_id=db.execute('INSERT INTO scans(at,start,end,digest) VALUES(?,?,?,?)',(at,start,end,fingerprint)).lastrowid
            new=[];changed=[];missing=[];reappeared=[]
            previous_ids=set()
            if previous:
                previous_ids={r[0] for r in db.execute('SELECT event_id FROM observations WHERE scan_id=?',(previous[0],))}
            for ident,e in events.items():
                h=hashes[ident]
                if ident not in last:
                    new.append(ident)
                elif last[ident][0]!=h:
                    changed.append(ident)
                elif ident not in previous_ids:
                    reappeared.append(ident)
                db.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?,?)',(ident,h,canonical(e),scan_id))
                db.execute('INSERT INTO observations VALUES(?,?,?)',(scan_id,ident,h))
            for ident,(_,e) in last.items():
                p=e['dates'].get('process_date')
                if ident not in events and (p is None or start<=p<=end):
                    missing.append(ident)
            # Different provider IDs may still refer to the same economic event.
            # Flag overlapping cash-dividend ex-dates; never merge or sum them here.
            clusters={}
            for ident,e in events.items():
                if e['type']=='cash_dividends' and e['dates'].get('ex_date'):
                    for s in e['symbols']:
                        clusters.setdefault((s,e['dates']['ex_date']),[]).append(ident)
            overlaps=[{'ticker':s,'ex_date':d,'event_ids':sorted(ids)} for (s,d),ids in sorted(clusters.items()) if len(ids)>1]
            revision_count=db.execute('SELECT count(*) FROM revisions').fetchone()[0]
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return dict(FLAGS,state='recorded',checked_utc=at,scan_id=scan_id,events_count=len(events),
                new_events=len(new),changed_events=len(changed),missing_in_window=len(missing),
                reappeared_events=len(reappeared),total_revisions=revision_count,
                changed_ids=sorted(changed),missing_ids=sorted(missing),overlapping_cash_events=overlaps,
                requires_review=bool(changed or missing or overlaps or any(e['state']=='manual_review_required' for e in events.values())))

def main():
    try:
        root=Path(os.environ['STATE_DIRECTORY'])
        if root.name!='gta-corporate-audit':
            raise ValueError('Unexpected audit state directory')
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        today=datetime.now(timezone.utc).astimezone(NY).date()
        report=scan(client(key,secret),SYMBOLS,today-timedelta(days=60),today+timedelta(days=30))
        result=record(root,report)
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps(dict(FLAGS,state='failed',error_type=type(exc).__name__)))
        raise SystemExit(1)

if __name__=='__main__':
    main()
