"""Small, dependency-free read-only dashboard helpers."""
from decimal import Decimal, InvalidOperation
from html import escape
from urllib.parse import urlsplit
from datetime import date

COLORS = {'A':'#60a5fa','B':'#34d399','C':'#fbbf24','D':'#c084fc','E':'#f87171'}


def request_path(target):
    # Only origin-form HTTP request targets. Never route external URLs.
    if not target.startswith('/') or target.startswith('//'):
        return ''
    try:
        return urlsplit(target).path
    except ValueError:
        return ''


def equity_chart(statistics):
    series = {}
    for row in statistics:
        account = row.get('account', '')
        if account not in {p + ':crypto' for p in COLORS}:
            continue
        try:
            initial = Decimal(row['initial_pln'])
            if not initial.is_finite() or initial <= 0:
                continue
        except (InvalidOperation, KeyError, TypeError):
            continue
        points = {}
        for item in row.get('history', []):
            try:
                day = date.fromisoformat(item['day'])
                value = Decimal(item['equity'])
                if not value.is_finite() or value < 0:
                    continue
                pct = float((value / initial - 1) * 100)
                if not -100 <= pct <= 1000000:
                    continue
                points[day] = pct
            except (ValueError, TypeError, KeyError, InvalidOperation, OverflowError):
                continue
        if points:
            series[account[0]] = points
    days = sorted({d for points in series.values() for d in points})
    if not days:
        return '<p>Brak zapisanych wycen do wykresu.</p>'
    values = [0.0] + [v for points in series.values() for v in points.values()]
    low, high = min(values), max(values)
    pad = max((high-low)*.08, .05)
    low, high = low-pad, high+pad
    span = max((days[-1]-days[0]).days, 1)
    def x(day):
        return 65 + (day-days[0]).days/span*685
    def y(value):
        return 215-(value-low)/(high-low)*190
    parts = ['<svg viewBox="0 0 780 255" role="img" aria-label="Zwrot portfeli kryptowalutowych od startu" xmlns="http://www.w3.org/2000/svg" style="width:100%;min-width:300px;max-width:100%">']
    for i in range(5):
        value = low+(high-low)*i/4
        yy = y(value)
        parts.append(f'<line x1="65" x2="750" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#34435a"/><text x="58" y="{yy+4:.2f}" text-anchor="end" fill="#b8c8da" font-size="12">{value:+.2f}%</text>')
    for label, points in sorted(series.items()):
        previous = None
        for day, value in sorted(points.items()):
            xx, yy = x(day), y(value)
            if previous and (day-previous[0]).days == 1:
                parts.append(f'<line x1="{x(previous[0]):.2f}" y1="{y(previous[1]):.2f}" x2="{xx:.2f}" y2="{yy:.2f}" stroke="{COLORS[label]}" stroke-width="2"/>')
            parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="{COLORS[label]}"><title>{label} {day.isoformat()}: {value:+.3f}%</title></circle>')
            previous = day,value
    parts.append(f'<text x="65" y="240" fill="#b8c8da" font-size="12">{days[0].isoformat()}</text>')
    if len(days)>1:
        parts.append(f'<text x="750" y="240" text-anchor="end" fill="#b8c8da" font-size="12">{days[-1].isoformat()}</text>')
    parts.append('</svg>')
    legend = ' · '.join(f'<span style="color:{COLORS[p]}">{escape(p)}</span>' for p in sorted(series))
    return ''.join(parts) + '<p>' + legend + ' | Ostatnia zapisana wycena dnia UTC; brak interpolacji luk. Benchmark ma inną ekspozycję.</p>'
