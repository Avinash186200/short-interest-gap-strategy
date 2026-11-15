"""
Settlement date & publication date analysis.
Exports:
  - output/settlement_pub_analysis.md   (markdown summary)
  - output/Trade_Level_Settlement_Analysis.xlsx  (trade-level data + analysis tabs)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path("<local-path>")
OUT_MD   = ROOT / "output" / "settlement_pub_analysis.md"
OUT_XLSX = ROOT / "output" / "Trade_Level_Settlement_Analysis.xlsx"

# ── load data ──────────────────────────────────────────────────────────────
df  = pd.read_parquet(ROOT / "data" / "trades_selected.parquet")
ds  = pd.read_parquet(ROOT / "data" / "drift_signal.parquet")

for c in ['trade_date', 'settlement_date', 'publication_date']:
    df[c] = pd.to_datetime(df[c])
ds['settlement_date'] = pd.to_datetime(ds['settlementDate'])
ds['pub_eve_date']    = pd.to_datetime(ds['pub_eve_date'])

# selected LONG trades
sel = df[(df['selected']) & (df['side'] == 'LONG')].copy()

# ── derived columns ─────────────────────────────────────────────────────────
sel['lag_cal_days']  = (sel['publication_date'] - sel['settlement_date']).dt.days
sel['days_since_pub'] = (sel['trade_date'] - sel['publication_date']).dt.days

# ── settlement→publication lag ──────────────────────────────────────────────
lag_counts = sel['lag_cal_days'].value_counts().sort_index()

# ── performance by days-since-publication (0-2, 3-5, 6-10, 11-15, 16-20, 21+)
bins   = [-1, 2, 5, 10, 15, 20, 100]
labels = ['0-2 days', '3-5 days', '6-10 days', '11-15 days', '16-20 days', '21+ days']
sel['pub_bucket'] = pd.cut(sel['days_since_pub'], bins=bins, labels=labels)

perf_by_pub = (sel.groupby(['pub_bucket', 'gap_threshold_pct'], observed=False)['realized_return_pct']
               .agg(n='count', hit=lambda x: (x > 0).mean() * 100, avg_ret='mean', sum_ret='sum')
               .round(2).reset_index())

# ── performance by price_drift_pct quartile ────────────────────────────────
q_vals = sel['price_drift_pct'].quantile([0, .25, .5, .75, 1])
q_bins   = [q_vals[0] - 0.001, q_vals[0.25], q_vals[0.5], q_vals[0.75], q_vals[1] + 0.001]
q_labels = [
    f'Q1 most-neg (< {q_vals[0.25]:.1f}%)',
    f'Q2 ({q_vals[0.25]:.1f}% to {q_vals[0.5]:.1f}%)',
    f'Q3 ({q_vals[0.5]:.1f}% to {q_vals[0.75]:.1f}%)',
    f'Q4 most-pos (> {q_vals[0.75]:.1f}%)'
]
sel['drift_q'] = pd.cut(sel['price_drift_pct'], bins=q_bins, labels=q_labels)

perf_by_drift = (sel.groupby(['drift_q', 'gap_threshold_pct'], observed=False)['realized_return_pct']
                 .agg(n='count', hit=lambda x: (x > 0).mean() * 100, avg_ret='mean', sum_ret='sum')
                 .round(2).reset_index())

# ── drift signal: lag distribution (settlement → pub_eve) ─────────────────
ds['lag_cal'] = (ds['pub_eve_date'] - ds['settlement_date']).dt.days
ds_valid = ds[ds['lag_cal'].between(8, 14)]  # clean records only
lag_pub_counts = ds_valid['lag_cal'].value_counts().sort_index()

# ── price_drift_pct distribution ───────────────────────────────────────────
drift_desc = sel['price_drift_pct'].describe(percentiles=[.05,.1,.25,.5,.75,.9,.95])

# ── drift by year ──────────────────────────────────────────────────────────
sel['year'] = sel['trade_date'].dt.year
drift_by_year = (sel.groupby('year')['price_drift_pct']
                 .agg(['count', 'mean', 'median', 'std'])
                 .round(2))

# =============================================================================
# MARKDOWN OUTPUT
# =============================================================================
lines = [
    "# Settlement Date & Publication Date Analysis",
    "",
    "## 1. What is `price_drift_pct`?",
    "",
    "The price drift is the percentage change in the stock price from the "
    "**settlement date close** to the **day before the publication date** (pub-eve close):",
    "",
    "```",
    "price_drift_pct = (pub_eve_close - settlement_close) / settlement_close × 100",
    "```",
    "",
    "This captures how much the price moved **during the ~8-business-day window** "
    "between when the FINRA short-interest position was measured and when we can "
    "first act on the data. A negative value means the stock fell before the signal "
    "became actionable.",
    "",
    "## 2. Settlement → Publication lag",
    "",
    "The lag is **consistently 8 business days**, which translates to either "
    "**10 or 12 calendar days** depending on where the weekends fall:",
    "",
]
for lag, cnt in lag_counts.items():
    lines.append(f"- {int(lag)} calendar days: {cnt:,} trades ({cnt/len(sel)*100:.1f}%)")
lines += [
    "",
    "> 10 cal days = settlement falls Mon/Tue (one weekend in the window).  ",
    "> 12 cal days = settlement falls Thu/Fri (two weekends in the window).",
    "",
    "## 3. `price_drift_pct` distribution (selected LONG trades, n=1,927)",
    "",
    "| Statistic | Value |",
    "|-----------|-------|",
]
for stat, val in drift_desc.items():
    lines.append(f"| {stat} | {val:.2f}% |")
lines += [
    "",
    "Key observations:",
    "- **Median = 0%** — half the selected trades have zero or negative drift (price flat or down between settlement and publication).",
    "- **Mean = +1.1%** — pulled up by a small number of extreme positive outliers (the max of +732.8% is almost certainly a stock-split artifact).",
    "- **Q1 = -7.1%** — the most-negative-drift quarter fell by more than 7% before the signal became actionable.",
    "- **Min = -72.9%** (SKYQ) — the most severe pre-publication blowdown; also the worst trade in the drawdown investigation.",
    "",
    "## 4. Drift distribution by year",
    "",
    "| Year | Count | Mean drift % | Median drift % | Std % |",
    "|------|-------|-------------|----------------|-------|",
]
for yr, row in drift_by_year.iterrows():
    lines.append(f"| {yr} | {int(row['count'])} | {row['mean']:.2f}% | {row['median']:.2f}% | {row['std']:.2f}% |")
lines += [
    "",
    "## 5. Performance by days since publication",
    "",
    "Within the 10-trading-day publication window, the gap day can fall on any day. "
    "Does timing within the window matter?",
    "",
    "| Days since pub | Gap threshold | Trades | Hit % | Avg ret % | Sum ret % |",
    "|----------------|--------------|--------|-------|-----------|-----------|",
]
for _, row in perf_by_pub.iterrows():
    lines.append(f"| {row['pub_bucket']} | {row['gap_threshold_pct']}% | {int(row['n'])} | "
                 f"{row['hit']:.1f}% | {row['avg_ret']:.2f}% | {row['sum_ret']:.2f}% |")
lines += [
    "",
    "**Key findings:**",
    "- **Days 0-2 (immediately after publication):** strongest performance (+67.1% / +86.2%). "
      "The signal is freshest; the SI data has not yet been priced in.",
    "- **Days 6-10:** second-best (+90.7% / +90.4%). A secondary window where the effect re-emerges.",
    "- **Days 11-15:** worst, both thresholds NEGATIVE (-44.0% / -81.2%). "
      "This is within the nominal '10 trading day' window but in calendar-day terms "
      "falls past 2 weekends (~15-16 cal days). The information may be stale here.",
    "- **Days 21+:** positive again — could be the *next* settlement cycle for the same ticker.",
    "",
    "## 6. Performance by price_drift_pct quartile",
    "",
    "Does bigger pre-publication price drop → bigger squeeze?",
    "",
    "| Drift quartile | Gap threshold | Trades | Hit % | Avg ret % | Sum ret % |",
    "|----------------|--------------|--------|-------|-----------|-----------|",
]
for _, row in perf_by_drift.iterrows():
    lines.append(f"| {row['drift_q']} | {row['gap_threshold_pct']}% | {int(row['n'])} | "
                 f"{row['hit']:.1f}% | {row['avg_ret']:.2f}% | {row['sum_ret']:.2f}% |")
lines += [
    "",
    "**Key findings:**",
    "- **Q1 (most negative drift < -7.1%) is NEGATIVE** at both thresholds (-20.0% / -16.8%). "
      "The extreme pre-publication drops are blowup situations (SKYQ, ZNTL, CAPR, RILY etc.) "
      "that continue to fall on the gap day rather than squeezing. The ranking selects these "
      "because they *look* most like squeeze candidates — but they are actually the most dangerous.",
    "- **Q2 (-7.1% to 0%) is the best performing quartile** (+124.5% / +59.1%). "
      "Moderate negative drift (price down 0-7% before publication) — squeezed enough to "
      "attract SI attention but not blown up.",
    "- **Q3 (0% to +3.5%)** also positive — price flat or mildly up before publication.",
    "- **Q4 (most positive drift > +3.5%)** weakest positive — the price already started moving "
      "before the publication date, leaving less upside on the gap day.",
    "",
    "**Implication:** The `drift_low` ranking selects into Q1 first (most negative). "
    "The positive full-sample result is driven by Q2 and Q3, which get selected on "
    "days where the most extreme names are not present. A refinement — add a floor on "
    "minimum price_drift_pct (e.g., > -10%) — might improve Q1 without losing Q2/Q3.",
    "",
    "## 7. File map",
    "",
    "- `data/drift_signal.parquet` — settlement_close, pub_eve_close, price_drift_pct, volume_ratio per ticker/settlement",
    "- `data/trades_selected.parquet` — full trade-level data incl. settlement_date, publication_date, days_since_pub, price_drift_pct, selected, selection_rule",
    "- `output/Trade_Level_Settlement_Analysis.xlsx` — trade-level export + all tables from this report",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT_MD}")

# =============================================================================
# EXCEL OUTPUT
# =============================================================================
THIN = Side(border_style="thin", color="D0D0D0")
BOX  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
H1   = Font(name="Calibri", size=14, bold=True, color="1F3864")
H2   = Font(name="Calibri", size=11, bold=True, color="1F3864")
TXT  = Font(name="Calibri", size=11)
TXT_BOLD = Font(name="Calibri", size=11, bold=True)
TXT_ITAL = Font(name="Calibri", size=11, italic=True, color="595959")
SHADE = PatternFill("solid", fgColor="EAF1FB")
ALT   = PatternFill("solid", fgColor="F7FAFD")
NEG   = Font(name="Calibri", size=11, color="9C2222")
POS   = Font(name="Calibri", size=11, color="1E6A2A")
WARN  = PatternFill("solid", fgColor="FBEAEA")
WRAP  = Alignment(wrap_text=True, vertical="top")

wb = Workbook()

def setw(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

def hdr(ws, txt, row=1, col=1, span=1):
    c = ws.cell(row, col, txt); c.font = H1
    if span > 1: ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col+span-1)

def sub(ws, txt, row, col=1, span=1):
    c = ws.cell(row, col, txt); c.font = H2; c.fill = SHADE
    if span > 1: ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col+span-1)

def body(ws, txt, row, col=1, span=1, italic=False):
    c = ws.cell(row, col, txt)
    c.font = TXT_ITAL if italic else TXT
    c.alignment = WRAP
    if span > 1: ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col+span-1)
    ws.row_dimensions[row].height = max(15, 15 * (1 + txt.count("\n")))

def tbl(ws, start_row, headers, rows, col=1, neg_cols=(), pos_cols=()):
    for i, h in enumerate(headers):
        c = ws.cell(start_row, col+i, h); c.font = TXT_BOLD; c.fill = SHADE; c.border = BOX
    for r_off, row_data in enumerate(rows):
        r = start_row + 1 + r_off
        bg = ALT if r_off % 2 == 1 else None
        for i, v in enumerate(row_data):
            c = ws.cell(r, col+i, v); c.border = BOX
            if bg: c.fill = bg
            if isinstance(v, float): c.number_format = "0.00"
            c.font = TXT
            if (col+i) in neg_cols and isinstance(v, (int,float)):
                c.font = NEG if v < 0 else POS
    return start_row + 1 + len(rows)

# ── Sheet 1: Settlement & Publication overview ──────────────────────────────
ws = wb.active
ws.title = "Settlement & Publication dates"
setw(ws, [30, 20, 20, 20, 20, 20])

hdr(ws, "Settlement Date & Publication Date Analysis", span=6)
body(ws, "price_drift_pct = (close on day BEFORE publication date − close on settlement date) / settlement close × 100\n"
         "This measures how much the stock moved during the ~8-business-day FINRA publication lag, "
         "before the signal is visible to us.", 2, span=6)

sub(ws, "Settlement → Publication lag", 4, span=6)
body(ws, "Lag is consistently 8 business days. Calendar-day count is 10 or 12 depending on weekends.", 5, span=6, italic=True)
tbl(ws, 6, ["Calendar days lag", "Trades", "% of total"],
    [(int(lag), cnt, round(cnt/len(sel)*100,1)) for lag, cnt in lag_counts.items()])

sub(ws, "price_drift_pct distribution (selected LONG, n=1,927)", 10, span=6)
body(ws, "Median = 0% | Mean = +1.1% (pulled up by +732.8% outlier) | Q1 = −7.1% | Q3 = +3.5% | Min = −72.9% (SKYQ)", 11, span=6, italic=True)
stats_rows = [(stat, round(val,2)) for stat, val in drift_desc.items()]
tbl(ws, 12, ["Statistic", "price_drift_pct %"], stats_rows, neg_cols=(2,))

sub(ws, "Drift distribution by year", 22, span=6)
yr_rows = [(int(yr), int(row['count']), round(row['mean'],2), round(row['median'],2), round(row['std'],2))
           for yr, row in drift_by_year.iterrows()]
tbl(ws, 23, ["Year", "Count", "Mean drift %", "Median drift %", "Std %"], yr_rows, neg_cols=(3,4))

# ── Sheet 2: Performance by days since publication ──────────────────────────
ws2 = wb.create_sheet("Days since publication")
setw(ws2, [18, 16, 10, 12, 12, 12])

hdr(ws2, "Performance by Days Since Publication Date", span=6)
body(ws2, "Within the 10-trading-day window, does when the gap falls matter?", 2, span=6, italic=True)

sub(ws2, "Results by days-since-publication bucket", 4, span=6)
pub_rows = [(str(row['pub_bucket']), row['gap_threshold_pct'], int(row['n']),
             round(row['hit'],1), round(row['avg_ret'],2), round(row['sum_ret'],2))
            for _, row in perf_by_pub.iterrows()]
nxt = tbl(ws2, 5, ["Days since pub", "Gap threshold %", "Trades", "Hit %", "Avg ret %", "Sum ret %"],
          pub_rows, neg_cols=(5,6))

sub(ws2, "Key interpretation", nxt+1, span=6)
findings = [
    ("Days 0-2 (freshest)", "Strongest: +67.1%/@2.5%, +86.2%/@3.0%. SI data just became public, not yet priced in."),
    ("Days 6-10", "Second-best: +90.7%/@2.5%, +90.4%/@3.0%. Secondary opportunity within the window."),
    ("Days 11-15 ⚠", "NEGATIVE at both thresholds (−44.0%/−81.2%). Information may be stale by this point."),
    ("Days 21+", "Positive again — likely the NEXT FINRA settlement cycle for the same ticker."),
]
ws2.cell(nxt+2, 1, "Days bucket").font = TXT_BOLD; ws2.cell(nxt+2, 1).fill = SHADE
ws2.cell(nxt+2, 2, "Interpretation").font = TXT_BOLD; ws2.cell(nxt+2, 2).fill = SHADE
ws2.merge_cells(start_row=nxt+2, start_column=2, end_row=nxt+2, end_column=6)
for i, (a, b) in enumerate(findings, nxt+3):
    ws2.cell(i, 1, a).font = TXT_BOLD; ws2.cell(i, 1).border = BOX
    ws2.cell(i, 2, b).font = TXT; ws2.cell(i, 2).alignment = WRAP; ws2.cell(i, 2).border = BOX
    ws2.merge_cells(start_row=i, start_column=2, end_row=i, end_column=6)
    ws2.row_dimensions[i].height = 40

# ── Sheet 3: Performance by drift quartile ─────────────────────────────────
ws3 = wb.create_sheet("Drift quartile analysis")
setw(ws3, [34, 16, 10, 12, 12, 12])

hdr(ws3, "Performance by price_drift_pct Quartile", span=6)
body(ws3, "Does bigger pre-publication price drop → bigger squeeze? Tested by quartile of price_drift_pct.", 2, span=6, italic=True)
body(ws3, f"Quartile boundaries: Q1 < {q_vals[0.25]:.1f}% | Q2 {q_vals[0.25]:.1f}% to {q_vals[0.5]:.1f}% | "
          f"Q3 {q_vals[0.5]:.1f}% to {q_vals[0.75]:.1f}% | Q4 > {q_vals[0.75]:.1f}%", 3, span=6, italic=True)

sub(ws3, "Results by drift quartile", 5, span=6)
dq_rows = [(str(row['drift_q']), row['gap_threshold_pct'], int(row['n']),
            round(row['hit'],1), round(row['avg_ret'],2), round(row['sum_ret'],2))
           for _, row in perf_by_drift.iterrows()]
nxt3 = tbl(ws3, 6, ["Drift quartile", "Gap threshold %", "Trades", "Hit %", "Avg ret %", "Sum ret %"],
           dq_rows, neg_cols=(5,6))

sub(ws3, "Key findings", nxt3+1, span=6)
body(ws3,
     "Q1 (most negative drift, < −7.1%) is NEGATIVE at both thresholds (−20.0%/−16.8%). "
     "These are blowup situations — SKYQ (−24.3% drift), ZNTL (−32.0%), CAPR (−10.4%), RILY (−12.0%) etc. "
     "They continued to fall on the gap day instead of squeezing.\n\n"
     "Q2 (−7.1% to 0%) is the BEST quartile (+124.5%/@2.5%, +59.1%/@3.0%). "
     "Moderate negative drift — price fell enough to attract SI attention but not blown up.\n\n"
     "IMPLICATION: The drift_low ranking selects Q1 first (most negative). "
     "The positive full-sample result is driven by Q2/Q3 being selected on days when extreme names are absent. "
     "A MINIMUM DRIFT FLOOR (e.g., price_drift_pct > −10%) might improve results "
     "by excluding the worst Q1 blowups — not yet tested train/test disciplined.",
     nxt3+2, span=6)

# ── Sheet 4: Trade-level data (selected LONG) ──────────────────────────────
ws4 = wb.create_sheet("Trade data (selected LONG)")
export_cols = [
    'trade_date', 'ticker', 'settlement_date', 'publication_date',
    'lag_cal_days', 'days_since_pub',
    'gap_threshold_pct', 'gap_pct',
    'price_drift_pct', 'volume_ratio',
    'entry_price', 'exit_price', 'exit_time_et', 'exit_rule',
    'realized_return_pct', 'selected', 'selection_rule'
]
export = sel[export_cols].copy()
export['trade_date']       = export['trade_date'].dt.strftime('%Y-%m-%d')
export['settlement_date']  = export['settlement_date'].dt.strftime('%Y-%m-%d')
export['publication_date'] = export['publication_date'].dt.strftime('%Y-%m-%d')

col_widths = [14, 8, 16, 18, 14, 16, 18, 10, 16, 14, 12, 12, 14, 14, 18, 10, 18]
setw(ws4, col_widths)
for i, col in enumerate(export_cols, 1):
    c = ws4.cell(1, i, col); c.font = TXT_BOLD; c.fill = SHADE; c.border = BOX

for r_off, (_, row) in enumerate(export.iterrows()):
    r = 2 + r_off
    bg = ALT if r_off % 2 == 1 else None
    for i, col in enumerate(export_cols, 1):
        val = row[col]
        if isinstance(val, (np.integer,)): val = int(val)
        elif isinstance(val, (np.floating,)): val = float(val)
        elif isinstance(val, bool): val = str(val)
        c = ws4.cell(r, i, val)
        c.font = TXT; c.border = BOX
        if bg: c.fill = bg
        if isinstance(val, float): c.number_format = "0.00"
        if col == 'realized_return_pct' and isinstance(val, float):
            c.font = NEG if val < 0 else POS

# ── Sheet 5: Full trade data (all trades, not just selected) ───────────────
ws5 = wb.create_sheet("All trades")
all_cols = [
    'trade_date', 'ticker', 'side', 'settlement_date', 'publication_date',
    'gap_threshold_pct', 'gap_pct', 'rule_name',
    'price_drift_pct', 'entry_price', 'exit_price',
    'exit_rule', 'realized_return_pct', 'selected', 'selection_rule'
]
all_df = df[all_cols].copy()
for c in ['trade_date', 'settlement_date', 'publication_date']:
    all_df[c] = pd.to_datetime(all_df[c]).dt.strftime('%Y-%m-%d')

col_widths2 = [14, 8, 8, 16, 18, 16, 10, 26, 16, 12, 12, 14, 18, 10, 18]
setw(ws5, col_widths2)
for i, col in enumerate(all_cols, 1):
    c = ws5.cell(1, i, col); c.font = TXT_BOLD; c.fill = SHADE; c.border = BOX

for r_off, (_, row) in enumerate(all_df.iterrows()):
    r = 2 + r_off
    bg = ALT if r_off % 2 == 1 else None
    for i, col in enumerate(all_cols, 1):
        val = row[col]
        if isinstance(val, (np.integer,)): val = int(val)
        elif isinstance(val, (np.floating,)): val = float(val)
        elif isinstance(val, bool): val = str(val)
        c = ws5.cell(r, i, val)
        c.font = TXT; c.border = BOX
        if bg: c.fill = bg
        if isinstance(val, float): c.number_format = "0.00"
        if col == 'realized_return_pct' and isinstance(val, float):
            c.font = NEG if val < 0 else POS

OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT_XLSX)
print(f"wrote {OUT_XLSX}")
