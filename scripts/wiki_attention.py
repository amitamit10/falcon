"""Free 'attention' data: daily Wikipedia page views of each TA-125 company (English article found by searching its Yahoo name). Point-in-time by construction (views of day t are known on day t+1)."""
import json, re, time, urllib.parse, requests, pandas as pd, yfinance as yf
from pathlib import Path
from tradingbot.data import load_tickers
UA = {"User-Agent": "tradingbot-research/0.1 (personal project)"}; out = Path("data/wiki"); rows = []
def clean(n): return re.sub(r"\b(ltd|inc|corp|corporation|company|co|plc|limited|holdings?|group|\.|,|\(.*?\))\b", " ", n.lower()).split()
for tk in load_tickers("data/tase125.txt"):
    f = out / f"{tk}.csv"
    if f.exists(): continue
    try:
        info = yf.Ticker(tk).info; name = info.get("longName") or info.get("shortName")
        if not name: continue
        r = requests.get("https://en.wikipedia.org/w/api.php", params=dict(action="opensearch", search=name, limit=3, format="json"), headers=UA, timeout=20).json()
        title = next((t for t in r[1] if set(clean(name)[:1]) <= set(clean(t))), None)
        if not title: rows.append(dict(ticker=tk, name=name, title=None)); continue
        u = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{urllib.parse.quote(title.replace(' ', '_'), safe='')}/daily/20150701/20261112"
        j = requests.get(u, headers=UA, timeout=30).json()
        d = pd.DataFrame(j["items"]); d["date"] = pd.to_datetime(d["timestamp"].str[:8]); d[["date", "views"]].to_csv(f, index=False)
        rows.append(dict(ticker=tk, name=name, title=title, days=len(d)))
    except Exception as e:
        rows.append(dict(ticker=tk, name=None, title=None, err=str(e)[:60]))
    time.sleep(0.2)
pd.DataFrame(rows).to_csv(out / "_map.csv", index=False); print(pd.DataFrame(rows).title.notna().sum(), "matched of", len(rows))
