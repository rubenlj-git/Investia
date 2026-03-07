import re
import requests
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timezone


urls = pd.read_excel("Data/FIN.xlsx", sheet_name="Links")["links"].to_list()

def get_rentabilidad_code(url: str, timeout: int = 30) -> str | None:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    }
    html = requests.get(url, headers=headers, timeout=timeout).text

    # Encuentra: id="rentabilidad-xxxxxxxx"
    m = re.search(r'id="rentabilidad-([a-z0-9]+)"', html, flags=re.IGNORECASE)
    return m.group(1) if m else None

def isin_from_finect_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].split("-")[0]

params = {"start": "2015-01-01"}

HEADERS = {
    "accept": "application/json",
    "key": os.environ["FINECT_API_KEY"],
    "referer": "https://www.finect.com/",
    "origin": "https://www.finect.com",
    "user-agent": "Mozilla/5.0",
}

COOKIES = {
    "_fi_s": os.environ["FINECT_COOKIE"]
}

df_final = pd.DataFrame()

for u in urls:
    print(u)
    code = get_rentabilidad_code(u)
    isin = isin_from_finect_url(u)

    url_api = f"https://api.finect.com/v4/products/collectives/funds/{code}/timeseries"
    r = requests.get(url_api, params={"start": "2015-01-01"},
                     headers=HEADERS, cookies=COOKIES)

    data = r.json()

    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["datetime"])
    df = df.drop(columns="datetime").set_index("date")
    df.columns = [isin]

    df_final = df if df_final.empty else df_final.join(df, how="outer")


df_yahoo = yf.download(
    ["BRYN.DE", "EGLN.L"],
    start="2015-01-01",
    progress=False
)["Close"]
df_yahoo.index = df_yahoo.index.tz_localize(None)
df_yahoo.columns = ["US0846707026", "IE00B4ND3602"]
df_final.index = df_final.index.tz_localize(None)
df_final = df_final.join(df_yahoo, how="outer")
df_final = df_final.reset_index().rename(columns={"index":"date"})
df_final["date"] = pd.to_datetime(df_final["date"], "%d/%m/%Y")
df_final["EUR"]= 1


df_final["_updated"] = datetime.now(timezone.utc).isoformat()
df_final.to_json("NAV.json", orient="records")
