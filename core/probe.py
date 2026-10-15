"""One-off check of outside data sources from GitHub's servers (the Claude sandbox can't reach them):
prints the HTTP status and the first lines of one sample of each, never any secret. Used to confirm
file formats before B0's code and the borrow logger depend on them.

Usage: python -m core.probe
"""
import ftplib
import io
import zipfile

import requests

UA = {"User-Agent": "edge-lab research probe"}


def show(name, fn):
    try:
        print(f"== {name}\n{fn()}\n")
    except Exception as e:  # noqa: BLE001 - a probe reports every failure and goes on
        print(f"== {name}\nFAILED: {type(e).__name__}: {str(e)[:300]}\n")


def http(url, n=600, headers=None, method="get", **kw):
    r = getattr(requests, method)(url, headers=headers or UA, timeout=60, **kw)
    return f"HTTP {r.status_code} {r.headers.get('content-type')}\n{r.text[:n]}"


def ibkr():
    buf = io.BytesIO()
    with ftplib.FTP("ftp3.interactivebrokers.com", timeout=60) as f:
        f.login("shortstock", "")
        names = f.nlst()
        f.retrbinary("RETR usa.txt", buf.write)
    lines = buf.getvalue().decode("latin-1").splitlines()
    return f"files: {names[:10]}\nlines: {len(lines)}\n" + "\n".join(lines[:4])


def ftd():
    import os
    r = requests.get("https://www.sec.gov/files/data/fails-deliver-data/cnsfails202401a.zip",
                     headers={"User-Agent": os.environ.get("SEC_USER_AGENT", "edge-lab probe")}, timeout=60)
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return f"{z.namelist()}\n" + z.read(z.namelist()[0]).decode("latin-1")[:400]


def main():
    show("IBKR stock-loan file", ibkr)
    show("FINRA Reg SHO daily short volume", lambda: http("https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240102.txt", 400))
    show("FINRA consolidated short interest API (no key)", lambda: http(
        "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest", 600, method="post",
        headers={**UA, "Accept": "application/json"},
        json={"limit": 3, "compareFilters": [{"compareType": "equal", "fieldName": "symbolCode", "fieldValue": "AAPL"}]}))
    show("FINRA short interest files page", lambda: http("https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files", 300))
    show("SEC fails-to-deliver 2024-01 first half", ftd)


if __name__ == "__main__":
    main()
