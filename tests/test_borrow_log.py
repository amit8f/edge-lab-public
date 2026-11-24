import pandas as pd

from tracks.tools.borrow_log import parse, save

TXT = """#BOF|2026.11.24|21:45:02
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|
AAPL|USD|APPLE INC|265598|US0378331005|3.5612|0.25|>10000000|BBG000B9XRY4|
ABOS|USD|ACUMEN PHARMA|1|US00509|-45.1|48.9|3000|X|
NONE|USD|NO SHARES|2|US1|NaN|NaN|0|Y|
#EOF|3
"""


def test_parse_reads_fee_and_availability():
    df = parse(TXT)
    assert list(df["symbol"]) == ["AAPL", "ABOS", "NONE"]
    a = df.set_index("symbol")
    assert a.loc["ABOS", "fee"] == 48.9 and a.loc["AAPL", "available"] == 10_000_000
    assert a.loc["NONE", "available"] == 0 and pd.isna(a.loc["NONE", "fee"])


def test_save_writes_one_file_per_day(tmp_path):
    p = save(parse(TXT), tmp_path, "2026-11-24")
    assert p.name == "2026-11-24.csv.gz" and len(pd.read_csv(p)) == 3
