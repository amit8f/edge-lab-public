"""DoD announcement parsing on made-up paragraphs (no network)."""
from tracks.main import dod_text as T

PAGE = """<h1>Contracts for Jan. 3, 2024</h1><p>NAVY</p>
<p>Kratos Unmanned Aerial Systems Inc., Sacramento, California, is awarded a $28,412,000 firm-fixed-price contract for
target drones. Work will be performed in Sacramento. The Naval Air Systems Command is the contracting activity.</p>
<p>Lockheed Martin Corp., Rotary and Mission Systems, Owego, New York, is awarded a $63,224,000 modification (P00012) to
previously awarded contract N00019-21-C-0001.</p>
<p>The following companies have been awarded a combined $1.2 billion multiple-award contract: Acme Inc., Dayton, Ohio;
Beta LLC, Austin, Texas.</p>
<p>AT&amp;T Corp., Oakton, Virginia, has been awarded a $9.5 million contract for network services.</p>"""


def test_awards_parse_name_value_and_flags():
    a = T.awards(PAGE)
    assert [x["name"] for x in a] == ["Kratos Unmanned Aerial Systems Inc.", "Lockheed Martin Corp.",
                                      "The following companies", "AT&T Corp."]
    assert a[0]["value"] == 28_412_000 and not a[0]["modification"] and not a[0]["multiple"]
    assert a[1]["modification"]
    assert a[2]["multiple"] and a[2]["value"] == 1.2e9
    assert a[3]["value"] == 9.5e6


def test_clean_name():
    assert T.clean_name("Kratos Unmanned Aerial Systems Inc.") == "KRATOS UNMANNED AERIAL SYSTEMS"
    assert T.clean_name("AT&T Corp.") == "AT AND T"
    assert T.clean_name("Lockheed Martin Corp") == T.clean_name("LOCKHEED MARTIN CORPORATION")
