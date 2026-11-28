# WIRE0: how much earlier the free wires are than Benzinga (descriptive)

Registered 2026-11-28. Source: Amit, 2026-11-28 ("All three"; the wire feeds from the pre-news plan). Descriptive,
not counted (count stays 205).

The listener (core/listener.py, running since late October) already logs PR Newswire and Business Wire every 20
seconds with a hash of each normalised title. `tracks/news/wire_speed.py` (workflow wire-speed.yml) matches the last
20 finished days with Benzinga's headlines from Alpaca by that hash and reports, per wire: matched headlines, median
lead (Benzinga's time minus our first sight; a day's first read is left out), and the share seen 30 s, 60 s and 5
minutes earlier. GlobeNewswire answers 403 from GitHub and stays out.

If the wires are a minute or more earlier on a large share of NX3's events, the next step is a registered rule that
lets NX3 decide on the wire's copy (a new idea); if not, the wire path is dropped.
