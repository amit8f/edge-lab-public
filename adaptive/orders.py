"""Plan v3 orders (section 3, addition 1): every entry is a bracket order, so the stop is held by Alpaca
even if our job dies. Limit entry, day order (cancelled by the watcher after 15 minutes unfilled),
regular hours only. Builds the JSON body only; core.alpaca.post sends it to the paper endpoint."""

CANCEL_AFTER_MIN = 15


def bracket(symbol, side, shares, limit, stop, take_profit=None, client_id=None):
    if shares <= 0:
        raise ValueError("shares must be positive")
    if side == "long" and not stop < limit:
        raise ValueError("long stop must sit below the limit")
    if side == "short" and not stop > limit:
        raise ValueError("short stop must sit above the limit")
    body = {
        "symbol": symbol, "qty": str(int(shares)), "side": "buy" if side == "long" else "sell",
        "type": "limit", "limit_price": f"{limit:.2f}" if limit >= 1 else f"{limit:.4f}",
        "time_in_force": "day", "extended_hours": False,
        "order_class": "bracket" if take_profit else "oto",
        "stop_loss": {"stop_price": f"{stop:.2f}" if stop >= 1 else f"{stop:.4f}"},
    }
    if take_profit:
        body["take_profit"] = {"limit_price": f"{take_profit:.2f}" if take_profit >= 1 else f"{take_profit:.4f}"}
    if client_id:
        body["client_order_id"] = client_id
    return body
