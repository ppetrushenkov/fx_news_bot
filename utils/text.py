from datetime import timezone, timedelta
from html import escape as html_escape
from typing import Dict, Literal

import numpy as np
import pandas as pd

from db.models import Events
from utils.datetime_utils import _to_user_tz, utc_now


def _esc(value) -> str:
    return html_escape(str(value)) if value is not None else "N/A"


def format_high_impact_event_html(ev: Events | dict, tz: timezone, *, time_only: bool = False) -> str:
    if ev.date:
        local_dt = _to_user_tz(ev.date, tz)
        fmt = "%H:%M" if time_only else "%Y-%m-%d %H:%M"
        event_time = f"{local_dt.strftime(fmt)} UTC{local_dt.strftime('%z')[:3]}:{local_dt.strftime('%z')[3:]}"
    else:
        event_time = "N/A"

    time_left = get_time_left(ev)

    importance_map = {-1: "Low", 0: "Medium", 1: "<b>High</b>"}

    lines = (
        f"• <b>{_esc(ev.title)}</b> (time left: {time_left})\n"
        f"  - When: {event_time}\n"
        f"  - Currency: <i>{_esc(ev.currency)}</i>\n"  # <code>{_esc(ev.currency)}</code>
        f"  - Previous: {_esc(ev.previous)}\n"
        f"  - Forecast: {_esc(ev.forecast)}\n"
        f"  - Importance: {importance_map.get(ev.importance, 'Unknown')}\n"
    )
    url = (ev.source_url or "").strip() if isinstance(ev.source_url, str) else ""
    if url:
        lines += f'  - Source: <a href="{html_escape(url)}">{_esc(ev.source)}</a>\n'
    return lines


def get_time_left(ev: Events | dict) -> str:
    now = utc_now()
    time_left = ev.date.replace(tzinfo=None) - now.replace(tzinfo=None)
    total_seconds = time_left.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)} hours {int(minutes)} minutes"


def _chunk_telegram_html(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, rest = [], text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return chunks


def escape_markdown_v2(s: str) -> str:
    """Converts a string to an escaped MarkdownV2 monospaced string."""
    escaped_table = ""
    for char in s:
        # if char in r"_*[]()~`>#+-=|{}.!":
        if char in r"_[]()~`>#+-=|{}.!":
            escaped_table += f"\\{char}"
        else:
            escaped_table += char

    # 3. Wrap inside a monospaced code block so columns align perfectly
    return f"\n{escaped_table}\n"


def formulate_daily_prediction_message(predictions: dict) -> str:
    market_state = transform_daily_predictions(predictions)

    impulse_bars = [
        ticker
        for ticker, data in market_state.items()
        if data["Impulse_bar"]
    ]

    bar_type = [
        ticker for ticker, data in market_state.items()
        if data["Bar_type"]
    ]

    any_alerts = any([impulse_bars, bar_type])

    if any_alerts:
        message_lines = []
        message_lines.append(
            '❗️<b>High chance for Impulse today:</b>\n'
            + ", ".join(impulse_bars)
        ) if impulse_bars else ""

        message_lines.append(
            '❗️<b>Expected type of bar:</b>\n'
            + ", ".join(bar_type)
        ) if bar_type else ""

        text = '\n'.join(message_lines)
        return text
    else:
        return ""




def formulate_prediction_message(predictions: dict, ml_risk: Literal['Conservative', 'Base', 'Aggressive']) -> str:
    """Based on predictions formulate final structure to send it to user"""

    market_state = transform_predictions(predictions)

    # +---------------- RANGES ------------------+
    movement_1h = [
        ticker
        for ticker, data in market_state.items()
        if data["forecast"]["range_1h"]["p50"] >= 4  # If we expect very big bar like 4 ATR
    ]

    # TODO: Mult on atr and divide on daily atr
    movement_24h = [
        ticker
        for ticker, data in market_state.items()
        if data["forecast"]["range_24h"]["p50"] >= 8
    ]

    # +----------------- REGIME -----------------+
    table = {'ticker': [], '1 day': [], '2 days': [], 'Swings': []}  # Swings for 24 hours (or 1 day)
    for ticker, data in market_state.items():
        dir_1d = data['regime']['short']
        dir_2d = data['regime']['long']

        table["ticker"].append(ticker)
        table["1 day"].append(dir_1d if dir_1d != 'None' else '-')
        table["2 days"].append(dir_2d if dir_2d != 'None' else '-')
        table["Swings"].append(data['stats']['Direction Changes'])

    regime = pd.DataFrame(table)
    regime.set_index("ticker", inplace=True)
    order = ["EUR/USD", "GBP/USD", "USD/CHF", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD"]
    regime = regime.loc[order]
    regime.reset_index(inplace=True)

    markdown_regime = f"<pre>{regime.to_markdown(index=False)}</pre>"

    # +---------- NOISE DETECTION ---------------+
    # Chaos
    chaos = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"][ml_risk]["Chaos"]
    ]

    # Extremum Breakouts
    expansion = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"][ml_risk]["Extremum Breakout"]
    ]

    # Spikes
    spikes = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"][ml_risk]["Big Spike"]
    ]

    # Swing Failure Pattern (SFP)
    sfp = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"][ml_risk]["SFP"]
    ]

    any_alerts = any([movement_1h, movement_24h, chaos, expansion, spikes, sfp])

    # +---------------- Stack all together ---------------+

    if any_alerts:
        # Ranges
        message_lines = []
        message_lines.append(
            '❗️<b>Currencies that can fluctuate more than their 1 hour ATR:</b>\n'
            + ", ".join(movement_1h)
        ) if movement_1h else ""

        message_lines.append(
            '\n‼️<b>Currencies that can fluctuate more than their daily ATR:</b>\n'
            + ", ".join(movement_24h)
        ) if movement_24h else ""

        # Regime
        message_lines.append("\n📈 <b>Regime</b>")
        message_lines.append(markdown_regime)

        # Noise
        message_lines.append("\n🚨 <b>Noise</b>") if any([chaos, expansion, spikes, sfp]) else ""
        message_lines.append(f"<b>Possible chaos:</b> {"; ".join(chaos)}") if chaos else ""
        message_lines.append(f"<b>Possible double expansion:</b> {"; ".join(expansion)}") if expansion else ""
        message_lines.append(f"<b>Possible spikes:</b> {"; ".join(spikes)}") if spikes else ""
        message_lines.append(f"<b>Possible false breakouts:</b> {"; ".join(sfp)}") if sfp else ""

        text = '\n'.join(message_lines)
        return text

    else:
        return "No alerts."


def transform_daily_predictions(predictions: dict) -> Dict[str, np.ndarray]:
    result = {}

    for i, ticker in enumerate(predictions["tickers"]):
        result[ticker] = {
            'Impulse_bar': predictions["Impulse_bar"][i],
            'Bar_type': predictions["Bar_type"][i]
        }

    return result


def transform_predictions(predictions: dict) -> dict:
    market_state = {}

    for i, ticker in enumerate(predictions["tickers"]):
        market_state[ticker] = {
            "forecast": {
                "range_1h": {
                    "p10": predictions["total_range_1h"][i][0],
                    "p50": predictions["total_range_1h"][i][1],
                    "p90": predictions["total_range_1h"][i][2],
                },
                "range_3h": {
                    "p10": predictions["total_range_3h"][i][0],
                    "p50": predictions["total_range_3h"][i][1],
                    "p90": predictions["total_range_3h"][i][2],
                },
                "range_6h": {
                    "p10": predictions["total_range_6h"][i][0],
                    "p50": predictions["total_range_6h"][i][1],
                    "p90": predictions["total_range_6h"][i][2],
                },
                "range_24h": {
                    "p10": predictions["total_range_24h"][i][0],
                    "p50": predictions["total_range_24h"][i][1],
                    "p90": predictions["total_range_24h"][i][2],
                },
            },
            "regime": {
                "short": predictions["Regime in 1 day"][i],
                "long": predictions["Regime in 2 days"][i],
            },
            "noise": {
                "Conservative": {
                    "Big Spike": predictions["Big Spike"]["Conservative"][i],
                    "Extremum Breakout": predictions["Extremum Breakout"]["Conservative"][i],
                    "Chaos": predictions["Chaos"]["Conservative"][i],
                    "SFP": predictions["SFP"]["Conservative"][i],
                },
                "Base": {
                    "Big Spike": predictions["Big Spike"]["Base"][i],
                    "Extremum Breakout": predictions["Extremum Breakout"]["Base"][i],
                    "Chaos": predictions["Chaos"]["Base"][i],
                    "SFP": predictions["SFP"]["Base"][i],
                },
                "Aggressive": {
                    "Big Spike": predictions["Big Spike"]["Aggressive"][i],
                    "Extremum Breakout": predictions["Extremum Breakout"]["Aggressive"][i],
                    "Chaos": predictions["Chaos"]["Aggressive"][i],
                    "SFP": predictions["SFP"]["Aggressive"][i],
                },
            },
            "stats": {
                "Direction Changes": predictions["Direction Changes"][i],
            }
        }

    return market_state


def get_most_important_events(title: pd.Series):
    title = str(title).upper()

    # Priority mappings for specific events
    if 'BALANCE OF TRADE' in title:
        return 'Balance_of_Trade'
    if 'CPI' in title or 'INFLATION RATE' in title or 'PPI' in title:
        if 'CORE' in title:
            return 'Core_Inflation_rate'
        return 'Inflation_rate'
    if 'INTEREST RATE DECISION' in title or 'DEPOSIT FACILITY RATE' in title:
        return 'Interest_Rate_Decision'
    if 'NON FARM PAYROLLS' in title or 'NONFARM PAYROLLS' in title:
        return 'NFP'
    if 'GDP' in title:
        return 'GDP'
    if 'FOMC' in title:
        return 'FOMC'
    if 'PMI' in title:
        if 'MANUFACTURING' in title:
            return 'PMI_Manufacturing'
        if 'SERVICES' in title:
            return 'PMI_Services'
        return 'PMI'
    if 'RETAIL SALES' in title:
        return 'Retail_Sales'
    if 'UNEMPLOYMENT RATE' in title:
        return 'Unemployment_rate'

    return None
