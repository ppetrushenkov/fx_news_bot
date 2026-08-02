"""
Load CSV macro series from ``dataset/economic_indicators`` and attach them to a
news aggregate frame by calendar day (UTC).

For each file, the level and its first difference are taken from the **raw**
rows (e.g. month-over-month for monthly prints, day-over-day for VIX) before
expanding to a daily grid via forward-fill.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _default_indicators_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "dataset" / "economic_indicators"


def _read_level_and_diff_from_csv(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")
    date_col, val_col = df.columns[0], df.columns[1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[val_col], errors="coerce")
    level = pd.Series(values.values, index=dates, name=f"macro_{path.stem}")
    level = level[~level.index.isna()].sort_index()
    level = level[~level.index.duplicated(keep="last")]
    diff = level.diff()
    diff.name = f"{level.name}_diff"
    return level, diff


def load_economic_indicators_daily(
    indicators_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a daily DataFrame (naive date index) with ``macro_*`` and
    ``macro_*_diff`` per CSV. Diffs are computed on native observation dates
    then forward-filled across calendar days like the levels.
    """
    root = indicators_dir or _default_indicators_dir()
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {root}")

    blocks: list[pd.DataFrame] = []
    for p in paths:
        level, diff = _read_level_and_diff_from_csv(p)
        if level.empty:
            continue
        idx = pd.date_range(level.index.min(), level.index.max(), freq="D")
        blocks.append(
            pd.DataFrame(
                {
                    level.name: level.reindex(idx).ffill(),
                    diff.name: diff.reindex(idx).ffill(),
                }
            )
        )

    wide = pd.concat(blocks, axis=1)
    return wide.sort_index().ffill()


def join_economic_indicators_to_news_agg(
    news_agg: pd.DataFrame,
    time_col: str = "cropped_date",
    indicators_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Left-join macro level and ``macro_*_diff`` columns onto ``news_agg`` using
    the UTC calendar date of ``time_col``.
    """
    daily = load_economic_indicators_daily(indicators_dir=indicators_dir)
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index).normalize()

    out = news_agg.copy()
    join_key = pd.to_datetime(out[time_col], utc=True).dt.normalize()
    join_key = join_key.dt.tz_localize(None)

    macro = daily.reindex(join_key.values).reset_index(drop=True)
    macro.columns = [
        c if c.startswith("macro_") else f"macro_{c}" for c in macro.columns
    ]

    dup = [c for c in macro.columns if c in out.columns]
    if dup:
        macro = macro.drop(columns=dup)

    return pd.concat([out.reset_index(drop=True), macro], axis=1)
