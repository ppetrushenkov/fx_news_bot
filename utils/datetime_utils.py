from datetime import datetime, timezone, date, timedelta, date as date_type
from typing import Optional


def utc_now() -> datetime:
    """Return current datetime in UTC."""
    return datetime.now(timezone.utc)


def _to_user_tz(when: datetime, tz: timezone) -> datetime:
    return when.replace(tzinfo=timezone.utc).astimezone(tz) if when.tzinfo is None else when.astimezone(tz)


def _weekly_high_impact_period_utc(now: datetime) -> tuple[datetime, datetime, date]:
    """[week_start, week_end_exclusive) in naive UTC through end of `_next_sunday_utc()` (inclusive Sunday)."""
    utc_today = now.astimezone(timezone.utc).date()
    week_start = datetime.combine(utc_today, datetime.min.time())
    last_sunday = next_sunday_utc(utc_today)
    week_end_exclusive = datetime.combine(last_sunday, datetime.min.time()) + timedelta(days=1)
    return week_start, week_end_exclusive, last_sunday


def next_sunday_utc(d: Optional[date_type] = None) -> date_type:
    d = d or utc_now().date()
    days_ahead = 6 - d.weekday()  # Monday=0 .. Sunday=6
    return d + timedelta(days=days_ahead if days_ahead != 0 else 7)


def _utc_calendar_day_bounds(when: datetime) -> tuple[datetime, datetime]:
    """Naive UTC midnight window [day_start, day_end) for the UTC calendar day of `when`."""
    if when.tzinfo is not None:
        d = when.astimezone(timezone.utc).date()
    else:
        d = when.date()
    day_start = datetime.combine(d, datetime.min.time())
    return day_start, day_start + timedelta(days=1)
