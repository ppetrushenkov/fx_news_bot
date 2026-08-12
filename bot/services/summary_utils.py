from datetime import date, datetime

import pandas as pd
from sqlalchemy import select, func

from utils.alerts import get_user_timezone, get_user_importance_settings
from utils.datetime_utils import _utc_calendar_day_bounds
from utils.text import format_high_impact_event_html
from db.database import SessionLocal
from db.models import Events


def get_events_for_date(requested_date: date) -> pd.DataFrame:
    """Return all events for specified date as a dataframe"""
    db = SessionLocal()

    try:
        stmt = select(Events).where(func.date(Events.date) == requested_date)
        today_events = pd.read_sql(stmt, con=db.connection())
        return today_events

    except Exception as e:
        print("Error in get_events_for_date(): %s", e)

    finally:
        db.close()


def get_summary_for(start_date: datetime, user_id: int) -> tuple[str | int | None, int | None]:
    db = SessionLocal()

    day_start, day_end = _utc_calendar_day_bounds(start_date)

    try:
        tz = get_user_timezone(db, user_id)
        importance = get_user_importance_settings(db, user_id)

        rows = db.execute(
            select(Events)
            .where(Events.date >= day_start, Events.date < day_end, Events.importance.in_(importance))
            .order_by(Events.date)
        ).scalars().all()

        if not rows:
            return None, None
        return "\n".join(format_high_impact_event_html(ev, tz) for ev in rows), len(rows)

    except Exception as e:
        print("Error in get_summary_for: %s", e)
        return -1, -1

    finally:
        db.close()
