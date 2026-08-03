from __future__ import annotations

from datetime import timezone, timedelta
import datetime
from typing import Optional

# from requests import Session

from bot.data_loader import get_economic_events
from db.data_handler import DBHandler

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# try:
#     from apscheduler.schedulers.asyncio import AsyncIOScheduler
#     from apscheduler.triggers.cron import CronTrigger
#     from apscheduler.triggers.date import DateTrigger
# except Exception:  # pragma: no cover
#     AsyncIOScheduler = None  # type: ignore
#     CronTrigger = None  # type: ignore
#     DateTrigger = None  # type: ignore

import pandas as pd
from sqlalchemy import text, select, delete

from config import Config
from db.database import SessionLocal
from db.models import Events

from bot.blocks import update_block

from bot.feature_engineer import _utc_now, _df_standardize_event_dates


async def run_ml_prediction_for_upcoming_events() -> None:
    """
    Placeholder.
    Triggered at (event_time - 1 hour). Here you’ll later:
    - collect all events coming out soon
    - fetch historical prices for tickers
    - run ML prediction
    """
    print('DO SOME ML')
    return


def _get_next_event_times_minus_1h() -> list[datetime.datetime]:
    """Return one-hour-before event times that are still in the future.

    The window starts at the current UTC moment and ends at the end of the
    next calendar day, so the scheduler only prepares jobs for upcoming events.
    """
    now_utc = _utc_now().replace(tzinfo=None)
    window_end = datetime.datetime.combine(
        (now_utc.date() + timedelta(days=1)),
        datetime.time(23, 59, 59, 999999),
    )

    with SessionLocal() as sess:
        q = sess.query(Events.date).filter(
            Events.date >= now_utc,
            Events.date < window_end,
            Events.importance >= 0  # Consider events with medium and high importance
        )
        events = pd.read_sql(q.statement, sess.bind)

    if events.empty:
        return []

    events['date'] = pd.to_datetime(events['date'], errors='coerce')
    events = events.dropna(subset=['date'])

    rounded_date = (
        events['date']
        .dt.floor('1h')
        .sort_values()
        .drop_duplicates()
    )

    scheduled_times = []
    for dt in rounded_date:
        candidate = (dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt) - timedelta(hours=1)
        if candidate >= now_utc:
            scheduled_times.append(candidate)

    return scheduled_times


def _reschedule_tomorrow_ml_jobs(scheduler: AsyncIOScheduler) -> None:
    """
    Ensures we have one-off ML prediction jobs scheduled for tomorrow at (event_time - 1 hour).

    This is intended to be called:
    - once on startup (optional convenience), and
    - once per day from the daily scheduler job (required for correctness).
    """
    # Remove previously created one-off jobs to avoid accumulating stale schedules.
    # We use a stable prefix for IDs so cleanup is deterministic.
    for job in scheduler.get_jobs():
        if job.id.startswith("ml_before_event_"):
            scheduler.remove_job(job.id)

    for t in _get_next_event_times_minus_1h():
        scheduler.add_job(
            run_ml_prediction_for_upcoming_events,
            DateTrigger(run_date=t),
            id=f"ml_before_event_{t.strftime('%Y%m%d_%H%M')}",
            replace_existing=True,
        )


def set_schedulers() -> AsyncIOScheduler:
    """
    Block 3:
    - Weekly scheduler: Sunday 18:00 UTC → weekly_scheduler_job
    - Daily scheduler: every day 00:05 UTC (Config default) → daily_scheduler_job
      After daily update, also sets one-off schedulers at (event_time - 1h) for tomorrow.
    """
    if AsyncIOScheduler is None or CronTrigger is None or DateTrigger is None:
        raise RuntimeError("Missing dependency `apscheduler`. Install it to run schedulers.")

    scheduler = AsyncIOScheduler(timezone=timezone.utc)

    # Weekly: Sunday 18:00 UTC
    scheduler.add_job(
        weekly_scheduler_job,
        CronTrigger(day_of_week="sun", hour=18, minute=0, timezone=timezone.utc),
        id="weekly_update_sun_1800_utc",
        replace_existing=False,
    )

    # # Daily: by config (UTC)
    # scheduler.add_job(
    #     daily_scheduler_job,
    #     CronTrigger(hour=Config.NEWS_UPDATE_HOUR, minute=Config.NEWS_UPDATE_MINUTE, timezone=timezone.utc),
    #     id="daily_update_tomorrow_events",
    #     kwargs={"scheduler": scheduler},
    #     replace_existing=True,
    # )

    # Convenience: also schedule tomorrow's one-off jobs immediately on startup.
    # The daily job will re-create these each day.
    _reschedule_tomorrow_ml_jobs(scheduler)

    scheduler.start()
    return scheduler


def weekly_scheduler_job() -> None:
    """
    Weekly (Sunday 18:00 UTC):
    - Refresh FutureEvents for next week
    - Move already released events to PastEvents
    - Update Prices
    - Populate EventRanges for the week
    """
    update_block()
    return


def daily_scheduler_job(
    *,
    scheduler: AsyncIOScheduler,
) -> None:
    """
    Daily:
    - Refresh events for tomorrow;
    - Create one-off schedulers at (event_time - 1h) aggregated by time
    """
    db = DBHandler()
    # 1. Get events for tomorrow
    now = _utc_now()
    start_date = (now + timedelta(days=1)).date()
    end_date = start_date + timedelta(days=1)

    events_for_tomorrow = get_economic_events(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    events_for_tomorrow = _df_standardize_event_dates(events_for_tomorrow)

    # 2. Replace old events with fresh ones
    with SessionLocal() as sess:
        q = delete(Events).where(Events.date == start_date)
        sess.execute(q)

        events_for_tomorrow.to_sql(
            Events.__tablename__, sess.bind, if_exists='append', index=False
        )
        
        sess.commit()

    _reschedule_tomorrow_ml_jobs(scheduler)