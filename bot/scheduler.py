from __future__ import annotations

import asyncio
from datetime import timezone, timedelta
import datetime

from bot.services.summary_utils import get_events_for_date

from db.data_loader import get_economic_events
from db.data_handler import DBHandler

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

import pandas as pd
from sqlalchemy import delete

from db.database import SessionLocal
from db.models import Events

from bot.blocks import update_block
from utils.alerts import get_users_for_daily_alert, get_user_timezone, get_user_importance_settings
from utils.text import format_high_impact_event_html

from utils.utils import df_standardize_event_dates
from utils.datetime_utils import utc_now


def _get_next_event_times_minus_1h() -> list[datetime.datetime]:
    """Return one-hour-before event times that are still in the future.

    The window starts at the current UTC moment and ends at the end of the
    next calendar day, so the scheduler only prepares jobs for upcoming events.
    """
    now_utc = utc_now().replace(tzinfo=None)
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


# def _reschedule_tomorrow_ml_jobs(scheduler: AsyncIOScheduler) -> None:
#     """
#     Ensures we have one-off ML prediction jobs scheduled for tomorrow at (event_time - 1 hour).
#
#     This is intended to be called:
#     - once on startup (optional convenience), and
#     - once per day from the daily scheduler job (required for correctness).
#     """
#     # Remove previously created one-off jobs to avoid accumulating stale schedules.
#     # We use a stable prefix for IDs so cleanup is deterministic.
#     for job in scheduler.get_jobs():
#         if job.id.startswith("ml_before_event_"):
#             scheduler.remove_job(job.id)
#
#     for t in _get_next_event_times_minus_1h():
#         scheduler.add_job(
#             run_ml_prediction_for_upcoming_events,
#             DateTrigger(run_date=t),
#             id=f"ml_before_event_{t.strftime('%Y%m%d_%H%M')}",
#             replace_existing=True,
#         )


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


# def daily_scheduler_job(
#     *,
#     scheduler: AsyncIOScheduler,
# ) -> None:
#     """
#     Daily:
#     - Refresh events for tomorrow;
#     - Create one-off schedulers at (event_time - 1h) aggregated by time
#     """
#     db = DBHandler()
#
#     # 1. Get events for tomorrow
#     now = utc_now()
#     start_date = (now + timedelta(days=1)).date()
#     end_date = start_date + timedelta(days=1)
#
#     events_for_tomorrow = get_economic_events(
#         start_date=start_date.strftime("%Y-%m-%d"),
#         end_date=end_date.strftime("%Y-%m-%d"),
#     )
#     events_for_tomorrow = df_standardize_event_dates(events_for_tomorrow)
#
#     # 2. Replace old events with fresh ones
#     with SessionLocal() as sess:
#         q = delete(Events).where(Events.date == start_date)
#         sess.execute(q)
#
#         events_for_tomorrow.to_sql(
#             Events.__tablename__, sess.bind, if_exists='append', index=False
#         )
#
#         sess.commit()
#
#     _reschedule_tomorrow_ml_jobs(scheduler)

# async def morning_alert_dispatcher():
#     """
#     Start every hour at :00 minutes.
#     Calculates the GMT time for the hour of 8:00. A
#     """
#     now_utc = utc_now()
#     current_utc_hour = now_utc.hour
#     current_date = now_utc.date()
#
#     TARGET_HOUR = 10
#
#     # Calculate GMT, where TARGET HOUR is right now
#     gmt = TARGET_HOUR - current_utc_hour
#
#     if gmt > 12:
#         gmt -= 24
#     elif gmt <= -12:
#         gmt += 24
#
#     print(f"Initialize daily alert. UTC time: {current_utc_hour:02d}:00. Search for users with UTC{gmt:+d}")
#
#     # Get users with UTC offset equal to target_offset and check if they are subscribed on daily alerts
#     users_to_alert = get_users_for_daily_alert(gmt)
#
#     # If there are users to alert, proceed with the process
#     if len(users_to_alert) > 0:
#         db = SessionLocal()
#         today_events = get_events_for_date(current_date)
#
#         for user_id in users_to_alert:
#             tz = get_user_timezone(db, user_id)
#             importance = get_user_importance_settings(db, user_id)
#
#             if today_events is not None:
#                 today_events = today_events[today_events["importance"].isin(importance)]
#
#                 if not today_events.empty:
#                     message = "☀️ Good morning! Here is the daily update. Below is a list of today's events. \n\n" + \
#                               "\n".join(format_high_impact_event_html(ev, tz) for _, ev in today_events.iterrows())
#
#                     try:
#                         await bot.send_message(
#                             chat_id=user_id,
#                             text=message,
#                             parse_mode="HTML",
#                             disable_web_page_preview=True
#                         )
#                         await asyncio.sleep(0.1)
#
#                     except Exception as e:
#                         print(f"Failed to send notification to user {user_id}: {e}")
#
#             else:
#                 message = "☀️ Good morning! Here is the daily summary. No events match your importance filters for today."
#                 await bot.send_message(chat_id=user_id, text=message)
#                 continue


