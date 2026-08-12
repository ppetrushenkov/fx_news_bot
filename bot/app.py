from apscheduler.triggers.date import DateTrigger

from bot.predictions import get_predictions_for_next_hour, get_daily_predictions
from bot.services.summary_utils import get_events_for_date
from config import Config

from db.data_handler import DBHandler
from db.database import SessionLocal
from ml.predictor import FxRangePredictor

from utils.alerts import get_users_for_chaos_predictions, get_user_importance_settings, get_user_timezone, \
    get_users_for_daily_alert
from utils.text import formulate_prediction_message, format_high_impact_event_html, formulate_daily_prediction_message
from bot.scheduler import weekly_scheduler_job, _get_next_event_times_minus_1h

from datetime import datetime, timedelta, timezone

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage


from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import asyncio
import logging

from bot.commands.base import router as base_router
from bot.commands.summaries import router as summary_router
from bot.commands.settings import router as settings_router
from bot.commands.predictions import router as predictions_router


# +=====================================+
# |       INITIALIZE VARIABLES          |
# +=====================================+
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the scheduler
scheduler = AsyncIOScheduler(timezone=timezone.utc)

# Initialize Routers
router = Router()

# Initialize Dispatcher and Bot
dp = Dispatcher(storage=MemoryStorage())
dp.include_routers(
    router,
    base_router,  # Start and Help
    summary_router,  # Event summaries (today, tomorrow, weekly summary)
    settings_router,  # Settings (set_gmt, set_alerts, set_importance)
    predictions_router  # Predictions (test_check_the_market)
)

bot = Bot(token=Config.TELEGRAM_TOKEN)

# +=====================================+
#     SET DAILY ALERTS AND SCHEDULER    |
# +=====================================+

async def morning_alert_dispatcher():
    """
    Start every hour at :00 minutes.
    Calculates the GMT time for the hour of 8:00. A
    """
    now_utc = datetime.now(timezone.utc)
    current_utc_hour = now_utc.hour
    today = now_utc.date()

    TARGET_HOUR = 22

    # Calculate GMT, where TARGET HOUR is right now
    gmt = TARGET_HOUR - current_utc_hour

    if gmt > 12:
        gmt -= 24
    elif gmt <= -12:
        gmt += 24

    logging.info(f"Initialize daily alert. UTC time: {current_utc_hour:02d}:00. Search for users with UTC{gmt:+d}")

    # Get users with UTC offset equal to target_offset and check if they are subscribed on daily alerts
    users_to_alert = get_users_for_daily_alert(gmt)

    # If there are users to alert, proceed with the process
    if len(users_to_alert) > 0:
        db = SessionLocal()
        today_events = get_events_for_date(today)
        agg_events, events, today_predictions = get_daily_predictions()
        prediction_answer = formulate_daily_prediction_message(today_predictions)

        for user_id in users_to_alert:
            tz = get_user_timezone(db, user_id)
            importance = get_user_importance_settings(db, user_id)

            if today_events is not None:
                today_events = today_events[today_events["importance"].isin(importance)]

                if not today_events.empty:
                    message = "☀️ Good morning! Here is the daily update. Below is a list of today's events. \n\n" + \
                              "\n".join(format_high_impact_event_html(ev, tz) for _, ev in today_events.iterrows())

                    try:
                        await bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )

                        await bot.send_message(
                            chat_id=user_id,
                            text=prediction_answer,
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        logging.error(f"Failed to send notification to user {user_id}: {e}")

            else:
                message = "☀️ Good morning! Here is the daily summary. No events match your importance filters for today."
                await bot.send_message(chat_id=user_id, text=message)
                continue


def _reschedule_tomorrow_ml_jobs(scheduler: AsyncIOScheduler) -> None:
    """
    Ensures we have one-off ML prediction jobs scheduled for tomorrow at (event_time - 1 hour).

    This is intended to be called:
    - once on startup (optional convenience), and
    - once per day from the daily scheduler job (required for correctness).
    """
    # Remove old jobs
    for job in scheduler.get_jobs():
        if job.id.startswith("ml_before_event_"):
            scheduler.remove_job(job.id)

    # Add new jobs for tomorrow
    for t in _get_next_event_times_minus_1h():
        scheduler.add_job(
            scheduled_check_the_market,
            DateTrigger(run_date=t),
            id=f"ml_before_event_{t.strftime('%Y%m%d_%H%M')}",
            replace_existing=True,
        )


async def daily_summary_dispatcher():
    """
    This function is intended to be scheduled to run once per day (e.g., at 00:00 UTC).
    It will reschedule the one-off ML prediction jobs for tomorrow's events.
    """
    db = DBHandler()
    db.update_events()
    _reschedule_tomorrow_ml_jobs(scheduler)


# +=====================================+
# |               HELP                  |
# +=====================================+

@dp.message(Command("show_jobs"))
async def show_jobs(message: types.Message):
    jobs = scheduler.get_jobs()
    if not jobs:
        await message.answer("No scheduled jobs.")
        return

    job_list = []
    for job in jobs:
        job_list.append(f"- [ID]: {job.id}, [Next Run]: {job.next_run_time}, [Trigger]: {job.trigger} \n")

    await message.answer("\n".join(job_list))


# ────────────────────────── Scheduler check the market ───────────────────────────────────────────────

async def scheduled_check_the_market():
    # Users subscribed on ML predictions
    users_to_alert = get_users_for_chaos_predictions()

    if not users_to_alert:
        print('[PREDICTIONS] No users subscribed on chaos alerts')
        return None

    # Get events, prices and predictions
    agg_events, events, predictions = await get_predictions_for_next_hour()

    # Send messages
    if len(users_to_alert) > 0 and predictions is not None:
        important_news = events[
            (events['importance'] >= 0) & (events['rounded_time'] == agg_events['rounded_time'].iloc[0])
        ]
        news_list = important_news['title'].unique().tolist()
        news_str = "\n ▫️ ".join(news_list) if news_list else "no important events in the next hour"

        intro = f"⚠️ <b>Market Prediction Update</b>\n\n" \
                "In the next hour, the following important news are expected:\n" \
                f"*{news_str}*\n\n" \
                "Please check the details below for potential market movements and noise detection. \n\n"

        text = formulate_prediction_message(events, agg_events, predictions)

        for user_id in users_to_alert:
            try:
                await bot.send_message(
                    chat_id=user_id,
                    text=intro,
                    parse_mode="HTML")

                await bot.send_message(
                    chat_id=user_id,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
                await asyncio.sleep(0.1)

            except Exception as e:
                logging.error(f"Can't send message to user {user_id}: {e}")
    return None


# +---------- RUN BOT -----------------+
async def main():
    # Weekly scheduled job to update the database with new events from the source
    scheduler.add_job(
        weekly_scheduler_job,
        CronTrigger(day_of_week="sun", hour=18, minute=0, timezone=timezone.utc),
        id="weekly_update_sun_1800_utc",
        replace_existing=True,
    )

    # Every hour runs a function to send daily summary to users who subscribed on daily alerts
    scheduler.add_job(
        morning_alert_dispatcher,
        trigger="cron",
        day_of_week='mon-fri',
        hour="*",
        minute=57
    )

    # Every day at 23:59 UTC runs a function to set schedulers on the next day for users who subscribed on chaos alerts (ML)
    scheduler.add_job(
        daily_summary_dispatcher,
        trigger=CronTrigger(hour=23, minute=59, timezone=timezone(timedelta(hours=0))),
        id="daily_summary_dispatcher",
        replace_existing=True,
    )
    await daily_summary_dispatcher()  # Run once at startup to ensure the next day's jobs are scheduled

    scheduler.start()

    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
