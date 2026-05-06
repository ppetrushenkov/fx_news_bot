from time import time
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy import func, select

from db.database import SessionLocal
from db.models import TodayEconomicNews, TodayEventsAggregated, UserSubscription
from db.data_handler import DBHandler

from ml.predictor import VolatilityPredictor

from utils import (convert_tz_to_moscow, 
                   fetch_economic_news, 
                   fetch_economic_news_for_a_week,
                   custom_datetime_crop_to_closest_half_hour, 
                   get_datetime_list_to_set_scheduler
                   )

from config import Config
from app import bot, dp

import asyncio
import pandas as pd


# Create session factory for scheduler
session = SessionLocal()

# Predictor
predictor = VolatilityPredictor()


async def send_alert_to_users(predictions: pd.DataFrame):
    """Send alert to users if there is some trigger alert."""
    subscribers = session.query(UserSubscription).all()
    
    # Format the prediction message
    message_text = "🔔 *Market Alert - News Coming Out!* 🔔\n\n"
    
    for _, pred in predictions.iterrows():
        message_text += f"📈 *Ticker: {pred['ticker']}*\n"
        message_text += f"⏰ *Time:* {pred['datetime']}\n\n"
        
        message_text += "*Predicted Movement (Quantile Ranges):*\n"
        message_text += f"• 1h Range: {pred['trg_future_range_1h']:.2f} pips\n"
        message_text += f"• 3h Range: {pred['trg_future_range_3h']:.2f} pips\n"
        message_text += f"• 6h Range: {pred['trg_future_range_6h']:.2f} pips\n"
        message_text += f"• 24h Range: {pred['trg_future_range_24h']:.2f} pips\n\n"
        
        message_text += "*Chaos Indicators:*\n"
        message_text += f"• 1h Chaos: {'Yes' if pred['trg_is_chaos_1h'] else 'No'}\n"
        message_text += f"• 3h Chaos: {'Yes' if pred['trg_is_chaos_3h'] else 'No'}\n"
        message_text += f"• 6h Chaos: {'Yes' if pred['trg_is_chaos_6h'] else 'No'}\n"
        message_text += f"• 24h Chaos: {'Yes' if pred['trg_is_chaos_24h'] else 'No'}\n\n"
        
        message_text += "*Movement Type:*\n"
        message_text += f"• Trend: {'Yes' if pred['trg_is_trend'] else 'No'}\n"
        message_text += f"• Flat: {'Yes' if pred['trg_is_flat'] else 'No'}\n"
        message_text += "----------------------------\n"

    try:
        for subscriber in subscribers:
            await bot.send_message(
                chat_id=subscriber.chat_id, 
                text=message_text,
                parse_mode='Markdown'
            )
    except Exception as e:
        print(f'Error sending alert to users: {e}')

# -----------------------------------------+
#    SCHEDULER: Populate table every day   |
# -----------------------------------------+
def populate_database():
    """Populate database with economic news for the coming week."""
    db_handler = DBHandler()
    need_to_update = db_handler.is_need_to_update_today_events()

    if not need_to_update:
        print("[INFO] There is no need to populate database yet")
        return

    news_df = fetch_economic_news_for_a_week()
    
    # If news data is not empty, insert into database
    if news_df.shape[0] > 0:
        db_handler.write_table_to_database(news_df, TodayEconomicNews.__tablename__)
        print("Database populated successfully with news for the coming week")

    # Aggregate news data and write to database
    print("Populating table with aggregated news...")
    # Using the predictor's method for aggregation
    agg_news = predictor.add_features_for_news(news_df)
    db_handler.write_table_to_database(df=agg_news, table_name=TodayEventsAggregated.__tablename__)
    print("Aggregation table populated successfully")

    print("Set up scheduler for checking the market before news is out")
    set_multi_hour_scheduler_for_a_day()
    print("Scheduler set up successfully")


async def scheduled_everyday_population():
    """Async function to run database population."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, populate_database)


def setup_everyday_population_scheduler():
    """Setup and start the scheduler."""
    scheduler = AsyncIOScheduler()
    
    # Schedule daily news population
    scheduler.add_job(
        scheduled_everyday_population,
        CronTrigger(hour=Config.NEWS_UPDATE_HOUR, minute=Config.NEWS_UPDATE_MINUTE),
        id='daily_news_population'
    )
    
    scheduler.start()
    return scheduler


# ---------------------------------------------+
#    SCHEDULER: Check the market before news   |
# ---------------------------------------------+
def get_predictions_for_current_market() -> pd.DataFrame:
    """Pipeline function to check the market before news is out.
    It will return the Pandas dataframe with predictions for every ticker.
    """
    db_handler = DBHandler()
    events = db_handler.get_aggregated_events_for_coming_hour()
    prices = db_handler.get_last_prices()
    predictions = predictor.get_predictions(events, prices)
    
    return predictions


async def check_the_market_and_alert_the_users() -> None:
    """Function to check the market when the news are coming.
    If there is some trigger alert, the bot notify you about it."""
    print('[INFO] Ready to check the market')
    predictions = get_predictions_for_current_market()

    # Alert if predictions are not empty
    if len(predictions) > 0:
        print('[INFO] Alert the users')
        await send_alert_to_users(predictions)


async def scheduled_check_the_market():
    """Async function to check the market berore news is out."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, check_the_market_and_alert_the_users)


def set_multi_hour_scheduler_for_a_day():
    """
    Setup and start the scheduler on the specific times, right 30min (can be also set on 1hour)
    before the event is out for TODAY only.
    """
    data_retriever = DBHandler()
    df = data_retriever.get_events_for_today()
    
    # Filter for today's events
    df['date'] = pd.to_datetime(df['date'])
    today = datetime.now().date()
    df_today = df[df['date'].dt.date == today]

    if df_today.empty:
        print("No events found for today. Nothing to schedule.")
        return None

    times_before_events = get_datetime_list_to_set_scheduler(df_today['date'], delta='30min')

    print('The scheduler will be set on this times: \n', '\n'.join([t.strftime('%Y-%m-%d %H:%M') for t in times_before_events]))

    multi_hour_scheduler = AsyncIOScheduler()
    for sch_time in times_before_events:
        multi_hour_scheduler.add_job(
            scheduled_check_the_market,
            CronTrigger(hour=sch_time.hour, minute=sch_time.minute),
            id=f'market_check_{sch_time.strftime("%H%M")}_{str(int(time()))}'
        )
    multi_hour_scheduler.start()
    return multi_hour_scheduler
