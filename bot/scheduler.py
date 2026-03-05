from time import time
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy import func, select

from db.database import SessionLocal
from db.models import TodayEconomicNews, TodayEventsAggregated
from db.data_handler import DBHandler

from ml.predictor import VolatilityPredictor

from utils import (convert_tz_to_moscow, 
                   fetch_economic_news, 
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


# -----------------------------------------+
#    SCHEDULER: Populate table every day   |
# -----------------------------------------+
def populate_database():
    """Populate database with daily economic news."""
    db_handler = DBHandler()
    need_to_update = db_handler.check_if_need_to_update_today_events()

    if not need_to_update:
        print("There is no need to populate database yet")
        return

    news_df = fetch_economic_news()
    
    # If news data is not empty, insert into database
    if news_df.shape[0] > 0:
        db_handler.write_table_to_database(news_df, TodayEconomicNews.__tablename__)
        print("Database populated successfully")

    # Aggregate news data and write to database
    print("Populating table with aggregated news...")
    agg_news = predictor.add_features_for_news(news_df)
    db_handler.write_table_to_database(df=agg_news, table_name=TodayEventsAggregated.__tablename__)
    print("Aggregation table populated successfully")


async def scheduled_population():
    """Async function to run database population."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, populate_database)


def setup_scheduler():
    """Setup and start the scheduler."""
    scheduler = AsyncIOScheduler()
    
    # Schedule daily news population
    scheduler.add_job(
        scheduled_population,
        CronTrigger(hour=Config.NEWS_UPDATE_HOUR, minute=Config.NEWS_UPDATE_MINUTE),
        id='daily_news_population'
    )
    
    scheduler.start()
    return scheduler


# ---------------------------------------------+
#    SCHEDULER: Check the market before news   |
# ---------------------------------------------+
def check_the_market():
    """Pipeline function to check the market before news is out."""
    data_retriever = DBHandler()
    events = data_retriever.get_aggregated_events_for_coming_hour()
    prices = data_retriever.get_last_prices()
    predictions = predictor.get_predictions(events, prices)
    
    return predictions


def check_the_market_and_alert_the_users():
    """Function to check the market when the news are coming.
    If there is some trigger alert, the bot notify you about it."""
    print('[INFO] Ready to check the market')
    predictions = check_the_market()

    # Alert if predictions are not empty
    if len(predictions) > 0:
        # TODO: make TG bot support to write if the market may spread out.
        # TODO: How to write message through telegram bot from here?
        pass


async def scheduled_check_the_market():
    """Async function to check the market berore news is out."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, check_the_market_and_alert_the_users)


def set_multi_hour_scheduler_for_a_day():
    """
    Setup and start the scheduler on the specific times, right 30min (can be also set on 1hour)
    before the event is out.
    """
    data_retriever = DBHandler()
    df = data_retriever.get_events_for_today()
    times_before_events = get_datetime_list_to_set_scheduler(df['date'], delta='30min')

    print('The scheduler will be set on this times: \n', '\n'.join(times_before_events))

    multi_hour_scheduler = AsyncIOScheduler()
    for sch_time in times_before_events:
        multi_hour_scheduler.add_job(
            scheduled_check_the_market,
            CronTrigger(hour=sch_time.hour, minute=sch_time.minute),
            run_date=sch_time.date,
            id=f'daily_news_population_{str(int(time()))}'
        )
    multi_hour_scheduler.start()
    return multi_hour_scheduler
