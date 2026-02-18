from time import time
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, select

from db.database import SessionLocal
from db.models import TodayEconomicNews
from db.data_handler import DataRetriever

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
    


def check_if_need_update() -> bool:
    """Return True if current date is not the same as the min date """
    q = select(func.min(TodayEconomicNews.date))
    min_date = session.execute(q).scalar()  # TODO: check this method or use .scalars().all()[0]
    min_date = pd.to_datetime(min_date).date()
    date_now = datetime.now().date()
    if min_date == date_now:
        return False
    return True


def populate_database():
    """Populate database with daily economic news."""
    need_to_update = check_if_need_update()
    if not need_to_update:
        print("There is no need to populate database yet")
        return

    news_df = fetch_economic_news()
    # TODO: What time need to be set globally here?
    # news_df['date'] = convert_tz_to_moscow(news_df['date'])
    news_df.sort_values('date', inplace=True)

    if 'scale' not in news_df.columns:
        news_df['scale'] = None
    
    if not news_df.empty:
        try:
            # Process news data and insert into database
            news_df.to_sql(TodayEconomicNews.__tablename__, session.bind, if_exists='replace', index=False)
            print(f"Added {len(news_df)} news items to database")

        except Exception as e:
            print(f"Error populating database: {e}")
            session.rollback()
        
        finally:
            session.close()


async def scheduled_population():
    """Async function to run database population."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, populate_database)


async def check_the_market():
    """Async function to check the market when the news are coming.
    If there is some trigger alert, the bot notify you about it."""
    data_retriever = DataRetriever()
    predictions = data_retriever.check_the_market()
    
    if len(predictions) > 0:
        # TODO: make TG bot support to write if the market may spread out.
        # TODO: How to write message through telegram bot from here?
        pass


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


def set_multi_hour_scheduler_for_a_day():
    """
    Setup and start the scheduler on the specific times, right 30min (can be also set on 1hour)
    before the event is out.
    """
    data_retriever = DataRetriever()
    df = data_retriever.get_events_for_today()
    times_before_events = get_datetime_list_to_set_scheduler(df['date'], delta='30min')

    multi_hour_scheduler = AsyncIOScheduler()
    for sch_time in times_before_events:
        multi_hour_scheduler.add_job(
            check_the_market,
            CronTrigger(hour=sch_time.hour, minute=sch_time.minute),
            run_date=sch_time.date,
            id=f'daily_news_population_{str(int(time()))}'
        )
    multi_hour_scheduler.start()
    return multi_hour_scheduler
