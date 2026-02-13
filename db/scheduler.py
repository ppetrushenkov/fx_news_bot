from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy.orm import sessionmaker

from db.database import engine, get_db
from db.handlers import DataRetriever
from db.models import TodayEconomicNews
from db.utils import convert_tz_to_moscow, fetch_economic_news, custom_datetime_crop_to_closest_half_hour

from config import Config

import asyncio
import pandas as pd


# Create session factory for scheduler
SchedulerSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    

def get_datetime_list_to_set_scheduler(event_times: pd.Series, delta: str = '30min') -> pd.Series:
    """
    Takes the raw datetime column from TodayEconomicNews table
    and return the list of unique datetime values, when the events comes out
    and shift it on `delta` back.
    
    :param event_times: Pandas datetime column from TodayEconomicNews table
    :type event_times: pd.Series
    :param delta: The value that the datetime will move back to
    :type delta: str
    :return: The pd.Series of datetime values, shifted back on `delta` value
    :rtype: Series[Any]
    """
    crop_dates = event_times.apply(custom_datetime_crop_to_closest_half_hour)
    unique_dates = crop_dates.unique()
    unique_dates_shifted = unique_dates - pd.Timedelta(delta)
    return unique_dates_shifted


def populate_database():
    """Populate database with daily economic news."""
    news_df = fetch_economic_news()
    news_df['date'] = convert_tz_to_moscow(news_df['date'])
    news_df.sort_values('date', inplace=True)
    
    if not news_df.empty:
        # Create database session
        db = SchedulerSession()
        
        try:
            # Process news data and insert into database
            news_df.to_sql(TodayEconomicNews.__tablename__, db.bind, if_exists='append', index=False)
            print(f"Added {len(news_df)} news items to database")

        except Exception as e:
            print(f"Error populating database: {e}")
            db.rollback()
        
        finally:
            db.close()


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
    for time in times_before_events:
        multi_hour_scheduler.add_job(
            check_the_market,
            CronTrigger(hour=time.hour, minute=time.minute),
            run_date=time.date,
            id=f'daily_news_population_{time.strftime("%H:%M")}'  # TODO: Change on unix timestamp
        )
    multi_hour_scheduler.start()
    return multi_hour_scheduler
