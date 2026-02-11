from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy.orm import sessionmaker

from db.database import engine, get_db
from db.models import TodayEconomicNews
from db.utils import convert_tz_to_moscow, fetch_economic_news, custom_datetime_crop_to_closest_half_hour

from config import Config

import asyncio
import pandas as pd


# Create session factory for scheduler
SchedulerSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    

def shift_dates_back_by_delta(event_times: pd.Series, delta: str = '30min') -> pd.Series:
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
    """Sets and start the scheduler for the current day.
    """
    db = next(get_db())
    query = db.query(TodayEconomicNews)
    df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)
    unique_dates_shifted = shift_dates_back_by_delta(df['date'])
    times = shift_dates_back_by_delta(unique_dates_shifted)
