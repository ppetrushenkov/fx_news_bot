from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy.orm import sessionmaker

from db.database import engine
from db.models import TodayEconomicNews
from db.utils import convert_tz_to_moscow, fetch_economic_news

from config import Config

import asyncio


# Create session factory for scheduler
SchedulerSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
