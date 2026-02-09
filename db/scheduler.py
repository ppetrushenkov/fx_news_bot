from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import asyncio
import requests
import pandas as pd

from datetime import datetime, timedelta

from sqlalchemy.orm import sessionmaker

from db.database import engine
from db.models import EconomicNews
from config import Config


# Create session factory for scheduler
SchedulerSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def fetch_economic_news():
    """Fetch economic news for the current day."""
    countries = ['US', 'FR', 'GB', 'EU', 'AU', 'DE', 'JP']  # TODO: add more countries
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    url = 'https://economic-calendar.tradingview.com/events'
    headers = {'Origin': 'https://in.tradingview.com'}
    payload = {
        'from': f'{today}T00:00:00.000Z',
        'to': f'{tomorrow}T00:00:00.000Z',
        'countries': ','.join(countries)
    }
    
    try:
        response = requests.get(url, headers=headers, params=payload)
        data = response.json()
        news_df = pd.DataFrame(data['result'])
        return news_df
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()


def populate_database():
    """Populate database with daily economic news."""
    news_df = fetch_economic_news()
    
    if not news_df.empty:
        # Create database session
        db = SchedulerSession()
        
        try:
            # Process news data and insert into database
            # TODO: Check Pandas version `to_sql()`
            news_df.to_sql(EconomicNews.__tablename__, db.bind, if_exists='append', index=False)
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
