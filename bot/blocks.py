from __future__ import annotations


from db.database import create_tables
from db.data_handler import DBHandler


def create_block() -> None:
    """
    Block 1: Create tables:
    - Events: Past and future events (UTC)
    - Prices: TwelveData historical prices
    - Ranges: Ranges, that produced every event in 1, 3, 6 and 24 hours
    - Predictions: Store predictions for every hour, that contains at least one event
    - UserSubscriptions: Store telegram user ids, that subscribed on alerts
    """
    create_tables()


def update_block() -> None:
    """
    Block 2: Check if any data in DataBase. If not, populate with all data for the last 2 months and next week further.
    Else check the last data in tables and populate with a fresh data.
    For the events also check the future data to ensure the data still the same.
    """
    db = DBHandler()

    db.update_events()
    db.update_hourly_prices()
    db.update_daily_prices()
    db.update_ranges()
    
    print('\n////////////////////////////////////')
    print('[INFO] The database has been updated')
    print('////////////////////////////////////\n')
    return
