# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram Bot
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'forex_news_bot.db')
    
    # Scheduling
    NEWS_UPDATE_HOUR = int(os.getenv('NEWS_UPDATE_HOUR', '0'))
    NEWS_UPDATE_MINUTE = int(os.getenv('NEWS_UPDATE_MINUTE', '5'))
    
    # Data APIs
    TRADINGVIEW_NEWS_API = 'https://economic-calendar.tradingview.com/events'
    TWELVE_API = os.getenv('TWELVE_API_KEY')
    FRED_API = os.getenv('FRED_API_KEY')
    TIINGO_API = os.getenv('TIINGO_API_KEY')

    # TimeZone
    TZ = os.getenv('TZ', 'Etc/UTC')
