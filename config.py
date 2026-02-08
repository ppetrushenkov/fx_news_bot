# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram Bot
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_API_KEY')
    TWELVE_API = os.getenv('TWELVE_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'forex_news_bot.db')
    
    # Scheduling  # TODO: Make Update hours and minutes configurable
    NEWS_UPDATE_HOUR = int(os.getenv('NEWS_UPDATE_HOUR', '0'))
    NEWS_UPDATE_MINUTE = int(os.getenv('NEWS_UPDATE_MINUTE', '5'))
    
    # Model paths
    VOLATILITY_MODEL_PATH = os.getenv('VOLATILITY_MODEL_PATH', 'models/volatility_model.cbm')
    
    # API endpoints
    TRADINGVIEW_NEWS_API = os.getenv('TRADINGVIEW_NEWS_API', 'https://economic-calendar.tradingview.com/events')
