from config import Config
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from db.database import get_db
from db.models import TodayEconomicNews, UserSubscription
from sqlalchemy.sql import func
import pandas as pd
import logging


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


bot = Bot(token=Config.TELEGRAM_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    user = message.from_user
    await message.answer(
            f'Hi {user.first_name}! Welcome to the Forex News Bot. '
            'I will notify you about important economic events and market predictions. \n'
            '\n'
            'To get started, use the /help command to see the list of available commands. \n'
            '\n'
            'Available commands:\n'
            '/start - Start the bot\n'
            '/help - Show this help message\n'
            '/subscribe - Subscribe to alerts\n'
            '/unsubscribe - Unsubscribe from alerts\n'
            '/daily_summary - Show daily market summary for today\n'
            '/set_daily_summary_time - Set the time, when the bot will send you a summary'
        )
    
@dp.message(Command('help'))
async def help(message: types.Message):
    await message.answer(
        'Available commands:\n'
        '/start - Start the bot\n'
        '/help - Show this help message\n'
        '/subscribe - Subscribe to alerts\n'
        '/unsubscribe - Unsubscribe from alerts\n'
        '/daily_summary - Show daily market summary for today\n'
        '/settime - Set (or change) user timezone'
        '/set_daily_summary_time - Set the time, when the bot will send you a summary'
    )

@dp.message(Command('subscribe'))
async def subscribe(message: types.Message):
    try:
        # Ask about user timezone
        # TODO: Make bot ask about user timezone before inserting into subscription table
        # Insert info about user into UserSubscription table
        db = next(get_db())
        db.add(
            UserSubscription(
                user_id=message.from_user.id,
                chat_id=message.chat.id,
                subscribed_to_alerts=True,
                subscribed_to_daily_summary=True,
                created_at=func.now()
            )
        )
        db.commit()
        await message.answer('You have successfully subscribed to alerts and daily summary.')
    except Exception as e:
        logger.error(f"Error in subscribe command: {e}")
        await message.answer('An error occurred while subscribing.')


@dp.message(Command('unsubscribe'))
async def unsubscribe(message: types.Message):
    try:
        db = next(get_db())
        db.query(UserSubscription).filter(UserSubscription.user_id == message.from_user.id).delete()
        db.commit()
        await message.answer('You have successfully unsubscribed from alerts and daily summary.')
    except Exception as e:
        logger.error(f"Error in unsubscribe command: {e}")
        await message.answer('An error occurred while unsubscribing.')
        
# TODO: add the function of setting time for daily summary in subscribe function

@dp.message(Command('daily_summary'))
async def daily_summary(message: types.Message):
    print('[INFO] Running daily_summary command')
    db = next(get_db())
    try:
        # Query all records from TodayEconomicNews table
        query = db.query(TodayEconomicNews)
        df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)

        # If dataframe is not empty -> form the summary
        if not df.empty:
            current_date = df.date.dt.date.iloc[0]
            high_impact_events_count = df[df['importance'] == 1].shape[0]
            high_impact_events = df[df['importance'] == 1]['title'].tolist()
            
            daily_summary = \
            f'Daily market summary for {current_date}: \n' \
            f'High impact events count: {high_impact_events_count} \n' \
            f'High impact events: {'\n -'.join([' '] + high_impact_events)}' 
            await message.answer(daily_summary)
        else:
            await message.answer('No daily summary found.')
    
    except Exception as e:
        logger.error(f"Error in daily_summary command: {e}")
        await message.answer('An error occurred while getting daily summary.')


@dp.message()
async def echo(message: types.Message):
    await message.answer(message.text)


async def monitoring_the_market():
    pass


async def main():
    print('Telegram bot started')
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    main()