from datetime import timedelta
from config import Config

from db.database import get_db
from db.models import Events, UserSubscriptions

from bot.feature_engineer import _utc_now, _next_sunday_utc

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command

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
        '/weekly_summary - Show market summary for a week (till next Sunday)\n'
    )

@dp.message(Command('subscribe'))
async def subscribe(message: types.Message):
    try:
        # Ask about user timezone
        # TODO: Make bot ask about user timezone before inserting into subscription table
        # Insert info about user into UserSubscription table
        db = next(get_db())
        db.add(
            UserSubscriptions(
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
        db.query(UserSubscriptions).filter(UserSubscriptions.user_id == message.from_user.id).delete()
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
    start_date = _utc_now()
    end_date = start_date + timedelta(days=1)
    try:
        # Query all records from Events table for today (UTC)
        query = db.query(Events).filter(
            Events.date >= start_date.strftime('%Y-%m-%d'),
            Events.date < end_date.strftime('%Y-%m-%d')
        )
        df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)

        # If dataframe is not empty -> form the summary
        if not df.empty:
            # Filter for high impact events (importance == 1)
            high_impact_df = df[df['importance'] == 1].copy()

            if not high_impact_df.empty:
                daily_summary_lines = [
                    f"📅 Daily high-impact market summary ({start_date.strftime('%Y-%m-%d')}):",
                    f"\nHigh impact events count: {len(high_impact_df)}\n",
                ]

                for _, row in high_impact_df.iterrows():
                    event_time = pd.to_datetime(row["date"]).strftime("%Y-%m-%d %H:%M UTC")
                    event_title = row["title"]
                    currency = row.get("currency", "N/A")
                    prev = row.get("previous", "N/A")
                    forecast = row.get("forecast", "N/A")
                    source_url = row.get("source_url", None)

                    event_info = (
                        f"• <b>{event_title}</b>\n"
                        f"  - When: {event_time}\n"
                        f"  - Currency: <code>{currency}</code>\n"
                        f"  - Previous: {prev}\n"
                        f"  - Forecast: {forecast}\n"
                    )
                    if source_url and isinstance(source_url, str) and source_url.strip():
                        event_info += f"  - <a href=\"{source_url}\">Source</a>\n"

                    daily_summary_lines.append(event_info)

                daily_summary_text = "\n".join(daily_summary_lines)
                await message.answer(daily_summary_text, parse_mode="HTML", disable_web_page_preview=True)
            else:
                await message.answer("No high impact events found for today.")
        else:
            await message.answer('No daily summary found.')
    
    except Exception as e:
        logger.error(f"Error in daily_summary command: {e}")
        await message.answer('An error occurred while getting daily summary.')


@dp.message(Command('weekly_summary'))
async def weekly_summary(message: types.Message):
    print('[INFO] Running weekly_summary command')
    db = next(get_db())
    start_date = _utc_now()
    end_date = _next_sunday_utc()
    try:
        # Query all records from Events table for the upcoming week
        query = db.query(Events).filter(
            Events.date >= start_date.strftime('%Y-%m-%d'),
            Events.date < end_date.strftime('%Y-%m-%d')
        )
        df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)

        # If dataframe is not empty -> form the summary
        if not df.empty:
            # Filter for high impact events (importance == 1)
            high_impact_df = df[df['importance'] == 1].copy()

            if not high_impact_df.empty:
                weekly_summary_lines = [
                    f"📅 Weekly high-impact market summary ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):",
                    f"\nHigh impact events count: {len(high_impact_df)}\n"
                ]
                for _, row in high_impact_df.iterrows():
                    # Format event date/time
                    event_time = pd.to_datetime(row['date']).strftime('%Y-%m-%d %H:%M UTC')
                    event_title = row['title']
                    currency = row.get('currency', 'N/A')
                    prev = row.get('previous', 'N/A')
                    forecast = row.get('forecast', 'N/A')
                    source_url = row.get('source_url', None)

                    # Event line with available info
                    event_info = (
                        f"• <b>{event_title}</b>\n"
                        f"  - When: {event_time}\n"
                        f"  - Currency: <code>{currency}</code>\n"
                        f"  - Previous: {prev}\n"
                        f"  - Forecast: {forecast}\n"
                    )
                    # Add hyperlink if source_url exists
                    if source_url and isinstance(source_url, str) and source_url.strip():
                        event_info += f"  - <a href=\"{source_url}\">Source</a>\n"
                    weekly_summary_lines.append(event_info)
                weekly_summary = "\n".join(weekly_summary_lines)
                await message.answer(weekly_summary, parse_mode="HTML", disable_web_page_preview=True)
            else:
                await message.answer('No high impact events found for the coming week.')
        else:
            await message.answer('No weekly summary found.')

    except Exception as e:
        logger.error(f"Error in weekly_summary command: {e}")
        await message.answer('An error occurred while getting weekly summary.')


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