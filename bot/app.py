# bot/bot.py
import asyncio
import datetime
import logging
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from db.database import get_db
from db.models import TodayEconomicNews, UserSubscription
from config import Config

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Send a message when the command /start is issued."""
#     user = update.effective_user
#     chat_id = update.effective_chat.id
    
#     # Add user to database
#     db_generator = get_db()
#     db = next(db_generator)
    
#     try:
#         # Check if user already exists
#         user_subscription = db.query(UserSubscription).filter(
#             UserSubscription.user_id == user.id
#         ).first()
        
#         if not user_subscription:
#             # Create new user subscription
#             user_subscription = UserSubscription(
#                 user_id=user.id,
#                 chat_id=chat_id
#             )
#             db.add(user_subscription)
#             db.commit()
        
#         await update.message.reply_text(
#             f'Hi {user.first_name}! Welcome to the Forex News Bot. '
#             'I will notify you about important economic events and market predictions.'
#         )
    
#     except Exception as e:
#         logger.error(f"Error in start command: {e}")
#         db.rollback()
    
#     finally:
#         # Close the generator to properly close the session
#         try:
#             next(db_generator)
#         except StopIteration:
#             pass
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    # user = update.effective_user
    # chat_id = update.effective_chat.id
    
    # Greet the user
    # await update.message.reply_text(
    #         f'Hi {user.first_name}! Welcome to the Forex News Bot. '
    #         'I will notify you about important economic events and market predictions.'
    #     )
    """Send a message when the command /start is issued."""
    if update.effective_chat:
        await update.effective_chat.send_message("Hello! I am your bot.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        'Available commands:\n'
        '/start - Start the bot\n'
        '/help - Show this help message\n'
        '/subscribe - Subscribe to alerts\n'
        '/unsubscribe - Unsubscribe from alerts\n'
        '/daily_summary - Show daily market summary for today\n'
    )


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Subscribe user to alerts."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        # Get or create user subscription
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).first()
        
        if user_subscription:
            user_subscription.subscribed_to_alerts = True
            user_subscription.chat_id = chat_id
        else:
            user_subscription = UserSubscription(
                user_id=user_id,
                chat_id=chat_id,
                subscribed_to_alerts=True
            )
            db.add(user_subscription)
        
        db.commit()
        await update.message.reply_text('You have been subscribed to alerts!')
    
    except Exception as e:
        logger.error(f"Error in subscribe command: {e}")
        db.rollback()
        await update.message.reply_text('An error occurred while subscribing.')
    
    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unsubscribe user from alerts."""
    user_id = update.effective_user.id
    
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        user_subscription = db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id
        ).first()
        
        if user_subscription:
            user_subscription.subscribed_to_alerts = False
            db.commit()
            await update.message.reply_text('You have been unsubscribed from alerts.')
        else:
            await update.message.reply_text('You are not subscribed to alerts.')
    
    except Exception as e:
        logger.error(f"Error in unsubscribe command: {e}")
        db.rollback()
        await update.message.reply_text('An error occurred while unsubscribing.')
    
    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass


async def daily_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send daily market summary."""

    db = next(get_db())
    
    try:
        # Query all records from TodayEconomicNews table
        query = db.query(TodayEconomicNews)
        # Convert to pandas DataFrame
        df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)
        # print(df)
        high_impact_events_count = df[df['impact'] == 1].shape[0]
        high_impact_events = df[df['impact'] == 1]['event'].tolist()
        
        if daily_summary:
            message = f'''
            Daily summary for {daily_summary.date}:
            High impact events count: {high_impact_events_count}
            High impact events: {high_impact_events}
            '''
            await update.message.reply_text(message)
        else:
            await update.message.reply_text('No daily summary found.')
    
    except Exception as e:
        logger.error(f"Error in daily_summary command: {e}")
        await update.message.reply_text('An error occurred while getting daily summary.')


async def main():
    """Start the bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(Config.TELEGRAM_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(CommandHandler("daily_summary", daily_summary))

    await application.initialize()
    await application.start()

    print("Telegram bot started")

    # запускаем получение обновлений
    await application.bot.initialize()
    # await application.start_polling()

    # держим процесс живым
    await asyncio.Event().wait()

    # await application.bot.initialize()
    # await application.updater.start_polling()

    # # держим процесс живым
    # await asyncio.Event().wait()


if __name__ == '__main__':
    # Initialize database
    from db.database import create_tables
    create_tables()
    main()
