import asyncio
import signal
import sys
from bot.app import main as start_bot
from db.scheduler import setup_scheduler
from bot.model_executor import check_for_new_events
from db.database import create_tables


async def main():
    """Main application entry point."""
    # Initialize database
    create_tables()
    print("Database initialized")
    
    # Setup scheduler
    scheduler = setup_scheduler()
    print("Scheduler started")
    
    # Start bot in a separate task
    # bot_task = asyncio.create_task(start_bot())
    # try:
    await start_bot()
    print("Bot started")
    
    # Keep the application running
    # try:
    #     await bot_task
    # except KeyboardInterrupt:
    #     print("Shutting down...")
    #     scheduler.shutdown()
    #     sys.exit(0)


if __name__ == '__main__':
    asyncio.run(main())