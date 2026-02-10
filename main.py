import asyncio
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

    await start_bot()
    print("Bot started")


if __name__ == '__main__':
    asyncio.run(main())