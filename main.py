import asyncio
from bot.app import main as start_bot
from bot.scheduler import setup_everyday_population_scheduler
from ml.predictor import check_for_new_events
from db.database import create_tables
from bot.scheduler import populate_database


async def main():
    """Main application entry point."""
    # Initialize database
    create_tables()
    print("Database initialized")

    # Populate database with data
    populate_database()
    print("Database populated")
    
    # Setup scheduler
    scheduler = setup_everyday_population_scheduler()
    print("Scheduler started")

    await start_bot()
    print("Bot started")


if __name__ == '__main__':
    asyncio.run(main())