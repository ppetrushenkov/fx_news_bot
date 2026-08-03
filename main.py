import asyncio
import os

from bot.app import main as start_bot
from bot.scheduler import set_schedulers
from bot.blocks import create_block, update_block

from rich.traceback import install
install(show_locals=True)



async def main():
    """Main application entry point."""
    # Block 1: Initialize database if not exists
    db_path = 'forex_news_bot.db'
    # if os.path.isfile(db_path):
    #     print(f"\n[INFO] Database {db_path} already exists")
    # else:
    #     create_block()
    #     print("\n[INFO] Database initialized. Create block completed")
    
    # Block 1: Initialize database if not exists
    create_block()
    print("\n[INFO] Database initialized. Create block completed")

    # Block 2: Update data
    update_block()
    print("\n[INFO] Update block completed")

    # Block 3: Setup schedulers
    # set_schedulers()
    # print("\n[INFO] Schedulers started")

    # Block 3: Start bot
    await start_bot()
    print("\n[INFO] Bot started")


if __name__ == '__main__':
    asyncio.run(main())