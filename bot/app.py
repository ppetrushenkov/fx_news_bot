from config import Config
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command


bot = Bot(token=Config.TELEGRAM_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message):
    user = message.from_user
    await message.answer(
            f'Hi {user.first_name}! Welcome to the Forex News Bot. '
            'I will notify you about important economic events and market predictions.'
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
    )


@dp.message()
async def echo(message: types.Message):
    await message.answer(message.text)


async def main():
    print('Telegram bot started')
    await dp.start_polling(bot)


if __name__ == '__main__':
    main()