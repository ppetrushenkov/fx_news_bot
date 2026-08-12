from aiogram import Router, F
from aiogram import types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command

from bot.keyboards import build_main_buttons


router = Router()


# ─────────────────────── Start ────────────────────────────────
@router.message(CommandStart())
async def start(message: types.Message):
    intro_text = """
👋 Welcome! I'm the FX Volatility Alert Bot.

I keep an eye on the major USD pairs and warn you about upcoming
volatility around scheduled macro news events — NFP, rate decisions,
CPI, and more. About an hour before high-impact news, my models
forecast the expected price range, chance of a choppy/"chaos" move,
and the risk of a false breakout (swing failure pattern) — so you
know whether it's better to sit the news out, trade it carefully,
or watch for a range-trade setup.

What I can do:
📅 Show today's, tomorrow's, and this week's economic calendar
🧠 Give you an on-demand market forecast
🔔 Send daily/weekly digests and volatility alerts
⚙️ Filter news by importance and set your own alert preferences

⚠️ Before I can send you alerts at the right local time, please set
your timezone with /set_gmt — this only takes a second and I'll use
it for every alert going forward."""
    await message.answer(intro_text, reply_markup=build_main_buttons())


# TODO: Add /statistics - command to show the events that most volatile or chaotic
# ─────────────────────── Help ────────────────────────────────
@router.message(Command('help'))
@router.message(F.text == "❓ Help")
async def help_(message: types.Message):
    await message.answer("""
<b>Available commands:</b>
Base:
/start - Start the bot
/help - Call help

Settings:
/set_alerts - Configure notifications
/set_importance - Configure importance filters
/set_gmt - Configure time zone
/set_importance - Configure importance filters

Summary:
/today_summary - events for today
/tomorrow_summary - events for tomorrow
/weekly_summary - events for the week

Prediction:
/check_the_market - Check the market now with closest events and current prices
""",
        parse_mode=ParseMode.HTML
    )
