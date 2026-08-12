from datetime import datetime, timezone, timedelta

from aiogram import Router, F
from aiogram import types
from aiogram.enums import ParseMode
from aiogram.filters import Command

from bot.predictions import get_predictions_for_next_hour
from utils.alerts import get_user_settings
from utils.datetime_utils import utc_now
from utils.text import formulate_prediction_message
from db.data_handler import DBHandler
# from ml.predictor import FxRangePredictor

router = Router()


# ────────────────────────── Commands ───────────────────────────────────────────────
@router.message(Command('check_the_market'))
@router.message(F.text == "🧠 Make a forecast")
async def test_check_the_market(message: types.Message):
    db = DBHandler()

    # Get events, prices and predictions
    agg_events, events, predictions = await get_predictions_for_next_hour()

    # Get important news
    next_hour = utc_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    important_events = events[(events['importance'] >= 0) & (events['rounded_time'] == next_hour)]

    # title = "🔔 <b>ATTENTION: High-Impact News in 1 Hour!</b>\n\n"
    title = "🧠 <b>Market Prediction Update</b>\n\n"

    if important_events.empty:
        intro = "No important events are expected in the next hour \n"
        news_list = ""
        footer = ""

    else:
        intro = "The following major economic events are scheduled to be released shortly:\n\n"

        # Form the list of important events
        news_list = ""
        for i, row in important_events[['title', 'country', 'date', 'rounded_time']].iterrows():
            news_list += f"• [{row['country']}] {row['title']} ({row['date'].strftime("%H:%M")})\n"

        footer = "\n⚠️ Expect high market volatility. Manage your risks accordingly.\n\n"

    user_settings = get_user_settings(db=db, user_id=message.from_user.id)
    user_risk = user_settings.ml_risk_level.capitalize()

    text_predictions = formulate_prediction_message(predictions, ml_risk=user_risk)

    full_message = f"{title}{intro}{news_list}{footer}{text_predictions}"

    try:
        await message.answer(text=full_message, parse_mode=ParseMode.HTML)
        print("Alert sent successfully!")

    except Exception as e:
        print(f"Error sending message: {e}")
