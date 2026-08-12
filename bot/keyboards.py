from aiogram import types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

from bot.parameters import TZ_OPTIONS
from db.models import UserSettings


def build_main_buttons() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="📅 Events for today"),
        KeyboardButton(text="📅 Events for tomorrow"),
    )
    builder.row(
            KeyboardButton(text="📊 Events for the week"),
            KeyboardButton(text="🧠 Make a forecast"),
        )

    builder.row(
        KeyboardButton(text="🔔 Notification settings"),
        KeyboardButton(text="📍 Timezone settings"),
    )

    builder.row(
            KeyboardButton(text="⚙️ Importance settings"),
            KeyboardButton(text="⚙️ Set ML risk"),
        )

    builder.row(KeyboardButton(text="❓ Help"))

    return builder.as_markup(resize_keyboard=True)


def get_importance_settings_keyboard(settings: UserSettings):
    builder = InlineKeyboardBuilder()

    # Теперь достаем значения через точку
    low_status = "🟢" if settings.show_low_importance else "🔴"
    medium_status = "🟢" if settings.show_medium_importance else "🔴"
    high_status = "🟢" if settings.show_high_importance else "🔴"

    builder.row(
        types.InlineKeyboardButton(text=f"{low_status} Low", callback_data="toggle_show_low_importance"),
        types.InlineKeyboardButton(text=f"{medium_status} Medium", callback_data="toggle_show_medium_importance")
    )
    builder.row(
        types.InlineKeyboardButton(text=f"{high_status} High", callback_data="toggle_show_high_importance")
    )

    return builder.as_markup()


def get_alert_settings_keyboard(settings: UserSettings):
    builder = InlineKeyboardBuilder()

    # Теперь достаем значения через точку
    daily_status = "🟢" if settings.daily_alerts else "🔴"
    weekly_status = "🟢" if settings.weekly_alerts else "🔴"
    vol_status = "🟢" if settings.chaos_alerts else "🔴"

    builder.row(
        types.InlineKeyboardButton(text=f"{daily_status} Daily", callback_data="toggle_daily_alerts"),
        types.InlineKeyboardButton(text=f"{weekly_status} Weekly", callback_data="toggle_weekly_alerts")
    )
    builder.row(
        types.InlineKeyboardButton(text=f"{vol_status} Volatility", callback_data="toggle_chaos_alerts")
    )

    return builder.as_markup()


def build_tz_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=label, callback_data=f"tz_{offset}")
        for label, offset in TZ_OPTIONS
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton(text="✏️ Another / Enter manually", callback_data="tz_manual")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_risk_keyboard() -> InlineKeyboardMarkup:
    buttons = [[
        InlineKeyboardButton(text="Conservative", callback_data="risk_conservative"),
        InlineKeyboardButton(text="Base", callback_data="risk_base"),
        InlineKeyboardButton(text="Aggressive", callback_data="risk_aggressive")
    ]]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
