# from flask import session
from collections import defaultdict
from config import Config

from db.database import SessionLocal, get_db
from db.data_handler import DBHandler
from db.models import Events, UserSettings

from ml.predictor import FxRangePredictor
from ml.news_featuring import extract_news_features_pipeline, aggregate_events, floor_or_ceil
from ml.price_featuring import add_features, get_base_and_quote_currency

from bot.feature_engineer import _utc_now, _next_sunday_utc
from bot.scheduler import _get_next_event_times_minus_1h

from sqlalchemy import (
    func, 
    select, 
    cast, 
    Date,
    Integer
)
from html import escape as html_escape

from datetime import (
    datetime, 
    timedelta, 
    timezone, 
    date, 
    time
)

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.filters.callback_data import CallbackData
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery, 
    Message, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup, 
    KeyboardButton
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

import asyncio
import pytz
import calendar
import logging
import pandas as pd


# +=====================================+
# |            LOGGING                  |
# +=====================================+
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

icons = "❌⬅️➡️📍✅"

db = next(get_db())


router = Router()
dp = Dispatcher(storage=MemoryStorage())
dp.include_router(router)

bot = Bot(token=Config.TELEGRAM_TOKEN)

# Initialize the scheduler
scheduler = AsyncIOScheduler(timezone=timezone.utc)


TZ_OPTIONS = [
    ("Лондон (UTC+0)", 0),
    ("Франкфурт/Париж (UTC+1)", 1),
    ("Италия (UTC+2)", 2),
    ("Москва/Стамбул (UTC+3)", 3),
    ("Нью-Йорк (UTC-5)", -5),
    ("Чикаго (UTC-6)", -6),
    ("Сингапур/Гонконг (UTC+8)", 8),
]


class OnboardingStates(StatesGroup):
    waiting_for_timezone = State()
    waiting_for_alert_preferences = State()


def build_buttons() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(
        KeyboardButton(text="📅 События на сегодня"),
        KeyboardButton(text="📅 События на завтра"),
    )
    builder.row(KeyboardButton(text="📊 События на неделю"))
    builder.row(KeyboardButton(text="🧠 Сделать прогноз на текущий момент"))
    
    builder.row(KeyboardButton(text="🔔 Настройки уведомлений"))
    builder.row(KeyboardButton(text="📍 Настройки часового пояса"))
    builder.row(KeyboardButton(text="⚙️ Настройка важности событий"))
    builder.row(KeyboardButton(text="❓ Помощь"))

    return builder.as_markup(resize_keyboard=True)


# +=====================================+
#           SUMMARY BUTTONS             |
# +=====================================+
@dp.message(F.text == "📅 События на сегодня")
async def btn_today_summary(message: types.Message):
    await today_summary(message)

@dp.message(F.text == "📅 События на завтра")
async def btn_tomorrow_summary(message: types.Message):
    await tomorrow_summary(message)

@dp.message(F.text == "📊 События на неделю")
async def btn_weekly_summary(message: types.Message):
    await weekly_summary(message)

@dp.message(F.text == "🧠 Сделать прогноз на текущий момент")
async def btn_forecast(message: types.Message):
    await test_check_the_market(message)

@dp.message(F.text == "⚙️ Настройка важности событий")
async def btn_importance(message: types.Message, state: FSMContext):
    await set_importance(message, state)

@dp.message(F.text == "🔔 Настройки уведомлений")
async def btn_alerts(message: types.Message, state: FSMContext):
    await set_alerts(message, state)

@dp.message(F.text == "📍 Настройки часового пояса")
async def btn_timezone(message: types.Message, state: FSMContext):
    await set_gmt(message, state)
    
@dp.message(F.text == "❓ Помощь")
async def btn_help(message: types.Message):
    await help_(message)


# +=====================================+
#      SET ALERT TIME AND SCHEDULER     |
# +=====================================+
def get_user_ids_for_daily_alert(target_offset: int) -> list:
    """Return the list of user ids, that should get daily alerts at specific time"""
    db = SessionLocal()
    try:        
        stmt = select(UserSettings.user_id).where(
            cast(func.floor(UserSettings.user_timezone), Integer) == target_offset,
            UserSettings.daily_alerts == True
        )
        
        user_ids = db.scalars(stmt).all()
        return list(user_ids)

    finally:
        db.close()


def get_user_importance_settings(db, user_id: int) -> list:
    user_settings = db.get(UserSettings, user_id)

    importance = [
        i for i, imp_flg in zip([-1, 0, 1], [
            user_settings.show_low_importance,
            user_settings.show_medium_importance,
            user_settings.show_high_importance
        ]) if imp_flg
    ]

    return importance


async def hourly_alert_dispatcher():
    """
    Запускается каждый час в :00 минут.
    Вычисляет, в каком часовом поясе сейчас 8:00 утра, и делает по ним рассылку.
    """
    now_utc = datetime.now(timezone.utc)
    current_utc_hour = now_utc.hour
    current_date = now_utc.date()
    
    # Целевое время для отправки
    TARGET_HOUR = 8
    
    # Вычисляем, для какого timezone_offset сейчас наступило TARGET_HOUR
    # Формула: Offset = Target - Current_UTC
    # Пример: Если в UTC сейчас 5 утра, то 8 утра сейчас там, где offset = 8 - 5 = 3 (UTC+3)
    target_offset = TARGET_HOUR - current_utc_hour
    
    # Корректируем под 24-часовой формат (если offset уходит в минус или больше 12)
    if target_offset > 12:
        target_offset -= 24
    elif target_offset <= -12:
        target_offset += 24

    logging.info(f"Инициализация рассылки. Время UTC: {current_utc_hour:02d}:00. Ищем пользователей с UTC{target_offset:+d}")

    # Получаем из БД только тех, у кого включены уведомления И чей часовой пояс совпал
    users_to_alert = get_user_ids_for_daily_alert(target_offset)

    # Get events
    if len(users_to_alert) > 0:
        db = SessionLocal()
        today_events = get_events_for_date(current_date)

        for user_id in users_to_alert:
            tz = get_user_timezone(db, user_id)
            importance = get_user_importance_settings(db, user_id)

            if today_events is not None:
                today_events = today_events[today_events["importance"].isin(importance)]

                if not today_events.empty:
                    message = "☀️ Доброе утро! Дневная сводка. Ниже список событий на сегодня. \n" + \
                        "\n".join(_format_high_impact_event_html(ev, tz) for _, ev in today_events.iterrows())

                    try:
                        await bot.send_message(
                            chat_id=user_id, 
                            text=message, 
                            parse_mode="HTML", 
                            disable_web_page_preview=True
                        )
                        await asyncio.sleep(0.1)
                    
                    except Exception as e:
                        logging.error(f"Не удалось отправить уведомление пользователю {user_id}: {e}")

            else:
                message = "☀️ Доброе утро! Дневная сводка: Сегодня нет событий, соответствующих вашим фильтрам по важности."
                await bot.send_message(chat_id=user_id, text=message)
                continue


# +=====================================+
#           SCHEDULER FOR EVENTS        |
# +=====================================+

def _reschedule_tomorrow_ml_jobs(scheduler: AsyncIOScheduler) -> None:
    """
    Ensures we have one-off ML prediction jobs scheduled for tomorrow at (event_time - 1 hour).

    This is intended to be called:
    - once on startup (optional convenience), and
    - once per day from the daily scheduler job (required for correctness).
    """
    # Remove old jobs
    for job in scheduler.get_jobs():
        if job.id.startswith("ml_before_event_"):
            scheduler.remove_job(job.id)
    # Add new jobs for tomorrow
    for t in _get_next_event_times_minus_1h():
        scheduler.add_job(
            scheduled_check_the_market,
            DateTrigger(run_date=t),
            id=f"ml_before_event_{t.strftime('%Y%m%d_%H%M')}",
            replace_existing=True,
        )


async def daily_summary_dispatcher():
    """
    This function is intended to be scheduled to run once per day (e.g., at 00:00 UTC).
    It will reschedule the one-off ML prediction jobs for tomorrow's events.
    """
    _reschedule_tomorrow_ml_jobs(scheduler)


@dp.message(Command("show_jobs"))
async def show_jobs(message: types.Message):
    jobs = scheduler.get_jobs()
    if not jobs:
        await message.answer("Нет запланированных задач.")
        return

    job_list = []
    for job in jobs:
        job_list.append(f"ID: {job.id}, Next Run: {job.next_run_time}, Trigger: {job.trigger}")

    await message.answer("\n".join(job_list))


# +=====================================+
# |            SET TIMEZONE             |
# +=====================================+

def build_tz_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=label, callback_data=f"tz_{offset}")
        for label, offset in TZ_OPTIONS
    ]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton(text="✏️ Другой / ввести вручную", callback_data="tz_manual")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(CommandStart())
async def start(message: types.Message, state: FSMContext):
    intro_text = (
        "👋 Привет! Я бот для отслеживания волатильности на Forex.\n\n"
        "Я слежу за основными парами USD и сигнализирую о потенциальном "
        "хаосе на рынке за час до важных макро-событий (NFP, решения по ставке и т.д.)\n\n"
        "Чтобы присылать тебе алерты в правильное локальное время, "
        "укажи свой часовой пояс с помощью команды /set_gmt"
    )
    await message.answer(intro_text, reply_markup=build_buttons())


@dp.message(Command("set_gmt"))
async def set_gmt(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    settings = db.get(UserSettings, user_id)
    if settings is not None and settings.user_timezone is not None:
        sign = "+" if settings.user_timezone >= 0 else ""
        await message.answer(
            f"Текущий часовой пояс: UTC{sign}{settings.user_timezone}\n\n"
            "Если хочешь изменить его, выбери из списка или введи вручную:",
            reply_markup=build_tz_keyboard()
        )
        await state.set_state(OnboardingStates.waiting_for_timezone)
    else:
        await message.answer(
            "Выбери свой часовой пояс из списка или введи вручную:",
            reply_markup=build_tz_keyboard()
        )
        await state.set_state(OnboardingStates.waiting_for_timezone)


@router.callback_query(F.data == "tz_manual")
async def tz_manual_requested(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_text(
        "Введи свой часовой пояс числом относительно UTC.\n"
        "Например: 3, -5, 5.5 (для UTC+5:30)\n\n"
        "Диапазон: от -12 до +14"
    )
    await state.set_state(OnboardingStates.waiting_for_timezone)


@router.callback_query(F.data.startswith("tz_"), OnboardingStates.waiting_for_timezone)
async def tz_button_chosen(callback: CallbackQuery, state: FSMContext):
    # tz_manual уже отработал выше, сюда попадают только числовые tz_<offset>
    offset = float(callback.data.split("_")[1])

    await save_timezone(callback.from_user.id, offset)  # твоя функция сохранения в БД
    await state.clear()

    sign = "+" if offset >= 0 else ""
    await callback.answer()
    await callback.message.edit_text(
        f"✅ Часовой пояс установлен: UTC{sign}{offset}\n\n"
        f"Теперь я буду присылать алерты с учётом твоего локального времени."
    )


@router.message(OnboardingStates.waiting_for_timezone)
async def tz_text_input(message: Message, state: FSMContext):
    text = message.text.strip().replace(",", ".")
    try:
        offset = float(text)
        if not -12 <= offset <= 14:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Не понял. Введи число от -12 до +14, например: 3 или -4.5")
        return

    await save_timezone(message.from_user.id, offset)  # твоя функция сохранения в БД
    await state.clear()

    sign = "+" if offset >= 0 else ""
    await message.answer(
        f"✅ Часовой пояс установлен: UTC{sign}{offset}\n\n"
        f"Теперь я буду присылать алерты с учётом твоего локального времени."
    )


async def save_timezone(
    user_id: int,
    offset: float
) -> UserSettings:
    """
    Сохраняет (или создаёт) часовой пояс пользователя в UserSettings.

    :param user_id: Telegram user_id (PK таблицы)
    :param offset: смещение часового пояса, например 3.0 для UTC+3
    :return: обновлённый/созданный объект UserSettings
    """
    settings = db.get(UserSettings, user_id)

    if settings is None:
        # пользователя ещё нет в таблице — создаём с дефолтными quiet hours
        settings = UserSettings(
            user_id=user_id,
            user_timezone=offset
        )
        db.add(settings)
    else:
        settings.user_timezone = offset
        settings.updated_at = func.now()

    db.commit()
    db.refresh(settings)
    return settings


# +=====================================+
# |            SET ALERTS               |
# +=====================================+

@dp.callback_query(lambda c: c.data.startswith('toggle_show'))
async def toggle_importance(callback_query: types.CallbackQuery):
    setting_key = callback_query.data.replace("toggle_", "")
    user_id = callback_query.from_user.id

    settings = db.get(UserSettings, user_id)
    
    if not settings:
        # На случай, если настроек почему-то не оказалось в БД
        await callback_query.answer("Настройки не найдены.")
        return

    # 2. Меняем значение динамически по имени атрибута
    current_value = getattr(settings, setting_key)  # Эквивалентно settings.daily
    setattr(settings, setting_key, not current_value) # Инвертируем и записываем в объект

    # 3. Сохраняем изменения в саму базу данных
    try:
        db.commit()  # Фиксируем изменения в БД
    
    except Exception as e:
        db.rollback()  # Если что-то пошло не так, откатываем изменения
        await callback_query.answer("Ошибка при сохранении настроек.")
        return

    # 4. Обновляем клавиатуру в интерфейсе Telegram
    await callback_query.message.edit_reply_markup(
        reply_markup=get_importance_settings_keyboard(settings) # Передаем уже обновленный объект settings
    )
    await callback_query.answer()


@dp.callback_query(lambda c: c.data.startswith('toggle_'))
async def toggle_notification(callback_query: types.CallbackQuery):
    setting_key = callback_query.data.replace("toggle_", "")
    user_id = callback_query.from_user.id

    settings = db.get(UserSettings, user_id)
    
    if not settings:
        # На случай, если настроек почему-то не оказалось в БД
        await callback_query.answer("Настройки не найдены.")
        return

    # 2. Меняем значение динамически по имени атрибута
    current_value = getattr(settings, setting_key)  # Эквивалентно settings.daily
    setattr(settings, setting_key, not current_value) # Инвертируем и записываем в объект

    # 3. Сохраняем изменения в саму базу данных
    try:
        db.commit()  # Фиксируем изменения в БД
    
    except Exception as e:
        db.rollback()  # Если что-то пошло не так, откатываем изменения
        await callback_query.answer("Ошибка при сохранении настроек.")
        return

    # 4. Обновляем клавиатуру в интерфейсе Telegram
    await callback_query.message.edit_reply_markup(
        reply_markup=get_alert_settings_keyboard(settings) # Передаем уже обновленный объект settings
    )
    await callback_query.answer()


def get_alert_settings_keyboard(settings: UserSettings):
    builder = InlineKeyboardBuilder()
    
    # Теперь достаем значения через точку
    daily_status = "🟢" if settings.daily_alerts else "🔴"
    weekly_status = "🟢" if settings.weekly_alerts else "🔴"
    vol_status = "🟢" if settings.chaos_alerts else "🔴"
    
    builder.row(
        types.InlineKeyboardButton(text=f"{daily_status} Ежедневные", callback_data="toggle_daily_alerts"),
        types.InlineKeyboardButton(text=f"{weekly_status} Еженедельные", callback_data="toggle_weekly_alerts")
    )
    builder.row(
        types.InlineKeyboardButton(text=f"{vol_status} Волатильность", callback_data="toggle_chaos_alerts")
    )

    return builder.as_markup()


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


@dp.message(Command('set_importance'))
async def set_importance(message: types.Message, state: FSMContext):
    settings = db.get(UserSettings, message.from_user.id)

    if not settings:
        settings = UserSettings(user_id=message.from_user.id, show_low_importance=False, show_medium_importance=False, show_high_importance=True)
        db.add(settings)
        db.commit()
    
    await message.answer(
        text="⚙️ Настройки важности событий:",
        reply_markup=get_importance_settings_keyboard(settings) # Передаем объект напрямую
    )


@dp.message(Command('set_alerts'))
async def set_alerts(message: types.Message, state: FSMContext):
    settings = db.get(UserSettings, message.from_user.id)

    if not settings:
        settings = UserSettings(user_id=message.from_user.id, daily_alerts=False, weekly_alerts=False, chaos_alerts=False)
        db.add(settings)
        db.commit()
    
    await message.answer(
        text="⚙️ Настройки уведомлений:",
        reply_markup=get_alert_settings_keyboard(settings) # Передаем объект напрямую
    )


# +=====================================+
# |               HELP                  |
# +=====================================+
@dp.message(Command('help'))
async def help_(message: types.Message):
    await message.answer("""
<b>Доступные команды:</b>
/start - Запустить бот
/help - вызвать help
/set_alerts - настроить уведомления
/set_importance - настроить фильтры важности событий
/set_gmt - настроить часовой пояс
/settings - пользовательские настройки
/today_summary - события на сегодня
/tomorrow_summary - события на завтра
/weekly_summary - события на неделю
/statistics - показать общую статистику по событиям
/set_language - выбрать язык / set language
/test_check_the_market - Check the market now with closest events and current prices (FOR TEST)
""",
        parse_mode=ParseMode.HTML
    )


@dp.message(Command('settings'))
async def show_settings(message: types.Message):
    await message.answer("Not realized yet. Use /set_alerts or /set_gmt to configure your settings.")


# +=====================================+
# |               SUMMARY               |
# +=====================================+
def _weekly_high_impact_period_utc(now: datetime) -> tuple[datetime, datetime, date]:
    """[week_start, week_end_exclusive) in naive UTC through end of `_next_sunday_utc()` (inclusive Sunday)."""
    utc_today = now.astimezone(timezone.utc).date()
    week_start = datetime.combine(utc_today, datetime.min.time())
    last_sunday = _next_sunday_utc(utc_today)
    week_end_exclusive = datetime.combine(last_sunday, datetime.min.time()) + timedelta(days=1)
    return week_start, week_end_exclusive, last_sunday

def _utc_calendar_day_bounds(when: datetime) -> tuple[datetime, datetime]:
    """Naive UTC midnight window [day_start, day_end) for the UTC calendar day of `when`."""
    if when.tzinfo is not None:
        d = when.astimezone(timezone.utc).date()
    else:
        d = when.date()
    day_start = datetime.combine(d, datetime.min.time())
    return day_start, day_start + timedelta(days=1)

# ────────────────────────── Часовой пояс ────────────────────────────────────────────────

def get_user_timezone(db, user_id: int) -> timezone:
    settings = db.get(UserSettings, user_id)
    try:
        offset = float(settings.user_timezone) if settings and settings.user_timezone is not None else 0.0
        assert -12.0 <= offset <= 14.0
        return timezone(timedelta(hours=offset))
    except (TypeError, ValueError, AssertionError):
        logger.warning("Invalid timezone for user %s, falling back to UTC", user_id)
        return timezone.utc


def _to_user_tz(when: datetime, tz: timezone) -> datetime:
    return when.replace(tzinfo=timezone.utc).astimezone(tz) if when.tzinfo is None else when.astimezone(tz)


# ────────────────────────── Форматирование события ──────────────────────────────────────

def _esc(value) -> str:
    return html_escape(str(value)) if value is not None else "N/A"


def _format_high_impact_event_html(ev: Events | pd.Series, tz: timezone, *, time_only: bool = False) -> str:
    if ev.date:
        local_dt = _to_user_tz(ev.date, tz)
        fmt = "%H:%M" if time_only else "%Y-%m-%d %H:%M"
        event_time = f"{local_dt.strftime(fmt)} UTC{local_dt.strftime('%z')[:3]}:{local_dt.strftime('%z')[3:]}"
    else:
        event_time = "N/A"

    importance_map = {-1: "Low", 0: "Medium", 1: "<b>High</b>"}

    lines = (
        f"• <b>{_esc(ev.title)}</b>\n"
        f"  - When: {event_time}\n"
        f"  - Currency: <code>{_esc(ev.currency)}</code>\n"
        f"  - Previous: {_esc(ev.previous)}\n"
        f"  - Forecast: {_esc(ev.forecast)}\n"
        f"  - Importance: {importance_map.get(ev.importance, 'Unknown')}\n"
    )
    url = (ev.source_url or "").strip() if isinstance(ev.source_url, str) else ""
    if url:
        lines += f'  - Source: <a href="{html_escape(url)}">{_esc(ev.source)}</a>\n'
    return lines


def _chunk_telegram_html(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, rest = [], text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return chunks


# ────────────────────────── Дневная сводка ───────────────────────────────────────────────

def get_events_for_date(requested_date: date) -> pd.DataFrame:
    """Return all events for specified date as a dataframe"""
    db = SessionLocal()

    try:
        stmt = select(Events).where(func.date(Events.date) == requested_date)
        today_events = pd.read_sql(stmt, con=db.connection())
        return today_events
    
    except Exception as e:
        logger.error("Error in get_events_for_date(): %s", e)

    finally:
        db.close()


def get_summary_for(start_date: datetime, user_id: int) -> tuple[str | int | None, int | None]:
    day_start, day_end = _utc_calendar_day_bounds(start_date)
    db = SessionLocal()
    try:
        tz = get_user_timezone(db, user_id)
        importance = get_user_importance_settings(db, user_id)
        
        rows = db.execute(
            select(Events)
            .where(Events.date >= day_start, Events.date < day_end, Events.importance.in_(importance))
            .order_by(Events.date)
        ).scalars().all()

        if not rows:
            return None, None
        return "\n".join(_format_high_impact_event_html(ev, tz) for ev in rows), len(rows)
    except Exception as e:
        logger.error("Error in get_summary_for: %s", e)
        return -1, -1
    finally:
        db.close()


async def _send_daily_summary_answer(message: types.Message, *, day: datetime, empty_label: str) -> None:
    summary, events_cnt = get_summary_for(day, message.from_user.id)
    if summary == -1:
        await message.answer("An error occurred while getting daily summary.")
        return
    if not summary:
        await message.answer(f"No daily summary found for {empty_label}.")
        return

    report_day = day.astimezone(timezone.utc).date() if day.tzinfo else day.date()
    text = (
        f"📅 Daily high-impact market summary ({report_day.isoformat()}):\n"
        f"\nHigh impact events count: {events_cnt}\n\n{summary}"
    )
    await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@dp.message(Command("today_summary"))
async def today_summary(message: types.Message):
    await _send_daily_summary_answer(message, day=_utc_now(), empty_label="today")


@dp.message(Command("tomorrow_summary"))
async def tomorrow_summary(message: types.Message):
    await _send_daily_summary_answer(message, day=_utc_now() + timedelta(days=1), empty_label="tomorrow")


# ────────────────────────── Недельная сводка ─────────────────────────────────────────────

@dp.message(Command("weekly_summary"))
async def weekly_summary(message: types.Message):
    now = _utc_now()
    week_start, week_end_exclusive, last_sunday = _weekly_high_impact_period_utc(now)
    db = SessionLocal()
    try:
        tz = get_user_timezone(db, message.from_user.id)

        # TODO: Add importance filter
        rows = db.execute(
            select(Events)
            .where(Events.date >= week_start, Events.importance == 1)
            .order_by(Events.date)
        ).scalars().all()

        if not rows:
            await message.answer(
                f"No high-impact events found for this week "
                f"({week_start.date().isoformat()} → {last_sunday.isoformat()} UTC)."
            )
            return

        by_day: dict[date, list[Events]] = defaultdict(list)
        for ev in rows:
            if ev.date is not None:
                by_day[_to_user_tz(ev.date, tz).date()].append(ev)

        header = (
            "📅 <b>Weekly high-impact summary</b>\n"
            f"<i>{week_start.date().isoformat()} → {last_sunday.isoformat()} UTC</i>\n"
            f"Total: <b>{len(rows)}</b> event{'s' if len(rows) != 1 else ''}\n"
        )

        sections = []
        for day_key in sorted(by_day):
            day_events = by_day[day_key]
            wd = calendar.day_name[day_key.weekday()]
            label = f"──────────────────────────\n<b>{wd}</b> · {day_key.isoformat()} · {len(day_events)} high-impact\n──────────────────────────"
            body = "\n".join(_format_high_impact_event_html(ev, tz, time_only=True) for ev in day_events)
            sections.append(f"{label}\n\n{body}")

        full_text = header + "\n\n" + "\n\n".join(sections)
        for chunk in _chunk_telegram_html(full_text):
            await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)

    except Exception as e:
        logger.error("Error in weekly_summary command: %s", e)
        await message.answer("An error occurred while getting weekly summary.")
    finally:
        db.close()

# +------------- TEST CHECK THE MARKET -------------------+

def preprocess_prices_with_features(
    prices: pd.DataFrame,
    *,
    period: int = 21
) -> pd.DataFrame:
    """Add per-ticker price features and drop raw OHLC / vol columns."""
    list_of_prices = []
    for ticker, group in prices.groupby('ticker'):
        group['instrument'] = ticker
        group[['base_currency', 'quote_currency']] = get_base_and_quote_currency(ticker)
        group = add_features(group, period=period)
        group.drop(
            ['open', 'high', 'low', 'close', 'realized_vol_long', 'realized_vol_short'],
            axis=1,
            inplace=True,
        )
        list_of_prices.append(group)
    return pd.concat(list_of_prices)


def preprocess_events(news: pd.DataFrame, datetime_crop_method: str = '1st'):
    news = extract_news_features_pipeline(news)

    print('[INFO] Cropping datetime hours...')
    if datetime_crop_method == '1st':
        news['date'] = pd.to_datetime(news['date'], utc=True)
        news['rounded_time'] = news['date'].apply(lambda x: x.floor('1h'))

    elif datetime_crop_method == '2nd':
        news['date'] = pd.to_datetime(news['date'], utc=True)
        news['rounded_time'] = news['date'].apply(lambda x: floor_or_ceil(x, freq='h'))
    
    print('[INFO] Aggregating events...')
    agg_events = aggregate_events(news, dt_col='rounded_time')
    agg_events['time_to_check'] = agg_events['rounded_time'] - pd.Timedelta(hours=1)
    print('[INFO] Done!')

    return agg_events


@dp.message(Command('test_check_the_market'))
async def test_check_the_market(message: types.Message):
    print('[CHECK THE MARKET] Start the function to checking the market')
    db = DBHandler()
    predictor = FxRangePredictor()
    
    # Preparing Events
    now = _utc_now()
    events = db.get_events_for_range(
        start=now - timedelta(days=2), 
        end=now + timedelta(days=2)
    )
    events['rounded_time'] = events['date'].apply(lambda x: x.floor('1h'))
    agg_events = predictor.event_transformer.transform(events)
    agg_events = agg_events.iloc[[-2], :]

    print('[CHECK THE MARKET] Aggregated events for the next hour')

    # Prices
    prices = db.get_last_prices(period=21*24+1)  # 21 days + 1 hour for ATR calculation
    agg_prices = predictor.price_transformer.transform(prices)
    agg_prices_latest = agg_prices.groupby('ticker').tail(1)

    # Unite all together
    df = agg_prices_latest.merge(agg_events, how='cross')

    # Add range features for each event
    df = db.add_ranges_for_each_event(df)

    FEATURES = [i for i in df.columns if i not in ['time', 'time_to_check', 'rounded_time', 'datetime', 'open', 'high', 'low', 'close']]
    df = df[FEATURES]

    # PREDICTIONS
    print('[CHECK THE MARKET] Add predictions to DataFrame for further processing')
    predictions = predictor.get_predictions(df)
    print('[CHECK THE MARKET] Predictions for the next hour')
    print(predictions)

    if predictions:
        await send_predictions(events, agg_events, df, predictions, message)
    else:
        await message.answer("No valid predictions generated from the market check.")


def get_user_ids_for_chaos_predictions() -> list:
    """Return the list of user ids, that should get daily alerts at specific time"""
    db = SessionLocal()
    try:        
        stmt = select(UserSettings.user_id).where(UserSettings.chaos_alerts == True)
        user_ids = db.scalars(stmt).all()
        return list(user_ids)

    finally:
        db.close()


async def scheduled_check_the_market() -> None:
    """Run the market check without a Telegram `message` (for scheduled jobs).

    This performs the same ML/prediction steps but logs results instead of
    replying to a user. Scheduled jobs should use this entrypoint.
    """
    try:
        print('[SCHEDULED CHECK THE MARKET] Start scheduled market check')
        db = DBHandler()
        predictor = FxRangePredictor()

        # Preparing Events
        now = _utc_now()
        events = db.get_events_for_range(
            start=now - timedelta(days=2),
            end=now + timedelta(days=2)
        )
        events['rounded_time'] = events['date'].apply(lambda x: x.floor('1h'))
        agg_events = predictor.event_transformer.transform(events)
        agg_events = agg_events.iloc[[-2], :]

        # Prices
        prices = db.get_last_prices(period=21*24+1)
        agg_prices = predictor.price_transformer.transform(prices)
        agg_prices_latest = agg_prices.groupby('ticker').tail(1)

        # Unite all together
        df = agg_prices_latest.merge(agg_events, how='cross')
        df = db.add_ranges_for_each_event(df)
        FEATURES = [i for i in df.columns if i not in ['time', 'time_to_check', 'rounded_time', 'datetime', 'open', 'high', 'low', 'close']]
        df = df[FEATURES]

        # Predictions
        print('[CHECK THE MARKET] Add predictions to DataFrame for further processing')
        predictions = predictor.get_predictions(df)
        print('[CHECK THE MARKET] Predictions for the next hour')
        print(predictions)

        # Users subscribed on ML predictions
        users_to_alert = get_user_ids_for_chaos_predictions()

        # Get events
        if len(users_to_alert) > 0 and predictions is not None:
            # db = SessionLocal()
            text = formulate_prediction_message(events, agg_events, predictions)
    
            for user_id in users_to_alert:
                try:
                    await bot.send_message(
                        chat_id=user_id, 
                        text=text, 
                        parse_mode="HTML", 
                        disable_web_page_preview=True
                    )
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logging.error(f"Не удалось отправить уведомление пользователю {user_id}: {e}")


    except Exception as e:
        logging.exception('Error in scheduled_check_the_market: %s', e)


def escape_markdown_v2(s: str) -> str:
    """Converts a string to an escaped MarkdownV2 monospaced string."""
    escaped_table = ""
    for char in s:
        # if char in r"_*[]()~`>#+-=|{}.!":
        if char in r"_[]()~`>#+-=|{}.!":
            escaped_table += f"\\{char}"
        else:
            escaped_table += char
            
    # 3. Wrap inside a monospaced code block so columns align perfectly
    return f"\n{escaped_table}\n"


def transform_predictions(predictions: dict) -> dict:
    market_state = {}

    for i, ticker in enumerate(predictions["tickers"]):

        market_state[ticker] = {
            "forecast": {
                "range_1h": {
                    "p10": predictions["total_range_1h"][i][0],
                    "p50": predictions["total_range_1h"][i][1],
                    "p90": predictions["total_range_1h"][i][2],
                },
                "range_3h": {
                    "p10": predictions["total_range_3h"][i][0],
                    "p50": predictions["total_range_3h"][i][1],
                    "p90": predictions["total_range_3h"][i][2],
                },
                "range_6h": {
                    "p10": predictions["total_range_6h"][i][0],
                    "p50": predictions["total_range_6h"][i][1],
                    "p90": predictions["total_range_6h"][i][2],
                },
                "range_24h": {
                    "p10": predictions["total_range_24h"][i][0],
                    "p50": predictions["total_range_24h"][i][1],
                    "p90": predictions["total_range_24h"][i][2],
                },
            },
            "regime": {
                "short": predictions["regime_1day"][i],
                "long": predictions["regime_2days"][i],
            },
            "noise": {
                "big_doji": bool(predictions["big_doji"][i]),
                "expansion": bool(predictions["expansion"][i]),
                "chaos": bool(predictions["chaos"][i]),
                "sfp": bool(predictions["sfp"][i]),
            },
            "stats": {
                "dir_changes": predictions["dir_changes"][i],
            }
        }
    
    return market_state


def formulate_prediction_message(events: pd.DataFrame, agg_events: pd.DataFrame, predictions: dict) -> str:
    important_news = events[(events['importance'] == 1) & (events['rounded_time'] == agg_events['rounded_time'].iloc[0])]
    news_list = important_news['title'].unique().tolist()
    news_str = "\n ▫️ ".join(news_list) if news_list else "no important events in the next hour"

    market_state = transform_predictions(predictions)    

    # +---------------- RANGES ------------------+
    movement_1h = [
        ticker
        for ticker, data in market_state.items()
        if data["forecast"]["range_1h"]["p90"] >= 1
    ]

    movement_24h = [
            ticker
            for ticker, data in market_state.items()
            if data["forecast"]["range_24h"]["p90"] >= 1
        ]

    # +----------------- REGIME -----------------+
    table = {'ticker': [], '1 day': [], '2 days': [], 'Swings': []}  # Swings for 24 hours (or 1 day)
    for ticker, data in market_state.items():
        dir_1d = data['regime']['short'][0]
        dir_2d = data['regime']['long'][0]

        table["ticker"].append(ticker)
        table["1 day"].append(dir_1d if dir_1d != 'None' else 'Unknown')
        table["2 days"].append(dir_2d if dir_2d != 'None' else 'Unknown')
        table["Swings"].append(data['stats']['dir_changes'])

    regime = pd.DataFrame(table)
    regime.set_index("ticker", inplace=True)
    order = ["EUR/USD", "GBP/USD", "USD/CHF", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD"]
    regime = regime.loc[order]
    regime.reset_index(inplace=True)

    markdown_regime = f"<pre>{regime.to_markdown(index=False)}</pre>"

    # +---------- NOISE DETECTION ---------------+
    # Chaos
    chaos = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["chaos"]
    ]

    # Expansion
    expansion = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["expansion"]
    ]

    # Spikes
    spikes = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["big_doji"]
    ]
    
    # Swing Failure Pattern (SFP)
    sfp = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["sfp"]
    ]

    any_alerts = any([movement_1h, movement_24h, chaos, expansion, spikes, sfp])

    # +---------------- Stack all together ---------------+

    if any_alerts:
        message_lines = []

        # Ranges
        message_lines.append(
            '❗️<b>Currencies that can fluctuate more than their 1 hour ATR:</b>\n'
            + ", ".join(movement_1h)
        ) if movement_1h else ""
    
        message_lines.append(
            '\n‼️<b>Currencies that can fluctuate more than their daily ATR:</b>\n'
            + ", ".join(movement_24h)
        ) if movement_24h else ""
    
        # Regime
        message_lines.append("\n📈 <b>Regime</b>")
        message_lines.append(markdown_regime)
    
        # Noise
        message_lines.append("\n🚨 <b>Noise</b>") if any([chaos, expansion, spikes, sfp]) else ""
        message_lines.append(f"<b>Possible chaos:</b> {chaos}") if chaos else ""
        message_lines.append(f"<b>Possible double expansion:</b> {expansion}") if expansion else ""
        message_lines.append(f"<b>Possible spikes:</b> {spikes}") if spikes else ""
        message_lines.append(f"<b>Possible false breakouts:</b> {sfp}") if sfp else ""
        
        text = '\n'.join(message_lines)
        return text

    else:
        return "No alerts."

async def send_predictions(
        events: pd.DataFrame, 
        agg_events: pd.DataFrame,
        df: pd.DataFrame,
        predictions: dict,
        message: types.Message
    ) -> None:
    """Send predictions to the user."""
    important_news = events[(events['importance'] == 1) & (events['rounded_time'] == agg_events['rounded_time'].iloc[0])]
    news_list = important_news['title'].unique().tolist()
    news_str = "\n ▫️ ".join(news_list) if news_list else "no important events in the next hour"

    market_state = transform_predictions(predictions)
    # daily_atr = df.groupby('ticker')['prev_daily_atr'].last().to_dict()
    # atr = df.groupby('ticker')['atr'].last().to_dict()
    

    # +---------------- RANGES ------------------+
    movement_1h = [
        ticker
        for ticker, data in market_state.items()
        if data["forecast"]["range_1h"]["p90"] >= 1
    ]

    movement_24h = [
            ticker
            for ticker, data in market_state.items()
            if data["forecast"]["range_24h"]["p90"] >= 1
        ]

    # +----------------- REGIME -----------------+
    table = {'ticker': [], '1 day': [], '2 days': [], 'Swings': []}  # Swings for 24 hours (or 1 day)
    for ticker, data in market_state.items():
        dir_1d = data['regime']['short'][0]
        dir_2d = data['regime']['long'][0]

        table["ticker"].append(ticker)
        table["1 day"].append(dir_1d if dir_1d != 'None' else 'Unknown')
        table["2 days"].append(dir_2d if dir_2d != 'None' else 'Unknown')
        table["Swings"].append(data['stats']['dir_changes'])

    regime = pd.DataFrame(table)
    regime.set_index("ticker", inplace=True)
    order = ["EUR/USD", "GBP/USD", "USD/CHF", "USD/JPY", "USD/CAD", "AUD/USD", "NZD/USD"]
    regime = regime.loc[order]
    regime.reset_index(inplace=True)

    markdown_regime = f"<pre>{regime.to_markdown(index=False)}</pre>"

    # +---------- NOISE DETECTION ---------------+
    # Chaos
    chaos = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["chaos"]
    ]

    # Expansion
    expansion = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["expansion"]
    ]

    # Spikes
    spikes = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["big_doji"]
    ]
    
    # Swing Failure Pattern (SFP)
    sfp = [
        ticker
        for ticker, data in market_state.items()
        if data["noise"]["sfp"]
    ]

    any_alerts = any([movement_1h, movement_24h, chaos, expansion, spikes, sfp])

    # +---------------- Stack all together ---------------+

    if any_alerts:
        await message.answer("⚠️ <b>Market Prediction Update</b>\n\n"
                             "In the next hour, the following important news are expected:\n"
                             f"*{news_str}*\n\n"
                             "Please check the details below for potential market movements and noise detection.",
                             parse_mode=ParseMode.HTML)

        message_lines = []

        # Ranges
        message_lines.append(
            '❗️<b>Currencies that can fluctuate more than their 1 hour ATR:</b>\n'
            + ", ".join(movement_1h)
        ) if movement_1h else ""
    
        message_lines.append(
            '\n‼️<b>Currencies that can fluctuate more than their daily ATR:</b>\n'
            + ", ".join(movement_24h)
        ) if movement_24h else ""
    
        # Regime
        message_lines.append("\n📈 <b>Regime</b>")
        message_lines.append(markdown_regime)
    
        # Noise
        message_lines.append("\n🚨 <b>Noise</b>") if any([chaos, expansion, spikes, sfp]) else ""
        message_lines.append(f"<b>Possible chaos:</b> {chaos}") if chaos else ""
        message_lines.append(f"<b>Possible double expansion:</b> {expansion}") if expansion else ""
        message_lines.append(f"<b>Possible spikes:</b> {spikes}") if spikes else ""
        message_lines.append(f"<b>Possible false breakouts:</b> {sfp}") if sfp else ""
        
        text = '\n'.join(message_lines)
        await message.answer(text, parse_mode=ParseMode.HTML)

    else:
        await message.answer("✅ <b>Market Prediction Update</b>\n\n"
                             "In the next hour, there are no significant news or expected market movements.\n"
                             "Everything seems stable for now.",
                             parse_mode=ParseMode.HTML)


# +------------- ECHO -------------------+
@dp.message(StateFilter(None))
async def echo(message: types.Message):
    await message.answer(message.text)


# +---------- RUN BOT -----------------+
async def main():
    scheduler.add_job(
        hourly_alert_dispatcher,
        trigger="cron",
        day_of_week='mon-fri',
        hour="*",
        minute=0
    )

    scheduler.add_job(
        daily_summary_dispatcher,
        trigger=CronTrigger(hour=20, minute=17, timezone=timezone(timedelta(hours=3))),
        id="daily_summary_dispatcher",
        replace_existing=True,
    )
    
    scheduler.start()

    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())

