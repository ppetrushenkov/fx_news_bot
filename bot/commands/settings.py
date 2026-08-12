from aiogram import Router, F
from aiogram import types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards import build_tz_keyboard, get_importance_settings_keyboard, get_alert_settings_keyboard, \
    build_risk_keyboard
from bot.states import OnboardingStates
from utils.alerts import save_timezone, save_risk_level
from db.database import SessionLocal
from db.models import UserSettings


router = Router()


# ────────────────────────── Commands ───────────────────────────────────────────────
@router.message(Command('set_importance'))
@router.message(F.text == "⚙️ Importance settings")
async def set_importance(message: types.Message):
    db = SessionLocal()
    settings = db.get(UserSettings, message.from_user.id)

    if not settings:
        settings = UserSettings(user_id=message.from_user.id, show_low_importance=False, show_medium_importance=False,
                                show_high_importance=True)
        db.add(settings)
        db.commit()

    await message.answer(
        text="⚙️ Settings for importance level:",
        reply_markup=get_importance_settings_keyboard(settings)  # Передаем объект напрямую
    )
    db.close()


@router.message(Command('set_alerts'))
@router.message(F.text == "🔔 Notification settings")
async def set_alerts(message: types.Message):
    db = SessionLocal()
    settings = db.get(UserSettings, message.from_user.id)

    if not settings:
        settings = UserSettings(user_id=message.from_user.id, daily_alerts=False, weekly_alerts=False,
                                chaos_alerts=False)
        db.add(settings)
        db.commit()

    await message.answer(
        text="⚙️ Settings for notifications:",
        reply_markup=get_alert_settings_keyboard(settings)  # Передаем объект напрямую
    )
    db.close()


@router.message(Command("set_gmt"))
@router.message(F.text == "📍 Timezone settings")
async def set_gmt(message: types.Message, state: FSMContext):
    db = SessionLocal()
    settings = db.get(UserSettings, message.from_user.id)

    if settings is not None and settings.user_timezone is not None:
        sign = "+" if settings.user_timezone >= 0 else ""
        await message.answer(
            f"📍Current time zone: UTC{sign}{settings.user_timezone}\n\n"
            "If you want to change it, select from the list or enter manually:",
            reply_markup=build_tz_keyboard()
        )
        await state.set_state(OnboardingStates.waiting_for_timezone)

    else:
        await message.answer(
            "📍Select your time zone from the list or enter manually:",
            reply_markup=build_tz_keyboard()
        )
        await state.set_state(OnboardingStates.waiting_for_timezone)


@router.message(Command("set_risk"))
@router.message(F.text == "⚙️ Set ML risk")
async def set_risk(message: types.Message, state: FSMContext):
    db = SessionLocal()
    settings = db.get(UserSettings, message.from_user.id)

    if settings is not None and settings.ml_risk_level is not None:
        # "• Conservative: You will get more precise predictions. There will be fewer, but more precise.\n"
        await message.answer(
            text=f"Current ML risk level: <b>{settings.ml_risk_level.capitalize()}</b>\n\n"
            "• <b>Conservative:</b> You will get more precise predictions (fewer alerts, but higher accuracy).\n"
            "• <b>Base:</b> Standard risk level. A balance between precision and recall.\n"
            "• <b>Aggressive:</b> More predictions, but with a higher rate of false positives.\n\n"
            "To change your settings, select an option from the list below:",
            reply_markup=build_risk_keyboard(),
            parse_mode="HTML"
        )
        await state.set_state(OnboardingStates.waiting_for_risk_level)

    else:
        await message.answer(
            text="Choose your risk level:",
        )
        await state.set_state(OnboardingStates.waiting_for_risk_level)


# ────────────────────────── Functions ───────────────────────────────────────────────
@router.callback_query(F.data == "tz_manual")
async def tz_manual_requested(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await callback.message.edit_text(
        "Enter your time zone as a number relative to UTC.\n"
        "For example: 3, -5, 5.5 (for UTC+5:30)\n\n"
        "Range: from -12 to +14"
    )
    await state.set_state(OnboardingStates.waiting_for_timezone)


@router.callback_query(F.data.startswith("tz_"), OnboardingStates.waiting_for_timezone)
async def tz_button_chosen(callback: CallbackQuery, state: FSMContext):
    offset = float(callback.data.split("_")[1])

    await save_timezone(callback.from_user.id, offset)  # твоя функция сохранения в БД
    await state.clear()

    sign = "+" if offset >= 0 else ""
    await callback.answer()
    await callback.message.edit_text(
        f"✅ Time zone set: UTC{sign}{offset}\n\n"
        f"Now I will send you alerts based on your local time."
    )


@router.message(OnboardingStates.waiting_for_timezone)
async def tz_text_input(message: Message, state: FSMContext):
    text = message.text.strip().replace(",", ".")
    try:
        offset = float(text)
        if not -12 <= offset <= 14:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Didn't understand. Enter a number from -12 to +14, for example: 3 or -4.5")
        return

    await save_timezone(message.from_user.id, offset)  # твоя функция сохранения в БД
    await state.clear()

    sign = "+" if offset >= 0 else ""
    await message.answer(
        f"✅ Time zone set: UTC{sign}{offset}\n\n"
        f"Now I will send you alerts based on your local time."
    )


@router.callback_query(lambda c: c.data.startswith('toggle_show'))
async def toggle_importance(callback_query: types.CallbackQuery):
    db = SessionLocal()
    setting_key = callback_query.data.replace("toggle_", "")
    user_id = callback_query.from_user.id

    settings = db.get(UserSettings, user_id)

    if not settings:
        # На случай, если настроек почему-то не оказалось в БД
        await callback_query.answer("Something went wrong while getting user settings.")
        return

    # 2. Меняем значение динамически по имени атрибута
    current_value = getattr(settings, setting_key)  # Эквивалентно settings.daily
    setattr(settings, setting_key, not current_value)  # Инвертируем и записываем в объект

    # 3. Сохраняем изменения в саму базу данных
    try:
        db.commit()  # Фиксируем изменения в БД

    except Exception as e:
        db.rollback()  # Если что-то пошло не так, откатываем изменения
        await callback_query.answer("Something went wrong while saving settings.")
        return

    # 4. Обновляем клавиатуру в интерфейсе Telegram
    await callback_query.message.edit_reply_markup(
        reply_markup=get_importance_settings_keyboard(settings)  # Передаем уже обновленный объект settings
    )
    await callback_query.answer()
    db.close()


@router.callback_query(lambda c: c.data.startswith('toggle_'))
async def toggle_notification(callback_query: types.CallbackQuery):
    db = SessionLocal()
    setting_key = callback_query.data.replace("toggle_", "")
    user_id = callback_query.from_user.id

    settings = db.get(UserSettings, user_id)

    if not settings:
        # На случай, если настроек почему-то не оказалось в БД
        await callback_query.answer("Something went wrong while getting user settings.")
        return

    # 2. Меняем значение динамически по имени атрибута
    current_value = getattr(settings, setting_key)  # Эквивалентно settings.daily
    setattr(settings, setting_key, not current_value)  # Инвертируем и записываем в объект

    # 3. Сохраняем изменения в саму базу данных
    try:
        db.commit()  # Фиксируем изменения в БД

    except Exception as e:
        db.rollback()  # Если что-то пошло не так, откатываем изменения
        await callback_query.answer("Something went wrong while saving settings.")
        return

    # 4. Обновляем клавиатуру в интерфейсе Telegram
    await callback_query.message.edit_reply_markup(
        reply_markup=get_alert_settings_keyboard(settings)  # Передаем уже обновленный объект settings
    )
    await callback_query.answer()
    db.close()


@router.callback_query(F.data.startswith("risk_"), OnboardingStates.waiting_for_risk_level)
async def tz_button_chosen(callback: CallbackQuery, state: FSMContext):
    risk = callback.data.split("_")[1]

    # await save_timezone(callback.from_user.id, offset)  # твоя функция сохранения в БД
    await save_risk_level(callback.from_user.id, risk)
    await state.clear()

    await callback.answer()
    await callback.message.edit_text(
        f"✅ Risk level set: {risk}\n\n"
    )