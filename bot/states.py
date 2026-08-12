from aiogram.fsm.state import StatesGroup, State


class OnboardingStates(StatesGroup):
    waiting_for_timezone = State()
    waiting_for_alert_preferences = State()
    waiting_for_risk_level = State()