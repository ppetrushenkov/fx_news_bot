from datetime import timezone, timedelta, date

import pandas as pd
from sqlalchemy import select, func, cast, Integer

from db.database import SessionLocal
from db.models import UserSettings, Events


def get_user_settings(db, user_id: int) -> UserSettings:
    """Return the UserSettings object for the given user_id"""
    try:
        return db.sess.get(UserSettings, user_id)
    except (
            Exception) as e: print(e)
    finally:
        db.sess.close()


def get_user_timezone(db, user_id: int) -> timezone:
    settings = db.get(UserSettings, user_id)

    try:
        offset = float(settings.user_timezone) if settings and settings.user_timezone is not None else 0.0
        assert -12.0 <= offset <= 14.0
        return timezone(timedelta(hours=offset))

    except (TypeError, ValueError, AssertionError):
        print("Invalid timezone for user %s, falling back to UTC", user_id)
        return timezone.utc


def get_users_for_daily_alert(gmt: int) -> list:
    """Return the list of user ids, that subscribed on daily alerts"""
    db = SessionLocal()
    try:
        stmt = select(UserSettings.user_id).where(
            cast(func.floor(UserSettings.user_timezone), Integer) == gmt,
            UserSettings.daily_alerts == True
        )

        user_ids = db.scalars(stmt).all()
        return list(user_ids)

    finally:
        db.close()


def get_user_importance_settings(db, user_id: int) -> list:
    """Return the list of user's importance settings.
    For example [-1, 0, 1] if user's importance settings are set to show low, medium and high.
    [1] is only for high importances."""
    user_settings = db.get(UserSettings, user_id)

    importance = [
        i for i, imp_flg in zip([-1, 0, 1], [
            user_settings.show_low_importance,
            user_settings.show_medium_importance,
            user_settings.show_high_importance
        ]) if imp_flg
    ]

    return importance


def get_users_for_chaos_predictions() -> list:
    """Return the list of user ids, that subscribed on chaos predictions alerts"""
    db = SessionLocal()
    try:
        stmt = select(UserSettings.user_id).where(UserSettings.chaos_alerts == True)
        user_ids = db.scalars(stmt).all()
        return list(user_ids)

    finally:
        db.close()


async def save_timezone(
    user_id: int,
    offset: float
) -> UserSettings:
    """
    Saves (or creates) the user's time zone in UserSettings table in the database.

    :param user_id: Telegram user_id (rk)
    :param offset: offset from UTC in hours (float, can be negative)
    :return: updated UserSettings object
    """
    db = SessionLocal()
    settings = db.get(UserSettings, user_id)

    if settings is None:
        # user does not exist in the database, create a new record
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


async def save_risk_level(
        user_id: int,
        risk_level: str
) -> UserSettings:
    db = SessionLocal()
    settings = db.get(UserSettings, user_id)

    if settings is None:
        settings = UserSettings(user_id=user_id, ml_risk_level=risk_level)
        db.add(settings)
    else:
        settings.ml_risk_level = risk_level
        settings.updated_at = func.now()

    db.commit()
    db.refresh(settings)
    return settings
