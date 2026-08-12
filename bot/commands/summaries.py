import calendar
from collections import defaultdict
from datetime import timedelta, date, datetime, timezone

from aiogram import Router, F
from aiogram import types
from aiogram.filters import Command
from sqlalchemy import select

from bot.services.summary_utils import get_summary_for
from utils.alerts import get_user_timezone, get_user_importance_settings
from utils.datetime_utils import utc_now, _to_user_tz, _weekly_high_impact_period_utc, next_sunday_utc
from utils.text import format_high_impact_event_html, _chunk_telegram_html
from db.database import SessionLocal
from db.models import Events


router = Router()


@router.message(Command("today_summary"))
@router.message(F.text == "📅 Events for today")
async def today_summary(message: types.Message):
    await _send_daily_summary_answer(message, day=utc_now(), empty_label="today")


@router.message(Command("tomorrow_summary"))
@router.message(F.text == "📅 Events for tomorrow")
async def tomorrow_summary(message: types.Message):
    await _send_daily_summary_answer(message, day=utc_now() + timedelta(days=1), empty_label="tomorrow")


@router.message(Command("weekly_summary"))
@router.message(F.text == "📊 Events for the week")
async def weekly_summary(message: types.Message):
    now = utc_now()
    week_start, week_end_exclusive, last_sunday = _weekly_high_impact_period_utc(now)
    db = SessionLocal()
    try:
        tz = get_user_timezone(db, message.from_user.id)
        importance = get_user_importance_settings(db, message.from_user.id)

        rows = db.execute(
            select(Events)
            .where(Events.date >= week_start, Events.date <= next_sunday_utc(), Events.importance.in_(importance))
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
            body = "\n".join(format_high_impact_event_html(ev, tz, time_only=True) for ev in day_events)
            sections.append(f"{label}\n\n{body}")

        full_text = header + "\n\n" + "\n\n".join(sections)
        for chunk in _chunk_telegram_html(full_text):
            await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)

    except Exception as e:
        print("Error in weekly_summary command: %s", e)
        await message.answer("An error occurred while getting weekly summary.")
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