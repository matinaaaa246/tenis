import time
import argparse
from datetime import datetime

from config import Config, logger
from database import AdaptiveLedgerDB
from harvester import MasterpieceTennisHarvester
from engine import PsychometricMonteCarloEngine
from llm_arbiter import LLMQuantArbiter
from telegram_bot import TelegramBroadcaster, TelegramBotDaemon


def run_morning_analysis_pipeline():
    """
    پایپلاین تحلیل صبحگاهی مسابقات تنیس جهانی — ورژن ۱
    مرتبسازی و ارائه قویترین پیشبینیهای روز بر اساس Power Score
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"☀️ شروع پایپلاین تحلیل صبحگاهی {Config.VERSION_NAME} برای تاریخ {today_str}...")

    ledger = AdaptiveLedgerDB(Config.DB_PATH)
    harvester = MasterpieceTennisHarvester()
    arbiter = LLMQuantArbiter()
    telegram = TelegramBroadcaster(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

    current_weights = ledger.get_current_ensemble_weights()
    w_mc, w_mkt, w_elo = current_weights
    logger.info(f"وزنهای مدل خودآموز بیزین: MC={int(w_mc*100)}% | Market={int(w_mkt*100)}% | Elo={int(w_elo*100)}%")

    fixtures = harvester.fetch_daily_fixtures(today_str)
    if not fixtures:
        empty_msg = (
            f"📅 <b>گزارش روزانه تنیس ({today_str}) — {Config.VERSION_NAME}</b>\n"
            f"در این لحظه مسابقه انفرادی شروعنشدهای در تقویم جهانی یافت نشد."
        )
        telegram.broadcast_to_all_chats(empty_msg)
        return

    analyzed_results = []

    for idx, m in enumerate(fixtures, start=1):
        p1_name = m.get("p1_name", "Player 1")
        p2_name = m.get("p2_name", "Player 2")
        logger.info(f"[{idx}/{len(fixtures)}] تحلیل عمیق: {p1_name} 🆚 {p2_name}...")

        p1_profile = harvester.build_player_complete_telemetry(
            m["p1_id"], p1_name, m.get("p1_country", "UNK"), m.get("host_country", "UNK"), m.get("p1_rank", 200), m.get("surface", "Hard")
        )
        p2_profile = harvester.build_player_complete_telemetry(
            m["p2_id"], p2_name, m.get("p2_country", "UNK"), m.get("host_country", "UNK"), m.get("p2_rank", 200), m.get("surface", "Hard")
        )

        h2h_p1, h2h_p2 = harvester.fetch_h2h(m["event_id"])
        market_micro = harvester.fetch_market_microstructure(m["event_id"])

        quant_report = PsychometricMonteCarloEngine.analyze_match(
            m, p1_profile, p2_profile, h2h_p1, h2h_p2, market_micro, current_weights
        )
        audit_result = arbiter.evaluate(p1_profile, p2_profile, quant_report)

        ledger.record_morning_prediction(today_str, p1_profile, p2_profile, quant_report, audit_result)

        analyzed_results.append({
            "p1": p1_profile,
            "p2": p2_profile,
            "quant": quant_report,
            "audit": audit_result,
            "sort_key": (
                1 if audit_result.status == "IDENTIFIED" else 0,
                quant_report.power_score,
                audit_result.confidence
            )
        })

    # مرتبسازی تمام بازیها بر اساس قدرت پیشبینی (قویترین پیشبینیها در صدر)
    analyzed_results.sort(key=lambda x: x["sort_key"], reverse=True)

    identified_picks = [r for r in analyzed_results if r["audit"].status == "IDENTIFIED"]
    unidentifiable_matches = [r for r in analyzed_results if r["audit"].status != "IDENTIFIED"]

    summary_lines = []
    for rank_idx, r in enumerate(identified_picks, 1):
        audit = r["audit"]
        opponent = r["p2"].name if audit.predicted_winner == r["p1"].name else r["p1"].name
        icon = "🔥" if audit.confidence >= 72.0 else "✅"
        summary_lines.append(
            f"{rank_idx}. {icon} <b>{audit.predicted_winner}</b> (مقابل {opponent})\n"
            f"   └ 🎯 اطمینان: <code>{audit.confidence}%</code> | 📌 پیشنهاد: <b>{audit.recommended_bet}</b> (ضریب منصفانه: <code>{audit.fair_odd}</code>)"
        )

    unidentified_lines = []
    for r in unidentifiable_matches:
        unidentified_lines.append(
            f"• {r['p1'].name} 🆚 {r['p2'].name} — <i>قابل شناسایی نیست ({r['quant'].prob_spread}% اختلاف)</i>"
        )

    master_header = (
        f"🏛 <b>سیستم جامع پیشبینی کوانت و هوش مصنوعی تنیس — {Config.VERSION_NAME}</b>\n"
        f"📅 <b>تاریخ:</b> <code>{today_str}</code> | ⚙️ <i>وزنها: MC={int(w_mc*100)}% | Mkt={int(w_mkt*100)}% | Elo={int(w_elo*100)}%</i>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔍 <b>کل مسابقات تحلیلشده امروز:</b> <code>{len(analyzed_results)}</code>\n"
        f"🔥 <b>پیشبینیهای قدرتمند تاییدشده:</b> <code>{len(identified_picks)}</code>\n"
        f"⚠️ <b>بازیهای غیرقابل شناسایی (نزدیک):</b> <code>{len(unidentifiable_matches)}</code>\n\n"
        f"🏆 <b>لیست قویترین پیشبینیهای امروز (به ترتیب قدرت سیگنال):</b>\n"
        + ("\n".join(summary_lines) if summary_lines else "<i>همه بازیهای فعلی در محدوده ۵۰-۵۰ قرار دارند.</i>")
        + "\n\n"
        f"🚫 <b>بازیهای غیرقابل شناسایی (رقابت نزدیک و تنگاتنگ):</b>\n"
        + ("\n".join(unidentified_lines[:10]) if unidentified_lines else "<i>موردی یافت نشد.</i>")
        + "\n━━━━━━━━━━━━━━━━━━\n"
        f"<i>کارتهای تحلیلی کامل تکتک مسابقات در ادامه ارسال میشود 👇</i>"
    )

    telegram.broadcast_to_all_chats(master_header)

    for idx, item in enumerate(analyzed_results, start=1):
        card = telegram.format_match_card(
            idx, item["p1"], item["p2"], item["quant"], item["audit"]
        )
        telegram.broadcast_to_all_chats(card)


def run_nightly_settlement_pipeline():
    """
    پایپلاین تسویهحساب شبانه، کارنامه برد و باخت و کالیبراسیون خودکار وزنها — ورژن ۱
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"🌙 شروع پایپلاین حسابرسی شبانه {Config.VERSION_NAME} برای تاریخ {today_str}...")

    ledger = AdaptiveLedgerDB(Config.DB_PATH)
    harvester = MasterpieceTennisHarvester()
    telegram = TelegramBroadcaster(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

    pending_rows = ledger.get_pending_predictions()
    for row in pending_rows:
        ev_id = row["event_id"]
        outcome = harvester.fetch_finished_match_outcome(ev_id)
        state = outcome.get("state")

        if state == "PENDING":
            continue
        elif state == "VOID":
            ledger.update_settlement(ev_id, "VOID", None, "فسخ/انصراف", outcome.get("reason", "Void"), 0)
        elif state == "FINISHED":
            winner_id = outcome.get("winner_id")
            winner_name = outcome.get("winner_name", "")
            score_str = outcome.get("score_str", "")
            sets_played = outcome.get("sets_played", 0)

            if row["prediction_status"] == "IDENTIFIED":
                if winner_id == row["predicted_winner_id"] or (winner_name and winner_name.strip().lower() == str(row["predicted_winner_name"]).strip().lower()):
                    stl_status = "WON"
                else:
                    stl_status = "LOST"
            else:
                stl_status = "NO_BET_SETTLED"

            ledger.update_settlement(ev_id, stl_status, winner_id, winner_name, score_str, sets_played)

    calib_info = ledger.recalibrate_weights_from_settled_matches()
    summary = ledger.get_date_summary(today_str)
    rows = summary["rows"]

    if not rows:
        telegram.broadcast_to_all_chats(
            f"🌙 <b>کارنامه شبانه ({today_str}) — {Config.VERSION_NAME}</b>\nپیشبینی ثبتشدهای برای امروز در دیتابیس یافت نشد."
        )
        return

    won_list, lost_list, pending_list = [], [], []

    for r in rows:
        match_title = f"{r['p1_name']} 🆚 {r['p2_name']}"
        stl = r["settlement_status"]

        if r["prediction_status"] == "IDENTIFIED":
            if stl == "WON":
                won_list.append(
                    f"✅ <b>{r['predicted_winner_name']}</b> ({match_title})\n"
                    f"   └ نتیجه نهایی: <code>{r['final_score']}</code> | اطمینان صبح: <code>{r['confidence']}%</code>"
                )
            elif stl == "LOST":
                lost_list.append(
                    f"❌ <b>{r['predicted_winner_name']}</b> ({match_title})\n"
                    f"   └ برنده واقعی: <b>{r['actual_winner_name']}</b> (<code>{r['final_score']}</code>)"
                )
            elif stl != "VOID":
                pending_list.append(f"⏳ {match_title} (پیشنهاد: <b>{r['predicted_winner_name']}</b>)")

    total_settled = len(won_list) + len(lost_list)
    daily_wr = round((len(won_list) / total_settled) * 100, 1) if total_settled > 0 else 0.0

    report_msg = (
        f"🌙 <b>کارنامه حسابرسی شبانه و خودآموزی مدل — {Config.VERSION_NAME}</b>\n"
        f"📅 <b>تاریخ:</b> <code>{today_str}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>نتایج پیشبینیهای امروز:</b>\n"
        f"• کل پیشبینیهای تعیینتکلیفشده: <code>{total_settled}</code>\n"
        f"• ✅ <b>پیشبینیهای درست (WON):</b> <code>{len(won_list)}</code>\n"
        f"• ❌ <b>پیشبینیهای غلط (LOST):</b> <code>{len(lost_list)}</code>\n"
        f"• ⏳ <b>در جریان یا معوق به فردا:</b> <code>{len(pending_list)}</code>\n"
        f"• 🎯 <b>نرخ برد امروز (Win Rate):</b> <code>{daily_wr}%</code>\n"
        f"• 🏆 <b>نرخ برد کل تاریخچه ربات:</b> <code>{summary['all_time_win_rate']}%</code> "
        f"(<code>{summary['all_time_won']}W - {summary['all_time_lost']}L</code>)\n\n"
        f"🧠 <b>وزنهای کالیبرهشده برای فردا ({calib_info.get('samples', 0)} بازی):</b>\n"
        f"<code>MC: {int(calib_info['w_mc']*100)}% | Market: {int(calib_info['w_market']*100)}% | Elo: {int(calib_info['w_elo']*100)}%</code>\n\n"
        f"🟢 <b>لیست پیشبینیهای درست امروز:</b>\n"
        + ("\n".join(won_list) if won_list else "<i>موردی ثبت نشد.</i>")
        + "\n\n"
        f"🔴 <b>لیست پیشبینیهای غلط امروز:</b>\n"
        + ("\n".join(lost_list) if lost_list else "<i>بدون خطا (۰ مورد).</i>")
    )

    telegram.broadcast_to_all_chats(report_msg)
    logger.info("✅ کارنامه شبانه با موفقیت به همه گروهها ارسال شد.")


def main():
    import schedule

    parser = argparse.ArgumentParser(description=f"{Config.APP_TITLE}")
    parser.add_argument("--mode", choices=["morning", "night", "daemon"], default="daemon")
    parser.add_argument("--morning-time", type=str, default=Config.MORNING_RUN_TIME)
    parser.add_argument("--night-time", type=str, default=Config.NIGHT_SETTLE_TIME)
    args = parser.parse_args()

    if args.mode == "morning":
        run_morning_analysis_pipeline()
    elif args.mode == "night":
        run_nightly_settlement_pipeline()
    else:
        logger.info(f"🚀 {Config.APP_TITLE} در حالت شبانهروزی (Daemon) فعال شد.")
        bot_daemon = TelegramBotDaemon(Config.TELEGRAM_BOT_TOKEN)
        bot_daemon.start()

        schedule.every().day.at(args.morning_time).do(run_morning_analysis_pipeline)
        schedule.every().day.at(args.night_time).do(run_nightly_settlement_pipeline)

        try:
            run_morning_analysis_pipeline()
        except Exception as e:
            logger.error(f"خطا در اجرای اولیه صبحگاهی: {e}")

        while True:
            schedule.run_pending()
            time.sleep(15)


if __name__ == "__main__":
    main()