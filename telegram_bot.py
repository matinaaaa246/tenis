import time
import threading
import requests
from typing import Optional
from datetime import datetime

from config import Config, logger
from database import AdaptiveLedgerDB
from harvester import PlayerTelemetry, QuantMatchReport, LLMAuditOutput


class TelegramBroadcaster:
    """
    سیستم پیامرسانی هوشمند و چندگروهی تلگرام — ورژن ۱
    """
    def __init__(self, bot_token: Optional[str] = None, main_chat_id: Optional[str] = None):
        self.bot_token = (bot_token or Config.TELEGRAM_BOT_TOKEN).strip()
        self.main_chat_id = (main_chat_id or Config.TELEGRAM_CHAT_ID).strip()
        self.session = requests.Session()
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
        self.db = AdaptiveLedgerDB()

    @staticmethod
    def _build_prob_bar(p1_prob: float) -> str:
        filled = max(1, min(9, int(round(p1_prob / 10.0))))
        return "🟦" * filled + "🟧" * (10 - filled)

    def format_match_card(
        self,
        idx: int,
        p1: PlayerTelemetry,
        p2: PlayerTelemetry,
        q: QuantMatchReport,
        audit: LLMAuditOutput
    ) -> str:
        bar = self._build_prob_bar(q.ensemble_p1_prob)
        m = q.market_micro

        market_str = "مبتنی بر مدل کوانت و مونتکارلو"
        if m.live_p1_prob is not None:
            steam_badge = ""
            if m.smart_money_signal == "P1_STEAM":
                steam_badge = f" 🔥 (پول هوشمند +{m.line_movement_p1}% به سمت {p1.name})"
            elif m.smart_money_signal == "P2_STEAM":
                steam_badge = f" 🔥 (پول هوشمند {m.line_movement_p1}% به سمت {p2.name})"
            market_str = f"{m.live_p1_prob}% vs {round(100 - m.live_p1_prob, 1)}%{steam_badge}"

        if audit.status == "UNIDENTIFIABLE":
            badge = "⚠️ <b>وضعیت: قابل شناسایی نیست (رقابت بسیار نزدیک)</b>"
            winner_section = (
                f"🚫 <b>نتیجه مدل:</b> <code>قابل شناسایی نیست</code>\n"
                f"📌 <b>علت فنی:</b> <i>{audit.risk_warning}</i>"
            )
            kelly_section = ""
        else:
            tier_label = "🔥 <b>پیشبینی فوققدرتمند (HIGH-POWER PICK)</b>" if audit.confidence >= 72.0 else "✅ <b>پیشبینی تاییدشده (STRONG PICK)</b>"
            badge = f"{tier_label}"
            winner_section = (
                f"🏆 <b>برنده پیشبینیشده:</b> <code>{audit.predicted_winner}</code>\n"
                f"🎯 <b>قدرت اطمینان مدل:</b> <code>{audit.confidence}%</code>\n"
                f"📌 <b>حالت پیشنهادی:</b> <code>{audit.recommended_bet}</code> | <b>ضریب منصفانه:</b> <code>{audit.fair_odd}</code>"
            )
            kelly_section = (
                f"💰 <b>برتری نسبت به بازار (+EV):</b> <code>{q.ev_edge_pct:+.1f}%</code> | <b>سهم کِلی:</b> <code>{q.kelly_fraction_pct}%</code>\n"
                if m.live_p1_prob is not None else ""
            )

        p1_home = " 🏠(میزبان)" if p1.psychometrics.is_home_crowd else ""
        p2_home = " 🏠(میزبان)" if p2.psychometrics.is_home_crowd else ""

        return (
            f"🎾 <b>مسابقه #{idx} | {q.tournament}</b>\n"
            f"زمین: <b>{q.surface}</b> (ضریب سرعت: <code>{q.court_speed_index}</code>)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>{p1.name}</b> (#{p1.ranking}{p1_home})\n"
            f"      🆚\n"
            f"👤 <b>{p2.name}</b> (#{p2.ranking}{p2_home})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{badge}\n\n"
            f"{winner_section}\n"
            f"{kelly_section}\n"
            f"📊 <b>خروجی مدلهای کمی (P1 vs P2):</b>\n"
            f"• برآیند نهایی (Ensemble): <code>{q.ensemble_p1_prob}%</code> vs <code>{q.ensemble_p2_prob}%</code>\n"
            f"  {bar} (اختلاف: <code>{q.prob_spread}%</code>)\n"
            f"• شبیهساز مونتکارلو (۱۰K): <code>{q.mc_p1_win_prob}%</code> vs <code>{q.mc_p2_win_prob}%</code>\n"
            f"• ضریب Elo سطح زمین: <code>{p1.elo_surface_rating}</code> vs <code>{p2.elo_surface_rating}</code>\n"
            f"• جریان بازار: <code>{market_str}</code>\n\n"
            f"🔬 <b>تلمتری سرویس، ریترن و مارکوف:</b>\n"
            f"• حفظ گیم سرویس (Hold): <code>{q.p1_markov_hold_prob}%</code> | <code>{q.p2_markov_hold_prob}%</code>\n"
            f"• امتیاز سرویس (SPW): <code>{round(p1.service_points_won_pct*100,1)}%</code> | <code>{round(p2.service_points_won_pct*100,1)}%</code>\n"
            f"• امتیاز ریترن (RPW): <code>{round(p1.return_points_won_pct*100,1)}%</code> | <code>{round(p2.return_points_won_pct*100,1)}%</code>\n"
            f"• امید ریاضی ستها: <code>{q.expected_total_sets}</code> | احتمال تایبریک: <code>{q.tiebreak_prob}%</code>\n\n"
            f"🧠 <b>روانشناسی، هیجانات و بیومتریک:</b>\n"
            f"• اعصاب تحت فشار (Clutch): <code>{p1.psychometrics.clutch_rating}</code> | <code>{p2.psychometrics.clutch_rating}</code>\n"
            f"• کامبک پس از باخت ست اول: <code>{round(p1.psychometrics.comeback_resilience_cri*100,1)}%</code> | <code>{round(p2.psychometrics.comeback_resilience_cri*100,1)}%</code>\n"
            f"• برد در تایبریکها: <code>{round(p1.psychometrics.tiebreak_win_rate*100,1)}%</code> | <code>{round(p2.psychometrics.tiebreak_win_rate*100,1)}%</code>\n"
            f"• خستگی (دقایق بازی ۷۲h اخیر): <code>{p1.court_minutes_last_72h}m</code> | <code>{p2.court_minutes_last_72h}m</code>\n\n"
            f"📝 <b>تحلیل تاکتیکی و روانی (ورژن ۱):</b>\n"
            f"<blockquote>{audit.psychological_and_tactical_synthesis}</blockquote>\n"
        )

    def format_welcome_message(self, group_title: str = "") -> str:
        title_display = f"در گروه «{group_title}»" if group_title else "در این چت"
        return (
            f"🎾 <b>ربات پیشبینی کوانت و هوش مصنوعی تنیس (ورژن ۱) فعال شد!</b> 🏆\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"حضور ربات {title_display} با موفقیت ثبت شد.\n\n"
            f"⚙️ <b>موتورهای تحلیلی فعال:</b>\n"
            f"• <b>شبیهساز مونتکارلو غیرمانا (۱۰,۰۰۰ تکرار در هر بازی)</b>\n"
            f"• <b>مدل زنجیره مارکوف در گیمهای سرویس و تایبریک</b>\n"
            f"• <b>پروفایل روانشناسی (Clutch، نرخ کامبک و اخبار روز)</b>\n"
            f"• <b>ردیاب جریان پول هوشمند و ضرایب جهانی</b>\n\n"
            f"📌 <b>دستورات تعاملی:</b>\n"
            f"• <code>/today</code> یا <code>/predict</code>: دریافت قویترین پیشبینیهای امروز\n"
            f"• <code>/stats</code>: مشاهده کارنامه برد و باخت ربات\n"
            f"• <code>/help</code>: راهنمای سیستم\n"
        )

    def send_html_message(self, text: str, target_chat_id: Optional[str] = None):
        chat_id = str(target_chat_id or self.main_chat_id).strip()
        if not self.bot_token or "YOUR_" in self.bot_token or not chat_id or "YOUR_" in chat_id:
            logger.info("پیشنمایش کنسولی پیام تلگرام:\n" + text)
            return

        max_len = 3900
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
        for chunk in chunks:
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            try:
                res = self.session.post(f"{self.api_base}/sendMessage", json=payload, timeout=12)
                if res.status_code != 200:
                    logger.error(f"خطای ارسال تلگرام ({chat_id}): {res.text}")
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"عدم ارتباط با سرور تلگرام: {e}")

    def broadcast_to_all_chats(self, text: str):
        active_chats = self.db.get_active_chats()
        all_targets = set()
        if self.main_chat_id and "YOUR_" not in self.main_chat_id:
            all_targets.add(str(self.main_chat_id))
        for c in active_chats:
            all_targets.add(str(c))

        if not all_targets:
            self.send_html_message(text)
            return

        for cid in all_targets:
            self.send_html_message(text, target_chat_id=cid)


class TelegramBotDaemon:
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = (bot_token or Config.TELEGRAM_BOT_TOKEN).strip()
        self.enabled = bool(self.bot_token and "YOUR_" not in self.bot_token and Config.ENABLE_TELEGRAM_LISTENER)
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
        self.broadcaster = TelegramBroadcaster(self.bot_token)
        self.db = AdaptiveLedgerDB()
        self.session = requests.Session()
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.bot_id: Optional[int] = None
        self.bot_username: Optional[str] = None

    def start(self):
        if not self.enabled:
            logger.info("شنونده پسزمینه تلگرام غیرفعال است.")
            return

        self._fetch_bot_info()
        self.is_running = True
        self.thread = threading.Thread(target=self._polling_loop, daemon=True, name="TelegramListener")
        self.thread.start()
        logger.info("🚀 شنونده تلگرام در پسزمینه فعال شد.")

    def _fetch_bot_info(self):
        try:
            r = self.session.get(f"{self.api_base}/getMe", timeout=10)
            if r.status_code == 200 and r.json().get("ok"):
                res = r.json().get("result", {})
                self.bot_id = res.get("id")
                self.bot_username = res.get("username", "")
                logger.info(f"ربات تلگرام متصل شد: @{self.bot_username}")
        except Exception as e:
            logger.warning(f"عدم دریافت اطلاعات ربات تلگرام: {e}")

    def _polling_loop(self):
        offset = 0
        while self.is_running:
            try:
                params = {
                    "offset": offset,
                    "timeout": 20,
                    "allowed_updates": ["message", "my_chat_member", "chat_member"]
                }
                res = self.session.get(f"{self.api_base}/getUpdates", params=params, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    for upd in data.get("result", []):
                        offset = upd["update_id"] + 1
                        self._process_update(upd)
                elif res.status_code == 409:
                    time.sleep(10)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(3)

    def _process_update(self, update: dict):
        try:
            if "my_chat_member" in update:
                mcm = update["my_chat_member"]
                chat = mcm.get("chat", {})
                chat_id = chat.get("id")
                chat_type = chat.get("type", "")
                title = chat.get("title", "")
                username = chat.get("username", "")
                new_status = mcm.get("new_chat_member", {}).get("status", "")

                if new_status in ["member", "administrator"]:
                    self.db.register_chat(chat_id, chat_type, title, username)
                    welcome_text = self.broadcaster.format_welcome_message(title)
                    self.broadcaster.send_html_message(welcome_text, target_chat_id=str(chat_id))
                elif new_status in ["left", "kicked"]:
                    self.db.deactivate_chat(chat_id)
                return

            if "message" in update:
                msg = update["message"]
                chat = msg.get("chat", {})
                chat_id = chat.get("id")
                chat_type = chat.get("type", "")
                title = chat.get("title", "")
                username = chat.get("username", "")

                if "new_chat_members" in msg:
                    for member in msg["new_chat_members"]:
                        if member.get("id") == self.bot_id or (member.get("is_bot") and member.get("username") == self.bot_username):
                            self.db.register_chat(chat_id, chat_type, title, username)
                            welcome_text = self.broadcaster.format_welcome_message(title)
                            self.broadcaster.send_html_message(welcome_text, target_chat_id=str(chat_id))
                            return

                text = (msg.get("text") or "").strip()
                if not text:
                    return

                cmd = text.split()[0].lower().replace(f"@{self.bot_username}".lower(), "")
                if cmd in ["/start", "/help"]:
                    welcome_text = self.broadcaster.format_welcome_message(title)
                    self.broadcaster.send_html_message(welcome_text, target_chat_id=str(chat_id))
                elif cmd in ["/today", "/predict"]:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    summary = self.db.get_date_summary(today_str)
                    rows = summary.get("rows", [])
                    if rows:
                        picks = [r for r in rows if r["prediction_status"] == "IDENTIFIED"]
                        resp = (
                            f"🎾 <b>قویترین پیشبینیهای امروز ({today_str}) — ورژن ۱</b>\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"کل بازیهای تحلیلشده: <code>{len(rows)}</code> | پیشبینیهای قطعی: <code>{len(picks)}</code>\n\n"
                        )
                        for idx, p in enumerate(picks, 1):
                            resp += f"{idx}. 🏆 <b>{p['predicted_winner_name']}</b> (اطمینان: <code>{p['confidence']}%</code>)\n"
                        self.broadcaster.send_html_message(resp, target_chat_id=str(chat_id))
                    else:
                        self.broadcaster.send_html_message(
                            f"🎾 <b>پیشبینیهای امروز ({today_str})</b>\nتحلیلهای امروز هنوز اجرا نشده یا در حال پردازش است.",
                            target_chat_id=str(chat_id)
                        )
                elif cmd in ["/stats"]:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    summary = self.db.get_date_summary(today_str)
                    stats_text = (
                        f"📊 <b>کارنامه عملکرد ربات پیشبینی تنیس (ورژن ۱)</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"✅ پیشبینیهای درست: <code>{summary['all_time_won']}</code>\n"
                        f"❌ پیشبینیهای غلط: <code>{summary['all_time_lost']}</code>\n"
                        f"🎯 نرخ برد کل (Win Rate): <code>{summary['all_time_win_rate']}%</code>\n"
                    )
                    self.broadcaster.send_html_message(stats_text, target_chat_id=str(chat_id))
        except Exception as e:
            logger.error(f"خطا در پردازش پیام تلگرام: {e}")
