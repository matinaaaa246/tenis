import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from config import Config, logger


class AdaptiveLedgerDB:
    """
    دفتر کل محاسباتی SQLite برای ذخیره پیش‌بینی‌ها، تسویه‌حساب شبانه،
    الگوریتم خودآموز وزن‌های بیزین و فهرست گروه‌های عضو در ورژن ۱
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or Config.DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._get_conn() as conn:
            # جدول ذخیره پیش‌بینی‌ها
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    event_id INTEGER PRIMARY KEY,
                    match_date TEXT NOT NULL,
                    tournament TEXT,
                    surface TEXT,
                    p1_id INTEGER,
                    p1_name TEXT,
                    p2_id INTEGER,
                    p2_name TEXT,
                    mc_p1_prob REAL,
                    elo_p1_prob REAL,
                    market_p1_prob REAL,
                    ensemble_p1_prob REAL,
                    ensemble_p2_prob REAL,
                    prob_spread REAL,
                    kelly_stake REAL,
                    prediction_status TEXT,
                    predicted_winner_id INTEGER,
                    predicted_winner_name TEXT,
                    confidence REAL,
                    settlement_status TEXT DEFAULT 'PENDING',
                    actual_winner_id INTEGER,
                    actual_winner_name TEXT,
                    final_score TEXT,
                    sets_played INTEGER DEFAULT 0,
                    created_at TEXT,
                    settled_at TEXT
                )
            """)

            # جدول وزن‌های یادگیرنده مدل بیزین
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_weights (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    w_mc REAL DEFAULT 0.45,
                    w_market REAL DEFAULT 0.30,
                    w_elo REAL DEFAULT 0.25,
                    total_calibrated_matches INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                INSERT OR IGNORE INTO adaptive_weights (id, w_mc, w_market, w_elo, total_calibrated_matches, updated_at)
                VALUES (1, 0.45, 0.30, 0.25, 0, ?)
            """, (datetime.now().isoformat(),))

            # جدول گروه‌ها، سوپرگروه‌ها و کانال‌های عضو شده در تلگرام
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registered_chats (
                    chat_id INTEGER PRIMARY KEY,
                    chat_type TEXT NOT NULL,
                    title TEXT,
                    username TEXT,
                    joined_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    last_interaction TEXT
                )
            """)
            conn.commit()

    # --- مدیریت گروه‌های تلگرام ---
    def register_chat(self, chat_id: int, chat_type: str, title: Optional[str] = None, username: Optional[str] = None) -> bool:
        """ثبت یا فعال‌سازی مجدد گروه تلگرامی در دیتابیس"""
        now = datetime.now().isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO registered_chats (chat_id, chat_type, title, username, joined_at, is_active, last_interaction)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        chat_type = excluded.chat_type,
                        title = excluded.title,
                        username = excluded.username,
                        is_active = 1,
                        last_interaction = excluded.last_interaction
                """, (chat_id, chat_type, title or "", username or "", now, now))
                conn.commit()
            logger.info(f"✅ گروه/چت با شناسه {chat_id} ({title}) با موفقیت ثبت/بروزرسانی شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در ثبت چت در دیتابیس: {e}")
            return False

    def deactivate_chat(self, chat_id: int):
        """غیرفعال‌سازی چت زمانی که ربات از گروه ریمو یا بن می‌شود"""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE registered_chats
                    SET is_active = 0, last_interaction = ?
                    WHERE chat_id = ?
                """, (datetime.now().isoformat(), chat_id))
                conn.commit()
            logger.info(f"گروه {chat_id} به حالت غیرفعال تغییر یافت.")
        except Exception as e:
            logger.error(f"خطا در غیرفعال‌سازی چت: {e}")

    def get_active_chats(self) -> List[int]:
        """دریافت تمام چت‌های فعال برای ارسال برادکست روزانه"""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT chat_id FROM registered_chats WHERE is_active = 1").fetchall()
                return [int(r["chat_id"]) for r in rows]
        except Exception as e:
            logger.error(f"خطا در دریافت لیست چت‌ها: {e}")
            return []

    # --- مدیریت پیش‌بینی‌ها و وزن‌های کوانت ---
    def get_current_ensemble_weights(self) -> Tuple[float, float, float]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT w_mc, w_market, w_elo FROM adaptive_weights WHERE id = 1").fetchone()
            if row:
                return float(row["w_mc"]), float(row["w_market"]), float(row["w_elo"])
        return 0.45, 0.30, 0.25

    def record_morning_prediction(
        self,
        match_date: str,
        p1,
        p2,
        quant,
        audit
    ):
        pred_winner_id = None
        if audit.status == "IDENTIFIED":
            if audit.predicted_winner.strip().lower() == p1.name.strip().lower():
                pred_winner_id = p1.player_id
            elif audit.predicted_winner.strip().lower() == p2.name.strip().lower():
                pred_winner_id = p2.player_id
            else:
                pred_winner_id = p1.player_id if quant.ensemble_p1_prob > quant.ensemble_p2_prob else p2.player_id

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions (
                    event_id, match_date, tournament, surface,
                    p1_id, p1_name, p2_id, p2_name,
                    mc_p1_prob, elo_p1_prob, market_p1_prob,
                    ensemble_p1_prob, ensemble_p2_prob, prob_spread, kelly_stake,
                    prediction_status, predicted_winner_id, predicted_winner_name,
                    confidence, settlement_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT settlement_status FROM predictions WHERE event_id = ? AND settlement_status != 'PENDING'),
                    'PENDING'
                ), ?)
            """, (
                quant.event_id, match_date, quant.tournament, quant.surface,
                p1.player_id, p1.name, p2.player_id, p2.name,
                quant.mc_p1_win_prob, quant.elo_p1_win_prob, quant.market_micro.live_p1_prob,
                quant.ensemble_p1_prob, quant.ensemble_p2_prob, quant.prob_spread, quant.kelly_fraction_pct,
                audit.status, pred_winner_id, audit.predicted_winner,
                audit.confidence, quant.event_id, datetime.now().isoformat()
            ))
            conn.commit()

    def get_pending_predictions(self) -> List[sqlite3.Row]:
        with self._get_conn() as conn:
            return conn.execute("SELECT * FROM predictions WHERE settlement_status = 'PENDING'").fetchall()

    def update_settlement(
        self,
        event_id: int,
        settlement_status: str,
        actual_winner_id: Optional[int],
        actual_winner_name: str,
        final_score: str,
        sets_played: int
    ):
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE predictions
                SET settlement_status = ?,
                    actual_winner_id = ?,
                    actual_winner_name = ?,
                    final_score = ?,
                    sets_played = ?,
                    settled_at = ?
                WHERE event_id = ?
            """, (
                settlement_status, actual_winner_id, actual_winner_name,
                final_score, sets_played, datetime.now().isoformat(), event_id
            ))
            conn.commit()

    def recalibrate_weights_from_settled_matches(self) -> Dict[str, float]:
        """
        یادگیری خودکار شبانه (Self-Healing Bayesian Calibration):
        ارزیابی خطای Brier Score هر زیرمدل روی بازی‌های تمام‌شده و اصلاح وزن‌ها برای روز بعد
        """
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT p1_id, actual_winner_id, mc_p1_prob, elo_p1_prob, market_p1_prob
                FROM predictions
                WHERE settlement_status IN ('WON', 'LOST', 'NO_BET_SETTLED')
                  AND actual_winner_id IS NOT NULL
                ORDER BY settled_at DESC LIMIT 80
            """).fetchall()

            w_mc, w_mkt, w_elo = self.get_current_ensemble_weights()
            if len(rows) < 5:
                return {"w_mc": w_mc, "w_market": w_mkt, "w_elo": w_elo, "samples": len(rows)}

            brier_mc, brier_elo, brier_mkt = [], [], []
            for r in rows:
                outcome_p1 = 1.0 if r["actual_winner_id"] == r["p1_id"] else 0.0
                if r["mc_p1_prob"] is not None:
                    brier_mc.append(((r["mc_p1_prob"] / 100.0) - outcome_p1) ** 2)
                if r["elo_p1_prob"] is not None:
                    brier_elo.append(((r["elo_p1_prob"] / 100.0) - outcome_p1) ** 2)
                if r["market_p1_prob"] is not None:
                    brier_mkt.append(((r["market_p1_prob"] / 100.0) - outcome_p1) ** 2)

            acc_mc = 1.0 / max(0.04, float(np.mean(brier_mc))) if brier_mc else 4.0
            acc_elo = 1.0 / max(0.04, float(np.mean(brier_elo))) if brier_elo else 3.5
            acc_mkt = 1.0 / max(0.04, float(np.mean(brier_mkt))) if brier_mkt else 4.0

            raw_sum = acc_mc + acc_mkt + acc_elo
            target_mc = acc_mc / raw_sum
            target_mkt = acc_mkt / raw_sum
            target_elo = acc_elo / raw_sum

            # بروزرسانی با نرخ یادگیری ۲۰ درصد جهت پایداری
            lr = 0.20
            new_mc = round((1 - lr) * w_mc + lr * target_mc, 3)
            new_mkt = round((1 - lr) * w_mkt + lr * target_mkt, 3)
            new_elo = round(1.0 - (new_mc + new_mkt), 3)

            conn.execute("""
                UPDATE adaptive_weights
                SET w_mc = ?, w_market = ?, w_elo = ?, total_calibrated_matches = ?, updated_at = ?
                WHERE id = 1
            """, (new_mc, new_mkt, new_elo, len(rows), datetime.now().isoformat()))
            conn.commit()

            return {"w_mc": new_mc, "w_market": new_mkt, "w_elo": new_elo, "samples": len(rows)}

    def get_date_summary(self, match_date: str) -> Dict:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM predictions WHERE match_date = ?", (match_date,)).fetchall()
            all_time = conn.execute("""
                SELECT
                    SUM(CASE WHEN settlement_status = 'WON' THEN 1 ELSE 0 END) as total_won,
                    SUM(CASE WHEN settlement_status = 'LOST' THEN 1 ELSE 0 END) as total_lost
                FROM predictions
                WHERE prediction_status = 'IDENTIFIED'
            """).fetchone()

        total_won_all = all_time["total_won"] or 0
        total_lost_all = all_time["total_lost"] or 0
        all_time_wr = round((total_won_all / (total_won_all + total_lost_all)) * 100, 1) if (total_won_all + total_lost_all) > 0 else 0.0

        return {
            "rows": rows,
            "all_time_won": total_won_all,
            "all_time_lost": total_lost_all,
            "all_time_win_rate": all_time_wr
        }
