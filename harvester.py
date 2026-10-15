import time
import math
import urllib.parse
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from config import Config, logger


# =====================================================================
# مدلهای داده ساختاریافته (PYDANTIC SCHEMAS)
# =====================================================================

class PsychometricProfile(BaseModel):
    clutch_rating: float = 50.0              # 0 تا 100 (توانایی واقعی بازی تحت فشار و امتیازهای حساس)
    comeback_resilience_cri: float = 0.25    # نرخ واقعی بردن مسابقه پس از باختن ست اول
    tiebreak_win_rate: float = 0.50          # نرخ واقعی پیروزی در تایبریکها
    deciding_set_win_rate: float = 0.50      # نرخ واقعی پیروزی در ست سرنوشتساز (ست سوم/پنجم)
    bagel_shock_score: float = 0.0           # مثبت: بردهای 6-0/6-1 اخیر | منفی: ضربه روحی باخت سنگین
    is_home_crowd: bool = False              # امتیاز میزبانی
    news_sentiment_score: float = 0.0        # -1.0 تا +1.0 (برآیند اخبار مصدومیت و آمادگی)
    recent_headlines: List[str] = Field(default_factory=list)

class MarketMicrostructure(BaseModel):
    opening_p1_prob: Optional[float] = None
    live_p1_prob: Optional[float] = None
    line_movement_p1: float = 0.0
    fair_decimal_odd_p1: Optional[float] = None
    fair_decimal_odd_p2: Optional[float] = None
    smart_money_signal: Literal["P1_STEAM", "P2_STEAM", "NEUTRAL"] = "NEUTRAL"

class PlayerTelemetry(BaseModel):
    player_id: int
    name: str
    country_code: str = "UNK"
    ranking: int = 350
    elo_surface_rating: float = 1500.0
    recent_win_rate_10: float = 0.50
    surface_win_rate: float = 0.50
    fatigue_index: float = 0.0
    court_minutes_last_72h: int = 0
    sets_played_last_7d: int = 0
    first_serve_in_pct: float = 0.62
    first_serve_won_pct: float = 0.71
    second_serve_won_pct: float = 0.51
    service_points_won_pct: float = 0.63
    return_points_won_pct: float = 0.37
    bp_saved_pct: float = 0.61
    bp_converted_pct: float = 0.40
    psychometrics: PsychometricProfile

class QuantMatchReport(BaseModel):
    event_id: int
    tournament: str
    surface: str
    court_speed_index: float
    is_best_of_five: bool = False
    p1_markov_hold_prob: float
    p2_markov_hold_prob: float
    mc_p1_win_prob: float
    mc_p2_win_prob: float
    expected_total_sets: float
    tiebreak_prob: float
    elo_p1_win_prob: float
    market_micro: MarketMicrostructure
    ensemble_p1_prob: float
    ensemble_p2_prob: float
    prob_spread: float
    shannon_entropy: float
    ev_edge_pct: float
    kelly_fraction_pct: float
    h2h_p1: int
    h2h_p2: int
    unidentifiable_flags: List[str]
    is_mathematically_unidentifiable: bool
    power_score: float = 0.0                 # شاخص قدرت پیشبینی (Prediction Power Score)

class LLMAuditOutput(BaseModel):
    status: Literal["IDENTIFIED", "UNIDENTIFIABLE"] = Field(
        description="اگر بازی کاملاً ۵۰-۵۰ و غیرقابل تشخیص است UNIDENTIFIABLE وگرنه IDENTIFIED"
    )
    predicted_winner: str = Field(
        description="نام دقیق بازیکن برنده یا عبارت 'قابل شناسایی نیست'"
    )
    confidence: float = Field(
        description="درصد اطمینان نهایی از 0 تا 100"
    )
    recommended_bet: str = Field(
        default="برد مستقیم",
        description="پیشنهاد شرط: برد مستقیم، برد ۲-۰، یا هندیکپ"
    )
    fair_odd: float = Field(
        default=1.50,
        description="ضریب عادلانه محاسبهشده برای برنده"
    )
    psychological_and_tactical_synthesis: str = Field(
        description="تحلیل عمیق فنی و روانی مسابقه به زبان فارسی"
    )
    risk_warning: str = Field(
        description="نکته کلیدی ریسک یا علت غیرقابل شناسایی بودن"
    )


# =====================================================================
# رادار استخراج اطلاعات چندمنبعی (MULTI-SOURCE HARVESTER)
# =====================================================================

class MasterpieceTennisHarvester:
    SURFACE_BASELINES = {
        "Grass": {"spw": 0.658, "csi": 1.18},
        "Hard":  {"spw": 0.634, "csi": 1.04},
        "Clay":  {"spw": 0.602, "csi": 0.86}
    }

    NEGATIVE_NEWS_LEXICON = [
        "injury", "injured", "withdraw", "pain", "blister", "fatigue", "exhausted",
        "struggle", "illness", "sick", "medical", "surgery", "doubt", "split",
        "mental", "burnout", "upset", "crisis", "slump", "retired", "back issue", "ankle", "wrist"
    ]
    POSITIVE_NEWS_LEXICON = [
        "title", "champion", "unstoppable", "dominant", "streak", "confident",
        "masterclass", "career-high", "flawless", "ready", "healthy", "thrash", "form"
    ]

    def __init__(self):
        if HAS_CURL_CFFI:
            self.session = cffi_requests.Session(impersonate="chrome120", timeout=12)
        else:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        self.plain_session = requests.Session()
        self.plain_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        self.sofascore_base = "https://api.sofascore.com/api/v1"
        Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.historical_df = self._load_or_fetch_historical_datasets()

    def _load_or_fetch_historical_datasets(self) -> pd.DataFrame:
        cache_file = Config.CACHE_DIR / "historical_matches_v1.parquet"
        if cache_file.exists():
            if (time.time() - cache_file.stat().st_mtime) / 86400 < 7:
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        logger.info("در حال دانلود دیتاستهای مرجع تاریخی ATP و WTA...")
        urls = [
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv",
        ]
        dfs = []
        cols = [
            "surface", "winner_name", "loser_name", "score",
            "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
            "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced"
        ]
        for url in urls:
            try:
                res = self.plain_session.get(url, timeout=12)
                if res.status_code == 200:
                    df = pd.read_csv(StringIO(res.text), usecols=lambda c: c in cols, low_memory=False)
                    dfs.append(df)
            except Exception:
                pass

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            try:
                combined.to_parquet(cache_file, index=False)
            except Exception:
                pass
            return combined
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _normalize_surface(raw: str) -> str:
        s = (raw or "").lower()
        if "clay" in s:
            return "Clay"
        if "grass" in s:
            return "Grass"
        return "Hard"

    def _estimate_court_speed_index(self, tournament_name: str, surface: str) -> float:
        base_csi = self.SURFACE_BASELINES.get(surface, {"csi": 1.0})["csi"]
        t_low = tournament_name.lower()
        if any(w in t_low for w in ["indoor", "basel", "vienna", "paris", "turin", "madrid", "gstaad", "stuttgart"]):
            base_csi += 0.08
        elif any(w in t_low for w in ["indian wells", "monte carlo", "rome", "rio", "buenos aires"]):
            base_csi -= 0.06
        return round(base_csi, 2)

    def fetch_daily_fixtures(self, date_str: str) -> List[Dict]:
        candidates = []

        # ۱. دریافت از SofaScore
        try:
            url = f"{self.sofascore_base}/sport/tennis/scheduled-events/{date_str}"
            res = self.session.get(url, timeout=8)
            if res.status_code == 200:
                events = res.json().get("events", [])
                for ev in events:
                    if ev.get("status", {}).get("code") != 0:
                        continue
                    home = ev.get("homeTeam", {})
                    away = ev.get("awayTeam", {})
                    p1_name, p2_name = home.get("name", ""), away.get("name", "")
                    if "/" in p1_name or "/" in p2_name:
                        continue
                    tourn = ev.get("tournament", {})
                    tourn_name = tourn.get("name", "Unknown Tournament")
                    cat = tourn.get("category", {})
                    category_name = cat.get("name", "")
                    if "exhibition" in tourn_name.lower() or "utr" in category_name.lower():
                        continue

                    host_country = cat.get("alpha2", "") or tourn.get("uniqueTournament", {}).get("category", {}).get("alpha2", "")
                    p1_rank = home.get("ranking") or 400
                    p2_rank = away.get("ranking") or 400

                    surface = self._normalize_surface(ev.get("groundType", "Hard"))
                    full_title = f"{category_name} - {tourn_name}".strip(" -")
                    csi = self._estimate_court_speed_index(full_title, surface)
                    is_bo5 = any(gs in tourn_name.lower() for gs in ["australian open", "roland garros", "wimbledon", "us open"]) and "women" not in tourn_name.lower()

                    priority = 1200 - (p1_rank + p2_rank)
                    if any(k in category_name.lower() for k in ["atp", "wta", "grand slam", "challenger"]):
                        priority += 500

                    candidates.append({
                        "event_id": ev.get("id"),
                        "tournament": full_title,
                        "host_country": host_country,
                        "surface": surface,
                        "court_speed_index": csi,
                        "is_bo5": is_bo5,
                        "p1_id": home.get("id"),
                        "p1_name": p1_name,
                        "p1_rank": p1_rank,
                        "p1_country": home.get("country", {}).get("alpha2", "UNK"),
                        "p2_id": away.get("id"),
                        "p2_name": p2_name,
                        "p2_rank": p2_rank,
                        "p2_country": away.get("country", {}).get("alpha2", "UNK"),
                        "priority": priority
                    })
        except Exception as e:
            logger.debug(f"SofaScore schedule fallback triggered: {e}")

        # ۲. پشتیبانی خودکار از ESPN در صورت مسدود بودن SofaScore
        if not candidates:
            logger.info("دریافت لیست مسابقات روز از ESPN Scoreboard API...")
            for tour in ["atp", "wta"]:
                try:
                    espn_url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
                    r = self.plain_session.get(espn_url, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        for ev in data.get("events", []):
                            tourn_title = ev.get("name", f"{tour.upper()} Event")
                            for grp in ev.get("groupings", []):
                                grp_name = grp.get("grouping", {}).get("displayName", "")
                                if "double" in grp_name.lower():
                                    continue
                                for comp in grp.get("competitions", []):
                                    st = comp.get("status", {}).get("type", {})
                                    if st.get("state") != "pre":
                                        continue
                                    comps = comp.get("competitors", [])
                                    if len(comps) != 2:
                                        continue

                                    p1_obj, p2_obj = comps[0], comps[1]
                                    p1_name = p1_obj.get("athlete", {}).get("displayName", "")
                                    p2_name = p2_obj.get("athlete", {}).get("displayName", "")
                                    if not p1_name or not p2_name or "TBD" in [p1_name, p2_name] or "/" in p1_name:
                                        continue

                                    comp_id = int(comp.get("id", int(time.time() * 1000) % 10000000))
                                    p1_id = int(p1_obj.get("id", abs(hash(p1_name)) % 1000000))
                                    p2_id = int(p2_obj.get("id", abs(hash(p2_name)) % 1000000))
                                    p1_rank = int(p1_obj.get("curatedRank", {}).get("current", 120))
                                    p2_rank = int(p2_obj.get("curatedRank", {}).get("current", 120))

                                    venue_str = comp.get("venue", {}).get("fullName", "")
                                    surface = "Clay" if any(k in tourn_title.lower() or k in venue_str.lower() for k in ["clay", "roland", "rome", "madrid"]) else "Hard"
                                    if "grass" in tourn_title.lower() or "wimbledon" in tourn_title.lower():
                                        surface = "Grass"

                                    csi = self._estimate_court_speed_index(tourn_title, surface)
                                    full_tournament = f"{tour.upper()} - {tourn_title}".strip()

                                    candidates.append({
                                        "event_id": comp_id,
                                        "tournament": full_tournament,
                                        "host_country": "UNK",
                                        "surface": surface,
                                        "court_speed_index": csi,
                                        "is_bo5": False,
                                        "p1_id": p1_id,
                                        "p1_name": p1_name,
                                        "p1_rank": p1_rank,
                                        "p1_country": "UNK",
                                        "p2_id": p2_id,
                                        "p2_name": p2_name,
                                        "p2_rank": p2_rank,
                                        "p2_country": "UNK",
                                        "priority": 1000 - (p1_rank + p2_rank)
                                    })
                except Exception as e:
                    logger.debug(f"ESPN fetch error: {e}")

        candidates.sort(key=lambda x: x["priority"], reverse=True)
        return candidates[:Config.MAX_DAILY_MATCHES]

    def fetch_h2h(self, event_id: int) -> Tuple[int, int]:
        try:
            res = self.session.get(f"{self.sofascore_base}/event/{event_id}/h2h", timeout=5)
            if res.status_code == 200:
                duel = res.json().get("teamDuel", {})
                if duel:
                    return int(duel.get("homeWins", 0)), int(duel.get("awayWins", 0))
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _fractional_to_decimal(frac_str: Optional[str]) -> Optional[float]:
        if not frac_str:
            return None
        try:
            if "/" in str(frac_str):
                num, den = map(float, str(frac_str).split("/"))
                if den > 0:
                    return 1.0 + (num / den)
            return float(frac_str)
        except Exception:
            return None

    def fetch_market_microstructure(self, event_id: int) -> MarketMicrostructure:
        micro = MarketMicrostructure()
        try:
            res = self.session.get(f"{self.sofascore_base}/event/{event_id}/odds/1/all", timeout=5)
            if res.status_code == 200:
                markets = res.json().get("markets", [])
                for m in markets:
                    if m.get("marketId") == 1 or m.get("marketName", "").lower() in ["full time", "match winner"]:
                        choices = m.get("choices", [])
                        if len(choices) == 2:
                            cur_odd1 = self._fractional_to_decimal(choices[0].get("fractionalValue"))
                            cur_odd2 = self._fractional_to_decimal(choices[1].get("fractionalValue"))
                            init_odd1 = self._fractional_to_decimal(choices[0].get("initialFractionalValue"))
                            init_odd2 = self._fractional_to_decimal(choices[1].get("initialFractionalValue"))

                            if cur_odd1 and cur_odd2 and cur_odd1 > 1.0 and cur_odd2 > 1.0:
                                inv1, inv2 = 1.0 / cur_odd1, 1.0 / cur_odd2
                                overround = inv1 + inv2
                                fair_p1 = (inv1 / overround) * 100.0
                                fair_p2 = 100.0 - fair_p1

                                micro.live_p1_prob = round(fair_p1, 2)
                                micro.fair_decimal_odd_p1 = round(100.0 / max(1.0, fair_p1), 2)
                                micro.fair_decimal_odd_p2 = round(100.0 / max(1.0, fair_p2), 2)

                                if init_odd1 and init_odd2 and init_odd1 > 1.0 and init_odd2 > 1.0:
                                    i_inv1, i_inv2 = 1.0 / init_odd1, 1.0 / init_odd2
                                    open_p1 = (i_inv1 / (i_inv1 + i_inv2)) * 100.0
                                    micro.opening_p1_prob = round(open_p1, 2)
                                    move = fair_p1 - open_p1
                                    micro.line_movement_p1 = round(move, 2)

                                    if move >= Config.STEAM_MOVE_ALERT_PCT:
                                        micro.smart_money_signal = "P1_STEAM"
                                    elif move <= -Config.STEAM_MOVE_ALERT_PCT:
                                        micro.smart_money_signal = "P2_STEAM"
                            break
        except Exception:
            pass
        return micro

    def fetch_player_news_sentiment(self, player_name: str) -> Tuple[float, List[str]]:
        query = urllib.parse.quote(f'"{player_name}" tennis when:7d')
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headlines = []
        sentiment_score = 0.0

        try:
            res = self.plain_session.get(rss_url, timeout=6)
            if res.status_code == 200 and res.text:
                root = ET.fromstring(res.text)
                for item in root.findall(".//item")[:5]:
                    title_el = item.find("title")
                    if title_el is not None and title_el.text:
                        clean_title = title_el.text.split(" - ")[0].strip()
                        headlines.append(clean_title)
                        t_low = clean_title.lower()
                        for neg in self.NEGATIVE_NEWS_LEXICON:
                            if neg in t_low:
                                sentiment_score -= 0.25
                        for pos in self.POSITIVE_NEWS_LEXICON:
                            if pos in t_low:
                                sentiment_score += 0.20
        except Exception:
            pass

        return round(float(np.clip(sentiment_score, -1.0, 1.0)), 2), headlines[:3]

    def _extract_single_match_telemetry(self, match_id: int, is_home: bool) -> Optional[Dict[str, float]]:
        """استخراج واقعی ریزآمار سرویس اول، سرویس دوم، ریترن و بریکپوینت از بازیهای اخیر بازیکن"""
        try:
            res = self.session.get(f"{self.sofascore_base}/event/{match_id}/statistics", timeout=5)
            if res.status_code != 200:
                return None
            stats_blocks = res.json().get("statistics", [])
            all_period = next((b for b in stats_blocks if b.get("period") == "ALL"), None)
            if not all_period:
                return None

            parsed = {}
            side_key = "homeValue" if is_home else "awayValue"
            side_total_key = "homeTotal" if is_home else "awayTotal"

            for group in all_period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    name = item.get("name", "").lower()
                    val = item.get(side_key)
                    tot = item.get(side_total_key)

                    if "first serve" == name and tot:
                        parsed["1st_in"] = val / tot
                    elif "first serve points" in name and tot:
                        parsed["1st_won"] = val / tot
                    elif "second serve points" in name and tot:
                        parsed["2nd_won"] = val / tot
                    elif "break points saved" in name and tot:
                        parsed["bp_saved"] = val / tot
                    elif "break points converted" in name and tot:
                        parsed["bp_conv"] = val / tot
                    elif "service points won" in name and tot:
                        parsed["spw"] = val / tot
                    elif "receiver points won" in name and tot:
                        parsed["rpw"] = val / tot

            if "1st_won" in parsed and "2nd_won" in parsed:
                first_in = parsed.get("1st_in", 0.62)
                if "spw" not in parsed:
                    parsed["spw"] = (first_in * parsed["1st_won"]) + ((1.0 - first_in) * parsed["2nd_won"])
                return parsed
        except Exception:
            pass
        return None

    def build_player_complete_telemetry(
        self,
        player_id: int,
        name: str,
        country_code: str,
        host_country: str,
        rank: int,
        surface: str
    ) -> PlayerTelemetry:
        """
        استخراج و ترکیب تلمتری واقعی بازیهای اخیر + روانشناسی ستبهست + دیتاست تاریخی
        """
        tour_base_spw = self.SURFACE_BASELINES.get(surface, {"spw": 0.634})["spw"]
        rank_factor = np.clip((250 - min(rank, 250)) / 250.0, -0.5, 1.0)

        default_spw = tour_base_spw + (rank_factor * 0.035)
        default_rpw = (1.0 - tour_base_spw) + (rank_factor * 0.028)

        recent_wins, recent_total = 0, 0
        sets_last_7d, minutes_last_72h = 0, 0
        surf_wins_live, surf_total_live = 0, 0

        tb_won, tb_total = 0, 0
        deciding_won, deciding_total = 0, 0
        lost_set1_total, won_after_losing_set1 = 0, 0
        bagel_shock = 0.0

        telemetry_samples = []
        now_ts = time.time()

        # ۱. استخراج زنده ۱۵ بازی آخر بازیکن برای محاسبه واقعی خستگی، فرم و روانشناسی
        try:
            res = self.session.get(f"{self.sofascore_base}/team/{player_id}/events/last/0", timeout=6)
            if res.status_code == 200:
                events = res.json().get("events", [])
                finished_events = [e for e in events if e.get("status", {}).get("type") == "finished"]

                for ev in finished_events[-15:]:
                    recent_total += 1
                    is_home = (ev.get("homeTeam", {}).get("id") == player_id)
                    winner_code = ev.get("winnerCode")
                    won_match = (is_home and winner_code == 1) or (not is_home and winner_code == 2)
                    if won_match:
                        recent_wins += 1

                    ev_surf = self._normalize_surface(ev.get("groundType", "Hard"))
                    if ev_surf == surface:
                        surf_total_live += 1
                        if won_match:
                            surf_wins_live += 1

                    h_score_obj = ev.get("homeScore", {})
                    a_score_obj = ev.get("awayScore", {})
                    my_score_obj = h_score_obj if is_home else a_score_obj
                    opp_score_obj = a_score_obj if is_home else h_score_obj

                    my_sets = int(my_score_obj.get("current", 0) or 0)
                    opp_sets = int(opp_score_obj.get("current", 0) or 0)
                    total_sets_match = my_sets + opp_sets

                    s1_my = my_score_obj.get("period1")
                    s1_opp = opp_score_obj.get("period1")
                    if s1_my is not None and s1_opp is not None and s1_my < s1_opp:
                        lost_set1_total += 1
                        if won_match:
                            won_after_losing_set1 += 1

                    if total_sets_match >= 3:
                        deciding_total += 1
                        if won_match:
                            deciding_won += 1

                    is_very_recent = (finished_events.index(ev) >= max(0, len(finished_events) - 5))
                    for p_idx in range(1, 6):
                        ms = my_score_obj.get(f"period{p_idx}")
                        os_ = opp_score_obj.get(f"period{p_idx}")
                        if ms is not None and os_ is not None:
                            if (ms == 7 and os_ == 6) or (ms == 6 and os_ == 7):
                                tb_total += 1
                                if ms > os_:
                                    tb_won += 1
                            if is_very_recent:
                                if ms == 6 and os_ <= 1:
                                    bagel_shock += 0.35
                                elif os_ == 6 and ms <= 1:
                                    bagel_shock -= 0.45

                    start_ts = ev.get("startTimestamp", 0)
                    days_ago = (now_ts - start_ts) / 86400.0
                    if days_ago <= 7.0:
                        sets_last_7d += total_sets_match
                    if days_ago <= 3.0:
                        minutes_last_72h += total_sets_match * 44

                for ev in reversed(finished_events[-Config.RECENT_MATCHES_TELEMETRY:]):
                    is_home = (ev.get("homeTeam", {}).get("id") == player_id)
                    t_stat = self._extract_single_match_telemetry(ev.get("id"), is_home)
                    if t_stat:
                        telemetry_samples.append(t_stat)
        except Exception:
            pass

        # ۲. استخراج آمار تاریخی از دیتاست مرجع (ATP/WTA)
        hist_spw, hist_rpw, hist_surf_wr, hist_bp = None, None, None, None
        if not self.historical_df.empty:
            last_name = name.split()[-1].lower()
            df = self.historical_df
            w_m = df[df["winner_name"].fillna("").str.lower().str.contains(last_name, regex=False)]
            l_m = df[df["loser_name"].fillna("").str.lower().str.contains(last_name, regex=False)]
            if (len(w_m) + len(l_m)) >= 4:
                sw = len(w_m[w_m["surface"] == surface])
                sl = len(l_m[l_m["surface"] == surface])
                if (sw + sl) >= 3:
                    hist_surf_wr = sw / (sw + sl)

                w_spw = ((w_m["w_1stWon"] + w_m["w_2ndWon"]) / w_m["w_svpt"].replace(0, np.nan)).dropna()
                l_spw = ((l_m["l_1stWon"] + l_m["l_2ndWon"]) / l_m["l_svpt"].replace(0, np.nan)).dropna()
                all_spw = pd.concat([w_spw, l_spw])
                if len(all_spw) > 0:
                    hist_spw = float(all_spw.mean())

                w_rpw = (1.0 - ((w_m["l_1stWon"] + w_m["l_2ndWon"]) / w_m["l_svpt"].replace(0, np.nan))).dropna()
                l_rpw = (1.0 - ((l_m["w_1stWon"] + l_m["w_2ndWon"]) / l_m["w_svpt"].replace(0, np.nan))).dropna()
                all_rpw = pd.concat([w_rpw, l_rpw])
                if len(all_rpw) > 0:
                    hist_rpw = float(all_rpw.mean())

                w_bp = (w_m["w_bpSaved"] / w_m["w_bpFaced"].replace(0, np.nan)).dropna()
                l_bp = (l_m["l_bpSaved"] / l_m["l_bpFaced"].replace(0, np.nan)).dropna()
                all_bp = pd.concat([w_bp, l_bp])
                if len(all_bp) > 0:
                    hist_bp = float(all_bp.mean())

        # ۳. ترکیب بیزین تلمتری زنده و دیتاست تاریخی
        if telemetry_samples:
            live_spw = float(np.mean([s.get("spw", default_spw) for s in telemetry_samples]))
            live_rpw = float(np.mean([s.get("rpw", default_rpw) for s in telemetry_samples]))
            live_1st_in = float(np.mean([s.get("1st_in", 0.62) for s in telemetry_samples]))
            live_1st_won = float(np.mean([s.get("1st_won", 0.71) for s in telemetry_samples]))
            live_2nd_won = float(np.mean([s.get("2nd_won", 0.51) for s in telemetry_samples]))
            live_bp_saved = float(np.mean([s.get("bp_saved", 0.61) for s in telemetry_samples]))
            live_bp_conv = float(np.mean([s.get("bp_conv", 0.40) for s in telemetry_samples]))
        else:
            live_spw = hist_spw if hist_spw else default_spw
            live_rpw = hist_rpw if hist_rpw else default_rpw
            live_1st_in = 0.62
            live_1st_won = round(live_spw * 1.10, 3)
            live_2nd_won = round(live_spw * 0.80, 3)
            live_bp_saved = hist_bp if hist_bp else 0.61
            live_bp_conv = 0.40

        final_spw = (0.65 * live_spw + 0.35 * hist_spw) if hist_spw else live_spw
        final_rpw = (0.65 * live_rpw + 0.35 * hist_rpw) if hist_rpw else live_rpw
        final_bp_saved = (0.60 * live_bp_saved + 0.40 * hist_bp) if hist_bp else live_bp_saved

        recent_wr = (recent_wins / recent_total) if recent_total > 0 else (0.55 if rank < 100 else 0.46)
        live_surf_wr = (surf_wins_live / surf_total_live) if surf_total_live >= 2 else recent_wr
        final_surf_wr = (0.55 * live_surf_wr + 0.45 * hist_surf_wr) if hist_surf_wr is not None else live_surf_wr

        tb_wr = (tb_won + 1.5) / (tb_total + 3.0) if tb_total > 0 else (0.50 + rank_factor * 0.05)
        dec_wr = (deciding_won + 1.5) / (deciding_total + 3.0) if deciding_total > 0 else (0.50 + rank_factor * 0.05)
        cri_rate = (won_after_losing_set1 + 0.8) / (lost_set1_total + 3.0) if lost_set1_total > 0 else (0.25 + rank_factor * 0.06)
        is_home_crowd = bool(country_code and host_country and country_code.upper() == host_country.upper())

        news_score, headlines = self.fetch_player_news_sentiment(name)

        raw_clutch = (
            (final_bp_saved * 35.0) +
            (tb_wr * 25.0) +
            (dec_wr * 25.0) +
            (cri_rate * 15.0) +
            (bagel_shock * 4.0) +
            (3.0 if is_home_crowd else 0.0) +
            (news_score * 5.0)
        )
        clutch_rating = round(float(np.clip(raw_clutch * 1.45, 18.0, 98.0)), 1)

        psych_profile = PsychometricProfile(
            clutch_rating=clutch_rating,
            comeback_resilience_cri=round(float(cri_rate), 3),
            tiebreak_win_rate=round(float(tb_wr), 3),
            deciding_set_win_rate=round(float(dec_wr), 3),
            bagel_shock_score=round(float(np.clip(bagel_shock, -1.0, 1.0)), 2),
            is_home_crowd=is_home_crowd,
            news_sentiment_score=news_score,
            recent_headlines=headlines
        )

        base_elo = 1850.0 - (np.log2(max(1, rank)) * 62.0)
        surface_elo = base_elo + ((final_surf_wr - 0.50) * 220.0) + ((recent_wr - 0.50) * 110.0) + (news_score * 25.0)

        acute_fatigue = max(0.0, (minutes_last_72h - 140) / 220.0)
        chronic_fatigue = max(0.0, (sets_last_7d - 7) / 14.0)
        fatigue_idx = float(np.clip(0.65 * acute_fatigue + 0.35 * chronic_fatigue, 0.0, 1.0))

        return PlayerTelemetry(
            player_id=player_id,
            name=name,
            country_code=country_code,
            ranking=rank,
            elo_surface_rating=round(float(surface_elo), 1),
            recent_win_rate_10=round(float(recent_wr), 3),
            surface_win_rate=round(float(final_surf_wr), 3),
            fatigue_index=round(fatigue_idx, 3),
            court_minutes_last_72h=minutes_last_72h,
            sets_played_last_7d=sets_last_7d,
            first_serve_in_pct=round(live_1st_in, 3),
            first_serve_won_pct=round(live_1st_won, 3),
            second_serve_won_pct=round(live_2nd_won, 3),
            service_points_won_pct=round(float(np.clip(final_spw, 0.50, 0.78)), 4),
            return_points_won_pct=round(float(np.clip(final_rpw, 0.25, 0.50)), 4),
            bp_saved_pct=round(float(np.clip(final_bp_saved, 0.40, 0.80)), 3),
            bp_converted_pct=round(float(np.clip(live_bp_conv, 0.25, 0.60)), 3),
            psychometrics=psych_profile
        )

    def fetch_finished_match_outcome(self, event_id: int) -> Dict:
        """بررسی وضعیت نهایی مسابقه در پایان شب از هر دو منبع SofaScore و ESPN"""
        # ۱. بررسی از SofaScore
        try:
            res = self.session.get(f"{self.sofascore_base}/event/{event_id}", timeout=6)
            if res.status_code == 200:
                ev = res.json().get("event", {})
                st = ev.get("status", {})
                st_type = st.get("type", "")
                st_desc = st.get("description", "").lower()

                if st_type in ["canceled", "postponed"] or any(w in st_desc for w in ["walkover", "retired", "abandoned"]):
                    return {"state": "VOID", "reason": st.get("description", "Canceled/Retired")}

                if st_type == "finished":
                    w_code = ev.get("winnerCode")
                    h = ev.get("homeTeam", {})
                    a = ev.get("awayTeam", {})
                    h_sc = ev.get("homeScore", {}).get("current", 0) or 0
                    a_sc = ev.get("awayScore", {}).get("current", 0) or 0

                    set_scores = []
                    for p_idx in range(1, 6):
                        hs = ev.get("homeScore", {}).get(f"period{p_idx}")
                        as_ = ev.get("awayScore", {}).get(f"period{p_idx}")
                        if hs is not None and as_ is not None:
                            set_scores.append(f"{hs}-{as_}")

                    score_str = f"{h_sc}-{a_sc}" + (f" ({', '.join(set_scores)})" if set_scores else "")
                    w_id = h.get("id") if w_code == 1 else a.get("id")
                    w_name = h.get("name") if w_code == 1 else a.get("name")
                    return {
                        "state": "FINISHED",
                        "winner_id": w_id,
                        "winner_name": w_name,
                        "score_str": score_str,
                        "sets_played": int(h_sc + a_sc)
                    }
        except Exception:
            pass

        # ۲. بررسی از ESPN در صورتی که بازی از فید ESPN ثبت شده باشد
        for tour in ["atp", "wta"]:
            try:
                espn_url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
                r = self.plain_session.get(espn_url, timeout=8)
                if r.status_code == 200:
                    for ev in r.json().get("events", []):
                        for grp in ev.get("groupings", []):
                            for comp in grp.get("competitions", []):
                                if int(comp.get("id", -1)) == int(event_id):
                                    st_state = comp.get("status", {}).get("type", {}).get("state", "")
                                    if st_state == "post":
                                        comps = comp.get("competitors", [])
                                        winner_comp = next((c for c in comps if c.get("winner") is True), None)
                                        if winner_comp:
                                            w_name = winner_comp.get("athlete", {}).get("displayName", "")
                                            w_id = int(winner_comp.get("id", abs(hash(w_name)) % 1000000))
                                            return {
                                                "state": "FINISHED",
                                                "winner_id": w_id,
                                                "winner_name": w_name,
                                                "score_str": "Finished (ESPN)",
                                                "sets_played": 2
                                            }
            except Exception:
                pass

        return {"state": "PENDING"}
