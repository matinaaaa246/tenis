import math
import numpy as np
from typing import Tuple, Dict

from config import Config
from harvester import (
    PlayerTelemetry,
    MarketMicrostructure,
    QuantMatchReport,
    MasterpieceTennisHarvester
)


class PsychometricMonteCarloEngine:
    """
    موتور شبیهسازی کوانت وابسته به وضعیت روانی و فیزیکی (Stateful Monte Carlo Engine) — ورژن ۱
    """

    @staticmethod
    def markov_game_hold_prob(p: float) -> float:
        p = float(np.clip(p, 0.05, 0.95))
        q = 1.0 - p
        pre_deuce = (p ** 4) * (1.0 + 4.0 * q + 10.0 * (q ** 2))
        deuce_loop = (20.0 * (p ** 5) * (q ** 3)) / max(1e-9, (1.0 - 2.0 * p * q))
        return float(np.clip(pre_deuce + deuce_loop, 0.01, 0.99))

    @staticmethod
    def markov_tiebreak_prob(p_a: float, p_b: float, clutch_diff_p1: float) -> float:
        p_tb = 0.5 * p_a + 0.5 * (1.0 - p_b) + (clutch_diff_p1 * 0.0012)
        p_tb = float(np.clip(p_tb, 0.1, 0.9))
        q_tb = 1.0 - p_tb
        pre_6_6 = (p_tb ** 7) * (
            1 + 7 * q_tb + 28 * (q_tb ** 2) + 84 * (q_tb ** 3) + 210 * (q_tb ** 4) + 462 * (q_tb ** 5)
        )
        deuce_6_6 = (924.0 * (p_tb ** 7) * (q_tb ** 6) * (p_tb ** 2)) / max(1e-9, (1.0 - 2.0 * p_tb * q_tb))
        return float(np.clip(pre_6_6 + deuce_6_6, 0.05, 0.95))

    @classmethod
    def run_stateful_monte_carlo(
        cls,
        base_hold_p1: float,
        base_hold_p2: float,
        tb_p1: float,
        p1: PlayerTelemetry,
        p2: PlayerTelemetry,
        is_bo5: bool = False,
        n_sims: int = 10000
    ) -> Tuple[float, float, float]:
        rng = np.random.default_rng(seed=42)
        sets_to_win = 3 if is_bo5 else 2
        max_sets = 5 if is_bo5 else 3

        p1_sets_won = np.zeros(n_sims, dtype=np.int32)
        p2_sets_won = np.zeros(n_sims, dtype=np.int32)
        had_tiebreak = np.zeros(n_sims, dtype=bool)

        for set_num in range(1, max_sets + 1):
            active = (p1_sets_won < sets_to_win) & (p2_sets_won < sets_to_win)
            if not np.any(active):
                break

            dyn_hold_p1 = np.full(n_sims, base_hold_p1, dtype=np.float64)
            dyn_hold_p2 = np.full(n_sims, base_hold_p2, dtype=np.float64)

            if set_num == 2:
                p1_down = (p1_sets_won == 0) & (p2_sets_won == 1)
                p2_down = (p2_sets_won == 0) & (p1_sets_won == 1)

                p1_resilience_shift = (p1.psychometrics.comeback_resilience_cri - 0.25) * 0.08
                p2_resilience_shift = (p2.psychometrics.comeback_resilience_cri - 0.25) * 0.08

                dyn_hold_p1 = np.where(p1_down, np.clip(dyn_hold_p1 + p1_resilience_shift, 0.1, 0.96), dyn_hold_p1)
                dyn_hold_p2 = np.where(p2_down, np.clip(dyn_hold_p2 + p2_resilience_shift, 0.1, 0.96), dyn_hold_p2)

            elif set_num >= 3:
                p1_dec_shift = ((p1.psychometrics.deciding_set_win_rate - 0.50) * 0.06) - (p1.fatigue_index * 0.045)
                p2_dec_shift = ((p2.psychometrics.deciding_set_win_rate - 0.50) * 0.06) - (p2.fatigue_index * 0.045)

                dyn_hold_p1 = np.clip(dyn_hold_p1 + p1_dec_shift, 0.1, 0.96)
                dyn_hold_p2 = np.clip(dyn_hold_p2 + p2_dec_shift, 0.1, 0.96)

            g1 = np.zeros(n_sims, dtype=np.int32)
            g2 = np.zeros(n_sims, dtype=np.int32)

            for game_idx in range(12):
                set_ongoing = active & (~((g1 >= 6) & (g1 - g2 >= 2))) & (~((g2 >= 6) & (g2 - g1 >= 2)))
                if not np.any(set_ongoing):
                    break

                if game_idx % 2 == 0:
                    won_game = rng.random(n_sims) < dyn_hold_p1
                    g1 = np.where(set_ongoing & won_game, g1 + 1, g1)
                    g2 = np.where(set_ongoing & (~won_game), g2 + 1, g2)
                else:
                    won_game = rng.random(n_sims) < dyn_hold_p2
                    g2 = np.where(set_ongoing & won_game, g2 + 1, g2)
                    g1 = np.where(set_ongoing & (~won_game), g1 + 1, g1)

            tiebreak_mask = active & (g1 == 6) & (g2 == 6)
            had_tiebreak = had_tiebreak | tiebreak_mask

            tb_winner_p1 = rng.random(n_sims) < tb_p1
            g1 = np.where(tiebreak_mask & tb_winner_p1, 7, g1)
            g2 = np.where(tiebreak_mask & (~tb_winner_p1), 7, g2)

            p1_sets_won = np.where(active & (g1 > g2), p1_sets_won + 1, p1_sets_won)
            p2_sets_won = np.where(active & (g2 > g1), p2_sets_won + 1, p2_sets_won)

        mc_win_prob_p1 = float(np.mean(p1_sets_won == sets_to_win) * 100.0)
        expected_sets = float(np.mean(p1_sets_won + p2_sets_won))
        tb_occurrence_prob = float(np.mean(had_tiebreak) * 100.0)

        return round(mc_win_prob_p1, 2), round(expected_sets, 2), round(tb_occurrence_prob, 1)

    @staticmethod
    def calculate_shannon_entropy(prob_pct: float) -> float:
        p = np.clip(prob_pct / 100.0, 1e-6, 1.0 - 1e-6)
        q = 1.0 - p
        return round(float(-(p * math.log2(p) + q * math.log2(q))), 4)

    @staticmethod
    def calculate_kelly_and_ev(
        ensemble_p1: float,
        ensemble_p2: float,
        market: MarketMicrostructure
    ) -> Tuple[float, float]:
        """محاسبه واقعی ارزش انتظاری (+EV) و سایزینگ Quarter-Kelly بدون اعداد ساختگی"""
        if market.live_p1_prob is None:
            fav_prob = max(ensemble_p1, ensemble_p2) / 100.0
            # اگر ضریب بازار موجود نباشد، بر اساس حاشیه برتری بالای 55% سایز پیشنهادی محاسبه میشود
            conviction_stake = max(0.0, (fav_prob - 0.55) * 10.0)
            return 0.0, round(min(4.0, conviction_stake), 2)

        fav_is_p1 = ensemble_p1 >= ensemble_p2
        model_prob = (ensemble_p1 if fav_is_p1 else ensemble_p2) / 100.0
        mkt_prob = (market.live_p1_prob if fav_is_p1 else (100.0 - market.live_p1_prob)) / 100.0

        ev_edge = (model_prob - mkt_prob) * 100.0
        if ev_edge <= 0:
            return round(ev_edge, 2), 0.0

        decimal_odd = 1.0 / max(0.05, mkt_prob)
        b = decimal_odd - 1.0
        full_kelly = ((b * model_prob) - (1.0 - model_prob)) / max(0.01, b)
        quarter_kelly_pct = max(0.0, full_kelly * 0.25 * 100.0)
        return round(ev_edge, 2), round(min(5.0, quarter_kelly_pct), 2)

    @classmethod
    def analyze_match(
        cls,
        match_meta: Dict,
        p1: PlayerTelemetry,
        p2: PlayerTelemetry,
        h2h_p1: int,
        h2h_p2: int,
        market: MarketMicrostructure,
        weights: Tuple[float, float, float]
    ) -> QuantMatchReport:
        surface = match_meta["surface"]
        csi = match_meta["court_speed_index"]
        mu_s = MasterpieceTennisHarvester.SURFACE_BASELINES.get(surface, {"spw": 0.634})["spw"]
        mu_r = 1.0 - mu_s

        speed_boost_p1 = (csi - 1.0) * (p1.first_serve_won_pct - 0.70) * 0.25
        speed_boost_p2 = (csi - 1.0) * (p2.first_serve_won_pct - 0.70) * 0.25

        home_boost_p1 = 0.008 if p1.psychometrics.is_home_crowd else 0.0
        home_boost_p2 = 0.008 if p2.psychometrics.is_home_crowd else 0.0

        news_shift_p1 = p1.psychometrics.news_sentiment_score * 0.012
        news_shift_p2 = p2.psychometrics.news_sentiment_score * 0.012

        fatigue_diff_p1 = (p2.fatigue_index - p1.fatigue_index) * 0.016

        p1_pt_serve = (
            mu_s + (p1.service_points_won_pct - mu_s) - (p2.return_points_won_pct - mu_r)
            + speed_boost_p1 + home_boost_p1 + news_shift_p1 + fatigue_diff_p1
        )
        p2_pt_serve = (
            mu_s + (p2.service_points_won_pct - mu_s) - (p1.return_points_won_pct - mu_r)
            + speed_boost_p2 + home_boost_p2 + news_shift_p2 - fatigue_diff_p1
        )

        p1_pt_serve = float(np.clip(p1_pt_serve, 0.48, 0.79))
        p2_pt_serve = float(np.clip(p2_pt_serve, 0.48, 0.79))

        hold_p1 = cls.markov_game_hold_prob(p1_pt_serve)
        hold_p2 = cls.markov_game_hold_prob(p2_pt_serve)

        clutch_diff_p1 = p1.psychometrics.clutch_rating - p2.psychometrics.clutch_rating
        tb_p1 = cls.markov_tiebreak_prob(p1_pt_serve, p2_pt_serve, clutch_diff_p1)

        mc_p1_prob, exp_sets, tb_prob = cls.run_stateful_monte_carlo(
            hold_p1, hold_p2, tb_p1, p1, p2,
            is_bo5=match_meta.get("is_bo5", False),
            n_sims=Config.MONTE_CARLO_SIMULATIONS
        )

        elo_diff = p1.elo_surface_rating - p2.elo_surface_rating
        h2h_adj = ((h2h_p1 - h2h_p2) / max(1, h2h_p1 + h2h_p2 + 2)) * 45.0
        clutch_elo_adj = clutch_diff_p1 * 1.1
        elo_p1_prob = (1.0 / (1.0 + 10.0 ** (-(elo_diff + h2h_adj + clutch_elo_adj) / 400.0))) * 100.0

        w_mc, w_mkt, w_elo = weights
        if market.live_p1_prob is not None:
            ensemble_p1 = (w_mc * mc_p1_prob) + (w_mkt * market.live_p1_prob) + (w_elo * elo_p1_prob)
        else:
            norm = w_mc + w_elo
            ensemble_p1 = ((w_mc / norm) * mc_p1_prob) + ((w_elo / norm) * elo_p1_prob)

        ensemble_p1 = round(float(np.clip(ensemble_p1, 3.0, 97.0)), 2)
        ensemble_p2 = round(100.0 - ensemble_p1, 2)
        spread = round(abs(ensemble_p1 - ensemble_p2), 2)
        entropy = cls.calculate_shannon_entropy(ensemble_p1)
        ev_edge, kelly_pct = cls.calculate_kelly_and_ev(ensemble_p1, ensemble_p2, market)

        # تشخیص بازیهای کاملاً نزدیک و غیرقابل پیشبینی
        flags = []
        if spread < Config.MIN_PROB_SPREAD:
            flags.append(f"رقابت کاملاً نزدیک ({spread}% اختلاف احتمال)")

        if entropy > Config.MAX_SHANNON_ENTROPY:
            flags.append(f"آنتروپی شانون بالا ({entropy} bits - نزدیک به ۵۰-۵۰)")

        if ensemble_p1 > 56.0 and market.smart_money_signal == "P2_STEAM":
            flags.append(f"چرخش سنگین پول هوشمند بازار ({market.line_movement_p1}%) به سود {p2.name}")
        elif ensemble_p2 > 56.0 and market.smart_money_signal == "P1_STEAM":
            flags.append(f"چرخش سنگین پول هوشمند بازار (+{market.line_movement_p1}%) به سود {p1.name}")

        fav_player = p1 if ensemble_p1 >= ensemble_p2 else p2
        if fav_player.psychometrics.news_sentiment_score <= -0.50:
            flags.append(f"هشدار مصدومیت در اخبار روز برای {fav_player.name}")

        is_unidentifiable = len(flags) > 0

        # محاسبه نمره قدرت پیشبینی (Prediction Power Score) برای رتبهبندی قویترین سیگنالها
        fav_prob = max(ensemble_p1, ensemble_p2)
        power_score = (
            (fav_prob * 1.2) +
            (spread * 0.8) +
            (fav_player.psychometrics.clutch_rating * 0.25) +
            (max(0.0, ev_edge) * 1.5)
        )

        return QuantMatchReport(
            event_id=match_meta["event_id"],
            tournament=match_meta["tournament"],
            surface=surface,
            court_speed_index=csi,
            is_best_of_five=match_meta.get("is_bo5", False),
            p1_markov_hold_prob=round(hold_p1 * 100, 1),
            p2_markov_hold_prob=round(hold_p2 * 100, 1),
            mc_p1_win_prob=mc_p1_prob,
            mc_p2_win_prob=round(100.0 - mc_p1_prob, 2),
            expected_total_sets=exp_sets,
            tiebreak_prob=tb_prob,
            elo_p1_win_prob=round(elo_p1_prob, 2),
            market_micro=market,
            ensemble_p1_prob=ensemble_p1,
            ensemble_p2_prob=ensemble_p2,
            prob_spread=spread,
            shannon_entropy=entropy,
            ev_edge_pct=ev_edge,
            kelly_fraction_pct=kelly_pct,
            h2h_p1=h2h_p1,
            h2h_p2=h2h_p2,
            unidentifiable_flags=flags,
            is_mathematically_unidentifiable=is_unidentifiable,
            power_score=round(power_score, 1)
        )
