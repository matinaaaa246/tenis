import json
import re
from openai import OpenAI
from typing import Optional

from config import Config, logger
from harvester import PlayerTelemetry, QuantMatchReport, LLMAuditOutput


class LLMQuantArbiter:
    """
    داور هوش مصنوعی مجهز به روانشناسی ورزشی، ریزساختار بازار و تلمتری تکنیکال — ورژن ۱
    """
    def __init__(self):
        self.enabled = bool(Config.LLM_API_KEY and "YOUR_" not in Config.LLM_API_KEY)
        if self.enabled:
            self.client = OpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                timeout=18.0,
                max_retries=1
            )
            self.active_model = Config.LLM_MODEL
            logger.info(f"داور هوش مصنوعی ورژن ۱ با مدل {self.active_model} فعال شد.")
        else:
            logger.warning("کلید API هوش مصنوعی تنظیم نشده است؛ استفاده از موتور تحلیلی داخلی.")

    def _call_api_with_fallback(self, messages: list) -> Optional[str]:
        models_to_try = [self.active_model]
        if Config.LLM_FALLBACK_MODEL and Config.LLM_FALLBACK_MODEL not in models_to_try:
            models_to_try.append(Config.LLM_FALLBACK_MODEL)

        for model_name in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.15,
                    max_tokens=650
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    if model_name != self.active_model:
                        self.active_model = model_name
                    return content
            except Exception as e:
                err_str = str(e)
                if any(k in err_str for k in ["410", "Gone", "not available", "404"]):
                    if Config.LLM_FALLBACK_MODEL and model_name != Config.LLM_FALLBACK_MODEL:
                        self.active_model = Config.LLM_FALLBACK_MODEL
                else:
                    logger.warning(f"خطا در مدل {model_name}: {err_str[:90]}")
        return None

    def evaluate(
        self,
        p1: PlayerTelemetry,
        p2: PlayerTelemetry,
        quant: QuantMatchReport
    ) -> LLMAuditOutput:
        if not self.enabled:
            return self._deterministic_fallback(p1, p2, quant)

        fav = p1 if quant.ensemble_p1_prob > quant.ensemble_p2_prob else p2
        fav_prob = max(quant.ensemble_p1_prob, quant.ensemble_p2_prob)
        fair_odd = round(100.0 / max(1.0, fav_prob), 2)
        rec_bet = "برد ۲-۰ (Straight Sets)" if fav_prob >= 76.0 and quant.expected_total_sets < 2.25 else "برد مستقیم مسابقه"

        prompt = f"""
You are an elite Tennis Quantitative & Psychometric Predictor (Version 1).
Your goal is to deliver powerful, accurate predictions backed by deep statistical and psychological telemetry.

Match: {quant.tournament} ({quant.surface} | Speed CSI: {quant.court_speed_index})
Player 1: {p1.name} (Rank #{p1.ranking}, Surface WinRate: {round(p1.surface_win_rate*100,1)}%, Markov Hold: {quant.p1_markov_hold_prob}%, SPW: {round(p1.service_points_won_pct*100,1)}%, RPW: {round(p1.return_points_won_pct*100,1)}%, Clutch: {p1.psychometrics.clutch_rating}, CRI Comeback: {round(p1.psychometrics.comeback_resilience_cri*100,1)}%, 72h Court Mins: {p1.court_minutes_last_72h}m)
Player 2: {p2.name} (Rank #{p2.ranking}, Surface WinRate: {round(p2.surface_win_rate*100,1)}%, Markov Hold: {quant.p2_markov_hold_prob}%, SPW: {round(p2.service_points_won_pct*100,1)}%, RPW: {round(p2.return_points_won_pct*100,1)}%, Clutch: {p2.psychometrics.clutch_rating}, CRI Comeback: {round(p2.psychometrics.comeback_resilience_cri*100,1)}%, 72h Court Mins: {p2.court_minutes_last_72h}m)

Quant Ensemble Win Prob: {p1.name} {quant.ensemble_p1_prob}% vs {p2.name} {quant.ensemble_p2_prob}% (Spread: {quant.prob_spread}%)
Monte Carlo (10K): {quant.mc_p1_win_prob}% vs {quant.mc_p2_win_prob}% | Expected Sets: {quant.expected_total_sets} | Tiebreak Prob: {quant.tiebreak_prob}%
Risk Flags: {quant.unidentifiable_flags or 'None'}

Rules:
- If the match is truly too close to call (is_mathematically_unidentifiable is True), set status to 'UNIDENTIFIABLE' and predicted_winner to 'قابل شناسایی نیست'.
- Otherwise, set status to 'IDENTIFIED' and predicted_winner to '{fav.name}'.
- Write 'psychological_and_tactical_synthesis' in fluent, authoritative Persian citing exact numbers (Hold%, SPW/RPW, Clutch rating, fatigue, and surface synergy).

Return ONLY valid JSON:
{{
  "status": "IDENTIFIED",
  "predicted_winner": "{fav.name}",
  "confidence": {fav_prob},
  "recommended_bet": "{rec_bet}",
  "fair_odd": {fair_odd},
  "psychological_and_tactical_synthesis": "تحلیل دقیق فنی و روانی به فارسی...",
  "risk_warning": "فاکتور کلیدی یا هشدار مسابقه"
}}
"""

        try:
            raw_content = self._call_api_with_fallback([{"role": "user", "content": prompt}])
            if raw_content:
                match = re.search(r"(\{.*\})", raw_content.strip(), re.DOTALL)
                clean_json_str = match.group(1) if match else raw_content.strip()
                data = json.loads(clean_json_str)
                res = LLMAuditOutput(**data)

                if quant.is_mathematically_unidentifiable or res.confidence < Config.MIN_LLM_CONFIDENCE:
                    res.status = "UNIDENTIFIABLE"
                    res.predicted_winner = "قابل شناسایی نیست"
                    if quant.unidentifiable_flags:
                        res.risk_warning = " | ".join(quant.unidentifiable_flags)

                return res
        except Exception as e:
            logger.warning(f"خطا در پردازش پاسخ LLM ({e}) -> استفاده از موتور تحلیلی داخلی.")

        return self._deterministic_fallback(p1, p2, quant)

    @staticmethod
    def _deterministic_fallback(
        p1: PlayerTelemetry,
        p2: PlayerTelemetry,
        quant: QuantMatchReport
    ) -> LLMAuditOutput:
        fav = p1 if quant.ensemble_p1_prob > quant.ensemble_p2_prob else p2
        fav_prob = max(quant.ensemble_p1_prob, quant.ensemble_p2_prob)
        fair_odd = round(100.0 / max(1.0, fav_prob), 2)
        rec_bet = "برد ۲-۰ (Straight Sets)" if fav_prob >= 76.0 and quant.expected_total_sets < 2.25 else "برد مستقیم (Match Winner)"

        if quant.is_mathematically_unidentifiable:
            return LLMAuditOutput(
                status="UNIDENTIFIABLE",
                predicted_winner="قابل شناسایی نیست",
                confidence=fav_prob,
                recommended_bet="قابل شناسایی نیست (رقابت نزدیک)",
                fair_odd=fair_odd,
                psychological_and_tactical_synthesis=(
                    f"آمارهای فنی و روانی دو بازیکن بسیار به هم نزدیک است (نرخ حفظ سرویس {quant.p1_markov_hold_prob}% در برابر "
                    f"{quant.p2_markov_hold_prob}% و اختلاف احتمال کلی تنها {quant.prob_spread}%). با توجه به احتمال تایبریک "
                    f"{quant.tiebreak_prob}%، برنده قطعی قابل شناسایی نیست."
                ),
                risk_warning=" | ".join(quant.unidentifiable_flags) if quant.unidentifiable_flags else "رقابت نزدیک و تنگاتنگ"
            )

        fav_hold = quant.p1_markov_hold_prob if fav == p1 else quant.p2_markov_hold_prob
        und_hold = quant.p2_markov_hold_prob if fav == p1 else quant.p1_markov_hold_prob

        return LLMAuditOutput(
            status="IDENTIFIED",
            predicted_winner=fav.name,
            confidence=fav_prob,
            recommended_bet=rec_bet,
            fair_odd=fair_odd,
            psychological_and_tactical_synthesis=(
                f"برتری قدرتمند تکنیکال و ذهنی برای {fav.name}: "
                f"نرخ حفظ گیم سرویس در مدل مارکوف {fav_hold}% در برابر {und_hold}% حریف است. "
                f"همچنین در شاخصهای روانشناسی، نمره اعصاب تحت فشار (Clutch) برابر {fav.psychometrics.clutch_rating} و "
                f"نرخ بازگشت به بازی (CRI) معادل {round(fav.psychometrics.comeback_resilience_cri*100, 1)}% است که در زمین {quant.surface} "
                f"کفه ترازو را به شکل قاطعی به سود او سنگین میکند."
            ),
            risk_warning="تاییدشده توسط مدل زنجیره مارکوف، شبیهساز مونتکارلو و شاخصهای روانی"
        )
