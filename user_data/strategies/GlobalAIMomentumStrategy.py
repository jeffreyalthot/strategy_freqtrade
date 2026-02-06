from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class GlobalAIMomentumStrategy(IStrategy):
    """
    Strategy "IA" multi-facteurs pour Freqtrade.

    Objectif:
    - Combiner momentum, tendance, volatilité et volume.
    - Construire un score d'entrée/sortie inspiré d'un modèle de ranking.
    - Fournir une base solide pour itération et optimisation manuelle.

    Remarque:
    Cette stratégie n'est PAS la "meilleure du monde" de manière universelle.
    Aucun modèle ne surperforme tous les marchés en permanence.
    Elle sert de base professionnelle pour entraînement / backtest / itération.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    startup_candle_count = 240

    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "90": 0.02,
        "180": 0,
    }

    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    trend_ema_slope_min = DecimalParameter(0.0, 0.05, default=0.006, space="buy", optimize=True)
    trend_lr_slope_min = DecimalParameter(0.0, 0.05, default=0.004, space="buy", optimize=True)
    range_bb_width_max = DecimalParameter(0.01, 0.08, default=0.035, space="buy", optimize=True)
    range_rsi_band = IntParameter(4, 18, default=10, space="buy", optimize=True)
    range_cci_abs_max = IntParameter(60, 160, default=120, space="buy", optimize=True)
    volatile_atr_pct_min = DecimalParameter(0.004, 0.05, default=0.018, space="buy", optimize=True)
    volatile_vol_rel_min = DecimalParameter(1.0, 2.5, default=1.25, space="buy", optimize=True)
    volatile_roll_vol_min = DecimalParameter(0.002, 0.03, default=0.012, space="buy", optimize=True)
    bear_macd_hist_max = DecimalParameter(-1.0, -0.02, default=-0.08, space="buy", optimize=True)
    bear_rsi_max = IntParameter(20, 45, default=38, space="buy", optimize=True)
    min_entry_edge = DecimalParameter(0.01, 0.08, default=0.035, space="buy", optimize=True)
    volume_rel_min = DecimalParameter(0.5, 1.5, default=0.8, space="buy", optimize=True)
    vol_spike_z_max = DecimalParameter(1.0, 3.0, default=2.0, space="buy", optimize=True)

    trend_weight_momentum = DecimalParameter(0.05, 0.4, default=0.18, space="buy", optimize=True)
    trend_weight_trend = DecimalParameter(0.1, 0.5, default=0.32, space="buy", optimize=True)
    trend_weight_volume = DecimalParameter(0.05, 0.3, default=0.14, space="buy", optimize=True)
    range_weight_reversion = DecimalParameter(0.1, 0.6, default=0.32, space="buy", optimize=True)
    range_weight_mfi = DecimalParameter(0.05, 0.4, default=0.18, space="buy", optimize=True)
    range_weight_volatility = DecimalParameter(0.05, 0.4, default=0.20, space="buy", optimize=True)
    volatile_weight_breakout = DecimalParameter(0.1, 0.6, default=0.35, space="buy", optimize=True)
    volatile_weight_volume = DecimalParameter(0.05, 0.4, default=0.22, space="buy", optimize=True)
    volatile_weight_trend = DecimalParameter(0.05, 0.4, default=0.18, space="buy", optimize=True)
    bear_weight_rebound = DecimalParameter(0.1, 0.6, default=0.30, space="buy", optimize=True)
    bear_weight_exhaust = DecimalParameter(0.05, 0.4, default=0.20, space="buy", optimize=True)
    bear_weight_risk = DecimalParameter(0.05, 0.4, default=0.18, space="buy", optimize=True)

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "orange"},
            "bb_mid": {"color": "green"},
        },
        "subplots": {
            "Momentum": {
                "rsi": {"color": "purple"},
                "mfi": {"color": "brown"},
            },
            "Scoring": {
                "entry_score": {"color": "green"},
                "exit_score": {"color": "red"},
            },
        },
    }

    @staticmethod
    def optimizer_loop(
        param_space: Dict[str, Iterable[Any]],
        evaluate: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
        max_workers: int = 8,
    ) -> Dict[str, Any]:
        """
        Lance une boucle d'optimisation parallèle (non bloquante) pour évaluer
        un maximum de combinaisons de paramètres. Retourne la meilleure config.
        """
        keys = list(param_space.keys())
        combos = (dict(zip(keys, values)) for values in product(*param_space.values()))

        best_score = float("-inf")
        best_params: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate, params): params for params in combos}
            for future in as_completed(futures):
                score, params = future.result()
                if score > best_score:
                    best_score = score
                    best_params = params

        return {"score": best_score, "params": best_params}

    @staticmethod
    def _normalize_series(series: pd.Series, window: int = 96) -> pd.Series:
        min_v = series.rolling(window=window, min_periods=window // 2).min()
        max_v = series.rolling(window=window, min_periods=window // 2).max()
        normalized = (series - min_v) / (max_v - min_v + 1e-9)
        return normalized.clip(0, 1)

    @staticmethod
    def _clip_threshold(series: pd.Series, lower: float, upper: float) -> pd.Series:
        return series.fillna(lower).clip(lower=lower, upper=upper)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # Trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["sma_long"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema_slope"] = ta.LINEARREG_SLOPE(dataframe["ema_fast"], timeperiod=14)

        # Momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        dataframe["lr_slope"] = ta.LINEARREG_SLOPE(dataframe["close"], timeperiod=14)
        dataframe["lr_forecast"] = dataframe["close"] + dataframe["lr_slope"]
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=9)

        # Volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        atr_pct_mean = dataframe["atr_pct"].rolling(48, min_periods=24).mean()
        atr_pct_std = dataframe["atr_pct"].rolling(48, min_periods=24).std().replace(0, np.nan)
        dataframe["atr_pct_z"] = (dataframe["atr_pct"] - atr_pct_mean) / atr_pct_std

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"].replace(0, np.nan)

        # Volume
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(30).mean()
        dataframe["volume_rel"] = dataframe["volume"] / dataframe["volume_mean_30"].replace(0, np.nan)
        dataframe["volatility_spike"] = dataframe["atr_pct_z"] > self.vol_spike_z_max.value

        # Returns features
        dataframe["ret_1"] = dataframe["close"].pct_change(1)
        dataframe["ret_6"] = dataframe["close"].pct_change(6)
        dataframe["ret_24"] = dataframe["close"].pct_change(24)
        dataframe["rolling_volatility"] = dataframe["ret_1"].rolling(48).std()

        # Trend persistence / regime features
        dataframe["trend_persistence"] = (
            (dataframe["close"] > dataframe["ema_fast"]).rolling(24).mean().fillna(0)
        )
        dataframe["downside_risk"] = dataframe["ret_1"].clip(upper=0).rolling(24).std()
        bb_width_quantile = dataframe["bb_width"].rolling(96, min_periods=48).quantile(0.7)
        atr_pct_quantile = dataframe["atr_pct"].rolling(96, min_periods=48).quantile(0.7)

        trend_regime = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slope"] > self.trend_ema_slope_min.value)
            & (dataframe["lr_slope"] > self.trend_lr_slope_min.value)
        )
        range_regime = (
            (dataframe["bb_width"] < self.range_bb_width_max.value)
            & (dataframe["rsi"].between(50 - self.range_rsi_band.value, 50 + self.range_rsi_band.value))
            & (dataframe["cci"].abs() < self.range_cci_abs_max.value)
        )
        dataframe["range_exclusion"] = (
            (dataframe["bb_width"] > self.range_bb_width_max.value * 1.35)
            | (dataframe["rsi"].sub(50).abs() > self.range_rsi_band.value * 1.35)
        )
        volatile_regime = (
            (dataframe["atr_pct"] > self.volatile_atr_pct_min.value)
            & (dataframe["volume_rel"] > self.volatile_vol_rel_min.value)
            & (dataframe["rolling_volatility"] > self.volatile_roll_vol_min.value)
        )
        bear_regime = (
            (dataframe["macdhist"] < self.bear_macd_hist_max.value)
            & (dataframe["rsi"] < self.bear_rsi_max.value)
            & (dataframe["close"] < dataframe["sma_long"])
        )
        bull_regime = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["adx"] > 25)
            & (dataframe["rsi"] > 60)
            & (dataframe["macdhist"] > 0)
        )
        recovery_regime = (
            (dataframe["ema_fast"] > dataframe["ema_mid"])
            & (dataframe["close"] > dataframe["ema_mid"])
            & (dataframe["rsi"].between(45, 62))
            & (dataframe["macd"] > dataframe["macdsignal"])
        )
        distribution_regime = (
            (dataframe["close"] > dataframe["sma_long"])
            & (dataframe["ema_fast"] < dataframe["ema_mid"])
            & (dataframe["rsi"].between(55, 70))
            & (dataframe["macdhist"] < 0)
        )
        capitulation_regime = (
            (dataframe["rsi"] < 25)
            & (dataframe["ret_6"] < -0.06)
            & (dataframe["atr_pct"] > atr_pct_quantile)
            & (dataframe["volume_rel"] > self.volatile_vol_rel_min.value)
            & (dataframe["close"] < dataframe["bb_lower"])
        )

        dataframe["market_regime"] = np.select(
            [
                capitulation_regime,
                bear_regime,
                distribution_regime,
                volatile_regime,
                bull_regime,
                trend_regime,
                recovery_regime,
                range_regime,
            ],
            [7, 3, 6, 2, 4, 1, 5, 0],
            default=0,
        )
        dataframe["market_regime_label"] = np.select(
            [
                dataframe["market_regime"] == 7,
                dataframe["market_regime"] == 6,
                dataframe["market_regime"] == 5,
                dataframe["market_regime"] == 4,
                dataframe["market_regime"] == 3,
                dataframe["market_regime"] == 2,
                dataframe["market_regime"] == 1,
            ],
            ["capitulation", "distribution", "recovery", "bull", "bear", "volatile", "trend"],
            default="range",
        )

        # AI-like ranking score (multi-factor weighted)
        trend_strength = self._normalize_series((dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["close"])
        trend_quality = self._normalize_series(dataframe["adx"])
        momentum_quality = self._normalize_series((dataframe["rsi"] - 50).abs())
        macd_quality = self._normalize_series(dataframe["macd"] - dataframe["macdsignal"])
        volume_quality = self._normalize_series(dataframe["volume_rel"])
        volatility_quality = 1 - self._normalize_series(dataframe["atr_pct"])  # lower ATR% = cleaner trend
        persistence_quality = self._normalize_series(dataframe["trend_persistence"])
        downside_penalty = self._normalize_series(dataframe["downside_risk"])

        dataframe["entry_score"] = (
            0.24 * trend_strength
            + 0.16 * trend_quality
            + 0.20 * macd_quality
            + 0.14 * volume_quality
            + 0.10 * volatility_quality
            + 0.08 * momentum_quality
            + 0.12 * persistence_quality
            - 0.04 * downside_penalty
        )
        dataframe["entry_score"] = dataframe["entry_score"].clip(lower=0, upper=1)
        dataframe["trend_entry_score"] = (
            self.trend_weight_trend.value * self._normalize_series(dataframe["ema_slope"])
            + self.trend_weight_momentum.value * self._normalize_series(dataframe["roc"])
            + self.trend_weight_volume.value * self._normalize_series(dataframe["volume_rel"])
        ).clip(0, 1)
        dataframe["range_entry_score"] = (
            self.range_weight_reversion.value * (1 - self._normalize_series(dataframe["rsi"].sub(50).abs()))
            + self.range_weight_mfi.value * (1 - self._normalize_series(dataframe["mfi"]))
            + self.range_weight_volatility.value * (1 - self._normalize_series(dataframe["bb_width"]))
        ).clip(0, 1)
        dataframe["volatile_entry_score"] = (
            self.volatile_weight_breakout.value * self._normalize_series(dataframe["atr_pct"])
            + self.volatile_weight_volume.value * self._normalize_series(dataframe["volume_rel"])
            + self.volatile_weight_trend.value * self._normalize_series(dataframe["lr_slope"])
        ).clip(0, 1)
        dataframe["bear_entry_score"] = (
            self.bear_weight_rebound.value * (1 - self._normalize_series(dataframe["rsi"]))
            + self.bear_weight_exhaust.value * (1 - self._normalize_series(dataframe["macdhist"].abs()))
            + self.bear_weight_risk.value * (1 - self._normalize_series(dataframe["downside_risk"]))
        ).clip(0, 1)
        dataframe["bull_entry_score"] = (
            0.40 * trend_strength
            + 0.30 * self._normalize_series(dataframe["roc"])
            + 0.30 * volume_quality
        ).clip(0, 1)
        dataframe["recovery_entry_score"] = (
            0.35 * self._normalize_series(dataframe["rsi"])
            + 0.30 * self._normalize_series(dataframe["macd"] - dataframe["macdsignal"])
            + 0.20 * (1 - downside_penalty)
            + 0.15 * persistence_quality
        ).clip(0, 1)
        dataframe["distribution_entry_score"] = (
            0.45 * (1 - self._normalize_series(dataframe["macdhist"]))
            + 0.35 * (1 - self._normalize_series(dataframe["rsi"]))
            + 0.20 * volatility_quality
        ).clip(0, 1)
        dataframe["capitulation_entry_score"] = (
            0.45 * (1 - self._normalize_series(dataframe["rsi"]))
            + 0.35 * self._normalize_series(dataframe["volume_rel"])
            + 0.20 * (1 - self._normalize_series(dataframe["atr_pct"]))
        ).clip(0, 1)

        dataframe["entry_threshold"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.70),
            lower=0.56,
            upper=0.76,
        )
        dataframe["entry_threshold_trend"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.72),
            lower=0.58,
            upper=0.78,
        )
        dataframe["entry_threshold_range"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.62),
            lower=0.52,
            upper=0.70,
        )
        dataframe["entry_threshold_volatile"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.76),
            lower=0.60,
            upper=0.82,
        )
        dataframe["entry_threshold_bear"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.78),
            lower=0.60,
            upper=0.84,
        )
        dataframe["entry_threshold_bull"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.74),
            lower=0.60,
            upper=0.80,
        )
        dataframe["entry_threshold_recovery"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.68),
            lower=0.56,
            upper=0.76,
        )
        dataframe["entry_threshold_distribution"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.66),
            lower=0.54,
            upper=0.72,
        )
        dataframe["entry_threshold_capitulation"] = self._clip_threshold(
            dataframe["entry_score"].rolling(288, min_periods=96).quantile(0.80),
            lower=0.60,
            upper=0.86,
        )
        dataframe["entry_edge"] = dataframe["entry_score"] - dataframe["entry_threshold"]
        dataframe["entry_edge_trend"] = dataframe["entry_score"] - dataframe["entry_threshold_trend"]
        dataframe["entry_edge_range"] = dataframe["entry_score"] - dataframe["entry_threshold_range"]
        dataframe["entry_edge_volatile"] = dataframe["entry_score"] - dataframe["entry_threshold_volatile"]
        dataframe["entry_edge_bear"] = dataframe["entry_score"] - dataframe["entry_threshold_bear"]
        dataframe["entry_edge_bull"] = dataframe["entry_score"] - dataframe["entry_threshold_bull"]
        dataframe["entry_edge_recovery"] = dataframe["entry_score"] - dataframe["entry_threshold_recovery"]
        dataframe["entry_edge_distribution"] = dataframe["entry_score"] - dataframe["entry_threshold_distribution"]
        dataframe["entry_edge_capitulation"] = dataframe["entry_score"] - dataframe["entry_threshold_capitulation"]

        # Exit score focuses on momentum decay + volatility expansion
        bearish_macd = self._normalize_series(dataframe["macdsignal"] - dataframe["macd"])
        weakening_trend = self._normalize_series((dataframe["ema_mid"] - dataframe["ema_fast"]) / dataframe["close"])
        overbought = self._normalize_series(dataframe["rsi"] - 50)
        vol_spike = self._normalize_series(dataframe["atr_pct"])
        regime_break = self._normalize_series((dataframe["ema_slow"] - dataframe["ema_fast"]) / dataframe["close"])

        dataframe["exit_score"] = (
            0.30 * bearish_macd
            + 0.22 * weakening_trend
            + 0.20 * overbought
            + 0.16 * vol_spike
            + 0.12 * regime_break
        )
        dataframe["exit_threshold"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.65),
            lower=0.52,
            upper=0.74,
        )
        dataframe["exit_threshold_trend"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.62),
            lower=0.50,
            upper=0.72,
        )
        dataframe["exit_threshold_range"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.58),
            lower=0.48,
            upper=0.68,
        )
        dataframe["exit_threshold_volatile"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.70),
            lower=0.54,
            upper=0.78,
        )
        dataframe["exit_threshold_bear"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.72),
            lower=0.56,
            upper=0.80,
        )
        dataframe["exit_threshold_bull"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.64),
            lower=0.52,
            upper=0.74,
        )
        dataframe["exit_threshold_recovery"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.66),
            lower=0.52,
            upper=0.76,
        )
        dataframe["exit_threshold_distribution"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.70),
            lower=0.54,
            upper=0.78,
        )
        dataframe["exit_threshold_capitulation"] = self._clip_threshold(
            dataframe["exit_score"].rolling(288, min_periods=96).quantile(0.76),
            lower=0.56,
            upper=0.82,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        base_conditions: List[pd.Series] = [
            dataframe["volume"] > 0,
            dataframe["entry_score"] > dataframe["entry_threshold"],
            dataframe["volume_rel"] > self.volume_rel_min.value,
        ]

        trend_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 1,
            dataframe["ema_fast"] > dataframe["ema_mid"],
            dataframe["ema_mid"] > dataframe["ema_slow"],
            dataframe["close"] > dataframe["ema_fast"],
            dataframe["close"] > dataframe["bb_mid"],
            dataframe["rsi"].between(52, 75),
            dataframe["mfi"].between(50, 85),
            dataframe["trend_entry_score"] > dataframe["entry_threshold_trend"],
            dataframe["trend_persistence"] > 0.60,
            dataframe["lr_slope"] > 0,
            dataframe["entry_edge_trend"] > self.min_entry_edge.value,
            ~dataframe["volatility_spike"],
        ]

        range_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 0,
            dataframe["close"] < dataframe["bb_mid"],
            dataframe["close"] > dataframe["bb_lower"] * 0.995,
            dataframe["rsi"].between(35, 58),
            dataframe["mfi"].between(25, 55),
            dataframe["range_entry_score"] > dataframe["entry_threshold_range"],
            dataframe["ret_1"] > -0.02,
            ~dataframe["range_exclusion"],
            dataframe["entry_edge_range"] > self.min_entry_edge.value,
            ~dataframe["volatility_spike"],
        ]

        volatile_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 2,
            dataframe["close"] > dataframe["bb_upper"],
            dataframe["volume_rel"] > 1.2,
            dataframe["lr_slope"] > 0,
            dataframe["ret_1"] > 0,
            dataframe["volatile_entry_score"] > dataframe["entry_threshold_volatile"],
            dataframe["entry_edge_volatile"] > self.min_entry_edge.value,
        ]

        bear_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 3,
            dataframe["close"] < dataframe["sma_long"],
            dataframe["rsi"] < self.bear_rsi_max.value,
            dataframe["bear_entry_score"] > dataframe["entry_threshold_bear"],
            dataframe["ret_1"] > -0.05,
            dataframe["entry_edge_bear"] > self.min_entry_edge.value,
        ]
        bull_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 4,
            dataframe["close"] > dataframe["ema_fast"],
            dataframe["rsi"].between(60, 80),
            dataframe["bull_entry_score"] > dataframe["entry_threshold_bull"],
            dataframe["ret_6"] > 0.02,
            dataframe["entry_edge_bull"] > self.min_entry_edge.value,
            ~dataframe["volatility_spike"],
        ]
        recovery_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 5,
            dataframe["close"] > dataframe["ema_mid"],
            dataframe["rsi"].between(45, 65),
            dataframe["recovery_entry_score"] > dataframe["entry_threshold_recovery"],
            dataframe["ret_1"] > -0.01,
            dataframe["entry_edge_recovery"] > self.min_entry_edge.value,
            ~dataframe["volatility_spike"],
        ]
        distribution_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 6,
            dataframe["close"] > dataframe["bb_lower"] * 0.99,
            dataframe["rsi"].between(40, 60),
            dataframe["distribution_entry_score"] > dataframe["entry_threshold_distribution"],
            dataframe["ret_1"] > -0.02,
            dataframe["entry_edge_distribution"] > self.min_entry_edge.value,
            ~dataframe["volatility_spike"],
        ]
        capitulation_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 7,
            dataframe["close"] > dataframe["bb_lower"],
            dataframe["rsi"] < 35,
            dataframe["capitulation_entry_score"] > dataframe["entry_threshold_capitulation"],
            dataframe["ret_1"] > 0,
            dataframe["entry_edge_capitulation"] > self.min_entry_edge.value,
        ]

        trend_mask = reduce(lambda x, y: x & y, trend_conditions)
        range_mask = reduce(lambda x, y: x & y, range_conditions)
        volatile_mask = reduce(lambda x, y: x & y, volatile_conditions)
        bear_mask = reduce(lambda x, y: x & y, bear_conditions)
        bull_mask = reduce(lambda x, y: x & y, bull_conditions)
        recovery_mask = reduce(lambda x, y: x & y, recovery_conditions)
        distribution_mask = reduce(lambda x, y: x & y, distribution_conditions)
        capitulation_mask = reduce(lambda x, y: x & y, capitulation_conditions)

        dataframe.loc[trend_mask, ["enter_long", "enter_tag"]] = (1, "ai_trend_follow")
        dataframe.loc[range_mask, ["enter_long", "enter_tag"]] = (1, "ai_range_reversion")
        dataframe.loc[volatile_mask, ["enter_long", "enter_tag"]] = (1, "ai_vol_breakout")
        dataframe.loc[bear_mask, ["enter_long", "enter_tag"]] = (1, "ai_bear_rebound")
        dataframe.loc[bull_mask, ["enter_long", "enter_tag"]] = (1, "ai_bull_momentum")
        dataframe.loc[recovery_mask, ["enter_long", "enter_tag"]] = (1, "ai_recovery")
        dataframe.loc[distribution_mask, ["enter_long", "enter_tag"]] = (1, "ai_distribution_rebound")
        dataframe.loc[capitulation_mask, ["enter_long", "enter_tag"]] = (1, "ai_capitulation_snap")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        base_conditions: List[pd.Series] = [
            dataframe["volume"] > 0,
        ]

        trend_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 1,
            (
                (dataframe["close"] < dataframe["ema_mid"])
                | (dataframe["rsi"] > 78)
                | (dataframe["exit_score"] > dataframe["exit_threshold_trend"])
            ),
        ]

        range_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 0,
            (
                (dataframe["close"] > dataframe["bb_mid"])
                | (dataframe["rsi"] > 62)
                | (dataframe["exit_score"] > dataframe["exit_threshold_range"])
            ),
        ]

        volatile_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 2,
            (
                (dataframe["close"] < dataframe["bb_mid"])
                | (dataframe["ret_1"] < -0.03)
                | (dataframe["exit_score"] > dataframe["exit_threshold_volatile"])
            ),
        ]

        bear_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 3,
            (
                (dataframe["close"] > dataframe["ema_mid"])
                | (dataframe["rsi"] > 55)
                | (dataframe["exit_score"] > dataframe["exit_threshold_bear"])
            ),
        ]
        bull_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 4,
            (
                (dataframe["close"] < dataframe["ema_mid"])
                | (dataframe["rsi"] > 82)
                | (dataframe["exit_score"] > dataframe["exit_threshold_bull"])
            ),
        ]
        recovery_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 5,
            (
                (dataframe["close"] < dataframe["ema_fast"])
                | (dataframe["rsi"] > 70)
                | (dataframe["exit_score"] > dataframe["exit_threshold_recovery"])
            ),
        ]
        distribution_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 6,
            (
                (dataframe["close"] < dataframe["bb_mid"])
                | (dataframe["rsi"] < 45)
                | (dataframe["exit_score"] > dataframe["exit_threshold_distribution"])
            ),
        ]
        capitulation_conditions: List[pd.Series] = base_conditions + [
            dataframe["market_regime"] == 7,
            (
                (dataframe["close"] < dataframe["bb_lower"])
                | (dataframe["rsi"] > 50)
                | (dataframe["exit_score"] > dataframe["exit_threshold_capitulation"])
            ),
        ]

        trend_mask = reduce(lambda x, y: x & y, trend_conditions)
        range_mask = reduce(lambda x, y: x & y, range_conditions)
        volatile_mask = reduce(lambda x, y: x & y, volatile_conditions)
        bear_mask = reduce(lambda x, y: x & y, bear_conditions)
        bull_mask = reduce(lambda x, y: x & y, bull_conditions)
        recovery_mask = reduce(lambda x, y: x & y, recovery_conditions)
        distribution_mask = reduce(lambda x, y: x & y, distribution_conditions)
        capitulation_mask = reduce(lambda x, y: x & y, capitulation_conditions)

        dataframe.loc[trend_mask, ["exit_long", "exit_tag"]] = (1, "ai_trend_exit")
        dataframe.loc[range_mask, ["exit_long", "exit_tag"]] = (1, "ai_range_exit")
        dataframe.loc[volatile_mask, ["exit_long", "exit_tag"]] = (1, "ai_vol_exit")
        dataframe.loc[bear_mask, ["exit_long", "exit_tag"]] = (1, "ai_bear_exit")
        dataframe.loc[bull_mask, ["exit_long", "exit_tag"]] = (1, "ai_bull_exit")
        dataframe.loc[recovery_mask, ["exit_long", "exit_tag"]] = (1, "ai_recovery_exit")
        dataframe.loc[distribution_mask, ["exit_long", "exit_tag"]] = (1, "ai_distribution_exit")
        dataframe.loc[capitulation_mask, ["exit_long", "exit_tag"]] = (1, "ai_capitulation_exit")

        return dataframe
