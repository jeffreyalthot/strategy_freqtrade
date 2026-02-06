from __future__ import annotations

from functools import reduce
from typing import Dict, List

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy


class GlobalAIMomentumStrategy(IStrategy):
    """
    Strategy "IA" multi-facteurs pour Freqtrade.

    Objectif:
    - Combiner momentum, tendance, volatilité et volume.
    - Construire un score d'entrée/sortie inspiré d'un modèle de ranking.
    - Fournir une base solide pour optimisation hyperopt.

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
    def _normalize_series(series: pd.Series, window: int = 96) -> pd.Series:
        min_v = series.rolling(window=window, min_periods=window // 2).min()
        max_v = series.rolling(window=window, min_periods=window // 2).max()
        normalized = (series - min_v) / (max_v - min_v + 1e-9)
        return normalized.clip(0, 1)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        # Trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # Momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        # Volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"].replace(0, np.nan)

        # Volume
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(30).mean()
        dataframe["volume_rel"] = dataframe["volume"] / dataframe["volume_mean_30"].replace(0, np.nan)

        # Returns features
        dataframe["ret_1"] = dataframe["close"].pct_change(1)
        dataframe["ret_6"] = dataframe["close"].pct_change(6)
        dataframe["ret_24"] = dataframe["close"].pct_change(24)

        # AI-like ranking score (multi-factor weighted)
        trend_strength = self._normalize_series((dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["close"])
        trend_quality = self._normalize_series(dataframe["adx"])
        momentum_quality = self._normalize_series((dataframe["rsi"] - 50).abs())
        macd_quality = self._normalize_series(dataframe["macd"] - dataframe["macdsignal"])
        volume_quality = self._normalize_series(dataframe["volume_rel"])
        volatility_quality = 1 - self._normalize_series(dataframe["atr_pct"])  # lower ATR% = cleaner trend

        dataframe["entry_score"] = (
            0.26 * trend_strength
            + 0.18 * trend_quality
            + 0.20 * macd_quality
            + 0.16 * volume_quality
            + 0.10 * volatility_quality
            + 0.10 * momentum_quality
        )

        # Exit score focuses on momentum decay + volatility expansion
        bearish_macd = self._normalize_series(dataframe["macdsignal"] - dataframe["macd"])
        weakening_trend = self._normalize_series((dataframe["ema_mid"] - dataframe["ema_fast"]) / dataframe["close"])
        overbought = self._normalize_series(dataframe["rsi"] - 50)
        vol_spike = self._normalize_series(dataframe["atr_pct"])

        dataframe["exit_score"] = (
            0.35 * bearish_macd
            + 0.25 * weakening_trend
            + 0.20 * overbought
            + 0.20 * vol_spike
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        conditions: List[pd.Series] = [
            dataframe["volume"] > 0,
            dataframe["ema_fast"] > dataframe["ema_mid"],
            dataframe["ema_mid"] > dataframe["ema_slow"],
            dataframe["close"] > dataframe["bb_mid"],
            dataframe["rsi"].between(52, 72),
            dataframe["mfi"].between(45, 80),
            dataframe["entry_score"] > 0.62,
            dataframe["ret_6"] > -0.03,
        ]

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["enter_long", "enter_tag"]] = (1, "ai_momentum_long")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        conditions: List[pd.Series] = [
            dataframe["volume"] > 0,
            (
                (dataframe["close"] < dataframe["ema_fast"])
                | (dataframe["rsi"] > 78)
                | (dataframe["exit_score"] > 0.58)
                | ((dataframe["ret_1"] < -0.02) & (dataframe["atr_pct"] > dataframe["atr_pct"].rolling(24).mean()))
            ),
        ]

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["exit_long", "exit_tag"]] = (1, "ai_risk_off")

        return dataframe
