from __future__ import annotations

from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair


class ClientFlowAIMarketStrategy(IStrategy):
    """
    Stratégie orientée "profilage" des flux acheteurs/vendeurs.

    ⚠️ Limite importante : Freqtrade ne fournit pas l'identité des clients
    ni les carnets d'ordre détaillés. Cette stratégie simule donc un
    "profil de client" à partir des micro-indices de flux (pression d'achat
    et de vente) dérivés des chandelles, du volume et de la volatilité.

    Objectif :
    - Surveiller plusieurs timeframes (1m -> 3d) pour détecter la cohérence
      des flux.
    - Construire un score analytique (type IA légère) pour estimer les zones
      de valeur basse (buy) et haute (sell).
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 300

    minimal_roi = {
        "0": 0.06,
        "60": 0.03,
        "180": 0.015,
        "360": 0,
    }

    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    plot_config = {
        "main_plot": {
            "value_score": {"color": "green"},
            "flow_coherence": {"color": "orange"},
        },
        "subplots": {
            "AI Flow": {
                "ai_entry_score": {"color": "blue"},
                "ai_exit_score": {"color": "red"},
            }
        },
    }

    def informative_pairs(self) -> List[Tuple[str, str]]:
        pairs = self.dp.current_whitelist() if self.dp else []
        timeframes = ["1m", "5m", "10m", "15m", "30m", "1h", "1d", "3d"]
        return [(pair, tf) for pair in pairs for tf in timeframes if tf != self.timeframe]

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window=window, min_periods=max(2, window // 2)).mean()
        std = series.rolling(window=window, min_periods=max(2, window // 2)).std().replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _tanh_clip(series: pd.Series, limit: float = 3.0) -> pd.Series:
        return np.tanh(series.clip(-limit, limit))

    def _flow_profile(self, dataframe: pd.DataFrame, suffix: str) -> pd.DataFrame:
        price_delta = dataframe[f"close{suffix}"] - dataframe[f"open{suffix}"]
        spread = (dataframe[f"high{suffix}"] - dataframe[f"low{suffix}"]).replace(0, np.nan)
        pressure = price_delta / spread
        volume_norm = self._zscore(dataframe[f"volume{suffix}"], 48).fillna(0)
        volatility = ta.ATR(
            {
                "high": dataframe[f"high{suffix}"],
                "low": dataframe[f"low{suffix}"],
                "close": dataframe[f"close{suffix}"],
            },
            timeperiod=14,
        ) / dataframe[f"close{suffix}"].replace(0, np.nan)
        volatility = volatility.fillna(0)

        dataframe[f"flow_pressure{suffix}"] = self._tanh_clip(pressure.fillna(0))
        dataframe[f"flow_volume{suffix}"] = self._tanh_clip(volume_norm)
        dataframe[f"flow_volatility{suffix}"] = self._tanh_clip(self._zscore(volatility, 48).fillna(0))
        dataframe[f"flow_coherence{suffix}"] = (
            0.5 * dataframe[f"flow_pressure{suffix}"]
            + 0.3 * dataframe[f"flow_volume{suffix}"]
            - 0.2 * dataframe[f"flow_volatility{suffix}"]
        )
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe = self._flow_profile(dataframe, "")

        if self.dp:
            for tf in ["1m", "10m", "15m", "30m", "1h", "1d", "3d"]:
                info = self.dp.get_pair_dataframe(metadata["pair"], tf)
                info = self._flow_profile(info, "")
                dataframe = merge_informative_pair(dataframe, info, self.timeframe, tf, ffill=True)

        dataframe["value_score"] = self._tanh_clip(
            self._zscore(dataframe["close"], 96)
            + self._zscore(dataframe["flow_coherence"], 96).fillna(0)
        )

        flow_columns = [
            "flow_coherence_1m",
            "flow_coherence_10m",
            "flow_coherence_15m",
            "flow_coherence_30m",
            "flow_coherence_1h",
            "flow_coherence_1d",
            "flow_coherence_3d",
        ]
        flow_available = [col for col in flow_columns if col in dataframe]
        if flow_available:
            dataframe["flow_coherence"] = reduce(
                lambda left, right: left + right,
                [dataframe[col].fillna(0) for col in flow_available],
            ) / len(flow_available)

        dataframe["ai_entry_score"] = self._tanh_clip(
            -dataframe["value_score"] + dataframe["flow_coherence"].fillna(0)
        )
        dataframe["ai_exit_score"] = self._tanh_clip(
            dataframe["value_score"] - dataframe["flow_coherence"].fillna(0)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["ai_entry_score"] > 0.4)
                & (dataframe["flow_coherence"] > 0.05)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["ai_exit_score"] > 0.35)
                | (dataframe["flow_coherence"] < -0.1)
            ),
            "exit_long",
        ] = 1

        return dataframe
