"""Named trading set-ups used in the simulations, so every number in the docs says which one produced it.

Falcon is the name of our trading model. Versions:
  Falcon v1 - the baseline. The first ranking policy that was run for a year: top 20 of the daily ranking are bought, held while inside the top 40, sold below that;
              2-ATR stop; equal slices; blend 70% 10-day question / 30% next-day question.
  Falcon v2 - improved portfolio management, settings chosen on 2024 only: 10 positions, hold while inside the top 30, 3-ATR stop.
  Falcon v3 - (planned) v2 plus better signals (retraining every 10 days, extra ranking models), added only if 2024 says it helps."""
from __future__ import annotations

from dataclasses import dataclass

from .simulate import Params


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    top_n: int
    drop_mult: int
    w10: float
    stop_atr: float
    trail_atr: float = 0.0
    sizing: str = "equal"

    @property
    def drop_rank(self) -> int:
        return self.top_n * self.drop_mult

    def params(self, capital: float = 100_000.0, cost: float = 0.001, regime=None, liquidate_on_risk_off: bool = False) -> Params:
        return Params(capital=capital, max_pos=self.top_n, cost=cost, stop_atr=self.stop_atr, trail_atr=self.trail_atr, sizing=self.sizing, regime=regime,
                      liquidate_on_risk_off=liquidate_on_risk_off)


FALCON_V1 = Strategy("Falcon v1", "baseline: top 20, hold inside the top 40, 2-ATR stop, equal slices, 70/30 blend", top_n=20, drop_mult=2, w10=0.7, stop_atr=2.0)
FALCON_V2 = Strategy("Falcon v2", "improved portfolio settings: top 10, hold inside the top 30, 3-ATR stop (settings chosen on 2024 only)", top_n=10, drop_mult=3, w10=0.7, stop_atr=3.0)
STRATEGIES = {s.name: s for s in (FALCON_V1, FALCON_V2)}
