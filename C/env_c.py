"""方向 C 的交易環境：規格完整、有動作遮罩、有交易成本、可選獎勵。

和 qfin-main/exp2.Env 相同的部分：10 維狀態（同樣的特徵）、動作 0 持有／1 買 1 單位／2 全部賣出、只做多、
初始資金 10 單位（10 × 期初價格）。
不同的部分（針對論文二的痛點）：
  - 動作遮罩：沒錢不能買、沒部位不能賣。原本無效動作只罰 −0.1，代理人會用「無效的賣出」來空手，遮罩後這條路被關掉。
  - 交易成本：買進付 p×(1+c)、賣出收 p×(1−c)，c 為單邊成本比例（例如 0.0005 = 0.05%）；損益與持倉成本都含成本。
  - 獎勵 reward：
      "paper"  ：論文 Table III（−0.02 時間成本、順勢進場 +0.5／逆勢 −2、獲利出場 +10+50×pnl%、虧損出場 −2−10×|pnl%|、
                 持有虧損 −5×pnl%²、截斷 [−15, 30]），pnl 為扣成本後。
      "logret" ：100 × ln(V_{t+1} / V_t)，V 為資產市值（現金 + 持倉 × 收盤價）；成本直接反映在 V 上，沒有其他人為加減分。
"""
from __future__ import annotations

import math

import numpy as np


class EnvC:
    def __init__(self, df, probs, cost=0.0005, reward="logret", units0=10):
        self.p = df["Close"].to_numpy()
        self.ma5, self.ma20, self.ma60 = (df[f"MA{n}"].to_numpy() for n in (5, 20, 60))
        self.volrel = df["VOLREL"].to_numpy()
        self.pb, self.pr = probs[:, 1], probs[:, 0]
        self.c, self.reward, self.units0 = cost, reward, units0
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.units0 * self.p[0]
        self.init = self.cash
        self.units = 0
        self.avg = 0.0
        self.trades, self.buys, self.cost_paid = [], 0, 0.0
        self.values = [self.cash]
        return self.state()

    def value(self, p):
        return self.cash + self.units * p

    def state(self):
        t, p = self.t, self.p[self.t]
        v = self.value(p)
        hold = self.units > 0
        upnl = (p / self.avg - 1) * 100 if hold else 0.0
        apr = p / self.avg if hold else 1.0
        return np.array([[self.pb[t], self.pr[t], self.cash / v, self.units * p / v, upnl, apr,
                          (p / self.ma20[t] - 1) * 100, (p / self.ma60[t] - 1) * 100,
                          (self.ma5[t] / self.ma20[t] - 1) * 100, self.volrel[t]]])

    def mask(self):
        p = self.p[self.t]
        return np.array([True, self.cash >= p * (1 + self.c), self.units > 0])

    def step(self, a):
        t, p, c = self.t, self.p[self.t], self.c
        v0 = self.value(p)
        paper = self.reward == "paper"
        r = -0.02 if paper else 0.0
        if a == 1 and self.cash >= p * (1 + c):
            pay = p * (1 + c)
            if paper:
                r += 0.5 if p > self.ma20[t] else -2.0
            self.avg = (self.avg * self.units + pay) / (self.units + 1)
            self.units += 1
            self.cash -= pay
            self.cost_paid += p * c
            self.buys += 1
        elif a == 2 and self.units > 0:
            pnl = (p * (1 - c) / self.avg - 1) * 100
            self.trades.append(pnl)
            if paper:
                r += (10 + 50 * pnl) if pnl > 0 else (-2 - 10 * abs(pnl))
            self.cash += self.units * p * (1 - c)
            self.cost_paid += self.units * p * c
            self.units, self.avg = 0, 0.0
        self.t += 1
        p2 = self.p[self.t]
        v1 = self.value(p2)
        if paper:
            if self.units > 0:
                upnl = (p2 / self.avg - 1) * 100
                if upnl < 0:
                    r -= 5 * upnl ** 2
            r = min(max(r, -15.0), 30.0)
        else:
            r = 100 * math.log(v1 / v0)
        self.values.append(v1)
        return r, self.t >= len(self.p) - 1


def evaluate(policy, df, probs, cost):
    """以 policy(state, mask) → 動作 跑完整段資料（不額外平倉，期末以市值計算）。回傳績效 dict。"""
    env = EnvC(df, probs, cost, "logret")
    s, done, acts, held = env.reset(), False, [], []
    while not done:
        a = int(policy(s, env.mask()))
        acts.append(a)
        _, done = env.step(a)
        held.append(env.units > 0)
        if not done:
            s = env.state()
    v = np.array(env.values)
    tr = np.array(env.trades)
    dr = np.diff(v) / v[:-1]
    return {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "days": int(len(df)), "cost": cost,
            "total_return_pct": float(100 * (v[-1] / v[0] - 1)),
            "max_drawdown_pct": float(100 * np.max(1 - v / np.maximum.accumulate(v))),
            "sharpe_ann": float(np.sqrt(252) * dr.mean() / dr.std()) if dr.std() > 0 else 0.0,
            "trades": int(len(tr)), "buys": int(env.buys),
            "win_rate_pct": float(100 * np.mean(tr > 0)) if len(tr) else None,
            "avg_trade_pct": float(tr.mean()) if len(tr) else None,
            "time_in_market_pct": float(100 * np.mean(held)), "cost_paid_pct": float(100 * env.cost_paid / v[0]),
            "action_counts": np.bincount(acts, minlength=3).tolist(),
            "buy_hold_pct": float(100 * (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1))}


# ---------------------------------------------------------------- 規則型基準


def policy_buy_hold(s, m):
    return 1 if m[1] else 0


def make_policy_ma(df):
    """均線策略：MA5 > MA20 時逐日加碼（每天買 1 單位直到資金用完），MA5 < MA20 時全部賣出。"""
    up = (df["MA5"] > df["MA20"]).to_numpy()
    state = {"t": 0}

    def pol(s, m):
        t = state["t"]; state["t"] += 1
        if up[t]:
            return 1 if m[1] else 0
        return 2 if m[2] else 0
    return pol


def make_policy_random(seed):
    rng = np.random.default_rng(seed)

    def pol(s, m):
        return int(rng.choice(np.flatnonzero(m)))
    return pol
