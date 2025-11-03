# pd_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import random
import statistics
from collections import defaultdict
# ========== Core primitives ==========

class Action(str, Enum):
    C = "C"
    D = "D"

@dataclass
class Payoffs:
    T: int = 5  # Temptation: D vs C
    R: int = 3  # Reward:     C vs C
    P: int = 1  # Punishment: D vs D
    S: int = 0  # Sucker:     C vs D
    def payoff(self, a_self: Action, a_opp: Action) -> int:
        if a_self == Action.C and a_opp == Action.C: return self.R
        if a_self == Action.C and a_opp == Action.D: return self.S
        if a_self == Action.D and a_opp == Action.C: return self.T
        return self.P

@dataclass
class DyadHistory:
    my_actions:  List[Action] = field(default_factory=list)
    opp_actions: List[Action] = field(default_factory=list)
    my_payoffs:  List[int]    = field(default_factory=list)

class Agent:
    def __init__(self, name: str):
        self.name   = name
        self.score  = 0
        self.memory: Dict[str, DyadHistory] = {}
    def _ensure(self, opp: "Agent"):
        if opp.name not in self.memory:
            self.memory[opp.name] = DyadHistory()
    def last_with(self, opp: "Agent") -> Tuple[Optional[Action], Optional[Action], Optional[int]]:
        if opp.name not in self.memory or not self.memory[opp.name].my_actions:
            return None, None, None
        h = self.memory[opp.name]
        return h.my_actions[-1], h.opp_actions[-1], h.my_payoffs[-1]
    def decide(self, opp: "Agent") -> Action:
        raise NotImplementedError
    def observe(self, opp: "Agent", my_action: Action, opp_action: Action, my_payoff: int):
        self._ensure(opp)
        h = self.memory[opp.name]
        h.my_actions.append(my_action)
        h.opp_actions.append(opp_action)
        h.my_payoffs.append(my_payoff)
        self.score += my_payoff

# ========== 19 固定/概率/互惠策略（与你之前一致；删除了 BAY） ==========

class AlwaysCooperate(Agent):
    def decide(self, opp: "Agent") -> Action: return Action.C

class AlwaysDefect(Agent):
    def decide(self, opp: "Agent") -> Action: return Action.D

class TitForTat(Agent):
    def decide(self, opp: "Agent") -> Action:
        _, opp_last, _ = self.last_with(opp)
        return Action.C if opp_last is None else opp_last

class WinStayLoseShift(Agent):
    def decide(self, opp: "Agent") -> Action:
        my_last, opp_last, _ = self.last_with(opp)
        if my_last is None: return Action.C
        good = (my_last == Action.C and opp_last == Action.C) or (my_last == Action.D and opp_last == Action.C)
        return my_last if good else (Action.C if my_last == Action.D else Action.D)

class GrimTrigger(Agent):
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        if Action.D in self.memory[opp.name].opp_actions:
            return Action.D
        return Action.C

class TitForTwoTats(Agent):
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        h = self.memory[opp.name]
        if len(h.opp_actions) < 2: return Action.C
        return Action.D if (h.opp_actions[-1] == Action.D and h.opp_actions[-2] == Action.D) else Action.C

class SuspiciousTitForTat(Agent):
    def decide(self, opp: "Agent") -> Action:
        _, opp_last, _ = self.last_with(opp)
        return Action.D if opp_last is None else opp_last

class GenerousTitForTat(Agent):
    def __init__(self, name: str, generosity: float = 0.1):
        super().__init__(name); self.generosity = max(0.0, min(1.0, generosity))
    def decide(self, opp: "Agent") -> Action:
        _, opp_last, _ = self.last_with(opp)
        if opp_last is None or opp_last == Action.C: return Action.C
        return Action.C if random.random() < self.generosity else Action.D

class SoftGrudger(Agent):
    def __init__(self, name: str, punish_rounds: int = 2):
        super().__init__(name); self.punish_rounds = max(1, punish_rounds); self._punish_left: Dict[str, int] = {}
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        if self._punish_left.get(opp.name, 0) > 0:
            self._punish_left[opp.name] -= 1; return Action.D
        _, opp_last, _ = self.last_with(opp)
        if opp_last == Action.D:
            self._punish_left[opp.name] = self.punish_rounds - 1
            return Action.D
        return Action.C

class Alternator(Agent):
    def __init__(self, name: str, start_with_C: bool = True):
        super().__init__(name); self.start_with_C = bool(start_with_C); self._count: Dict[str, int] = {}
    def decide(self, opp: "Agent") -> Action:
        n = self._count.get(opp.name, 0); self._count[opp.name] = n + 1
        want_C = (n % 2 == 0) if self.start_with_C else (n % 2 == 1)
        return Action.C if want_C else Action.D

class RandomCooperator(Agent):
    def __init__(self, name: str, coop_prob: float = 0.5):
        super().__init__(name); self.p = max(0.0, min(1.0, coop_prob))
    def decide(self, opp: "Agent") -> Action:
        return Action.C if random.random() < self.p else Action.D

class Prober(Agent):
    def __init__(self, name: str):
        super().__init__(name); self._mode: Dict[str, str] = {}
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        h = self.memory[opp.name]; mode = self._mode.get(opp.name, "probe")
        if mode == "probe":
            t = len(h.my_actions)
            if t == 0: return Action.D
            if t in (1, 2): return Action.C
            retaliated = Action.D in h.opp_actions[1:3]
            self._mode[opp.name] = "tft" if retaliated else "exploit"
            return self.decide(opp)
        if mode == "tft":
            _, opp_last, _ = self.last_with(opp)
            return Action.C if opp_last is None else opp_last
        return Action.D

class StochasticWSLS(Agent):
    def __init__(self, name: str, epsilon: float = 0.2):
        super().__init__(name); self.eps = max(0.0, min(1.0, epsilon))
    def decide(self, opp: "Agent") -> Action:
        my_last, opp_last, _ = self.last_with(opp)
        if my_last is None: return Action.C
        won = (my_last == Action.C and opp_last == Action.C) or (my_last == Action.D and opp_last == Action.C)
        if won: return my_last
        return (Action.C if my_last == Action.D else Action.D) if random.random() < self.eps else my_last

class Joss(Agent):
    def __init__(self, name: str, p_defect_after_CC: float = 0.1):
        super().__init__(name); self.p = max(0.0, min(1.0, p_defect_after_CC))
    def decide(self, opp: "Agent") -> Action:
        _, opp_last, _ = self.last_with(opp)
        if opp_last is None: return Action.C
        if opp_last == Action.C: return Action.D if random.random() < self.p else Action.C
        return Action.D

class ContriteTitForTat(Agent):
    def __init__(self, name: str):
        super().__init__(name); self.contrite: Dict[str, bool] = {}
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        self.contrite.setdefault(opp.name, False)
        my_last, opp_last, _ = self.last_with(opp)
        if opp_last is None: return Action.C
        if my_last == Action.D and opp_last == Action.C: self.contrite[opp.name] = True
        if self.contrite[opp.name]:
            self.contrite[opp.name] = False; return Action.C
        return opp_last

class Gradual(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.punish_count: Dict[str, int] = {}
        self.remaining:    Dict[str, int] = {}
        self.repairing:    Dict[str, int] = {}
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        self.punish_count.setdefault(opp.name, 0)
        self.remaining.setdefault(opp.name, 0)
        self.repairing.setdefault(opp.name, 0)
        if self.remaining[opp.name] > 0:
            self.remaining[opp.name] -= 1
            if self.remaining[opp.name] == 0: self.repairing[opp.name] = 2
            return Action.D
        if self.repairing[opp.name] > 0:
            self.repairing[opp.name] -= 1; return Action.C
        _, opp_last, _ = self.last_with(opp)
        if opp_last == Action.D:
            self.punish_count[opp.name] += 1
            self.remaining[opp.name] = self.punish_count[opp.name]
            return Action.D
        return Action.C

class Tester(Agent):
    def __init__(self, name: str):
        super().__init__(name); self.mode: Dict[str, str] = {}
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp)
        mode = self.mode.get(opp.name, "probe"); h = self.memory[opp.name]
        if mode == "probe":
            if len(h.my_actions) == 0: return Action.D
            opp_last = h.opp_actions[-1]
            self.mode[opp.name] = "tft" if opp_last == Action.D else "exploit"
            return self.decide(opp)
        if mode == "tft":
            _, opp_last, _ = self.last_with(opp)
            return Action.C if opp_last is None else opp_last
        return Action.D

class Majority(Agent):
    def decide(self, opp: "Agent") -> Action:
        self._ensure(opp); h = self.memory[opp.name]
        if not h.opp_actions: return Action.C
        return Action.C if h.opp_actions.count(Action.C)/len(h.opp_actions) >= 0.5 else Action.D

class MemoryOne(Agent):
    def __init__(self, name: str, pCC: float, pCD: float, pDC: float, pDD: float):
        super().__init__(name)
        clip = lambda x: max(0.0, min(1.0, float(x)))
        self.p = {
            (Action.C, Action.C): clip(pCC),
            (Action.C, Action.D): clip(pCD),
            (Action.D, Action.C): clip(pDC),
            (Action.D, Action.D): clip(pDD),
        }
    def decide(self, opp: "Agent") -> Action:
        my_last, opp_last, _ = self.last_with(opp)
        if opp_last is None:
            return Action.C if random.random() < self.p[(Action.C, Action.C)] else Action.D
        probC = self.p[(my_last, opp_last)]
        return Action.C if random.random() < probC else Action.D

# ========== USER 策略（由外部设置 next_action） ==========

class UserAgent(Agent):
    """交互式用户策略：每轮从外部注入 next_action（C/D），若未设置则默认 C。"""
    def __init__(self, name: str = "USER"):
        super().__init__(name)
        self.next_action: Optional[Action] = None
    def decide(self, opp: "Agent") -> Action:
        return self.next_action if self.next_action is not None else Action.C

# ========== Engine（按步推进的模拟器） ==========

def _make_random_pairs(agents, rng):
    pool = list(agents)
    rng.shuffle(pool)  # 使用传入的 rng
    return [(pool[i], pool[i+1]) for i in range(0, len(pool) - 1, 2)]

def _play_pair_with_override(a1: Agent, a2: Agent, pay: Payoffs,
                             overrides: Optional[Dict[str, Action]] = None) -> Tuple[Action, Action, int, int]:
    if overrides and a1.name in overrides:
        m1 = overrides[a1.name]
    else:
        m1 = a1.decide(a2)
    if overrides and a2.name in overrides:
        m2 = overrides[a2.name]
    else:
        m2 = a2.decide(a1)
    p1 = pay.payoff(m1, m2); p2 = pay.payoff(m2, m1)
    a1.observe(a2, m1, m2, p1); a2.observe(a1, m2, m1, p2)
    return m1, m2, p1, p2

class Simulator:
    """支持 δ 的逐轮推进，支持对特定代理（USER）注入动作 overrides。"""
    def __init__(self, agents: List[Agent], pay: Payoffs, seed: int = 0, delta: float = 0.0):
        assert len(agents) % 2 == 0, "Agents count must be even."
        self.agents = agents
        self.pay = pay
        self.delta = max(0.0, min(1.0, float(delta)))
        self.rng = random.Random(seed)
        self.round = 0
        self.last_pairs: List[Tuple[Agent, Agent]] = []
        self.coop_rates: List[float] = []
        self.per_agent_cumsum: Dict[str, int] = {a.name: 0 for a in agents}
        self.per_agent_cumavg: Dict[str, List[float]] = {a.name: [] for a in agents}
        self.per_round_avg_all: List[float] = []
        self.action_counts = defaultdict(lambda: {"C": 0, "D": 0})

    def reset(self):
        self.round = 0; self.last_pairs = []; self.action_counts.clear()
        self.coop_rates = []
        self.per_round_avg_all = []
        self.per_agent_cumsum = {a.name: 0 for a in self.agents}
        self.per_agent_cumavg = {a.name: [] for a in self.agents}
        for a in self.agents:
            a.score = 0; a.memory.clear()

    def step(self, overrides: Optional[Dict[str, Action]] = None) -> Dict[str, Any]:
        """推进一轮；如提供 overrides={'USER': Action.C} 则强制 USER 的动作。返回本轮信息。"""
        # 维持你原本“先自增”的设计（前端若想马上刷新轮次，请在点击后 st.rerun()）
        self.round += 1

        # 复用上轮配对（以 δ 概率）或随机重配
        if self.last_pairs and self.rng.random() < self.delta:
            pairs = self.last_pairs
        else:
            pairs = _make_random_pairs(self.agents, self.rng)

        coop_count = 0
        interactions = 0
        round_sum = 0
        results = []  # [(a1.name, act1, a2.name, act2, p1, p2)]

        for a, b in pairs:
            m1, m2, p1, p2 = _play_pair_with_override(a, b, self.pay, overrides)

            # —— 累计合作次数、交互计数、总收益 —— #
            coop_count += (1 if m1 == Action.C else 0) + (1 if m2 == Action.C else 0)
            interactions += 2
            round_sum += (p1 + p2)

            # —— 累计每个智能体的累计收益（你原来的逻辑） —— #
            self.per_agent_cumsum[a.name] += p1
            self.per_agent_cumsum[b.name] += p2

            # —— 新增：全局 C/D 计数（用于右侧“全局历史百分比”） —— #
            # m1/m2 可能是枚举，转为 'C'/'D' 键
            k1 = 'C' if m1 == Action.C else 'D'
            k2 = 'C' if m2 == Action.C else 'D'
            self.action_counts[a.name][k1] += 1
            self.action_counts[b.name][k2] += 1

            # —— 记录本轮对局详情，用于前端弹窗 —— #
            # 若你更喜欢把动作存成字符串，可改成 k1/k2
            results.append((a.name, m1, b.name, m2, p1, p2))

        # —— 汇总指标（你原来的逻辑） —— #
        rate = (coop_count / interactions) if interactions else 0.0
        self.coop_rates.append(rate)
        self.per_round_avg_all.append(round_sum / len(self.agents))

        # 这里 self.round 已加 1，作为分母可以直接用（与你原实现一致）
        for a in self.agents:
            self.per_agent_cumavg[a.name].append(self.per_agent_cumsum[a.name] / self.round)

        # 记录“上一轮配对”
        self.last_pairs = pairs

        # 返回给前端的本轮信息（结构保持不变）
        return {
            "round": self.round,
            "pairs": results,
            "coop_rate": rate,
            "round_avg_payoff": round_sum / len(self.agents),
        }

    def summary(self) -> List[Tuple[str, int, float]]:
        out = [(a.name, a.score, a.score / max(1, self.round)) for a in self.agents]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def overall_coop(self) -> float:
        return statistics.mean(self.coop_rates) if self.coop_rates else 0.0

    def _make_pairs(self):
        """
        生成“本轮”的配对（不修改 self.round、不计分）。
        逻辑：
          1) 复制当前参赛智能体列表；
          2) 用 self.rng 就地打乱；
          3) 每 2 个组成一对；若数量为奇数，最后 1 个本轮轮空。
        返回：
          list[tuple[Agent, Agent]]，例如 [(a,b), (c,d), ...]
        """
        # 1) 取得参赛列表（视你的实现，可能是 self.agents 或 self.pool）
        try:
            pool = list(self.agents)  # 大多数实现都叫 self.agents
        except AttributeError:
            pool = list(self.pool)  # 如果你用的是 self.pool

        if not pool:
            return []

        # 2) 使用统一的 rng 打乱（兼容 random.Random / np.random.*）
        # random.Random、RandomState、Generator 都有 .shuffle(x)
        self.rng.shuffle(pool)

        # 3) 两两成对；奇数时最后一个轮空
        pairs = []
        n = len(pool)
        for i in range(0, n - 1, 2):
            pairs.append((pool[i], pool[i + 1]))

        return pairs

    def preview_pairs(self):
        """无副作用预览下一轮配对：保存 RNG 状态 → 复刻 step() 的配对逻辑 → 恢复 RNG 状态"""
        rng = self.rng

        # --- 保存 RNG 状态（按类型分支，不用 try/elif 了） ---
        if hasattr(rng, "bit_generator"):  # numpy.Generator
            state = rng.bit_generator.state
            kind = "numpy"
        elif hasattr(rng, "get_state") and hasattr(rng, "set_state"):  # numpy.RandomState
            state = rng.get_state()
            kind = "randomstate"
        elif hasattr(rng, "getstate") and hasattr(rng, "setstate"):  # python random.Random
            state = rng.getstate()
            kind = "pyrandom"
        else:
            raise TypeError(f"Unsupported RNG type: {type(rng)}")

        # --- 复刻 step() 的配对逻辑（包括那次 rng.random() 的消耗）---
        if self.last_pairs and rng.random() < self.delta:
            pairs = self.last_pairs
        else:
            pairs = _make_random_pairs(self.agents, rng)

        # --- 恢复 RNG 状态 ---
        if kind == "numpy":
            rng.bit_generator.state = state
        elif kind == "randomstate":
            rng.set_state(state)
        else:  # pyrandom
            rng.setstate(state)

        return pairs

# ========== 工厂：构建19个固定策略 + USER ==========

def build_agents_without_bay_and_with_user(
    gTFT_generosity: float = 0.1,
    SG_punish: int = 2,
    ALT_startC: int = 1,
    R50_p: float = 0.5,
    sWSLS_eps: float = 0.2,
    Joss_p: float = 0.1,
    M1_pCC: float = 0.95, M1_pCD: float = 0.2, M1_pDC: float = 0.9, M1_pDD: float = 0.1
) -> Tuple[List[Agent], UserAgent]:
    agents: List[Agent] = [
        AlwaysCooperate("AC"),
        AlwaysDefect("AD"),
        TitForTat("TFT"),
        WinStayLoseShift("WSLS"),
        GrimTrigger("GRIM"),
        TitForTwoTats("TF2T"),
        SuspiciousTitForTat("sTFT"),
        GenerousTitForTat(f"gTFT{gTFT_generosity}", generosity=gTFT_generosity),
        SoftGrudger(f"SG{SG_punish}", punish_rounds=SG_punish),
        Alternator("ALT", start_with_C=bool(ALT_startC)),
        RandomCooperator(f"R{int(R50_p*100)}", coop_prob=R50_p),
        Prober("PROB"),
        StochasticWSLS(f"sWSLS{int(sWSLS_eps*100)}", epsilon=sWSLS_eps),
        Joss(f"Joss{int(Joss_p*100)}", p_defect_after_CC=Joss_p),
        ContriteTitForTat("CTFT"),
        Gradual("Gradual"),
        Tester("Tester"),
        Majority("Majority"),
        MemoryOne("M1", pCC=M1_pCC, pCD=M1_pCD, pDC=M1_pDC, pDD=M1_pDD),
    ]
    user = UserAgent("USER")
    agents.append(user)  # 变成20人（偶数）
    return agents, user
