"""Avaliação funcional das capacidades em core/cognitivo (critérios pré-registrados).

Critérios: /workspace/nexus/cognitivo/AUDITORIA_2026-10-22.md (sha256 3d24a63d…), escritos
antes da execução. Mede propósito, não só ausência de crash.

Uso: python tools/cognitivo_eval.py --capability {episodic,predictive,mcts,causal,counterfactual,working_memory,attention,incremental,episodic_future,mental_simulation,analogical_reasoning,all} [--json PATH]
Saída: exit 0 se todo critério passa, 1 se algum falha, 2 para entrada ausente/vazia/desconhecida.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INVALID_SCENARIO = 0, 1, 2, 3
CAPABILITIES = ("episodic", "predictive", "mcts", "causal", "counterfactual", "working_memory", "attention", "incremental", "episodic_future", "mental_simulation", "analogical_reasoning", "contextual_memory")

TOPICS = [
    ("database", "migration", "rollback"), ("network", "latency", "spike"), ("disk", "quota", "exceeded"),
    ("login", "password", "lockout"), ("cache", "eviction", "storm"), ("certificate", "expiry", "renewal"),
    ("memory", "leak", "worker"), ("deploy", "canary", "failure"), ("backup", "restore", "drill"),
    ("billing", "invoice", "mismatch"), ("search", "index", "rebuild"), ("email", "bounce", "queue"),
    ("firewall", "rule", "audit"), ("dns", "resolution", "timeout"), ("payment", "gateway", "error"),
    ("kernel", "panic", "reboot"), ("thread", "deadlock", "detector"), ("api", "rate", "limit"),
    ("sensor", "drift", "calibration"), ("report", "export", "pdf"),
]
FILLER = ["incident", "team", "notes", "followup", "review", "monday", "urgent", "ticket"]


def _check(cid, value, passed, limit, **extra):
    return {"id": cid, "value": value, "limit": limit, "passed": bool(passed), **extra}


# ── Memória episódica ────────────────────────────────────────────────────────

def _episodic_corpus(seed=7):
    rng = random.Random(seed)  # nosec B311 - dados sintéticos de avaliação
    items = []
    for topic_index, words in enumerate(TOPICS):
        for _ in range(10):
            title = list(words) + rng.sample(FILLER, 2)
            rng.shuffle(title)
            items.append((topic_index, " ".join(title)))
    rng.shuffle(items)
    queries = []
    for topic_index, words in enumerate(TOPICS):
        for pair in ((words[2], words[0]), (words[1], words[2])):
            queries.append((topic_index, " ".join(pair)))
    return items, queries


def eval_episodic():
    from core.cognitivo.episodic_memory import RealEpisodicMemory

    checks = []
    items, queries = _episodic_corpus()
    with RealEpisodicMemory() as mem:
        topic_of = {}
        for topic, title in items:
            eid = mem.start_episode(title, importance=0.5)
            mem.close_episode(eid)
            topic_of[eid] = topic

        def precision(results, topic):
            return sum(topic_of[e.episode_id] == topic for e in results[:5]) / 5

        p_system = sum(precision(mem.search_episodes(q, limit=5), t) for t, q in queries) / len(queries)
        p_recency = sum(precision(mem.search_episodes(None, limit=5), t) for t, q in queries) / len(queries)
        checks.append(_check("EM1", round(p_system, 4), p_system >= 0.80 and p_system >= p_recency + 0.30,
                             ">=0.80 e >= recência+0.30", baseline_recency=round(p_recency, 4)))

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "episodic.sqlite3")
        with RealEpisodicMemory(path) as mem:
            ids = []
            for i in range(50):
                eid = mem.start_episode(f"persist episode {i}")
                mem.close_episode(eid)
                ids.append(eid)
            stats_before = mem.get_statistics()
        with RealEpisodicMemory(path) as mem:
            found = {e.episode_id for e in mem.search_episodes(None, limit=100)}
            same = found == set(ids) and mem.get_statistics() == stats_before
        checks.append(_check("EM2", same, same, "ids e estatísticas iguais após reabrir"))

    rng = random.Random(8)  # nosec B311 - dados sintéticos de avaliação
    with RealEpisodicMemory() as mem:
        for i in range(10_000):
            words = TOPICS[i % len(TOPICS)]
            eid = mem.start_episode(f"{words[i % 3]} {words[(i + 1) % 3]} {rng.choice(FILLER)} {i}")
            mem.close_episode(eid)
        latencies = []
        for i in range(200):
            words = TOPICS[i % len(TOPICS)]
            start = time.perf_counter()
            mem.search_episodes(f"{words[1]} {words[0]}", limit=10)
            latencies.append(time.perf_counter() - start)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1] * 1000
    checks.append(_check("EM3", round(p95, 3), p95 <= 50.0, "p95 <= 50 ms"))

    with RealEpisodicMemory() as mem:
        low = mem.start_episode("shared topic low", importance=0.1)
        mem.close_episode(low)
        high = mem.start_episode("shared topic high", importance=0.9)
        mem.close_episode(high)
        old_high = mem.start_episode("shared topic older", importance=0.95)
        mem.close_episode(old_high)
        order = [e.episode_id for e in mem.search_episodes("shared topic", limit=3)]
    checks.append(_check("EM4", order[0] == old_high, True, "informativo: maior importance primeiro?",
                         informative=True))
    return checks


# ── Codificação preditiva ────────────────────────────────────────────────────

def _categorical(rng, probs):
    x, acc = rng.random(), 0.0
    for symbol, p in zip("abc", probs):
        acc += p
        if x < acc:
            return symbol
    return "c"


def _run_pc(stream):
    from core.cognitivo.predictive_coding import RealPredictiveCodingSystem

    with contextlib.redirect_stdout(io.StringIO()):
        pc = RealPredictiveCodingSystem()
    errors, confidences = [], []
    for obs in stream:
        result = pc.hierarchical_prediction_cycle({"sensory": obs})
        err = result["errors"][0]
        errors.append(err.error_magnitude)
        confidences.append(result["predictions"][-1].confidence)
    return errors, confidences


def _persistence(stream):
    return [0.0 if i and stream[i - 1] == x else 1.0 for i, x in enumerate(stream)]


def _ece(confidences, errors, bins=10):
    total, n = 0.0, len(errors)
    for b in range(bins):
        idx = [i for i, c in enumerate(confidences) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
        if idx:
            acc = sum(1 - errors[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def _closed_form_posterior(prior, transition, likelihood):
    n = len(prior)
    predicted = [sum(prior[i] * transition[i][j] for i in range(n)) for j in range(n)]
    weights = [predicted[j] * likelihood[j] for j in range(n)]
    z = sum(weights)
    return [w / z for w in weights]


def eval_predictive(criteria="v2"):
    from core.cognitivo.predictive_coding import RealPredictiveCodingSystem

    checks = []
    rng = random.Random(11)  # nosec B311 - dados sintéticos de avaliação
    stream = [_categorical(rng, (0.7, 0.2, 0.1)) for _ in range(500)]
    errors, confidences = _run_pc(stream)
    rate = sum(errors[-200:]) / 200
    rate_persist = sum(_persistence(stream)[-200:]) / 200
    checks.append(_check("PC1", round(rate, 4), rate <= 0.35 and rate <= rate_persist - 0.05,
                         "<=0.35 e <= persistência-0.05", baseline_persistence=round(rate_persist, 4),
                         baseline_no_learning=1.0))

    rng = random.Random(12)  # nosec B311 - dados sintéticos de avaliação
    shifted = [_categorical(rng, (0.7, 0.2, 0.1)) for _ in range(300)]
    shifted += [_categorical(rng, (0.1, 0.2, 0.7)) for _ in range(300)]
    errors_shift, _ = _run_pc(shifted)
    rate_shift = sum(errors_shift[400:600]) / 200
    checks.append(_check("PC2", round(rate_shift, 4), rate_shift <= 0.40, "<=0.40",
                         baseline_persistence=round(sum(_persistence(shifted)[400:600]) / 200, 4)))

    ece = _ece(confidences, errors)
    checks.append(_check("PC3", round(ece, 4), ece <= 0.10, "ECE <= 0.10"))

    rng = random.Random(13)  # nosec B311 - dados sintéticos de avaliação
    worst = 0.0
    for _ in range(100):
        n = rng.randint(2, 6)

        def simplex():
            v = [rng.random() + 1e-3 for _ in range(n)]
            s = sum(v)
            return [x / s for x in v]

        prior, transition = simplex(), [simplex() for _ in range(n)]
        likelihood = [rng.random() + 1e-3 for _ in range(n)]
        states = [f"s{i}" for i in range(n)]
        got = RealPredictiveCodingSystem.update_belief_state(states, prior, transition, likelihood)
        expected = _closed_form_posterior(prior, transition, likelihood)
        worst = max(worst, max(abs(got[s] - e) for s, e in zip(states, expected)))
    checks.append(_check("PC4", worst, worst <= 1e-9, "erro máx <= 1e-9"))
    if criteria in ("v2", "v3"):
        checks = [c for c in checks if c["id"] not in ("PC1", "PC3")] + _predictive_v2()
    return checks


PC_V2_SEEDS = tuple(range(100, 120))


def _predictive_v2():
    """PC1/PC3 v2: médias em 20 seeds fixas (AUDITORIA v2)."""
    rates, persists, eces = [], [], []
    for seed in PC_V2_SEEDS:
        rng = random.Random(seed)  # nosec B311 - dados sintéticos de avaliação
        stream = [_categorical(rng, (0.7, 0.2, 0.1)) for _ in range(500)]
        errors, confidences = _run_pc(stream)
        rates.append(sum(errors[-200:]) / 200)
        persists.append(sum(_persistence(stream)[-200:]) / 200)
        eces.append(_ece(confidences, errors))
    n = len(PC_V2_SEEDS)
    rate, persist, ece = sum(rates) / n, sum(persists) / n, sum(eces) / n
    return [
        _check("PC1", round(rate, 4), rate <= 0.35 and rate <= persist - 0.05,
               "v2: média 20 seeds <=0.35 e <= persistência-0.05", criteria="v2", seeds=list(PC_V2_SEEDS),
               baseline_persistence=round(persist, 4),
               per_seed_pass_rate=round(sum(r <= 0.35 for r in rates) / n, 4)),
        _check("PC3", round(ece, 4), ece <= 0.10, "v2: ECE médio 20 seeds <= 0.10", criteria="v2",
               seeds=list(PC_V2_SEEDS), per_seed_pass_rate=round(sum(e <= 0.10 for e in eces) / n, 4)),
    ]


# ── MCTS ─────────────────────────────────────────────────────────────────────

GOALS = ("secure the system", "improve stability", "reduce cost", "keep things running")


def _value(planner, goal, sequence):
    state = planner._get_initial_state(goal)
    for action in sequence:
        state = planner._apply_action(state, action)
    return planner._evaluate_state(state, goal)


def _all_sequences(planner, goal, depth):
    state0 = planner._get_initial_state(goal)
    out = [[]]
    frontier = [([], state0)]
    for d in range(depth):
        nxt = []
        for seq, state in frontier:
            for action in planner._get_available_actions(state, d):
                new_seq = seq + [action]
                nxt.append((new_seq, planner._apply_action(state, action)))
                out.append(new_seq)
        frontier = nxt
    return out


def eval_mcts(criteria="v2"):
    from core.cognitivo.mcts_planner import RealMCTSPlanner

    checks = []
    near_opt, total = 0, 0
    sums = {"mcts": 0.0, "mcts50": 0.0, "greedy": 0.0, "random": 0.0, "opt": 0.0}
    rng = random.Random(21)  # nosec B311 - dados sintéticos de avaliação
    for goal, depth in itertools.product(GOALS, (2, 3)):
        ref = RealMCTSPlanner(max_depth=depth)
        sequences = _all_sequences(ref, goal, depth)
        opt = max(_value(ref, goal, s) for s in sequences)
        greedy, state = [], ref._get_initial_state(goal)
        for d in range(depth):
            best = max(ref._get_available_actions(state, d),
                       key=lambda a: ref._evaluate_state(ref._apply_action(state, a), goal))
            greedy.append(best)
            state = ref._apply_action(state, best)
        full = [s for s in sequences if len(s) == depth]
        rand_mean = sum(_value(ref, goal, rng.choice(full)) for _ in range(200)) / 200
        for seed in range(10):
            random.seed(seed)
            v = _value(ref, goal, RealMCTSPlanner(max_depth=depth).plan(goal, max_iterations=2000).best_action_sequence)
            random.seed(seed)
            v50 = _value(ref, goal, RealMCTSPlanner(max_depth=depth).plan(goal, max_iterations=50).best_action_sequence)
            near_opt += v >= 0.98 * opt
            total += 1
            sums["mcts"] += v
            sums["mcts50"] += v50
            sums["greedy"] += _value(ref, goal, greedy)
            sums["random"] += rand_mean
            sums["opt"] += opt
    means = {k: round(v / total, 4) for k, v in sums.items()}
    frac = near_opt / total
    checks.append(_check("M1", round(frac, 4), frac >= 0.90, ">=0.90 com valor >= 0.98*ótimo"))
    checks.append(_check("M2", means["mcts"], means["mcts"] >= means["greedy"] and means["mcts"] > means["random"] + 0.02,
                         ">= guloso e > aleatório+0.02", greedy=means["greedy"], random=means["random"],
                         optimum=means["opt"]))
    regret, regret50 = means["opt"] - means["mcts"], means["opt"] - means["mcts50"]
    checks.append(_check("M3", round(regret, 4), regret < regret50, "regret(2000) < regret(50)",
                         regret_50=round(regret50, 4)))

    if criteria == "v2":
        checks = [c for c in checks if c["id"] != "M3"] + [_mcts_m3_v2()]
    elif criteria == "v3":
        checks = [c for c in checks if c["id"] != "M3"] + _mcts_m3_v3()

    planner = RealMCTSPlanner(max_depth=3)
    effects_before = set(planner._action_effects)
    ok = True
    for seed in range(10):
        random.seed(seed)
        allowed = ["MONITOR", "ROLLBACK", "ESCALATE"]
        snapshot = list(allowed)
        plan = planner.plan("secure the system", available_actions=allowed, max_iterations=200)
        ok &= allowed == snapshot and all(a in snapshot for a in plan.best_action_sequence)
    random.seed(0)
    planner.plan("secure the system", available_actions=["CUSTOM_X"], max_iterations=20)
    ok &= set(planner._action_effects) == effects_before
    checks.append(_check("M4", ok, ok, "plano ⊆ available_actions, lista intacta, sem vazamento de estado"))

    times = []
    for seed in range(20):
        random.seed(seed)
        start = time.perf_counter()
        RealMCTSPlanner().plan("secure the system", max_iterations=200)
        times.append(time.perf_counter() - start)
    p95 = sorted(times)[18]
    checks.append(_check("M5", round(p95, 4), p95 <= 0.10, "p95 <= 0.10 s"))
    return checks


def _mcts_m3_v2(depth=8):
    """M3 v2: horizonte 8; regret contra o melhor valor observado no cenário (AUDITORIA v2)."""
    from core.cognitivo.mcts_planner import RealMCTSPlanner

    ref = RealMCTSPlanner(max_depth=depth)
    values = {2000: [], 50: []}
    best = 0.0
    for goal in GOALS:
        per_goal = {2000: [], 50: []}
        for seed in range(10):
            for iterations in (2000, 50):
                random.seed(seed)  # nosec B311 - seed fixa pré-registrada
                plan = RealMCTSPlanner(max_depth=depth).plan(goal, max_iterations=iterations)
                per_goal[iterations].append(_value(ref, goal, plan.best_action_sequence))
        best = max(per_goal[2000] + per_goal[50])
        for iterations in (2000, 50):
            values[iterations] += [best - v for v in per_goal[iterations]]
    regret, regret50 = (sum(values[k]) / len(values[k]) for k in (2000, 50))
    return _check("M3", round(regret, 4), regret < regret50, "v2: max_depth=8, regret(2000) < regret(50)",
                  criteria="v2", regret_50=round(regret50, 4))


# ── M3 v3: árvores sintéticas com ótimo exato (AUDITORIA, seção v3) ─────────

V3_INSTANCES = tuple(range(20))
V3_RUN_SEEDS = tuple(range(10))
V3_BUDGETS = (50, 200, 800, 2000)
V3_BRANCHING, V3_DEPTH = 6, 6
V3_ACTIONS = tuple(f"A{k}" for k in range(V3_BRANCHING))


class SyntheticTree:
    """Árvore determinística b=6, d=6 com ramo enganoso e agulha (ótimo único = 1,0)."""

    def __init__(self, instance):
        if isinstance(instance, bool) or not isinstance(instance, int) or instance < 0:
            raise ValueError("instance must be a non-negative integer")
        rng = random.Random(1000 + instance)  # nosec B311 - instância sintética de avaliação
        self.leaves = {}
        self._fill(rng, (), 0.0)
        self.deceptive = V3_ACTIONS[rng.randrange(V3_BRANCHING)]
        self.needle = (self.deceptive,) + tuple(V3_ACTIONS[rng.randrange(V3_BRANCHING)]
                                                for _ in range(V3_DEPTH - 1))
        for path in self.leaves:
            if path[0] == self.deceptive:
                self.leaves[path] *= 0.5
        self.leaves[self.needle] = 1.0
        self.optimum = max(self.leaves.values())  # força bruta sobre as 46 656 folhas

    def _fill(self, rng, path, acc):
        if len(path) == V3_DEPTH:
            self.leaves[path] = acc / V3_DEPTH
            return
        for action in V3_ACTIONS:
            self._fill(rng, path + (action,), acc + rng.random())

    def value(self, sequence):
        return self.leaves.get(tuple(sequence), 0.0)


def _synthetic_planner(tree):
    """Subclasse do planejador com domínio injetado (efeitos/avaliador) e instrumentação de folhas."""
    from core.cognitivo.mcts_planner import RealMCTSPlanner

    class SyntheticPlanner(RealMCTSPlanner):
        def __init__(self):
            super().__init__(max_depth=V3_DEPTH)
            self.leaf_evaluations = []

        def _get_initial_state(self, goal):
            return {"steps_taken": []}

        def _get_available_actions(self, state, depth):
            return [] if depth >= self.max_depth else list(V3_ACTIONS)

        def _apply_action(self, state, action):
            return {"steps_taken": list(state.get("steps_taken", [])) + [action]}

        def _evaluate_state(self, state, goal):
            steps = tuple(state.get("steps_taken", []))
            if len(steps) == V3_DEPTH:
                self.leaf_evaluations.append(steps)
            return tree.value(steps)

    return SyntheticPlanner()


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _m3_precheck(trees):
    """M3P: validade do cenário, independente da qualidade do planejador."""
    if not trees:
        raise ValueError("M3 v3 requires at least one instance")
    ok_unique = ok_deceptive = ok_budget = 0
    for tree in trees:
        others = [v for p, v in tree.leaves.items() if p != tree.needle]
        ok_unique += tree.leaves[tree.needle] == tree.optimum and max(others) < tree.optimum
        means = {}
        for path, v in tree.leaves.items():
            means.setdefault(path[0], []).append(v)
        mean_d = sum(means[tree.deceptive]) / len(means[tree.deceptive])
        ok_deceptive += all(mean_d < sum(v) / len(v) for a, v in means.items() if a != tree.deceptive)
        random.seed(0)
        planner = _synthetic_planner(tree)
        planner.plan("synthetic", max_iterations=V3_BUDGETS[0])
        ok_budget += len(set(planner.leaf_evaluations)) <= V3_BUDGETS[0]
    subtree = V3_BRANCHING ** (V3_DEPTH - 1)
    bound = 1 - V3_BUDGETS[0] / subtree
    n = len(trees)
    passed = ok_unique == ok_deceptive == ok_budget == n and bound >= 0.5
    return _check("M3P", round(bound, 4), passed, "v3: (a)(b)(c) em todas as instâncias e cota subótimo@50 >= 0.5",
                  criteria="v3", unique_optimum=f"{ok_unique}/{n}", deceptive_branch=f"{ok_deceptive}/{n}",
                  leaves_at_50_bounded=f"{ok_budget}/{n}", scenario_invalid=not passed)


def _mcts_m3_v3(instances=V3_INSTANCES, seeds=V3_RUN_SEEDS, budgets=V3_BUDGETS):
    if not instances or not seeds or not budgets:
        raise ValueError("M3 v3 requires non-empty instances, seeds and budgets")
    trees = [SyntheticTree(i) for i in instances]
    pre = _m3_precheck(trees)
    regrets = {b: [] for b in budgets}
    to_optimum = []
    for tree in trees:
        for seed in seeds:
            for budget in budgets:
                random.seed(seed)  # nosec B311 - seed fixa pré-registrada
                planner = _synthetic_planner(tree)
                plan = planner.plan("synthetic", max_iterations=budget)
                regrets[budget].append(tree.optimum - tree.value(plan.best_action_sequence))
                if budget == budgets[-1] and tree.needle in planner.leaf_evaluations:
                    to_optimum.append(planner.leaf_evaluations.index(tree.needle) + 1)
    means = {b: sum(r) / len(r) for b, r in regrets.items()}
    medians = {b: _median(r) for b, r in regrets.items()}
    meds = [medians[b] for b in budgets]
    ok = (pre["passed"] and means[budgets[-1]] < means[budgets[0]]
          and all(a >= b for a, b in zip(meds, meds[1:])))
    last = regrets[budgets[-1]]
    return [
        pre,
        _check("M3", round(means[budgets[-1]], 4), ok,
               "v3: média(2000) < média(50) e mediana não crescente", criteria="v3",
               regret_50=round(means[budgets[0]], 4),
               mean_regret={b: round(v, 4) for b, v in means.items()},
               median_regret={b: round(v, 4) for b, v in medians.items()},
               scenario_invalid=not pre["passed"]),
        _check("M3I", round(sum(r == 0 for r in last) / len(last), 4), True,
               "informativo: fração no ótimo @2000", criteria="v3", informative=True,
               evaluations_to_optimum=sorted(to_optimum)),
    ]


# ── Raciocínio causal + modelo de mundo (critérios C, pré-registrados em 285851b) ──

CAUSAL_L = [
    ("Smoking causes cancer", ("smoking", "cancer")),
    ("Heavy rain causes floods", ("heavy rain", "floods")),
    ("Memory leaks cause crashes", ("memory leaks", "crashes")),
    ("Overheating and dust cause fan failure", ("overheating and dust", "fan failure")),
    ("Expired certificates cause handshake errors", ("expired certificates", "handshake errors")),
    ("Bad configs cause outages", ("bad configs", "outages")),
    ("The service crashed because the disk was full", ("the disk was full", "the service crashed")),
    ("Latency increased because of the cache eviction", ("the cache eviction", "latency increased")),
    ("Logins failed because the token expired", ("the token expired", "logins failed")),
    ("Stress leads to insomnia", ("stress", "insomnia")),
    ("A missing index leads to slow queries", ("a missing index", "slow queries")),
    ("Packet loss leads to retransmissions", ("packet loss", "retransmissions")),
    ("Deforestation results in erosion", ("deforestation", "erosion")),
    ("The misconfigured rule results in blocked traffic", ("the misconfigured rule", "blocked traffic")),
    ("Unpatched systems result in breaches", ("unpatched systems", "breaches")),
    ("Delays occurred due to the storm", ("the storm", "delays occurred")),
    ("Data loss happened due to a failed backup", ("a failed backup", "data loss happened")),
    ("Timeouts are due to DNS failures", ("dns failures", "timeouts are")),
    ("Coffee does not cause cancer", None),
    ("Vaccines never cause autism", None),
    ("Logging doesn't cause outages", None),
    ("There is no evidence that caffeine causes anxiety", None),
    ("Ice cream sales are correlated with drownings", None),
    ("Alerts and deploys co-occur", None),
    ("Sleep is associated with mood", None),
    ("Rain often appears with traffic", None),
    ("We left early because", None),
    ("The causeway was closed", None),
    ("The cause is unknown", None),
    ("Causal models are hard", None),
]
CAUSAL_H = [  # held-out pré-registrado em 396164e (R3), nunca usado para ajustar padrões
    ("Poor sanitation causes cholera", ("poor sanitation", "cholera")),
    ("Frequent deploys cause regressions", ("frequent deploys", "regressions")),
    ("The outage happened because the router failed", ("the router failed", "the outage happened")),
    ("Prices rose because of the drought", ("the drought", "prices rose")),
    ("Overfishing leads to population collapse", ("overfishing", "population collapse")),
    ("Misuse of antibiotics leads to resistance", ("misuse of antibiotics", "resistance")),
    ("The leak results in pressure loss", ("the leak", "pressure loss")),
    ("Cancellations were due to the strike", ("the strike", "cancellations were")),
    ("The fire was caused by faulty wiring", ("faulty wiring", "the fire")),
    ("Heavy load triggers throttling", ("heavy load", "throttling")),
    ("The disk filled up, so the job failed", ("the disk filled up", "the job failed")),
    ("Inflation was driven by energy costs", ("energy costs", "inflation")),
    ("As a result of the patch, crashes stopped", ("the patch", "crashes stopped")),
    ("Stress contributes to heart disease", ("stress", "heart disease")),
    ("Video games do not cause violence", None),
    ("It is false that vaccines cause autism", None),
    ("Coffee is not known to cause ulcers", None),
    ("Nothing suggests that 5G causes illness", None),
    ("Shark attacks rise with ice cream sales", None),
    ("Height is correlated with income", None),
    ("Umbrellas and rain appear together", None),
    ("Because is a conjunction", None),
    ("He gave no cause for alarm", None),
    ("Causality is a philosophical topic", None),
]
CAUSAL_ACTION_VOCAB = ("APPLY_PATCH", "IGNORE_ALERT", "ROLLBACK", "ESCALATE", "MONITOR", "ISOLATE_SYSTEM",
                       "RESTORE_BACKUP", "DEPLOY_UPDATE", "INCREASE_LOGGING", "SHUTDOWN_SERVICE")
CAUSAL_C3_SETS = [
    ([["IGNORE_ALERT"], ["APPLY_PATCH"]], 1),
    ([["APPLY_PATCH"], ["IGNORE_ALERT"]], 0),
    ([["MONITOR"], ["APPLY_PATCH"], ["IGNORE_ALERT"]], 1),
    ([["IGNORE_ALERT"], ["MONITOR"]], 1),
    ([["IGNORE_ALERT"], ["ESCALATE"]], 1),
    ([["IGNORE_ALERT", "MONITOR"], ["MONITOR", "MONITOR"], ["IGNORE_ALERT"]], 1),
]


def _norm(text):
    return text.strip().lower().rstrip(".!?;,")


def _score_extraction(cls, dataset):
    tp = fp = fn = 0
    neg_fp = 0
    per_form = {}
    for idx, (text, label) in enumerate(dataset):
        pairs = {(_norm(r.cause), _norm(r.effect)) for r in cls().extract_causal_relations(text)}
        if label is None:
            fp += len(pairs)
            neg_fp += bool(pairs)
            continue
        hit = label in pairs
        tp += hit
        fn += not hit
        fp += len(pairs - {label})
        per_form.setdefault(idx // 3, []).append(hit)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negatives = sum(label is None for _, label in dataset)
    return f1, precision, recall, neg_fp / negatives, per_form


def eval_causal(criteria="v1"):
    from core.cognitivo.causal_world_model import RealCausalReasoning, RealWorldModelSimulator
    f1, precision, recall, fp_rate, per_form = _score_extraction(RealCausalReasoning, CAUSAL_L)
    negatives = sum(label is None for _, label in CAUSAL_L)
    f1_h, precision_h, recall_h, fp_h, _ = _score_extraction(RealCausalReasoning, CAUSAL_H)
    wm = RealWorldModelSimulator()
    initial = wm.current_state.state
    covered = []
    for action in CAUSAL_ACTION_VOCAB:
        r = wm.simulate([action])
        if r.final_state.state != initial or r.warnings:
            covered.append(action)
    coverage = len(covered) / len(CAUSAL_ACTION_VOCAB)
    unknown_warned = bool(wm.simulate(["UNKNOWN_X"]).warnings)
    correct = sum(wm.compare_futures(seqs)["best_future_index"] == best for seqs, best in CAUSAL_C3_SETS)
    baseline_first = sum(best == 0 for _, best in CAUSAL_C3_SETS)
    cr = RealCausalReasoning()
    world = RealWorldModelSimulator(causal_reasoning=cr)
    cr.extract_causal_relations("DEPLOY_FIX causes system_status secure")
    integrated = world.simulate(["DEPLOY_FIX"]).final_state.state.get("system_status") == "secure"
    forms = ["causes", "cause(pl)", "because", "leads to", "results in", "due to"]
    return [
        _check("C1a", round(f1, 4), f1 >= 0.70, "F1 em L >= 0.70", criteria="v1", precision=round(precision, 4),
               recall=round(recall, 4), recall_by_form={forms[k]: f"{sum(v)}/{len(v)}" for k, v in per_form.items()}),
        _check("C1b", round(fp_rate, 4), fp_rate <= 0.10, "FP nas 12 negativas <= 0.10", criteria="v1",
               baseline_keyword_cause=round(sum("cause" in t.lower() for t, lab in CAUSAL_L if lab is None)
                                            / negatives, 4)),
        _check("C2a", round(coverage, 4), coverage >= 0.80, "ações do vocabulário com efeito ou aviso >= 0.80",
               criteria="v1", covered=covered),
        _check("C2b", unknown_warned, unknown_warned, "ação desconhecida gera warning", criteria="v1"),
        _check("C3", f"{correct}/{len(CAUSAL_C3_SETS)}", correct >= 5, ">= 5/6 top-1", criteria="v1",
               baseline_first_index=f"{baseline_first}/{len(CAUSAL_C3_SETS)}"),
        _check("C4", integrated, integrated, "relação extraída altera simulação", criteria="v1"),
        _check("H-F1", round(f1_h, 4), f1_h >= 0.60 and f1 - f1_h <= 0.20,
               "guarda R3: F1(H) >= 0.60 e F1(L) - F1(H) <= 0.20", criteria="v1",
               precision=round(precision_h, 4), recall=round(recall_h, 4), gap=round(f1 - f1_h, 4)),
        _check("H-FP", round(fp_h, 4), fp_h <= 0.20, "guarda R3: FP nas negativas de H <= 0.20", criteria="v1"),
    ]



CF_WHAT_IF = [
    ("What if we apply the patch?", "we apply the patch"),
    ("what if the server reboots?", "the server reboots"),
    ("WHAT IF isolation is enabled?", "isolation is enabled"),
    ("What if?", ""),
    ("And what if DNS fails", "dns fails"),
]
CF_PAIRS = [
    ("system vulnerable", "apply patch"),
    ("alarm", "patch"),
    ("outage", "rollback"),
    ("host", "isolate"),
]

CF_EXPLICIT = [
    ("during the outage", "we had patched"),
    ("on host db-1", "isolate the host"),
    ("system vulnerable", "apply patch"),
    ("alarm firing", "rollback"),
]
CF_JACCARD = [
    ("the cat sat on the mat", "the cat sat on a mat", 1 - 5 / 6),
    ("outcome without intervention", "outcome with apply patch", 1 - 1 / 6),
    ("alpha beta", "gamma delta", 1.0),
]


def eval_counterfactual(criteria="v1"):
    """Auditoria CF — critérios pré-registrados em AUDITORIA (v1)."""
    import ast
    import contextlib
    import io
    from pathlib import Path as _P

    from core.cognitivo.causal_world_model import RealCausalReasoning
    from core.cognitivo.counterfactual import RealCounterfactualReasoningEngine

    def _silent(factory):
        with contextlib.redirect_stdout(io.StringIO()):
            return factory()

    class _WM:
        def predict_outcome(self, situation, intervention):
            return f"wm:{situation}:{intervention}"

    eng_proto = _silent(lambda: RealCounterfactualReasoningEngine(
        world_model=_WM(), causal_reasoning=RealCausalReasoning()))
    cf1 = eng_proto._predict_outcome("sit", "act") == "wm:sit:act"
    cf2 = eng_proto._build_causal_chain("alarm", "patch") == ["alarm causes patch"]

    src = _P("core/cognitivo/counterfactual.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    planner_used = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RealCounterfactualReasoningEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Attribute) and sub.attr == "planner":
                            if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                                planner_used = True
    cf3 = planner_used

    eng = _silent(lambda: RealCounterfactualReasoningEngine())
    parse_hits = sum(
        1 for q, exp in CF_WHAT_IF if eng._parse_what_if_question(q)[1] == exp
    )
    cf4_ok = parse_hits >= 5

    lens = (1, 3, 5, 10)
    ps = [eng._calculate_plausibility("x", ["a"] * n) for n in lens]
    cf5 = all(ps[i] > ps[i + 1] for i in range(len(ps) - 1))

    altered = 0
    for sit, inter in CF_PAIRS:
        sc = eng.generate_counterfactual_scenario(sit, inter)
        if sc.counterfactual_outcome != sc.actual_outcome:
            altered += 1
    cf6_ok = altered == len(CF_PAIRS)

    explicit_ok = 0
    for sit, inter in CF_EXPLICIT:
        out = eng.explore_what_if(situation=sit, intervention=inter)
        sc = out["scenario"]
        if sc.original_situation == sit and sc.intervention == inter:
            explicit_ok += 1
    cf7_ok = explicit_ok == len(CF_EXPLICIT) and cf4_ok

    jacc_ok = 0
    if eng._calculate_outcome_difference("", "") == 0.0 and eng._calculate_outcome_difference("x", "x") == 0.0:
        jacc_ok += 1
    import math
    pair_hits = 0
    for a, b, expected in CF_JACCARD:
        if math.isclose(eng._calculate_outcome_difference(a, b), expected, rel_tol=0, abs_tol=1e-9):
            pair_hits += 1
    if pair_hits == len(CF_JACCARD):
        jacc_ok += 1
    shared = eng._calculate_outcome_difference(
        "outcome_without_intervention", "outcome_with_apply_patch")
    if shared < 1.0 and math.isclose(shared, 1 - 1 / 6, rel_tol=0, abs_tol=1e-9):
        jacc_ok += 1

    return [
        _check("CF1", cf1, cf1, "world_model.predict_outcome honrado", criteria="v1"),
        _check("CF2", cf2, cf2, "causal.extract_causal_relations honrado", criteria="v1"),
        _check("CF3", cf3, cf3, "planner injetado é referenciado fora de __init__", criteria="v1"),
        _check("CF4", f"{parse_hits}/{len(CF_WHAT_IF)}", cf4_ok, "parse what-if >= 5/5", criteria="v1"),
        _check("CF5", cf5, cf5, "plausibilidade monótona em (1,3,5,10)", criteria="v1",
               values=ps),
        _check("CF6", f"{altered}/{len(CF_PAIRS)}", cf6_ok,
               "intervenção altera counterfactual_outcome em S", criteria="v1"),
        _check("CF7", f"{explicit_ok}/{len(CF_EXPLICIT)}", cf7_ok,
               "API explícita preserva situation (E) e CF4 intacto", criteria="v1"),
        _check("CF8", f"{jacc_ok}/3", jacc_ok == 3,
               "diferença = 1-Jaccard(tokens alnum) no conjunto J", criteria="v1"),
    ]


# SHA-256 do fingerprint WM6, medido no monólito em b6da175 e congelado no
# pré-registro dcfa180 (docs/cognitivo/AUDITORIA_2026-10-22.md). Não recalibrar.
WM_GOLDEN_SHA = "3b6b63cb69d6ffced16ffec0c1bbbbeb8fe92bf7cecd13e20f4989b0d17051ca"
_WM_NAMES = (
    "WorkingMemoryItem", "RehearsalEvent", "PhonologicalLoop",
    "VisuospatialSketchpad", "CentralExecutive", "RealWorkingMemorySystem",
)


def working_memory_fingerprint(mod):
    """Protocolo WM6. O dict, não o texto-fonte, entra no SHA."""
    import contextlib

    CentralExecutive = mod.CentralExecutive
    PhonologicalLoop = mod.PhonologicalLoop
    VisuospatialSketchpad = mod.VisuospatialSketchpad
    RealWorkingMemorySystem = mod.RealWorkingMemorySystem
    WorkingMemoryItem = mod.WorkingMemoryItem
    ex = CentralExecutive()
    clamp = {}
    for strength in (2.0, -1.0, 0.5, 0.0, 1.0, 1, True, False, float("inf"), float("-inf"), float("nan")):
        clamp[repr(strength)] = ex.allocate_attention("planning", strength=strength)["strength"]
    same = ex.switch_task("planning")
    other = ex.switch_task("other")
    plain = CentralExecutive().allocate_attention("planning", strength=2.0)
    delegated = CentralExecutive(attention_system=object()).allocate_attention("monitoring", strength=-1.0)
    errors = {}
    for bad in ("abc", None, [], {}):
        try:
            CentralExecutive().allocate_attention("t", strength=bad)
            errors[repr(bad)] = None
        except Exception as exc:
            errors[repr(bad)] = type(exc).__name__
    item = WorkingMemoryItem("i", "c", "verbal", 0.5, 0.0, 0.0)
    CentralExecutive().prioritize_item(item, 1.0)
    prioritized = item.activation
    CentralExecutive().prioritize_item(item, 10.0)
    prioritized_cap = item.activation
    pad = VisuospatialSketchpad()
    visual = pad.store_visual("pic")
    visual.activation = 0.5
    pad.refresh(visual.item_id)
    refreshed = visual.activation
    visual.activation = 0.9
    pad.refresh(visual.item_id)
    refreshed_cap = visual.activation
    refresh_miss = pad.refresh("nope")
    spatial_pad = VisuospatialSketchpad(capacity=4)
    spatial = []
    for index, (activation, entered) in enumerate(((0.9, 1.0), (0.1, 4.0), (0.8, 2.0), (0.5, 3.0))):
        stored = spatial_pad.store_spatial(f"s{index}")
        stored.activation = activation
        stored.entered_at = entered
        spatial.append(stored)
    spatial_pad.store_spatial("s4")
    spatial_ids = {stored.item_id for stored in spatial_pad.buffer}
    loop = PhonologicalLoop()
    loop.store("Alpha Beta")
    loop.store("other")
    empty_hit = loop.retrieve("")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        system = RealWorkingMemorySystem()
        stats0 = system.get_statistics()
        for index in range(4):
            stored = system.store_verbal(f"m{index}")
            stored.activation = 1.0
            stored.last_rehearsed = time.time()
        system.maintain_items()
        maintain_count = system.rehearsal_count
        system.rehearse_verbal("missing")
        miss_count = system.rehearsal_count
        spatial_item = system.store_spatial({"x": 1})
        integrated = []
        for index, (entered, activation) in enumerate(((1.0, 0.9), (2.0, 0.8), (3.0, 0.4), (4.0, 0.2))):
            created = system.integrate_multimodal(f"v{index}", index)
            created.entered_at = entered
            created.activation = activation
            integrated.append(created)
        system.integrate_multimodal("v4", 4)
        integrated_ids = {stored.item_id for stored in system.episodic_buffer}

        class _LTM:
            def __init__(self):
                self.calls = []

            def store(self, content, kind, importance):
                self.calls.append([content, kind, importance])

        ltm = _LTM()
        with_ltm = RealWorkingMemorySystem(long_term_memory=ltm)
        with_ltm.transfer_to_ltm(with_ltm.store_verbal("memo"))
        without_ltm = RealWorkingMemorySystem()
        without_ltm.transfer_to_ltm(without_ltm.store_verbal("z"))

        class _Consciousness:
            def __init__(self):
                self.calls = []

            def attend_to(self, *args, **kwargs):
                self.calls.append({"args": list(args), "kwargs": kwargs})

        consciousness = _Consciousness()
        aware = RealWorkingMemorySystem(consciousness=consciousness)
        aware.store_verbal("hello consciousness buffer text that is definitely longer than fifty characters!!")
        ordered = RealWorkingMemorySystem()
        low = ordered.store_verbal("low")
        low.activation = 0.2
        high = ordered.store_visual("high")
        high.activation = 0.9
        order = [stored.content for stored in ordered.get_active_items()]
        boosted = system.store_verbal("BoostMe")
        boosted.activation = 0.5
        found = system.retrieve_from_wm("boostme")
        missed = system.retrieve_from_wm("zzz")
    return {
        "banner_printed": "POST 6" in captured.getvalue(),
        "capacities": [PhonologicalLoop().capacity, VisuospatialSketchpad().capacity, system.episodic_capacity],
        "clamp": clamp,
        "consciousness": consciousness.calls[0] if consciousness.calls else None,
        "decay_rate": PhonologicalLoop().decay_rate,
        "delegated": delegated,
        "errors": errors,
        "get_active_order": order,
        "integrate_evicted_oldest_entered_at": integrated[0].item_id not in integrated_ids,
        "integrate_len": len(integrated_ids),
        "integrate_low_activation_remains": integrated[3].item_id in integrated_ids,
        "ltm_calls": ltm.calls,
        "ltm_count": with_ltm.items_transferred_to_ltm,
        "maintain_count": maintain_count,
        "no_ltm_count": without_ltm.items_transferred_to_ltm,
        "plain": plain,
        "prioritize": [prioritized, prioritized_cap],
        "refresh": [refreshed, refreshed_cap, refresh_miss],
        "rehearse_verbal_miss_count": miss_count,
        "retrieve_boost": None if found is None else found.activation,
        "retrieve_empty": None if empty_hit is None else empty_hit.content,
        "retrieve_miss": missed is None,
        "spatial_modality": spatial_item.modality,
        "spatial_oldest_remains": spatial[0].item_id in spatial_ids,
        "spatial_victim_is_low_activation": spatial[1].item_id not in spatial_ids,
        "stats0": stats0,
        "switch_other": other,
        "switch_same": same,
    }


def _wm_sha(payload) -> str:
    import hashlib
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _wm_standalone() -> bool:
    """O subprocesso que bloqueia o monólito fica no contrato (bandit não varre test_*.py).

    Aqui só a árvore de imports: um import de constitutional_brain, numpy ou vita
    no módulo extraído faz WM6 falhar. Menção no docstring não conta, e tools/
    não ganha B404/B603.
    """
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/working_memory.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" or name.startswith("core.constitutional_brain."):
                return False
    return True


def eval_working_memory(criteria="v1"):
    """WM1–WM6. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import inspect

    import core.cognitivo.working_memory as wm
    import core.constitutional_brain as brain

    loop = wm.PhonologicalLoop(capacity=7)
    stored = [loop.store(f"w{i}") for i in range(7)]
    for index, item in enumerate(stored):
        item.last_rehearsed = float(index + 1)
    eighth = loop.store("w7")
    kept = {item.item_id for item in loop.buffer}
    wm1_len = len(loop.buffer) == 7 and eighth.item_id in kept and stored[0].item_id not in kept
    tie = wm.PhonologicalLoop(capacity=7)
    tied = [tie.store(f"t{i}") for i in range(7)]
    for item in tied:
        item.last_rehearsed = 5.0
    tie.store("t7")
    tie_ids = {item.item_id for item in tie.buffer}
    wm1_tie = tied[0].item_id not in tie_ids and tied[1].item_id in tie_ids and len(tie.buffer) == 7

    pad = wm.VisuospatialSketchpad(capacity=4)
    visuals = [pad.store_visual(f"v{i}") for i in range(4)]
    for item, activation in zip(visuals, (0.9, 0.2, 0.8, 0.4)):
        item.activation = activation
    fifth = pad.store_visual("v4")
    visual_ids = {item.item_id for item in pad.buffer}
    wm2 = len(pad.buffer) == 4 and visuals[1].item_id not in visual_ids and fifth.item_id in visual_ids
    tie_pad = wm.VisuospatialSketchpad(capacity=4)
    tied_visual = [tie_pad.store_visual(f"e{i}") for i in range(4)]
    tie_pad.store_visual("e4")
    tie_visual_ids = {item.item_id for item in tie_pad.buffer}
    wm2_tie = tied_visual[0].item_id not in tie_visual_ids and tied_visual[1].item_id in tie_visual_ids

    decay = wm.PhonologicalLoop()
    kept_item = decay.store("keep")
    kept_item.activation = 1.0
    kept_item.last_rehearsed = time.time() - 0.4
    epoch = decay.store("epoch")
    epoch.activation = 1.0
    epoch.last_rehearsed = 0.0
    boundary = decay.store("boundary")
    boundary.activation = 0.1
    boundary.last_rehearsed = time.time()
    decay.decay_items()
    decay_ids = {item.item_id for item in decay.buffer}
    init_params = inspect.signature(wm.PhonologicalLoop.__init__).parameters
    decay_params = list(inspect.signature(wm.PhonologicalLoop.decay_items).parameters)
    no_clock = decay_params == ["self"] and "clock" not in init_params and "now" not in init_params
    wm3 = (
        epoch.item_id not in decay_ids
        and boundary.item_id not in decay_ids
        and kept_item.item_id in decay_ids
        and kept_item.activation < 1.0
        and kept_item.activation > 0.1
    )

    pair = wm.PhonologicalLoop()
    rehearsed = pair.store("rehearsed")
    twin = pair.store("twin")
    rehearsed.activation = 0.4
    twin.activation = 0.4
    rehearsed.last_rehearsed = 0.0
    twin.last_rehearsed = 0.0
    before = time.time()
    rehearse_ok = pair.rehearse(rehearsed.item_id)
    boost = rehearsed.activation
    rehearse_miss = pair.rehearse("missing")
    pair.decay_items()
    pair_ids = {item.item_id for item in pair.buffer}
    cap_loop = wm.PhonologicalLoop()
    near = cap_loop.store("near")
    near.activation = 0.9
    cap_loop.rehearse(near.item_id)
    topped = cap_loop.store("top")
    topped.activation = 1.0
    cap_loop.rehearse(topped.item_id)
    wm4 = (
        rehearse_ok is True
        and boost == 0.7
        and rehearsed.last_rehearsed >= before
        and rehearse_miss is False
        and near.activation == 1.0
        and topped.activation == 1.0
        and rehearsed.item_id in pair_ids
        and twin.item_id not in pair_ids
    )

    found_loop = wm.PhonologicalLoop()
    found_loop.store("Alpha Beta")
    found_loop.store("other")
    alpha = found_loop.retrieve("alpha")
    beta = found_loop.retrieve("BETA")
    alp = found_loop.retrieve("alp")
    miss = found_loop.retrieve("zzz")
    wm5 = (
        alpha is not None and alpha.content == "Alpha Beta"
        and beta is not None and beta.content == "Alpha Beta"
        and alp is not None and alp.content == "Alpha Beta"
        and miss is None
    )

    import core.cognitivo as package
    identity = all(
        getattr(brain, name) is getattr(wm, name) is getattr(package, name)
        and getattr(wm, name).__module__ == "core.cognitivo.working_memory"
        for name in _WM_NAMES
    )
    sha_module = _wm_sha(working_memory_fingerprint(wm))
    sha_brain = _wm_sha(working_memory_fingerprint(brain))
    standalone = _wm_standalone()
    wm6 = identity and standalone and sha_module == sha_brain == WM_GOLDEN_SHA

    return [
        _check("WM1", {"len": len(loop.buffer), "evicted_min": wm1_len, "tie_evicts_first": wm1_tie},
               wm1_len and wm1_tie, "len==7; vítima = menor last_rehearsed; empate = primeiro"),
        _check("WM2", {"len": len(pad.buffer), "evicted_low_activation": wm2, "tie_evicts_first": wm2_tie},
               wm2 and wm2_tie, "len==4; vítima = menor activation; empate = primeiro"),
        _check("WM3", {"clock": "wall" if no_clock else "injected", "kept_activation": kept_item.activation,
                       "removed_epoch": epoch.item_id not in decay_ids,
                       "removed_boundary": boundary.item_id not in decay_ids},
               wm3, "decay reduz activation; activation<=0.1 sai; sem exigir igualdade de relógio",
               clock="wall" if no_clock else "injected"),
        _check("WM4", {"boost": boost, "cap": near.activation, "at_cap": topped.activation,
                       "miss": rehearse_miss, "rehearsed_survives": rehearsed.item_id in pair_ids,
                       "twin_removed": twin.item_id not in pair_ids},
               wm4, "rehearse +0.3 teto 1.0; ensaiado sobrevive; gêmeo sai"),
        _check("WM5", {"alpha": None if alpha is None else alpha.content,
                       "BETA": None if beta is None else beta.content,
                       "alp": None if alp is None else alp.content, "miss": miss},
               wm5, "substring sem maiúsculas; miss é None"),
        _check("WM6", {"sha_module": sha_module, "sha_brain": sha_brain, "identity": identity,
                       "standalone": standalone, "golden": WM_GOLDEN_SHA},
               wm6, "reexport idêntico e fingerprint SHA pré-registrado"),
    ]


AT_SOURCE_SHA = "96090684b99d2568ea61f040c5eed9bbf6de774f5bad9d781f60c2a566ea2378"
AT_BEHAVIOR_SHA = "07877c7f811006b479729aa7faa7d6b82f824afc0d2bf2dc7514d059ce5f990b"
_AT_NAMES = (
    "StimulusItem",
    "AttentionalFocus",
    "AttentionShift",
    "BottomUpAttention",
    "TopDownAttention",
    "AttentionalSpotlight",
    "RealAttentionSystem",
)


def attention_fingerprint(mod):
    """Protocolo AT6. O dict, não o texto-fonte, entra no SHA comportamental."""
    import contextlib
    import inspect

    StimulusItem = mod.StimulusItem
    BottomUpAttention = mod.BottomUpAttention
    TopDownAttention = mod.TopDownAttention
    AttentionalSpotlight = mod.AttentionalSpotlight
    RealAttentionSystem = mod.RealAttentionSystem

    def stim(**kwargs):
        values = {
            "stimulus_id": "s1",
            "stimulus_type": "visual",
            "content": "x",
            "salience": 0.5,
            "priority": 0.5,
            "timestamp": 1.0,
        }
        values.update(kwargs)
        return StimulusItem(**values)

    def typed(fn):
        try:
            return fn()
        except Exception as exc:
            return type(exc).__name__

    bottom = BottomUpAttention()
    weights = dict(bottom.salience_factors)
    novel = stim(stimulus_id="new", contrast=1.0, motion=1.0)
    familiar = stim(stimulus_id="old", contrast=1.0, motion=1.0)
    contrast_cases = {
        "fallback": bottom._assess_contrast(stim(), {}),
        "stim_high": bottom._assess_contrast(stim(contrast=1.4), {}),
        "stim_low": bottom._assess_contrast(stim(contrast=-0.2), {}),
        "stim_true": bottom._assess_contrast(stim(contrast=True), {}),
        "stim_false": bottom._assess_contrast(stim(contrast=False), {}),
        "stim_nan": bottom._assess_contrast(stim(contrast=float("nan")), {}),
        "stim_inf": bottom._assess_contrast(stim(contrast=float("inf")), {}),
        "stim_ninf": bottom._assess_contrast(stim(contrast=float("-inf")), {}),
        "ctx": bottom._assess_contrast(stim(), {"contrast": 0.25}),
        "ctx_str": bottom._assess_contrast(stim(), {"contrast": "0.25"}),
        "ctx_none": bottom._assess_contrast(stim(), {"contrast": None}),
        "ctx_true": bottom._assess_contrast(stim(), {"contrast": True}),
        "ctx_nan": bottom._assess_contrast(stim(), {"contrast": float("nan")}),
        "stim_beats_ctx": bottom._assess_contrast(stim(contrast=0.1), {"contrast": 0.9}),
    }
    motion_cases = {
        "fallback": bottom._assess_motion(stim()),
        "high": bottom._assess_motion(stim(motion=1.4)),
        "low": bottom._assess_motion(stim(motion=-0.2)),
        "true": bottom._assess_motion(stim(motion=True)),
        "false": bottom._assess_motion(stim(motion=False)),
        "nan": bottom._assess_motion(stim(motion=float("nan"))),
        "inf": bottom._assess_motion(stim(motion=float("inf"))),
        "ninf": bottom._assess_motion(stim(motion=float("-inf"))),
    }
    errors = {
        "contrast_str": typed(lambda: bottom._assess_contrast(stim(contrast="0.2"), {})),
        "contrast_ctx_abc": typed(lambda: bottom._assess_contrast(stim(), {"contrast": "abc"})),
        "motion_str": typed(lambda: bottom._assess_motion(stim(motion="0.2"))),
    }
    intensity = {
        kind: bottom._assess_intensity(stim(stimulus_type=kind))
        for kind in ("visual", "auditory", "internal", "tactile")
    }
    top = TopDownAttention()
    no_goals = top.compute_goal_relevance(stim(content="find the cat"))
    top.set_goals(["find cat"])
    matches = {
        "full": top.compute_goal_relevance(stim(content="please find the cat now")),
        "one": top.compute_goal_relevance(stim(content="find the dog")),
        "none": top.compute_goal_relevance(stim(content="zzzz")),
        "case": top.compute_goal_relevance(stim(content="FIND CAT")),
    }
    cat = TopDownAttention()
    cat.set_goals(["cat"])
    substring = cat.compute_goal_relevance(stim(content="category"))
    empty = TopDownAttention()
    empty.set_goals([""])
    empty_goal = empty.compute_goal_relevance(stim(content="anything"))
    multi = TopDownAttention()
    multi.set_goals(["alpha", "beta gamma"])
    multi.update_goal_priority("alpha", 1.0)
    multi.update_goal_priority("beta gamma", 0.5)
    multi_scores = {
        "beta": multi.compute_goal_relevance(stim(content="beta only")),
        "alpha": multi.compute_goal_relevance(stim(content="alpha")),
    }
    multi.set_goals(["alpha"])
    priority_kept = multi.goal_relevance["alpha"]
    clamps = {}
    for label, value in (
        ("2", 2.0), ("-1", -1.0), ("0.5", 0.5), ("true", True), ("false", False),
        ("inf", float("inf")), ("-inf", float("-inf")), ("nan", float("nan")),
    ):
        holder = TopDownAttention()
        holder.update_goal_priority("g", value)
        clamps[label] = holder.goal_relevance["g"]
    priority_errors = {}
    for bad in ("abc", None, [], {}):
        holder = TopDownAttention()
        priority_errors[repr(bad)] = typed(lambda bad=bad, holder=holder: holder.update_goal_priority("g", bad))

    def spotlight_ids(pairs, extra):
        spot = AttentionalSpotlight(capacity=3)
        for index, salience in pairs:
            spot.focus_on(stim(
                stimulus_id=f"id{index}", content=f"c{index}", salience=salience,
                priority=0.0, timestamp=float(index),
            ))
        spot.focus_on(stim(
            stimulus_id=f"id{extra[0]}", content=f"c{extra[0]}", salience=extra[1],
            priority=0.0, timestamp=float(extra[0]),
        ))
        return [item.stimulus_id for item in spot.spotlight]

    evicted = spotlight_ids([(0, 0.9), (1, 0.2), (2, 0.8)], (3, 0.5))
    tie = spotlight_ids([(0, 0.4), (1, 0.4), (2, 0.9)], (3, 0.7))
    spot = AttentionalSpotlight(capacity=1)
    held = stim(stimulus_id="id0", content="c0", salience=0.1, priority=0.0, timestamp=0.0)
    spot.focus_on(held, strength=0.2)
    spot.focus_on(held, strength=0.9)
    twin = StimulusItem("id0", "visual", "c0", 0.1, 0.0, 0.0)
    spot.focus_on(twin, strength=0.9)
    refocus = {
        "len": len(spot.spotlight),
        "history": len(spot.focus_history),
        "strength": spot.current_focus.focus_strength,
        "twin_equal": twin == held,
    }
    broad = AttentionalSpotlight(capacity=3)
    for index, salience in ((0, 0.9), (1, 0.2), (2, 0.8)):
        broad.focus_on(stim(
            stimulus_id=f"id{index}", content=f"c{index}", salience=salience,
            priority=0.0, timestamp=float(index),
        ))
    broad.release_focus("id1")
    released_low = [item.stimulus_id for item in broad.spotlight]
    current_id = None if broad.current_focus is None else broad.current_focus.focused_item.stimulus_id
    lone = AttentionalSpotlight()
    lone.focus_on(stim(stimulus_id="gone", content="g", salience=0.4, priority=0.0, timestamp=1.0))
    lone.release_focus("gone")
    released_current = {
        "current_none": lone.current_focus is None,
        "duration_set": lone.focus_history[-1].duration is not None and lone.focus_history[-1].duration >= 0,
        "focused": lone.is_focused("gone"),
        "copy": lone.get_focused_items() is not lone.spotlight,
    }

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        RealAttentionSystem()
    banner = "POST 25" in captured.getvalue()

    def add(system, sid, salience, priority, kind="visual", content="z"):
        item = StimulusItem(sid, kind, content, salience, priority, 1.0)
        system.available_stimuli.append(item)
        return item

    with contextlib.redirect_stdout(io.StringIO()):
        ordered = RealAttentionSystem()
    for sid, sal, pri in (("a", 0.2, 0.2), ("b", 0.9, 0.9), ("c", 0.4, 0.8), ("d", 1.0, 0.0)):
        add(ordered, sid, sal, pri)
    selected = ordered.attend()
    order = [[item.stimulus_id, item.attention_score] for item in selected]
    with contextlib.redirect_stdout(io.StringIO()):
        tied = RealAttentionSystem()
    for index, (sal, pri) in enumerate(((0.5, 0.5), (0.4, 0.6), (0.6, 0.4), (0.0, 0.0))):
        add(tied, f"t{index}", sal, pri)
    tie_order = [item.stimulus_id for item in tied.attend()]
    with contextlib.redirect_stdout(io.StringIO()):
        stale = RealAttentionSystem()
    stale_item = StimulusItem("z", "visual", "z", 0.1, 0.1, 1.0)
    stale.available_stimuli.append(stale_item)
    stale.attend(goals=["zzz"])
    no_recompute = {
        "salience": stale_item.salience,
        "priority": stale_item.priority,
        "score": stale_item.attention_score,
        "goals": list(stale.top_down.active_goals),
        "episodes": stale.total_attentional_episodes,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        registered = RealAttentionSystem()
    registered.top_down.set_goals(["cat"])
    created = registered.register_stimulus("visual", "a cat")
    registered_values = {"salience": created.salience, "priority": created.priority}
    with contextlib.redirect_stdout(io.StringIO()):
        empty_system = RealAttentionSystem()
    empty_attend = {"selected": empty_system.attend(), "episodes": empty_system.total_attentional_episodes}
    with contextlib.redirect_stdout(io.StringIO()):
        shifter = RealAttentionSystem()
    visual_a = shifter.register_stimulus("visual", "one")
    visual_b = shifter.register_stimulus("visual", "two")
    auditory = shifter.register_stimulus("auditory", "three")
    missing = shifter.shift_attention("missing")
    shifts_after_miss = len(shifter.attention_shifts)
    first = shifter.shift_attention(visual_a.stimulus_id, "voluntary")
    second = shifter.shift_attention(visual_b.stimulus_id, "goal")
    third = shifter.shift_attention(auditory.stimulus_id, "salience")
    shift = {
        "missing": missing,
        "shifts_after_miss": shifts_after_miss,
        "ok": [first, second, third],
        "costs": [item.cost for item in shifter.attention_shifts],
        "from_first_none": shifter.attention_shifts[0].from_focus is None,
        "from_later_set": shifter.attention_shifts[1].from_focus is not None,
        "voluntary": shifter.voluntary_shifts,
        "involuntary": shifter.involuntary_shifts,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        fresh = RealAttentionSystem()
    alone = fresh.register_stimulus("auditory", "beep")
    fresh.shift_attention(alone.stimulus_id)
    no_focus_cost = fresh.attention_shifts[-1].cost

    class _Conscious:
        def __init__(self):
            self.calls = []

        def attend_to(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    with contextlib.redirect_stdout(io.StringIO()):
        aware = RealAttentionSystem(consciousness=_Conscious())
    aware.available_stimuli.append(StimulusItem("z", "visual", "z", 1.0, 1.0, 1.0))
    aware.attend()
    call_args, call_kwargs = aware.consciousness.calls[0]
    consciousness = {
        "args": list(call_args),
        "salience": call_kwargs.get("salience"),
        "n": len(aware.consciousness.calls),
    }

    with contextlib.redirect_stdout(io.StringIO()):
        cleaner = RealAttentionSystem()
    now = time.time()
    cleaner.available_stimuli.extend([
        StimulusItem("old", "visual", "o", 0.1, 0.1, 0.0),
        StimulusItem("recent", "visual", "r", 0.1, 0.1, now - 61.0),
        StimulusItem("live", "visual", "l", 0.1, 0.1, now),
    ])
    cleaner.clear_attended_stimuli()
    kept = [item.stimulus_id for item in cleaner.available_stimuli]
    init_params = inspect.signature(RealAttentionSystem.__init__).parameters
    clear_params = list(inspect.signature(RealAttentionSystem.clear_attended_stimuli).parameters)
    clock = "wall" if clear_params == ["self"] and "clock" not in init_params and "now" not in init_params else "injected"
    with contextlib.redirect_stdout(io.StringIO()):
        stats_system = RealAttentionSystem()
    stats0 = stats_system.get_statistics()
    return {
        "banner": banner,
        "clamps": clamps,
        "clear_kept": kept,
        "clock": clock,
        "consciousness": consciousness,
        "contrast": contrast_cases,
        "current_after_release": current_id,
        "empty_attend": empty_attend,
        "empty_goal": empty_goal,
        "errors": errors,
        "evicted": evicted,
        "intensity": intensity,
        "matches": matches,
        "motion": motion_cases,
        "multi": multi_scores,
        "no_focus_cost": no_focus_cost,
        "no_goals": no_goals,
        "no_recompute": no_recompute,
        "novelty_absent": bottom._assess_novelty(stim(), {}),
        "novelty_present": bottom._assess_novelty(stim(), {"seen_stimuli": ["s1"]}),
        "order": order,
        "priority_errors": priority_errors,
        "priority_kept": priority_kept,
        "refocus": refocus,
        "registered": registered_values,
        "released_current": released_current,
        "released_low": released_low,
        "salience_familiar": bottom.compute_salience(familiar, {"seen_stimuli": ["old"]}),
        "salience_novel": bottom.compute_salience(novel, {}),
        "shift": shift,
        "stats0": stats0,
        "substring": substring,
        "tie_order": tie_order,
        "tie_spotlight": tie,
        "weights": weights,
        "weights_sum": sum(weights.values()),
    }


def _at_sha(payload) -> str:
    import hashlib
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def attention_source_text(src: str) -> str:
    marker = "# MÓDULO 11A3: ATTENTION & FOCUS SYSTEM (POST 25)\n"
    idx = src.index(marker)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end_marker = '            "focus_capacity": self.spotlight.capacity\n        }\n'
    end = src.index(end_marker, idx) + len(end_marker)
    return src[line_start:end]


def _attention_source_sha() -> str:
    import hashlib
    text = (ROOT / "core/cognitivo/attention.py").read_text(encoding="utf-8")
    return hashlib.sha256(attention_source_text(text).encode()).hexdigest()


def _attention_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha AT6.

    Menção no docstring não conta. O bloqueio em runtime fica no contrato, fora de tools/.
    """
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/attention.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" or name.startswith("core.constitutional_brain."):
                return False
    return True


def eval_attention(criteria="v1"):
    """AT1–AT6. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import core.cognitivo.attention as attention
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = attention_fingerprint(attention)
    sha_module = _at_sha(payload)
    sha_brain = _at_sha(attention_fingerprint(brain))
    source_sha = _attention_source_sha()
    standalone = _attention_standalone()
    identity = all(
        getattr(brain, name) is getattr(attention, name) is getattr(package, name)
        and getattr(attention, name).__module__ == "core.cognitivo.attention"
        for name in _AT_NAMES
    )
    weights = payload["weights"]
    novel_formula = 0.3 * 0.9 + 0.25 * 0.7 + 0.25 * 1.0 + 0.2 * 1.0
    familiar_formula = 0.3 * 0.2 + 0.25 * 0.7 + 0.25 * 1.0 + 0.2 * 1.0
    at1 = (
        payload["weights_sum"] == 1.0
        and weights == {"novelty": 0.3, "intensity": 0.25, "contrast": 0.25, "motion": 0.2}
        and payload["novelty_absent"] == 0.9
        and payload["novelty_present"] == 0.2
        and payload["intensity"] == {"visual": 0.7, "auditory": 0.6, "internal": 0.4, "tactile": 0.4}
        and payload["salience_novel"] == novel_formula
        and payload["salience_familiar"] == familiar_formula
    )
    contrast = payload["contrast"]
    motion = payload["motion"]
    at2 = (
        contrast["stim_high"] == 1.0
        and contrast["stim_low"] == 0.0
        and contrast["fallback"] == 0.5
        and contrast["ctx"] == 0.25
        and contrast["ctx_str"] == 0.25
        and contrast["ctx_none"] == 0.5
        and contrast["stim_beats_ctx"] == 0.1
        and motion["fallback"] == 0.6
        and motion["high"] == 1.0
        and motion["low"] == 0.0
        and payload["errors"] == {
            "contrast_str": "TypeError",
            "contrast_ctx_abc": "ValueError",
            "motion_str": "TypeError",
        }
    )
    at3 = (
        payload["no_goals"] == 0.3
        and payload["matches"] == {"full": 0.5, "one": 0.25, "none": 0.05, "case": 0.5}
        and payload["substring"] == 0.5
        and payload["empty_goal"] == 0.05
        and payload["multi"] == {"beta": 0.25, "alpha": 1.0}
        and payload["priority_kept"] == 1.0
        and payload["clamps"]["2"] == 1.0
        and payload["clamps"]["-1"] == 0.0
        and payload["clamps"]["0.5"] == 0.5
        and payload["priority_errors"] == {
            "'abc'": "TypeError", "None": "TypeError", "[]": "TypeError", "{}": "TypeError",
        }
    )
    released = payload["released_current"]
    at4 = (
        payload["evicted"] == ["id0", "id2", "id3"]
        and payload["tie_spotlight"] == ["id1", "id2", "id3"]
        and payload["refocus"] == {"len": 1, "history": 1, "strength": 0.2, "twin_equal": True}
        and payload["released_low"] == ["id0", "id2"]
        and payload["current_after_release"] == "id2"
        and released["current_none"] is True
        and released["duration_set"] is True
        and released["focused"] is False
        and released["copy"] is True
    )
    shift = payload["shift"]
    at5 = (
        [row[0] for row in payload["order"]] == ["b", "c", "d"]
        and payload["order"][1][1] == (0.4 + 0.8) / 2.0
        and payload["tie_order"] == ["t0", "t1", "t2"]
        and payload["no_recompute"]["salience"] == 0.1
        and payload["no_recompute"]["priority"] == 0.1
        and payload["no_recompute"]["goals"] == ["zzz"]
        and shift["missing"] is False
        and shift["shifts_after_miss"] == 0
        and shift["costs"] == [0.2, 0.2, 0.5]
        and shift["voluntary"] == 1
        and shift["involuntary"] == 2
        and payload["no_focus_cost"] == 0.2
        and payload["empty_attend"] == {"selected": [], "episodes": 0}
    )
    at6 = (
        identity
        and standalone
        and source_sha == AT_SOURCE_SHA
        and sha_module == sha_brain == AT_BEHAVIOR_SHA
    )
    return [
        _check("AT1", {"weights_sum": payload["weights_sum"], "novelty_absent": payload["novelty_absent"],
                       "novelty_present": payload["novelty_present"], "intensity": payload["intensity"],
                       "salience_novel": payload["salience_novel"], "salience_familiar": payload["salience_familiar"]},
               at1, "pesos somam 1; novidade 0.9/0.2; intensidade 0.7/0.6/0.4"),
        _check("AT2", {"contrast_high": contrast["stim_high"], "contrast_low": contrast["stim_low"],
                       "fallback": contrast["fallback"], "ctx": contrast["ctx"], "motion": motion["fallback"],
                       "errors": payload["errors"]},
               at2, "clamp [0,1]; context; fallbacks 0.5/0.6; str inválida"),
        _check("AT3", {"no_goals": payload["no_goals"], "matches": payload["matches"],
                       "substring": payload["substring"], "empty_goal": payload["empty_goal"],
                       "multi": payload["multi"], "priority_kept": payload["priority_kept"],
                       "clamp_high": payload["clamps"]["2"], "priority_errors": payload["priority_errors"]},
               at3, "sem goals 0.3; substring; clamp [0,1]; TypeError"),
        _check("AT4", {"evicted": payload["evicted"], "tie": payload["tie_spotlight"],
                       "refocus": payload["refocus"], "released_low": payload["released_low"]},
               at4, "capacity 3; vítima = menor saliência; empate = primeiro"),
        _check("AT5", {"order": [row[0] for row in payload["order"]], "tie_order": payload["tie_order"],
                       "missing": shift["missing"], "costs": shift["costs"],
                       "no_recompute_priority": payload["no_recompute"]["priority"],
                       "empty_episodes": payload["empty_attend"]["episodes"]},
               at5, "top capacity estável; id ausente False; cross-modal 0.5"),
        _check("AT6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": AT_SOURCE_SHA, "behavior_golden": AT_BEHAVIOR_SHA},
               at6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
    ]



class _Mem:
    def __init__(self, mode="echo"):
        self.stores = []
        self.retrieves = []
        self.mode = mode

    def store(self, content, tier, importance):
        self.stores.append([content, tier, importance])
        return 1000 + len(self.stores)

    def retrieve(self, query, limit=1):
        self.retrieves.append([query, limit])
        if self.mode == "echo":
            return [{"content": item[0]} for item in self.stores]
        if self.mode == "empty":
            return []
        if self.mode == "tag":
            return [{"content": query}]
        return []


class _Log:
    def __init__(self):
        self.events = []

    def log_event(self, kind, payload):
        self.events.append([kind, payload])


class _Epi:
    def __init__(self, memories):
        self.memories = memories
        self.calls = []

    def get_episode_memories(self, episode_id):
        self.calls.append(episode_id)
        return self.memories


class _Book:
    def __init__(self, episodes):
        self.episodes = episodes
        self.searches = []

    def search_episodes(self, status="closed", limit=10):
        self.searches.append({"status": status, "limit": limit})
        return list(self.episodes)[:limit]

    def get_episode_memories(self, episode_id):
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode.memories
        return []


def _view(event):
    return {
        "episode_id": event.episode_id,
        "memory_id": event.memory_id,
        "content": event.content,
        "old_importance": event.old_importance,
        "new_importance": event.new_importance,
        "promotion": event.promotion,
        "reason": event.reason,
        "timestamp_is_float": isinstance(event.timestamp, float),
    }


def _run(mod, memories, log=None, mode="echo"):
    memory = _Mem(mode=mode)
    learner = mod.RealIncrementalLearner(_Epi(memories), memory, log)
    events = learner.learn_from_episode("ep")
    return learner, memory, [_view(event) for event in events]


def _typed(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return None


def incremental_fingerprint(mod):
    """Protocolo IL6. O dict, não o texto-fonte, entra no SHA comportamental.

    Timestamps e duration numérica ficam de fora: vêm de time.time().
    """
    import inspect
    from types import SimpleNamespace

    learner_cls = mod.RealIncrementalLearner
    hyper = {
        "promotion": learner_cls.PROMOTION_RELEVANCE_THRESHOLD,
        "boost": learner_cls.REINFORCEMENT_IMPORTANCE_BOOST,
        "cap": learner_cls.IMPORTANCE_CAP,
        "decay": learner_cls.DECAY_RATE,
        "pattern_freq": learner_cls.PATTERN_MIN_FREQ,
        "pattern_importance": learner_cls.PATTERN_IMPORTANCE,
    }
    init_params = list(inspect.signature(learner_cls.__init__).parameters)
    clock = "wall" if "clock" not in init_params and "now" not in init_params else "injected"

    promo_learner, promo_mem, promo_events = _run(mod, [{
        "id": 7, "content": "alpha fact", "relevance_to_episode": 0.75, "importance": 0.8,
    }])
    promo_stores = [list(row) for row in promo_mem.stores]
    promo_stats = promo_learner.get_statistics()
    promo_ids = sorted(promo_learner.promoted_ids)
    again = promo_learner.learn_from_episode("ep")
    below_learner, below_mem, below_events = _run(mod, [{
        "id": 7, "content": "alpha fact", "relevance_to_episode": 0.749, "importance": 0.4,
    }])
    _cap_learner, cap_mem, cap_events = _run(mod, [{
        "id": 7, "content": "alpha fact", "relevance_to_episode": 0.9, "importance": 0.9,
    }])
    none_learner, none_mem, none_events = _run(mod, [
        {"id": None, "content": "ghost", "relevance_to_episode": 0.9, "importance": 0.9},
        {"content": "also", "relevance_to_episode": 1.0, "importance": 0.9},
    ])
    long_content = "L" * 100
    long_learner, long_mem, long_events = _run(mod, [{
        "id": 3, "content": long_content, "relevance_to_episode": 0.8, "importance": 0.5,
    }])
    _zero_learner, zero_mem, zero_events = _run(mod, [{
        "id": 0, "content": "zero", "relevance_to_episode": 0.8, "importance": 0.5,
    }])
    _missing_imp_learner, missing_imp_mem, missing_imp_events = _run(mod, [{
        "id": 4, "content": "no-imp", "relevance_to_episode": 0.8,
    }])
    missing_rel_learner, missing_rel_mem, missing_rel_events = _run(mod, [{
        "id": 5, "content": "no-rel", "importance": 0.9,
    }])
    _missing_content_learner, missing_content_mem, missing_content_events = _run(mod, [{
        "id": 6, "relevance_to_episode": 0.8, "importance": 0.5,
    }])

    def boundary(relevance, importance):
        _learner, memory, events = _run(mod, [{
            "id": 1, "content": "r", "relevance_to_episode": relevance, "importance": importance,
        }])
        return {"events": events, "stores": memory.stores, "promoted": sorted(_learner.promoted_ids)}

    preset_mem = _Mem()
    preset = learner_cls(_Epi([{
        "id": 9, "content": "z", "relevance_to_episode": 0.9, "importance": 0.8,
    }]), preset_mem, None)
    preset.promoted_ids.add(9)
    preset_events = [_view(event) for event in preset.learn_from_episode("ep")]

    freq1_learner, freq1_mem, _freq1_events = _run(mod, [{
        "id": 1, "content": "Repeat Me", "relevance_to_episode": 0.1, "importance": 0.1,
    }])
    freq1_new = freq1_learner.detect_and_store_patterns()
    freq1_stores = [list(row) for row in freq1_mem.stores]
    freq1_retrieves = [list(row) for row in freq1_mem.retrieves]
    freq1_learner.episodic.memories = [{
        "id": 2, "content": "  REPEAT ME  ", "relevance_to_episode": 0.1, "importance": 0.1,
    }]
    freq1_learner.learn_from_episode("ep2")
    own_new = freq1_learner.detect_and_store_patterns()
    own_again = freq1_learner.detect_and_store_patterns()

    pattern_log = _Log()
    pattern_mem = _Mem(mode="empty")
    pattern_learner = learner_cls(_Epi([{
        "id": 1, "content": "Repeat Me", "relevance_to_episode": 0.1, "importance": 0.1,
    }]), pattern_mem, pattern_log)
    pattern_learner.learn_from_episode("a")
    pattern_learner.episodic.memories = [{
        "id": 2, "content": "repeat me", "relevance_to_episode": 0.1, "importance": 0.1,
    }]
    pattern_learner.learn_from_episode("b")
    logged_new = pattern_learner.detect_and_store_patterns()
    tag_mem = _Mem(mode="tag")
    tag_learner = learner_cls(_Epi([{
        "id": 1, "content": "repeat me", "relevance_to_episode": 0.1, "importance": 0.1,
    }]), tag_mem, _Log())
    tag_learner.learn_from_episode("a")
    tag_learner.learn_from_episode("b")
    tag_new = tag_learner.detect_and_store_patterns()

    plan_learner, plan_mem, _ = _run(mod, [])
    plan_none = plan_learner.learn_from_plan(None)
    plan_low = plan_learner.learn_from_plan(SimpleNamespace(
        expected_reward=0.699, goal="g", best_action_sequence=["a"],
    ))
    plan_missing = plan_learner.learn_from_plan(SimpleNamespace(goal="g", best_action_sequence=["a"]))
    plan_log = _Log()
    plan_ok_mem = _Mem()
    plan_ok = learner_cls(_Epi([]), plan_ok_mem, plan_log)
    plan_event = plan_ok.learn_from_plan(SimpleNamespace(
        expected_reward=0.7, goal="G" * 80, best_action_sequence=["look", "go"],
    ))
    plan_plain_mem = _Mem()
    plan_plain = learner_cls(_Epi([]), plan_plain_mem, None)
    plan_high = plan_plain.learn_from_plan(SimpleNamespace(
        expected_reward=1.2, goal="g", best_action_sequence=[],
    ))
    plan_bool = plan_plain.learn_from_plan(SimpleNamespace(
        expected_reward=True, goal="g", best_action_sequence=["a"],
    ))
    plan_false = plan_plain.learn_from_plan(SimpleNamespace(
        expected_reward=False, goal="g", best_action_sequence=["a"],
    ))
    plan_nan = plan_plain.learn_from_plan(SimpleNamespace(
        expected_reward=float("nan"), goal="g", best_action_sequence=["a"],
    ))

    decay_plain = learner_cls(_Epi([]), _Mem(), None)
    decay_no_log = {"returned": decay_plain.apply_decay({1}), "stores": decay_plain.memory.stores}
    decay_empty_log = _Log()
    decay_empty_learner = learner_cls(_Epi([]), _Mem(), decay_empty_log)
    decay_empty = {
        "returned": decay_empty_learner.apply_decay(set()),
        "log": decay_empty_log.events,
        "stores": decay_empty_learner.memory.stores,
    }
    decay_counts = {}
    for count in (1, 4, 5, 6):
        log = _Log()
        holder = learner_cls(_Epi([]), _Mem(), log)
        decay_counts[str(count)] = {
            "returned": holder.apply_decay(set(range(count))),
            "log": log.events,
            "stores": holder.memory.stores,
        }

    episodes = [
        SimpleNamespace(episode_id="ep-a", memories=[{
            "id": 1, "content": "solo", "relevance_to_episode": 0.8, "importance": 0.5,
        }]),
        SimpleNamespace(episode_id="ep-b", memories=[{
            "id": 2, "content": "solo", "relevance_to_episode": 0.6, "importance": 0.8,
        }]),
        SimpleNamespace(episode_id="ep-c", memories=[{
            "id": 3, "content": "quiet", "relevance_to_episode": 0.1, "importance": 0.1,
        }]),
    ]

    def consolidate(episodes_in, last_n=10, patterns=True, decay=True, log=True):
        book = _Book(episodes_in)
        memory = _Mem()
        logger = _Log() if log else None
        holder = learner_cls(book, memory, logger)
        report = holder.consolidate_learning(
            last_n_episodes=last_n, run_pattern_detection=patterns, run_decay=decay,
        ) if last_n is not None else holder.consolidate_learning()
        return {
            "episodes_processed": report.episodes_processed,
            "memories_promoted": report.memories_promoted,
            "memories_reinforced": report.memories_reinforced,
            "patterns_detected": report.patterns_detected,
            "total_learning_events": report.total_learning_events,
            "duration_ok": (
                isinstance(report.duration, float)
                and report.duration >= 0
                and report.duration == round(report.duration, 4)
            ),
            "stores": memory.stores,
            "retrieves": memory.retrieves,
            "searches": book.searches,
            "log": [] if logger is None else logger.events,
            "stats": holder.get_statistics(),
        }

    full = consolidate(episodes, last_n=10, patterns=True, decay=True, log=True)
    flags_off = consolidate(episodes, last_n=2, patterns=False, decay=False, log=True)
    no_log = consolidate(episodes, last_n=10, patterns=True, decay=True, log=False)
    empty = consolidate([], last_n=None, patterns=True, decay=True, log=True)

    def gap(item):
        try:
            _learner, memory, events = _run(mod, [item])
        except Exception as exc:
            return {"err": type(exc).__name__}
        return {"events": events, "stores": memory.stores, "patterns": dict(_learner.pattern_registry)}

    errors = {
        "str_relevance": _typed(lambda: _run(mod, [{
            "id": 1, "content": "b", "relevance_to_episode": "0.9", "importance": 0.5,
        }])),
        "str_importance_promotion": _typed(lambda: _run(mod, [{
            "id": 1, "content": "b", "relevance_to_episode": 0.9, "importance": "0.5",
        }])),
        "str_importance_reinforcement": _typed(lambda: _run(mod, [{
            "id": 1, "content": "b", "relevance_to_episode": 0.6, "importance": "0.8",
        }])),
        "plan_str_reward": _typed(lambda: learner_cls(_Epi([]), _Mem(), None).learn_from_plan(
            SimpleNamespace(expected_reward="0.9", goal="g", best_action_sequence=["a"])
        )),
        "plan_missing_sequence": _typed(lambda: learner_cls(_Epi([]), _Mem(), None).learn_from_plan(
            SimpleNamespace(expected_reward=0.9, goal="g")
        )),
        "plan_goal_none": _typed(lambda: learner_cls(_Epi([]), _Mem(), None).learn_from_plan(
            SimpleNamespace(expected_reward=0.9, goal=None, best_action_sequence=["a"])
        )),
        "memory_item_str": _typed(lambda: _run(mod, ["nope"])),
    }

    return {
        "hyper": hyper,
        "clock": clock,
        "init_params": init_params,
        "promotion": {
            "events": promo_events,
            "stores": promo_stores,
            "promoted": promo_ids,
            "stats": promo_stats,
            "again": [_view(event) for event in again],
            "again_stores": [list(row) for row in promo_mem.stores],
            "again_promoted": sorted(promo_learner.promoted_ids),
            "again_stats": promo_learner.get_statistics(),
        },
        "below": {"events": below_events, "stores": below_mem.stores, "patterns": dict(below_learner.pattern_registry)},
        "cap": {"events": cap_events, "stores": cap_mem.stores},
        "skip_none": {
            "events": none_events,
            "stores": none_mem.stores,
            "patterns": dict(none_learner.pattern_registry),
        },
        "long": {
            "event_content": long_events[0]["content"],
            "stored": long_mem.stores,
            "key": next(iter(long_learner.pattern_registry)),
        },
        "zero_id": {"events": zero_events, "stores": zero_mem.stores},
        "defaults": {
            "missing_importance": {"events": missing_imp_events, "stores": missing_imp_mem.stores},
            "missing_relevance": {
                "events": missing_rel_events,
                "stores": missing_rel_mem.stores,
                "patterns": dict(missing_rel_learner.pattern_registry),
            },
            "missing_content": {"events": missing_content_events, "stores": missing_content_mem.stores},
        },
        "boundaries": {
            "rel_0_5_imp_0_7": boundary(0.5, 0.7),
            "rel_0_5_imp_under": boundary(0.5, 0.699999999999),
            "rel_under": boundary(0.499, 0.9),
            "reinforce_cap": boundary(0.6, 0.95),
            "just_under_promotes_not": boundary(0.749, 0.8),
            "both_would_match": boundary(0.8, 0.8),
        },
        "preset_promoted": {"events": preset_events, "stores": preset_mem.stores, "promoted": sorted(preset.promoted_ids)},
        "patterns": {
            "freq1_new": freq1_new,
            "freq1_stores": freq1_stores,
            "freq1_retrieves": freq1_retrieves,
            "registry": dict(freq1_learner.pattern_registry),
            "own_synthetic_first": own_new,
            "own_synthetic_second": own_again,
            "own_stores": freq1_mem.stores,
            "logged_new": logged_new,
            "logged_stores": pattern_mem.stores,
            "logged_events": pattern_log.events,
            "tag_suppresses": tag_new,
            "tag_stores": tag_mem.stores,
            "tag_patterns": tag_learner.total_patterns,
        },
        "plans": {
            "none": plan_none is None and plan_mem.stores == [] and plan_learner.total_promotions == 0,
            "low": plan_low is None and plan_mem.stores == [],
            "missing_reward": plan_missing is None and plan_mem.stores == [],
            "ok": None if plan_event is None else _view(plan_event),
            "ok_store": plan_ok_mem.stores,
            "ok_promoted": sorted(plan_ok.promoted_ids),
            "ok_stats": plan_ok.get_statistics(),
            "ok_log": plan_log.events,
            "high": None if plan_high is None else _view(plan_high),
            "high_store": plan_plain_mem.stores[0],
            "bool_true": None if plan_bool is None else _view(plan_bool),
            "bool_false": plan_false is None,
            "nan": None if plan_nan is None else _view(plan_nan),
            "plain_stores": plan_plain_mem.stores,
        },
        "decay": {"no_log": decay_no_log, "empty": decay_empty, "counts": decay_counts},
        "consolidate": {"full": full, "flags_off": flags_off, "no_log": no_log, "empty": empty},
        "gaps": {
            "bool_true": gap({"id": 1, "content": "b", "relevance_to_episode": True, "importance": 0.5}),
            "bool_false": gap({"id": 1, "content": "b", "relevance_to_episode": False, "importance": 0.9}),
            "bool_importance": gap({"id": 1, "content": "b", "relevance_to_episode": 0.6, "importance": True}),
            "nan_relevance": gap({"id": 1, "content": "b", "relevance_to_episode": float("nan"), "importance": 0.9}),
            "nan_importance": gap({"id": 1, "content": "b", "relevance_to_episode": 0.9, "importance": float("nan")}),
            "inf_relevance": gap({"id": 1, "content": "b", "relevance_to_episode": float("inf"), "importance": 0.5}),
            "ninf_importance": gap({"id": 1, "content": "b", "relevance_to_episode": 0.9, "importance": float("-inf")}),
            "negative_relevance": gap({"id": 1, "content": "b", "relevance_to_episode": -1, "importance": 0.9}),
        },
        "errors": errors,
    }


IL_SOURCE_SHA = "e09c2d75df964dd52915eaf0c0660dd45a6ac34cbf476804c786423910057542"
IL_BEHAVIOR_SHA = "636c95bfaebbd60ca55a5e14f12e9f5ff7b6f7868f5a476b31fa479528db7767"
_IL_MARKER = "# MÓDULO 11: INCREMENTAL LEARNER (POST12)\n"
_IL_END = '            "pattern_registry_size": len(self.pattern_registry),\n        }\n'
_IL_NAMES = ("LearningEvent", "ConsolidationReport", "RealIncrementalLearner")


def incremental_source_text(src: str) -> str:
    idx = src.index(_IL_MARKER)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end = src.index(_IL_END, idx) + len(_IL_END)
    return src[line_start:end]


def _il_sha(payload) -> str:
    import hashlib
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _incremental_source_sha() -> str:
    import hashlib
    text = (ROOT / "core/cognitivo/incremental_learner.py").read_text(encoding="utf-8")
    return hashlib.sha256(incremental_source_text(text).encode()).hexdigest()


def _incremental_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha IL6.

    Menção no docstring não conta. O bloqueio em runtime fica no contrato, fora de tools/.
    """
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/incremental_learner.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" or name.startswith("core.constitutional_brain."):
                return False
    return True


def eval_incremental(criteria="v1"):
    """IL1–IL6. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import core.cognitivo.incremental_learner as incremental
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = incremental_fingerprint(incremental)
    sha_module = _il_sha(payload)
    sha_brain = _il_sha(incremental_fingerprint(brain))
    source_sha = _incremental_source_sha()
    standalone = _incremental_standalone()
    identity = all(
        getattr(brain, name) is getattr(incremental, name) is getattr(package, name)
        and getattr(incremental, name).__module__ == "core.cognitivo.incremental_learner"
        for name in _IL_NAMES
    )
    promo = payload["promotion"]
    event = promo["events"][0]
    again = promo["again"][0]
    il1 = (
        payload["hyper"]["promotion"] == 0.75
        and payload["hyper"]["cap"] == 0.99
        and event["promotion"] is True
        and event["memory_id"] == 7
        and event["content"] == "alpha fact"
        and event["old_importance"] == 0.8
        and event["new_importance"] == min(0.99, 0.8 * 1.15)
        and event["reason"] == "relevance=0.75 >= 0.75"
        and event["timestamp_is_float"] is True
        and promo["stores"] == [["alpha fact", "long_term", event["new_importance"]]]
        and promo["promoted"] == [7]
        and promo["stats"]["total_promotions"] == 1
        and promo["stats"]["total_reinforcements"] == 0
        and promo["stats"]["pattern_registry_size"] == 1
        and again["promotion"] is False
        and again["new_importance"] == min(0.99, 0.8 + 0.08)
        and promo["again_promoted"] == [7]
        and promo["again_stats"]["total_promotions"] == 1
        and promo["again_stats"]["total_reinforcements"] == 1
        and payload["below"]["events"] == []
        and payload["below"]["stores"] == []
        and payload["below"]["patterns"] == {"alpha fact": 1}
        and payload["cap"]["events"][0]["new_importance"] == 0.99
        and payload["cap"]["events"][0]["promotion"] is True
        and payload["skip_none"]["events"] == []
        and payload["skip_none"]["stores"] == []
        and payload["skip_none"]["patterns"] == {}
        and payload["zero_id"]["events"][0]["memory_id"] == 0
        and payload["zero_id"]["events"][0]["promotion"] is True
        and len(payload["long"]["event_content"]) == 80
        and len(payload["long"]["stored"][0][0]) == 100
        and payload["long"]["key"] == "l" * 60
        and payload["long"]["stored"][0][1] == "long_term"
        and payload["defaults"]["missing_importance"]["events"][0]["old_importance"] == 0.5
        and payload["defaults"]["missing_relevance"]["events"] == []
        and payload["defaults"]["missing_relevance"]["patterns"] == {"no-rel": 1}
        and payload["defaults"]["missing_content"]["events"][0]["content"] == ""
    )
    bounds = payload["boundaries"]
    il2 = (
        payload["hyper"]["boost"] == 0.08
        and bounds["rel_0_5_imp_0_7"]["events"][0]["promotion"] is False
        and bounds["rel_0_5_imp_0_7"]["events"][0]["new_importance"] == min(0.99, 0.7 + 0.08)
        and bounds["rel_0_5_imp_0_7"]["stores"] == [[
            "r", "long_term", bounds["rel_0_5_imp_0_7"]["events"][0]["new_importance"],
        ]]
        and bounds["rel_0_5_imp_0_7"]["promoted"] == []
        and bounds["rel_0_5_imp_under"]["events"] == []
        and bounds["rel_under"]["events"] == []
        and bounds["reinforce_cap"]["events"][0]["new_importance"] == 0.99
        and bounds["reinforce_cap"]["events"][0]["promotion"] is False
        and bounds["just_under_promotes_not"]["events"][0]["promotion"] is False
        and bounds["just_under_promotes_not"]["events"][0]["reason"] == "reinforcement (old_imp=0.80, rel=0.75)"
        and bounds["just_under_promotes_not"]["promoted"] == []
        and len(bounds["both_would_match"]["events"]) == 1
        and bounds["both_would_match"]["events"][0]["promotion"] is True
        and bounds["both_would_match"]["promoted"] == [1]
        and payload["preset_promoted"]["events"][0]["promotion"] is False
        and payload["preset_promoted"]["promoted"] == [9]
        and payload["preset_promoted"]["events"][0]["new_importance"] == min(0.99, 0.8 + 0.08)
    )
    patterns = payload["patterns"]
    il3 = (
        payload["hyper"]["pattern_freq"] == 2
        and payload["hyper"]["pattern_importance"] == 0.92
        and patterns["freq1_new"] == 0
        and patterns["freq1_stores"] == []
        and patterns["freq1_retrieves"] == []
        and patterns["registry"] == {"repeat me": 2}
        and patterns["own_synthetic_first"] == 1
        and patterns["own_synthetic_second"] == 1
        and patterns["own_stores"] == [
            ["[PATTERN freq=2] repeat me", "long_term", 0.92],
            ["[PATTERN freq=2] repeat me", "long_term", 0.92],
        ]
        and patterns["logged_new"] == 1
        and patterns["logged_stores"] == [["[PATTERN freq=2] repeat me", "long_term", 0.92]]
        and patterns["logged_events"] == [["PATTERN_DETECTED", {"content": "repeat me", "frequency": 2}]]
        and patterns["tag_suppresses"] == 0
        and patterns["tag_stores"] == []
        and patterns["tag_patterns"] == 0
    )
    plans = payload["plans"]
    ok = plans["ok"]
    il4 = (
        plans["none"] is True
        and plans["low"] is True
        and plans["missing_reward"] is True
        and ok["promotion"] is True
        and ok["episode_id"] == "mcts_direct"
        and ok["memory_id"] == 1001
        and ok["old_importance"] == 0.0
        and ok["new_importance"] == min(0.99, 0.6 + 0.7 * 0.35)
        and ok["content"] == plans["ok_store"][0][0][:80]
        and len(ok["content"]) == 80
        and plans["ok_store"][0][1] == "long_term"
        and len(plans["ok_store"][0][0]) == 113
        and plans["ok_promoted"] == []
        and plans["ok_stats"]["total_promotions"] == 1
        and plans["ok_stats"]["pattern_registry_size"] == 0
        and plans["ok_log"][0][0] == "MCTS_PLAN_LEARNED"
        and plans["ok_log"][0][1]["reward"] == 0.7
        and plans["ok_log"][0][1]["sequence"] == ["look", "go"]
        and len(plans["ok_log"][0][1]["goal"]) == 80
        and plans["high"]["new_importance"] == 0.99
        and plans["high_store"][0].endswith("Sequence: ")
        and payload["errors"]["str_relevance"] == "TypeError"
        and payload["errors"]["str_importance_promotion"] == "TypeError"
        and payload["errors"]["str_importance_reinforcement"] == "TypeError"
        and payload["errors"]["plan_str_reward"] == "TypeError"
        and payload["errors"]["plan_missing_sequence"] == "AttributeError"
        and payload["errors"]["plan_goal_none"] == "TypeError"
        and payload["errors"]["memory_item_str"] == "AttributeError"
    )
    full = payload["consolidate"]["full"]
    flags_off = payload["consolidate"]["flags_off"]
    no_log = payload["consolidate"]["no_log"]
    empty = payload["consolidate"]["empty"]
    decay = payload["decay"]
    il5 = (
        full["episodes_processed"] == 3
        and full["memories_promoted"] == 1
        and full["memories_reinforced"] == 1
        and full["patterns_detected"] == 1
        and full["total_learning_events"] == 2
        and full["duration_ok"] is True
        and full["searches"] == [{"status": "closed", "limit": 10}]
        and full["stores"] == [
            ["solo", "long_term", 0.575],
            ["solo", "long_term", 0.88],
            ["[PATTERN freq=2] solo", "long_term", 0.92],
        ]
        and [row[0] for row in full["log"]] == ["PATTERN_DETECTED", "DECAY_APPLIED", "LEARNING_CONSOLIDATION"]
        and full["log"][1][1] == {"accessed_count": 2, "decay_rate": 0.02}
        and full["stats"]["total_consolidations"] == 1
        and full["stats"]["pattern_registry_size"] == 2
        and flags_off["episodes_processed"] == 2
        and flags_off["patterns_detected"] == 0
        and flags_off["retrieves"] == []
        and flags_off["searches"] == [{"status": "closed", "limit": 2}]
        and [row[0] for row in flags_off["log"]] == ["LEARNING_CONSOLIDATION"]
        and flags_off["log"][0][1]["patterns_found"] == 0
        and no_log["log"] == []
        and no_log["stores"][-1][0].startswith("[PATTERN freq=2]")
        and empty["episodes_processed"] == 0
        and empty["searches"] == [{"status": "closed", "limit": 10}]
        and empty["stores"] == []
        and empty["stats"]["total_consolidations"] == 1
        and [row[0] for row in empty["log"]] == ["LEARNING_CONSOLIDATION"]
        and decay["no_log"] == {"returned": 0, "stores": []}
        and decay["empty"]["returned"] == 0
        and decay["empty"]["log"] == []
        and decay["empty"]["stores"] == []
        and decay["counts"]["1"]["returned"] == 4
        and decay["counts"]["4"]["returned"] == 1
        and decay["counts"]["5"]["returned"] == 0
        and decay["counts"]["6"]["returned"] == 0
        and all(decay["counts"][key]["stores"] == [] for key in ("1", "4", "5", "6"))
        and payload["hyper"]["decay"] == 0.02
    )
    il6 = (
        identity
        and standalone
        and source_sha == IL_SOURCE_SHA
        and sha_module == sha_brain == IL_BEHAVIOR_SHA
        and payload["clock"] == "wall"
    )
    return [
        _check("IL1", {"new": event["new_importance"], "again_promotion": again["promotion"],
                       "cap": payload["cap"]["events"][0]["new_importance"],
                       "skipped": payload["skip_none"]["events"], "zero_id": payload["zero_id"]["events"][0]["memory_id"]},
               il1, "promove em 0.75 com teto 0.99; o mesmo id não promove de novo"),
        _check("IL2", {"boosted": bounds["rel_0_5_imp_0_7"]["events"][0]["new_importance"],
                       "both": bounds["both_would_match"]["events"][0]["promotion"],
                       "under": bounds["rel_under"]["events"]},
               il2, "reforço +0.08 só no elif; 0.8/0.8 promove uma vez"),
        _check("IL3", {"registry": patterns["registry"], "own_second": patterns["own_synthetic_second"],
                       "tag": patterns["tag_suppresses"], "logged": patterns["logged_new"]},
               il3, "freq 2 grava 0.92; a tag achada não duplica; o sintético próprio duplica"),
        _check("IL4", {"none": plans["none"], "low": plans["low"], "importance": ok["new_importance"],
                       "errors": payload["errors"]},
               il4, "None e reward<0.7 devolvem None; 0.7 grava com importância 0.845"),
        _check("IL5", {"promoted": full["memories_promoted"], "reinforced": full["memories_reinforced"],
                       "patterns": full["patterns_detected"], "decay_one": decay["counts"]["1"]["returned"],
                       "flags_patterns": flags_off["patterns_detected"]},
               il5, "contagens, flags e decay heurístico sem store"),
        _check("IL6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": IL_SOURCE_SHA, "behavior_golden": IL_BEHAVIOR_SHA},
               il6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
    ]


def _ef_err(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "no-error"


class _EfBoom:
    def __getattr__(self, name):
        raise AssertionError("called " + name)


def _ef_copy_seq(value):
    if isinstance(value, (list, tuple)):
        return type(value)(value)
    return value


def _ef_event(event):
    return {
        "description": event.description,
        "time_point": event.time_point,
        "time_point_type": type(event.time_point).__name__,
        "location": event.location,
        "participants": _ef_copy_seq(event.participants),
        "participants_type": type(event.participants).__name__,
        "expected_emotions": _ef_copy_seq(event.expected_emotions),
        "vividness": event.vividness,
        "plausibility": event.plausibility,
        "personal_significance": event.personal_significance,
        "significance_type": type(event.personal_significance).__name__,
        "event_id_is_uuid": isinstance(event.event_id, str) and len(event.event_id) == 36,
        "created_at_is_float": isinstance(event.created_at, float),
    }


def _ef_intention(item):
    return {
        "what": item.what,
        "what_type": type(item.what).__name__,
        "when": item.when,
        "where": item.where,
        "trigger": item.trigger,
        "trigger_type": type(item.trigger).__name__,
        "importance": item.importance,
        "importance_type": type(item.importance).__name__,
        "fulfilled": item.fulfilled,
        "id_is_uuid": isinstance(item.intention_id, str) and len(item.intention_id) == 36,
    }


def _ef_pre(result):
    sensory = result["sensory_details"]
    return {
        "event_id_is_uuid": isinstance(result["event_id"], str) and len(result["event_id"]) == 36,
        "sensory_details": {
            "visual": sensory["visual"],
            "emotional": _ef_copy_seq(sensory["emotional"]),
            "social": _ef_copy_seq(sensory["social"]),
        },
        "vividness": result["vividness"],
        "simulation_success": result["simulation_success"],
        "keys": sorted(result),
    }


def episodic_future_fingerprint(ns):
    import inspect
    if isinstance(ns, dict):
        cls = ns["RealEpisodicFutureThinkingSystem"]
        future_event = ns["FutureEvent"]
    else:
        cls = ns.RealEpisodicFutureThinkingSystem
        future_event = ns.FutureEvent
    params = list(inspect.signature(cls.__init__).parameters)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh = cls()
    banner = buf.getvalue()

    def quiet(memory=None, emotion=None, simulation=None, consciousness=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(memory, simulation, consciousness, emotion)

    def open_system():
        system = quiet()
        system.set_prospective_memory("keep", "later", where="room", trigger="bell")
        return system

    mem = _EfBoom()
    emo = _EfBoom()
    sim = _EfBoom()
    cons = _EfBoom()
    both = quiet(mem, emo, sim, cons)
    quiet_day = both.imagine_future_event("Quiet day", "tomorrow")
    override = both.imagine_future_event(
        "SUCCESS parade", "tomorrow",
        {"location": "hall", "participants": ["a"], "significance": 0.9},
    )
    challenge = both.imagine_future_event("challenge ahead", "next week")
    quiet_view = _ef_event(quiet_day)
    override_view = _ef_event(override)
    challenge_view = _ef_event(challenge)
    quiet_day.participants.append("x")
    again = both.imagine_future_event("Quiet day", "tomorrow")
    none_sim = quiet(mem, emo, None, None)
    none_view = _ef_event(none_sim.imagine_future_event("Quiet day", "tomorrow"))

    memory_only = quiet(_EfBoom(), None)
    memory_success = _ef_event(memory_only.imagine_future_event("success today", "tomorrow"))
    emotion_only = quiet(None, _EfBoom())
    emotion_cases = {
        name: _ef_event(emotion_only.imagine_future_event(text, "t"))
        for name, text in (
            ("success", "success"),
            ("achievement", "big achievement"),
            ("challenge", "The Challenge"),
            ("meeting", "team meeting"),
            ("empty", ""),
            ("homework", "homework"),
            ("network", "network"),
            ("lunch", "lunch"),
            ("travel", "travel"),
            ("presentation", "presentation"),
            ("workshop", "workshop"),
            ("working", "Working late"),
        )
    }
    neither = quiet(None, None)
    neither_success = _ef_event(neither.imagine_future_event("success today", "tomorrow"))
    falsy = {}
    for label, value in (("empty_str", ""), ("zero", 0), ("list", []), ("false", False), ("none", None)):
        falsy[label] = _ef_event(quiet(value, None).imagine_future_event("success", "t"))
    truthy = {
        "one": _ef_event(quiet(1, None).imagine_future_event("plain", "t")),
        "object": _ef_event(quiet(object(), None).imagine_future_event("plain", "t")),
    }
    plain = quiet(None, None)
    omitted = _ef_event(plain.imagine_future_event("plain", "later"))
    explicit_none = _ef_event(plain.imagine_future_event("plain", "later", None))
    empty_details = _ef_event(plain.imagine_future_event("plain", "later", {}))
    detail_system = quiet(None, None)
    sig0 = _ef_event(detail_system.imagine_future_event("plain", "t", {"significance": 0}))
    sig_false = _ef_event(detail_system.imagine_future_event("plain", "t", {"significance": False}))
    loc_empty = _ef_event(detail_system.imagine_future_event("plain", "t", {"location": ""}))
    parts_none = _ef_event(detail_system.imagine_future_event("plain", "t", {"participants": None}))
    parts_tuple = _ef_event(detail_system.imagine_future_event("plain", "t", {"participants": ("a", "b")}))
    sig_nan = _ef_event(detail_system.imagine_future_event("plain", "t", {"significance": float("nan")}))
    sig_inf = _ef_event(detail_system.imagine_future_event("plain", "t", {"significance": float("inf")}))
    sig_neg = _ef_event(detail_system.imagine_future_event("plain", "t", {"significance": -1}))
    details_zero = _ef_event(detail_system.imagine_future_event("plain", "t", 0))
    details_false = _ef_event(detail_system.imagine_future_event("plain", "t", False))
    details_blank = _ef_event(detail_system.imagine_future_event("plain", "t", ""))
    time_int = _ef_event(detail_system.imagine_future_event("plain", 7))
    stored_same = both.imagined_events[0] is quiet_day

    pre = quiet(_EfBoom(), _EfBoom())
    event = pre.imagine_future_event("Quiet day", "tomorrow")
    before = _ef_event(event)
    first = pre.preexperience_event(event)
    second = pre.preexperience_event(event)
    after = _ef_event(event)

    empty_stats = quiet().get_statistics()
    stats_system = quiet(_EfBoom(), None)
    stats_system.imagine_future_event("a", "t")
    stats_system.imagine_future_event("b", "t")
    stats_system.imagined_events[1].vividness = 0.25
    stats_system.preexperience_event(stats_system.imagined_events[0])
    two_stats = stats_system.get_statistics()

    prospective = quiet()
    default_item = prospective.set_prospective_memory("call", "tomorrow")
    blank_trigger = prospective.set_prospective_memory("x", "tonight", trigger="")
    explicit = prospective.set_prospective_memory(
        "review", "monday", where="lab", trigger="bell", importance=0.8,
    )
    none_trigger = prospective.set_prospective_memory("n", "soon", trigger=None)
    zero_trigger = prospective.set_prospective_memory("z", "soon", trigger=0)
    none_what = prospective.set_prospective_memory(None, "soon")
    importance_cases = {
        "high": _ef_intention(prospective.set_prospective_memory("h", "t", importance=1.5)),
        "negative": _ef_intention(prospective.set_prospective_memory("n", "t", importance=-1)),
        "true": _ef_intention(prospective.set_prospective_memory("b", "t", importance=True)),
        "nan": _ef_intention(prospective.set_prospective_memory("q", "t", importance=float("nan"))),
    }
    same_intention = prospective.prospective_memories[0] is default_item

    triggers = quiet()
    first_intention = triggers.set_prospective_memory(
        "call Ana", "tomorrow", where="office", trigger="phone", importance=0.9,
    )
    second_intention = triggers.set_prospective_memory(
        "send notes", "today", where="", trigger="email", importance=0.2,
    )
    done = triggers.set_prospective_memory(
        "already", "tomorrow", where="lab", trigger="bell", importance=0.4,
    )
    day_item = triggers.set_prospective_memory("day-match", "day", where="nowhere", trigger="nope")
    triggers.fulfill_intention(done.intention_id)
    contexts = {
        "empty": {},
        "time_tomorrow": {"time": "tomorrow"},
        "time_today": {"time": "today"},
        "location_office": {"location": "office"},
        "location_home_office": {"location": "home office"},
        "event_phone": {"event": "phone call"},
        "event_email": {"event": "email"},
        "blank_strings": {"time": "", "location": "", "event": ""},
        "zeros": {"time": 0, "location": 0, "event": 0},
        "all_ana": {"time": "tomorrow", "location": "office", "event": "phone"},
    }
    triggered = {}
    same_object = {}
    for name, context in contexts.items():
        found = triggers.check_prospective_triggers(context)
        triggered[name] = [item.what for item in found]
        same_object[name] = all(item in triggers.prospective_memories for item in found)

    fulfill = quiet()
    missing_return = fulfill.fulfill_intention("missing")
    missing_counter = fulfill.fulfilled_intentions
    kept = fulfill.set_prospective_memory("keep", "later", where="room", trigger="bell")
    other = fulfill.set_prospective_memory("other", "later", where="room", trigger="bell")
    once_return = fulfill.fulfill_intention(kept.intention_id)
    once = {
        "return": once_return is None,
        "fulfilled": kept.fulfilled,
        "other_open": other.fulfilled is False,
        "stats": fulfill.get_statistics(),
    }
    twice_return = fulfill.fulfill_intention(kept.intention_id)
    twice = {"return": twice_return is None, "stats": fulfill.get_statistics(), "flag": kept.fulfilled}
    lone = quiet()
    lone_item = lone.set_prospective_memory("only", "later")
    lone.fulfill_intention(lone_item.intention_id)
    lone.fulfill_intention(lone_item.intention_id)
    lone_stats = lone.get_statistics()

    horizons = {}
    growth = {}
    for horizon in (
        "1 year", "YEAR", "yearly", "next month", "Month", "1 year and 1 month",
        "tomorrow", "", "week", "monthly",
    ):
        projected = quiet().project_future_self(horizon)
        horizons[horizon] = projected["projected_changes"]
        growth[horizon] = {
            "growth": projected["anticipated_growth"],
            "continuity": projected["continuity_with_current_self"],
            "horizon": projected["time_horizon"],
        }

    left = future_event("a", "d", "t", "loc", [], [], 0.9, 0.4, 0.2, 1.0)
    right = future_event("b", "d", "t", "loc", [], [], 0.3, 0.8, 0.5, 1.0)
    tied = future_event("c", "d", "t", "loc", [], [], 0.5, 0.4, 0.4, 1.0)
    tied_other = future_event("d", "d", "t", "loc", [], [], 0.5, 0.4, 0.5, 1.0)
    product_tie = future_event("e", "d", "t", "loc", [], [], 0.4, 0.1, 0.5, 1.0)
    nan_left = future_event("n", "d", "t", "loc", [], [], float("nan"), 0.2, 0.9, 1.0)
    bool_left = future_event("t", "d", "t", "loc", [], [], 0.4, 0.2, True, 1.0)
    bool_right = future_event("u", "d", "t", "loc", [], [], 0.4, 0.2, False, 1.0)
    cmp_system = quiet()

    def cmp(one_event, other_event):
        return cmp_system.compare_future_scenarios(one_event, other_event)

    int_with_both = quiet(_EfBoom(), _EfBoom())
    int_event = _ef_event(int_with_both.imagine_future_event(1, "t"))

    return {
        "clock": "wall" if "clock" not in params else "injected",
        "init_params": params,
        "banner": banner,
        "fields": {
            "future": [field for field in future_event.__dataclass_fields__],
            "prospective": [field for field in (
                ns["ProspectiveMemory"] if isinstance(ns, dict) else ns.ProspectiveMemory
            ).__dataclass_fields__],
        },
        "wired": {
            "memory": quiet(mem, emo, sim, cons).memory is mem,
            "mental_simulation": quiet(mem, emo, sim, cons).mental_simulation is sim,
            "consciousness": quiet(mem, emo, sim, cons).consciousness is cons,
            "emotion": quiet(mem, emo, sim, cons).emotion is emo,
            "counters": [fresh.total_events_imagined, fresh.total_preexperiences, fresh.fulfilled_intentions],
            "empty_lists": fresh.imagined_events == [] and fresh.prospective_memories == [],
        },
        "both": {
            "quiet": quiet_view,
            "override": override_view,
            "challenge": challenge_view,
            "again_participants": again.participants,
            "ids_differ": len({quiet_day.event_id, override.event_id, challenge.event_id, again.event_id}) == 4,
            "count": both.total_events_imagined,
            "stored_same": stored_same,
            "simulation_ignored": none_view == _ef_event(quiet(mem, emo).imagine_future_event("Quiet day", "tomorrow")),
        },
        "memory_only": memory_success,
        "emotion_only": emotion_cases,
        "neither": neither_success,
        "falsy_memory": falsy,
        "truthy_memory": truthy,
        "details": {
            "omitted": omitted,
            "none": explicit_none,
            "empty": empty_details,
            "views_equal": omitted == explicit_none == empty_details,
            "significance_zero": sig0["personal_significance"],
            "significance_false": sig_false["personal_significance"],
            "significance_false_type": sig_false["significance_type"],
            "location_empty": loc_empty["location"],
            "participants_none": parts_none["participants"],
            "participants_tuple": parts_tuple["participants"],
            "participants_tuple_type": parts_tuple["participants_type"],
            "nan": sig_nan["personal_significance"],
            "inf": sig_inf["personal_significance"],
            "negative": sig_neg["personal_significance"],
            "details_zero": details_zero,
            "details_false": details_false,
            "details_blank": details_blank,
            "time_int": time_int["time_point"],
            "time_int_type": time_int["time_point_type"],
        },
        "preexperience": {
            "first": _ef_pre(first),
            "second": _ef_pre(second),
            "same_event_id": first["event_id"] == second["event_id"] == event.event_id,
            "count": pre.total_preexperiences,
            "imagined": pre.total_events_imagined,
            "unchanged": before == after,
        },
        "stats": {"empty": empty_stats, "two": two_stats},
        "prospective": {
            "default": _ef_intention(default_item),
            "blank_trigger": _ef_intention(blank_trigger),
            "explicit": _ef_intention(explicit),
            "none_trigger": _ef_intention(none_trigger),
            "zero_trigger": _ef_intention(zero_trigger),
            "none_what": _ef_intention(none_what),
            "importance": importance_cases,
            "same_object": same_intention,
            "default_trigger_is_when": default_item.trigger == default_item.when,
        },
        "triggers": {
            "found": triggered,
            "same_object": same_object,
            "fulfilled_skipped": done.what not in triggered["time_tomorrow"],
            "open": [first_intention.fulfilled, second_intention.fulfilled, day_item.fulfilled],
        },
        "fulfill": {
            "missing_return_none": missing_return is None,
            "missing_counter": missing_counter,
            "once": once,
            "twice": twice,
            "lone_double": lone_stats,
        },
        "project": {"changes": horizons, "growth": growth},
        "compare": {
            "differ": cmp(left, right),
            "vivid_tie": cmp(tied, tied_other),
            "product_tie": cmp(tied, product_tie),
            "nan": cmp(nan_left, right),
            "bool": cmp(bool_left, bool_right),
        },
        "gaps": {
            "int_description_with_memory_and_emotion": int_event,
            "memory_not_called": both.imagined_events[0].location == "familiar_place",
        },
        "errors": {
            "desc_int": _ef_err(lambda: quiet(None, None).imagine_future_event(1, "t")),
            "details_list": _ef_err(lambda: quiet().imagine_future_event("plain", "t", [1])),
            "details_true": _ef_err(lambda: quiet().imagine_future_event("plain", "t", True)),
            "horizon_int": _ef_err(lambda: quiet().project_future_self(1)),
            "horizon_none": _ef_err(lambda: quiet().project_future_self(None)),
            "context_none_empty": _ef_err(lambda: quiet().check_prospective_triggers(None)),
            "context_str_empty": _ef_err(lambda: quiet().check_prospective_triggers("time")),
            "context_list_empty": _ef_err(lambda: quiet().check_prospective_triggers([])),
            "context_none": _ef_err(lambda: open_system().check_prospective_triggers(None)),
            "context_str": _ef_err(lambda: open_system().check_prospective_triggers("time")),
            "context_list": _ef_err(lambda: open_system().check_prospective_triggers([])),
            "context_omitted": _ef_err(lambda: open_system().check_prospective_triggers()),
            "preexperience_none": _ef_err(lambda: quiet().preexperience_event(None)),
            "compare_missing": _ef_err(lambda: quiet().compare_future_scenarios(
                type("E", (), {"event_id": "a", "vividness": 0.1, "plausibility": 0.1})(),
                right,
            )),
            "compare_str_significance": _ef_err(lambda: quiet().compare_future_scenarios(
                future_event("s", "d", "t", "loc", [], [], 0.2, 0.2, "high", 1.0),
                right,
            )),
        },
    }


EF_SOURCE_SHA = "a72a34bcc303f98b37cd5e33e030ab69bea8579d4f111c95981b2cf9785cb3e7"
EF_BEHAVIOR_SHA = "062870d6a044174d49150fc6008521105c05557c40c2f46a4e83cd116b761f27"
_EF_MARKER = "# MÓDULO 31: EPISODIC FUTURE THINKING SYSTEM (POST 31)\n"
_EF_END = (
    '            "avg_event_vividness": sum(e.vividness for e in self.imagined_events) '
    "/ max(1, len(self.imagined_events))\n        }\n"
)
_EF_NAMES = ("FutureEvent", "ProspectiveMemory", "RealEpisodicFutureThinkingSystem")
_EF_YEAR = ["increased_experience", "new_skills", "evolved_goals"]
_EF_MONTH = ["incremental_progress", "learning"]
_EF_MINOR = ["minor_changes"]


def episodic_future_source_text(src: str) -> str:
    idx = src.index(_EF_MARKER)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end = src.index(_EF_END, idx) + len(_EF_END)
    return src[line_start:end]


def _ef_sha(payload) -> str:
    import hashlib
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _episodic_future_source_sha() -> str:
    import hashlib
    text = (ROOT / "core/cognitivo/episodic_future_thinking.py").read_text(encoding="utf-8")
    return hashlib.sha256(episodic_future_source_text(text).encode()).hexdigest()


def _episodic_future_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha EF6.

    Menção no docstring não conta. O bloqueio em runtime fica no contrato, fora de tools/.
    """
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/episodic_future_thinking.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" or name.startswith("core.constitutional_brain."):
                return False
    return True


def eval_episodic_future(criteria="v1"):
    """EF1–EF6. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import math

    import core.cognitivo.episodic_future_thinking as episodic_future
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = episodic_future_fingerprint(episodic_future)
    sha_module = _ef_sha(payload)
    sha_brain = _ef_sha(episodic_future_fingerprint(brain))
    source_sha = _episodic_future_source_sha()
    standalone = _episodic_future_standalone()
    identity = all(
        getattr(brain, name) is getattr(episodic_future, name) is getattr(package, name)
        and getattr(episodic_future, name).__module__ == "core.cognitivo.episodic_future_thinking"
        for name in _EF_NAMES
    )
    both = payload["both"]
    quiet = both["quiet"]
    emotions = payload["emotion_only"]
    details = payload["details"]
    common = ("homework", "network", "workshop", "working", "lunch", "travel", "presentation")
    ef1 = (
        quiet["location"] == "familiar_place"
        and quiet["participants"] == ["friend", "colleague"]
        and quiet["expected_emotions"] == ["happy", "excited"]
        and quiet["vividness"] == min(1.0, 0.3 + 3 * 0.15)
        and quiet["plausibility"] == 0.8
        and quiet["personal_significance"] == 0.5
        and both["override"]["location"] == "hall"
        and both["override"]["participants"] == ["a"]
        and both["override"]["personal_significance"] == 0.9
        and both["override"]["expected_emotions"] == ["happy", "excited"]
        and both["challenge"]["expected_emotions"] == ["happy", "excited"]
        and both["again_participants"] == ["friend", "colleague"]
        and both["ids_differ"] is True
        and both["count"] == 4
        and both["stored_same"] is True
        and both["simulation_ignored"] is True
        and payload["memory_only"]["expected_emotions"] == ["pride", "joy"]
        and payload["memory_only"]["location"] == "familiar_place"
        and payload["memory_only"]["vividness"] == 0.75
        and payload["memory_only"]["plausibility"] == 0.8
        and payload["neither"]["location"] == "unknown"
        and payload["neither"]["vividness"] == 0.3
        and payload["neither"]["plausibility"] == 0.5
        and payload["neither"]["expected_emotions"] == ["pride", "joy"]
        and emotions["challenge"]["expected_emotions"] == ["anxiety", "excitement"]
        and emotions["challenge"]["plausibility"] == 0.5
        and emotions["meeting"]["expected_emotions"] == ["curiosity", "neutral"]
        and emotions["meeting"]["plausibility"] == 0.7
        and emotions["empty"]["expected_emotions"] == ["neutral"]
        and emotions["empty"]["plausibility"] == 0.5
        and emotions["empty"]["vividness"] == 0.3
        and emotions["success"]["expected_emotions"] == ["pride", "joy"]
        and emotions["achievement"]["expected_emotions"] == ["pride", "joy"]
        and all(emotions[name]["plausibility"] == 0.7 for name in common)
        and all(payload["falsy_memory"][name]["location"] == "unknown" for name in payload["falsy_memory"])
        and payload["truthy_memory"]["one"]["location"] == "familiar_place"
        and payload["truthy_memory"]["object"]["location"] == "familiar_place"
        and details["views_equal"] is True
        and details["significance_zero"] == 0
        and details["significance_false"] is False
        and details["significance_false_type"] == "bool"
        and details["location_empty"] == ""
        and details["participants_none"] is None
        and details["participants_tuple_type"] == "tuple"
        and details["time_int"] == 7
        and details["time_int_type"] == "int"
        and details["details_false"]["location"] == "unknown"
        and details["details_zero"]["expected_emotions"] == ["neutral"]
        and details["details_blank"]["plausibility"] == 0.5
        and payload["gaps"]["int_description_with_memory_and_emotion"]["expected_emotions"] == ["happy", "excited"]
        and payload["gaps"]["memory_not_called"] is True
        and payload["wired"]["memory"] is True
        and payload["wired"]["mental_simulation"] is True
        and payload["wired"]["consciousness"] is True
        and payload["wired"]["emotion"] is True
    )
    pre = payload["preexperience"]
    empty_stats = payload["stats"]["empty"]
    two = payload["stats"]["two"]
    ef2 = (
        pre["first"]["simulation_success"] is True
        and pre["second"]["simulation_success"] is True
        and pre["first"]["sensory_details"]["visual"] == "Imagining familiar_place"
        and pre["first"]["sensory_details"]["emotional"] == ["happy", "excited"]
        and pre["first"]["sensory_details"]["social"] == ["friend", "colleague"]
        and pre["first"]["vividness"] == 0.75
        and pre["count"] == 2
        and pre["imagined"] == 1
        and pre["unchanged"] is True
        and pre["same_event_id"] is True
        and empty_stats["total_events_imagined"] == 0
        and empty_stats["total_preexperiences"] == 0
        and empty_stats["intention_fulfillment_rate"] == 0.0
        and empty_stats["avg_event_vividness"] == 0.0
        and two["avg_event_vividness"] == 0.5
        and two["total_preexperiences"] == 1
        and two["total_events_imagined"] == 2
    )
    prospective = payload["prospective"]
    found = payload["triggers"]["found"]
    ef3 = (
        prospective["default"]["trigger"] == "tomorrow"
        and prospective["default"]["importance"] == 0.5
        and prospective["default"]["where"] == ""
        and prospective["default_trigger_is_when"] is True
        and prospective["blank_trigger"]["trigger"] == "tonight"
        and prospective["none_trigger"]["trigger"] == "soon"
        and prospective["zero_trigger"]["trigger"] == "soon"
        and prospective["explicit"]["trigger"] == "bell"
        and prospective["explicit"]["importance"] == 0.8
        and prospective["none_what"]["what"] is None
        and prospective["importance"]["high"]["importance"] == 1.5
        and prospective["importance"]["negative"]["importance"] == -1
        and prospective["importance"]["true"]["importance"] is True
        and math.isnan(prospective["importance"]["nan"]["importance"])
        and prospective["same_object"] is True
        and found["empty"] == []
        and found["time_tomorrow"] == ["call Ana"]
        and found["time_today"] == ["send notes", "day-match"]
        and found["location_office"] == ["call Ana", "send notes"]
        and found["location_home_office"] == ["call Ana", "send notes"]
        and found["event_phone"] == ["call Ana"]
        and found["event_email"] == ["send notes"]
        and found["blank_strings"] == []
        and found["zeros"] == []
        and found["all_ana"] == ["call Ana", "send notes"]
        and payload["triggers"]["fulfilled_skipped"] is True
        and payload["triggers"]["open"] == [False, False, False]
        and all(payload["triggers"]["same_object"].values())
    )
    changes = payload["project"]["changes"]
    growth_ok = all(
        row["growth"] == {"knowledge": 0.8, "skills": 0.7, "wisdom": 0.6} and row["continuity"] == 0.7
        for row in payload["project"]["growth"].values()
    )
    ef4 = (
        payload["fulfill"]["missing_return_none"] is True
        and payload["fulfill"]["missing_counter"] == 0
        and payload["fulfill"]["once"]["return"] is True
        and payload["fulfill"]["once"]["stats"]["intention_fulfillment_rate"] == 0.5
        and payload["fulfill"]["once"]["stats"]["prospective_memories_active"] == 1
        and payload["fulfill"]["once"]["stats"]["fulfilled_intentions"] == 1
        and payload["fulfill"]["twice"]["stats"]["fulfilled_intentions"] == 2
        and payload["fulfill"]["twice"]["stats"]["intention_fulfillment_rate"] == 1.0
        and payload["fulfill"]["twice"]["stats"]["prospective_memories_active"] == 1
        and payload["fulfill"]["lone_double"]["intention_fulfillment_rate"] == 2.0
        and payload["fulfill"]["lone_double"]["fulfilled_intentions"] == 2
        and payload["fulfill"]["lone_double"]["prospective_memories_active"] == 0
        and changes["1 year"] == _EF_YEAR
        and changes["YEAR"] == _EF_YEAR
        and changes["yearly"] == _EF_YEAR
        and changes["1 year and 1 month"] == _EF_YEAR
        and changes["next month"] == _EF_MONTH
        and changes["Month"] == _EF_MONTH
        and changes["monthly"] == _EF_MONTH
        and changes["tomorrow"] == _EF_MINOR
        and changes[""] == _EF_MINOR
        and changes["week"] == _EF_MINOR
        and growth_ok
        and len(payload["project"]["growth"]) == 10
        and payload["compare"]["differ"] == {
            "more_vivid": "a", "more_plausible": "b", "more_significant": "b", "preferred": "a",
        }
        and payload["compare"]["vivid_tie"] == {
            "more_vivid": "d", "more_plausible": "d", "more_significant": "d", "preferred": "d",
        }
        and payload["compare"]["product_tie"] == {
            "more_vivid": "c", "more_plausible": "c", "more_significant": "e", "preferred": "e",
        }
        and payload["compare"]["nan"]["more_vivid"] == "b"
        and payload["compare"]["nan"]["preferred"] == "b"
        and payload["compare"]["nan"]["more_significant"] == "n"
        and payload["compare"]["bool"]["preferred"] == "t"
        and payload["compare"]["bool"]["more_vivid"] == "u"
        and payload["compare"]["bool"]["more_significant"] == "t"
    )
    errors = payload["errors"]
    ef5 = (
        errors["desc_int"] == "AttributeError"
        and errors["details_list"] == "AttributeError"
        and errors["details_true"] == "AttributeError"
        and errors["horizon_int"] == "AttributeError"
        and errors["horizon_none"] == "AttributeError"
        and errors["preexperience_none"] == "AttributeError"
        and errors["compare_missing"] == "AttributeError"
        and errors["compare_str_significance"] == "TypeError"
        and errors["context_none_empty"] == "no-error"
        and errors["context_str_empty"] == "no-error"
        and errors["context_list_empty"] == "no-error"
        and errors["context_none"] == "AttributeError"
        and errors["context_str"] == "AttributeError"
        and errors["context_list"] == "AttributeError"
        and errors["context_omitted"] == "TypeError"
    )
    ef6 = (
        identity
        and standalone
        and source_sha == EF_SOURCE_SHA
        and sha_module == sha_brain == EF_BEHAVIOR_SHA
        and payload["clock"] == "wall"
        and payload["init_params"] == ["self", "memory", "mental_simulation", "consciousness", "emotion"]
        and payload["banner"].endswith("Future imagination active)\n")
    )
    return [
        _check("EF1", {"vivid": quiet["vividness"], "plausible": quiet["plausibility"],
                       "homework": emotions["homework"]["plausibility"],
                       "network": emotions["network"]["plausibility"]},
               ef1, "passado fixo 0.75/0.8; substring work vale 0.7; emoção não é consultada"),
        _check("EF2", {"preexperiences": pre["count"], "avg": two["avg_event_vividness"],
                       "empty_avg": empty_stats["avg_event_vividness"]},
               ef2, "pré-vivência soma 2; média vazia é 0.0 e a do par é 0.5"),
        _check("EF3", {"today": found["time_today"], "office": found["location_office"],
                       "importance": prospective["importance"]["high"]["importance"]},
               ef3, "substring no gatilho; where vazio casa; importância 1.5 não é cortada"),
        _check("EF4", {"rate": payload["fulfill"]["lone_double"]["intention_fulfillment_rate"],
                       "year_and_month": changes["1 year and 1 month"],
                       "preferred_tie": payload["compare"]["product_tie"]["preferred"]},
               ef4, "taxa 2.0 na segunda conclusão; year antes de month; empate fica com o segundo"),
        _check("EF5", errors, ef5, "tipo errado quebra no uso; contexto sem intenção não lê o contexto"),
        _check("EF6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": EF_SOURCE_SHA, "behavior_golden": EF_BEHAVIOR_SHA},
               ef6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
    ]


# Goldens do move-only (#118, c856e1a); registro histórico, não são mais comparados.
MS_SOURCE_SHA_BASE = "43803b05cafb6461911aeba7b94e067e3bb83e0a0e3eb0b295b11f1306e9df67"
MS_BEHAVIOR_SHA_BASE = "15dadf71b6aac11280a10d72533065ab7f6ffb5e7a9ce29d688eefe1273ede76"
# Re-fixados depois do GREEN do crédito ciente do objetivo (declarado no pré-registro 82308d0); iguais em 3.11 e 3.12.
MS_SOURCE_SHA = "4b2356be6ff2543acb1ed8e9d32348c5bb4fbcb502e0d58249a4b02c69ae4c26"
MS_BEHAVIOR_SHA = "c8d6f43d0b58f7dcce277fa713df7d499601b9408e91c212d51f4704cadf48b4"
_MS_NAMES = ("MentalScenario", "MentalSimulationResult", "RealMentalSimulationSystem")
_MS_MARKER = "# MÓDULO 30: MENTAL SIMULATION SYSTEM (POST 30)"
_MS_END = '            "cached_simulations": len(self.simulation_cache)\n        }'
_MS_VOCAB = ("move forward", "acquire resource", "build structure", "wait", "rest", "inspect")
_MS_GOALS = ("increase position", "gather resources", "build structure", "survive")
_MS_GRID_SEED = 2026
_MS_GRID_SIZE = 120
_MS_ROLLOUT_SEEDS = tuple(range(20))
_MS_ROLLOUT_N = 100
_MS_ROLLOUT_CASES = {
    "reach": ({"position": 7}, ["move", "wait", "acquire"], "reach position", "move", 0.90),
    "increase": ({"position": 0}, ["move", "wait", "acquire"], "increase position", "move", 0.80),
    "gather": ({"resources": 3}, ["acquire resource", "wait", "build structure"], "gather resources",
               "acquire resource", 0.90),
    # MS3-v2 (pré-registro 82308d0): `move` não é a primeira ação; controles iguais ao código base
    "increase_late": ({"position": 3}, ["wait", "acquire", "move"], "increase position", "move", 0.80),
    "survive_control": ({"position": 0}, ["move", "wait", "acquire"], "survive", "acquire", 1.00),
    "mixed_control": ({"position": 0}, ["move", "wait", "acquire"], "increase position and gather resources",
                      "acquire", 1.00),
}
# MS1-v2: SHA-256 das 93 linhas da grade que o crédito novo não pode tocar (medido no código base c856e1a)
MS_INVARIANT_SHA = "acf3d3c2d4b7e0d3e46facfe1dc9b0e055a6232e7a936e21d8c700a3ec81dd09"
MS_INVARIANT_ROWS = 93
# MS7 (pré-registro 82308d0): (estado, ações, objetivo) -> (qualidade, sucesso)
_MS_CREDIT_CASES = {
    "pos0": ({"position": 0}, [], "increase position", 0.30, False),
    "pos5": ({"position": 5}, [], "increase position", 0.50, False),
    "pos9": ({"position": 9}, [], "increase position", 0.66, False),
    "pos_negative": ({"position": -3}, [], "increase position", 0.30, False),
    "no_resource_credit": ({"position": 0, "resources": 3, "structures": 2}, [], "increase position", 0.30, False),
    "pos_none": ({"position": None}, [], "increase position", 0.30, False),
    "pos_str": ({"position": "a"}, [], "increase position", 0.30, False),
    "pos_nan": ({"position": float("nan")}, [], "increase position", 0.30, False),
    "wait_pos5": ({"position": 5}, ["wait"], "increase position", 0.50, False),
    "mixed_legacy": ({"resources": 1, "position": 0}, [], "increase position and gather resources", 0.50, False),
    "survive_legacy": ({"structures": 1}, [], "survive", 0.50, False),
    "success": ({"position": 9}, ["move"], "increase position", 0.90, True),
}


def _ms_position_only(goal):
    low = goal.lower() if isinstance(goal, str) else ""
    return "position" in low and "resource" not in low and "structure" not in low


class _MsBoom:
    def __getattr__(self, name):
        raise AssertionError("called " + name)


def _ms_err(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "no-error"


def _ms_grid():
    import random
    rng = random.Random(_MS_GRID_SEED)  # nosec B311 - grade sintética de avaliação
    rows = []
    for i in range(_MS_GRID_SIZE):
        state = {"position": rng.randint(0, 9), "resources": rng.randint(0, 3), "structures": rng.randint(0, 2)}
        if rng.random() < 0.25:
            state["safety"] = rng.choice([0.2, 0.5, 0.8])
        actions = [rng.choice(_MS_VOCAB) for _ in range(rng.randint(1, 8))]
        rows.append(("g%d" % i, state, actions, rng.choice(_MS_GOALS)))
    return rows


def _ms_reference(state, actions, goal):
    """Referência independente, escrita só da leitura do bloco: o eval não importa a lógica medida."""
    cur = dict(state)
    risks, opps = [], []
    steps, success = 0, False
    for action in actions:
        steps += 1
        low = action.lower()
        nxt = dict(cur)
        if "move" in low:
            nxt["position"] = nxt.get("position", 0) + 1
        if "acquire" in low:
            nxt["resources"] = nxt.get("resources", 0) + 1
        if "build" in low:
            nxt["structures"] = nxt.get("structures", 0) + 1
        if "move" in low and cur.get("safety", 1.0) < 0.5:
            risks.append("unsafe_movement")
        if "acquire" in low and cur.get("resources", 0) > 10:
            risks.append("resource_overflow")
        if nxt.get("position", 0) < 0:
            risks.append("invalid_position")
        if nxt.get("resources", 0) > cur.get("resources", 0):
            opps.append("resource_gain")
        if nxt.get("structures", 0) > cur.get("structures", 0):
            opps.append("infrastructure_growth")
        cur = nxt
        g = goal.lower()
        if ("resource" in g and cur.get("resources", 0) >= 5) or ("structure" in g and cur.get("structures", 0) >= 3) \
                or ("position" in g and cur.get("position", 0) >= 10):
            success = True
            break
    if success:
        quality = 0.9
    elif _ms_position_only(goal):
        position = cur.get("position", 0)
        progress = 0.0
        if isinstance(position, (int, float)) and position == position:
            progress = min(1.0, max(0.0, position) / 10.0)
        quality = min(1.0, 0.3 + 0.4 * progress)
    else:
        quality = 0.3
        if cur.get("resources", 0) > 0:
            quality += 0.2
        if cur.get("structures", 0) > 0:
            quality += 0.2
        quality = min(1.0, quality)
    return {"final_state": cur, "success": success, "outcome_quality": quality, "steps_taken": steps,
            "predicted_utility": quality * (1.0 if success else 0.5), "risks": risks, "opportunities": opps}


def _ms_view(result):
    return {"final_state": dict(result.final_state), "success": result.success,
            "outcome_quality": result.outcome_quality, "steps_taken": result.steps_taken,
            "predicted_utility": result.predicted_utility, "risks": list(result.risks),
            "opportunities": list(result.opportunities)}


def _ms_copy_state(state):
    return {key: value for key, value in state.items()}


def mental_simulation_fingerprint(ns):
    import contextlib
    import copy
    import inspect
    import io
    import math
    import random
    if isinstance(ns, dict):
        cls, scenario_cls, result_cls = (ns["RealMentalSimulationSystem"], ns["MentalScenario"],
                                         ns["MentalSimulationResult"])
    else:
        cls, scenario_cls, result_cls = (ns.RealMentalSimulationSystem, ns.MentalScenario,
                                         ns.MentalSimulationResult)
    rng_state = random.getstate()
    try:
        def quiet(*args, **kwargs):
            with contextlib.redirect_stdout(io.StringIO()):
                return cls(*args, **kwargs)

        def scenario(sid, state, actions, goal, constraints=None):
            return scenario_cls(sid, state, actions, [] if constraints is None else constraints, goal)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fresh = cls()
        banner = buf.getvalue()
        fresh_stats = fresh.get_statistics()

        # grade de simulação
        grid = _ms_grid()
        system = quiet()
        grid_rows, mutated, same_cached = [], 0, 0
        for sid, state, actions, goal in grid:
            before = copy.deepcopy(state)
            result = system.simulate_scenario(scenario(sid, state, actions, goal))
            if state != before:
                mutated += 1
            if system.simulation_cache.get(sid) is result:
                same_cached += 1
            grid_rows.append(_ms_view(result))
        grid_stats = system.get_statistics()
        grid_digest = _ms_sha(grid_rows)
        invariant_rows = [row for row, (_sid, _state, _actions, goal) in zip(grid_rows, grid)
                          if not _ms_position_only(goal) or row["success"]]
        reference_matches = sum(
            _ms_view_equal(row, _ms_reference(state, actions, goal))
            for row, (sid, state, actions, goal) in zip(grid_rows, grid))

        # tabela de risco, oportunidade, qualidade
        def one(state, actions, goal="survive"):
            return _ms_view(quiet().simulate_scenario(scenario("t", state, actions, goal)))

        table = {
            "unsafe": one({"safety": 0.4}, ["move"]),
            "safe_edge": one({"safety": 0.5}, ["move"]),
            "overflow": one({"resources": 11}, ["acquire"]),
            "no_overflow": one({"resources": 10}, ["acquire"]),
            "invalid_once": one({"position": -1}, ["wait"]),
            "invalid_twice": one({"position": -1}, ["wait", "wait"]),
            "build": one({}, ["build"]),
            "goal_position": one({"position": 9}, ["move", "move"], "position"),
            "goal_resources": one({"resources": 4}, ["acquire", "acquire"], "RESOURCES"),
            "goal_structures": one({"structures": 2}, ["build", "build"], "structure"),
            "no_progress": one({"position": 0}, ["wait"]),
            "both_partial": one({"resources": 1, "structures": 1}, ["wait"]),
            "empty_actions": one({"position": 3}, []),
            "unknown_goal": one({"position": 99}, ["wait"], "survive"),
        }

        # MS7: crédito parcial ciente do objetivo
        credit = {}
        for key, (state, actions, goal, _quality, _success) in _MS_CREDIT_CASES.items():
            credited = quiet().simulate_scenario(scenario("c", dict(state), list(actions), goal))
            credit[key] = {"quality": credited.outcome_quality, "success": credited.success}

        # rollout com seeds explícitas
        rollout = {}
        for key, (state, actions, goal, _want, _limit) in _MS_ROLLOUT_CASES.items():
            picks, totals, caches = [], [], []
            for seed in _MS_ROLLOUT_SEEDS:
                rollout_system = quiet()
                random.seed(seed)  # nosec B311 - seed fixa pré-registrada
                picks.append(rollout_system.rollout_planning(dict(state), list(actions), goal, _MS_ROLLOUT_N))
                totals.append(rollout_system.total_simulations)
                caches.append(len(rollout_system.simulation_cache))
            rollout[key] = {"picks": picks, "totals": sorted(set(totals)), "caches": sorted(set(caches))}
        small = quiet()
        random.seed(7)
        small_pick = small.rollout_planning({"position": 0}, ["move", "wait"], "survive", 5)
        small_stats = small.get_statistics()

        # contrafactual
        cf_rows, cf_ok_id, cf_cache_kept, cf_inputs_kept = [], 0, 0, 0
        cf_system = quiet()
        alternatives = _MS_VOCAB[:3]
        for sid, state, actions, goal in grid[:40]:
            actual = scenario(sid, state, list(actions), goal, ["c"])
            original = cf_system.simulate_scenario(actual)
            for alt in alternatives:
                result = cf_system.counterfactual_simulation(actual, alt)
                cf_rows.append(_ms_view(result))
                cf_ok_id += result.scenario_id == "counterfactual_" + sid
                cf_cache_kept += cf_system.simulation_cache.get(sid) is original
                cf_inputs_kept += actual.actions == actions and actual.initial_state == state
        cf_reference = sum(
            _ms_view_equal(row, _ms_reference(state, [alt] + actions[1:], goal))
            for row, (alt, (sid, state, actions, goal)) in zip(
                cf_rows, [(a, g) for g in grid[:40] for a in alternatives]))
        cf_empty = quiet().counterfactual_simulation(scenario("e", {"position": 0}, [], "survive"), "move")

        # comparação
        cmp_system = quiet()
        win = cmp_system.compare_scenarios(
            scenario("a", {"position": 9}, ["move"], "position"), scenario("b", {}, ["wait"], "survive"))
        tie = cmp_system.compare_scenarios(
            scenario("c", {}, ["wait"], "survive"), scenario("d", {}, ["wait"], "survive"))
        risky = cmp_system.compare_scenarios(
            scenario("r", {"safety": 0.1, "position": -1}, ["move", "move"], "survive"),
            scenario("s", {}, ["wait"], "survive"))
        compare = {"win": win, "tie": tie, "risky": risky, "stats": cmp_system.get_statistics()}

        # colaboradores
        class Hooked:
            def __init__(self):
                self.calls = []

            def predict_next_state(self, state, action):
                self.calls.append((dict(state), action))
                return {"position": 99}

        class Falsy(Hooked):
            def __len__(self):
                return 0

        hooked = Hooked()
        hook_result = quiet(world_model=hooked).simulate_scenario(
            scenario("h", {"position": 1}, ["wait"], "position"))
        falsy = Falsy()
        falsy_result = quiet(world_model=falsy).simulate_scenario(
            scenario("f", {"position": 1}, ["move"], "survive"))
        plain_result = quiet(world_model=object()).simulate_scenario(
            scenario("p", {"position": 1}, ["move"], "survive"))
        boom = quiet(planner=_MsBoom(), reasoning=_MsBoom(), causal=_MsBoom())
        boom.simulate_scenario(scenario("b", {}, ["move"], "survive"))
        boom.rollout_planning({}, ["move"], "survive", 2)
        wired_world, wired_planner, wired_reasoning, wired_causal = object(), object(), object(), object()
        wired = quiet(wired_world, wired_planner, wired_reasoning, wired_causal)
        collaborators = {
            "hook_calls": hooked.calls, "hook_state": _ms_view(hook_result),
            "falsy_calls": falsy.calls, "falsy_state": _ms_view(falsy_result),
            "plain_state": _ms_view(plain_result),
            "untouched": True,
            "wired": [wired.world_model is wired_world, wired.planner is wired_planner,
                      wired.reasoning is wired_reasoning, wired.causal is wired_causal],
        }

        # gaps medidos
        gaps = {
            "remove_moves": one({"position": 0}, ["remove"])["final_state"],
            "rebuild_builds": one({}, ["rebuild"])["final_state"],
            "upper_case_move": one({}, ["MOVE"])["final_state"],
            "bool_position": one({"position": True}, ["move"])["final_state"]["position"],
            "nan_position_type": type(one({"position": math.nan}, ["wait"])["final_state"]["position"]).__name__,
            "nan_stays_valid": one({"position": math.nan}, ["wait"])["risks"],
            "goal_none_empty_actions": one({"position": 3}, [], None)["success"],
            "stats_after_grid": grid_stats,
            "fresh_stats": fresh_stats,
            "rollout_small": {"pick": small_pick, "stats": small_stats},
        }

        errors = {
            "action_int": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {}, [1], "g"))),
            "action_none": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {}, [None], "g"))),
            "state_none": _ms_err(lambda: quiet().simulate_scenario(scenario("x", None, [], "g"))),
            "state_list": _ms_err(lambda: quiet().simulate_scenario(scenario("x", [], ["move"], "g"))),
            "position_none": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {"position": None}, ["move"], "g"))),
            "position_str": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {"position": "a"}, ["move"], "g"))),
            "compare_position_str": _ms_err(lambda: quiet().simulate_scenario(
                scenario("x", {"position": "a"}, ["wait"], "survive"))),
            "actions_none": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {}, None, "g"))),
            "goal_none": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {}, ["wait"], None))),
            "goal_int": _ms_err(lambda: quiet().simulate_scenario(scenario("x", {}, ["wait"], 3))),
            "scenario_none": _ms_err(lambda: quiet().simulate_scenario(None)),
            "rollout_empty_actions": _ms_err(lambda: quiet().rollout_planning({}, [], "g", 3)),
            "rollout_zero": _ms_err(lambda: quiet().rollout_planning({}, ["move"], "g", 0)),
            "rollout_negative": _ms_err(lambda: quiet().rollout_planning({}, ["move"], "g", -1)),
            "rollout_none_actions": _ms_err(lambda: quiet().rollout_planning({}, None, "g", 1)),
            "rollout_state_none": _ms_err(lambda: quiet().rollout_planning(None, ["move"], "g", 1)),
            "counterfactual_none": _ms_err(lambda: quiet().counterfactual_simulation(None, "move")),
            "counterfactual_alt_int": _ms_err(lambda: quiet().counterfactual_simulation(
                scenario("x", {}, ["wait"], "g"), 7)),
            "compare_missing": _ms_err(lambda: quiet().compare_scenarios(scenario("x", {}, [], "g"), None)),
            "construct_extra": _ms_err(lambda: cls(1, 2, 3, 4, 5)),
        }
        return {
            "clock": "none",
            "randomness": "global random.choices; uuid4 ids",
            "init_params": list(inspect.signature(cls.__init__).parameters),
            "banner": banner,
            "scenario_fields": list(inspect.signature(scenario_cls).parameters),
            "result_fields": list(inspect.signature(result_cls).parameters),
            "grid": {"digest": grid_digest, "count": len(grid_rows), "mutated_inputs": mutated,
                     "same_cached": same_cached, "reference_matches": reference_matches,
                     "invariant_rows": len(invariant_rows), "invariant_digest": _ms_sha(invariant_rows)},
            "credit": credit,
            "table": table,
            "rollout": rollout,
            "counterfactual": {"digest": _ms_sha(cf_rows), "count": len(cf_rows), "ids": cf_ok_id,
                               "cache_kept": cf_cache_kept, "inputs_kept": cf_inputs_kept,
                               "reference_matches": cf_reference, "empty_actions": _ms_view(cf_empty),
                               "empty_id": cf_empty.scenario_id},
            "compare": compare,
            "collaborators": collaborators,
            "gaps": gaps,
            "errors": errors,
        }
    finally:
        random.setstate(rng_state)


def _ms_view_equal(left, right):
    return all(left[key] == right[key] for key in right)


def _ms_sha(payload) -> str:
    import hashlib
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def mental_simulation_source_text(src: str) -> str:
    idx = src.index(_MS_MARKER)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end = src.index(_MS_END, idx) + len(_MS_END)
    return src[line_start:end]


def _mental_simulation_source_sha() -> str:
    import hashlib
    text = (ROOT / "core/cognitivo/mental_simulation.py").read_text(encoding="utf-8")
    return hashlib.sha256(mental_simulation_source_text(text).encode()).hexdigest()


def _mental_simulation_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha MS6."""
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/mental_simulation.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" or name.startswith("core.constitutional_brain."):
                return False
    return True


def _ms_close(left, right):
    return abs(left - right) < 1e-9


# (estado final, sucesso, passos, qualidade, utilidade, riscos, oportunidades) esperados pela leitura do bloco
_MS_TABLE_EXPECTED = {
    "unsafe": ({"safety": 0.4, "position": 1}, False, 1, 0.3, 0.15, ["unsafe_movement"], []),
    "safe_edge": ({"safety": 0.5, "position": 1}, False, 1, 0.3, 0.15, [], []),
    "overflow": ({"resources": 12}, False, 1, 0.5, 0.25, ["resource_overflow"], ["resource_gain"]),
    "no_overflow": ({"resources": 11}, False, 1, 0.5, 0.25, [], ["resource_gain"]),
    "invalid_once": ({"position": -1}, False, 1, 0.3, 0.15, ["invalid_position"], []),
    "invalid_twice": ({"position": -1}, False, 2, 0.3, 0.15, ["invalid_position", "invalid_position"], []),
    "build": ({"structures": 1}, False, 1, 0.5, 0.25, [], ["infrastructure_growth"]),
    "goal_position": ({"position": 10}, True, 1, 0.9, 0.9, [], []),
    "goal_resources": ({"resources": 5}, True, 1, 0.9, 0.9, [], ["resource_gain"]),
    "goal_structures": ({"structures": 3}, True, 1, 0.9, 0.9, [], ["infrastructure_growth"]),
    "no_progress": ({"position": 0}, False, 1, 0.3, 0.15, [], []),
    "both_partial": ({"resources": 1, "structures": 1}, False, 1, 0.7, 0.35, [], []),
    "empty_actions": ({"position": 3}, False, 0, 0.3, 0.15, [], []),
    "unknown_goal": ({"position": 99}, False, 1, 0.3, 0.15, [], []),
}
_MS_ERRORS_EXPECTED = {
    "action_int": "AttributeError", "action_none": "AttributeError", "state_none": "AttributeError",
    "state_list": "AttributeError", "position_none": "TypeError", "position_str": "TypeError",
    "compare_position_str": "TypeError", "actions_none": "TypeError", "goal_none": "AttributeError",
    "goal_int": "AttributeError", "scenario_none": "AttributeError", "rollout_empty_actions": "ValueError",
    "rollout_zero": "ZeroDivisionError", "rollout_negative": "no-error", "rollout_none_actions": "TypeError",
    "rollout_state_none": "AttributeError", "counterfactual_none": "AttributeError",
    "counterfactual_alt_int": "AttributeError", "compare_missing": "AttributeError", "construct_extra": "TypeError",
}


def eval_mental_simulation(criteria="v1"):
    """MS1–MS6. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import core.cognitivo.mental_simulation as mental_simulation
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = mental_simulation_fingerprint(mental_simulation)
    sha_module = _ms_sha(payload)
    sha_brain = _ms_sha(mental_simulation_fingerprint(brain))
    source_sha = _mental_simulation_source_sha()
    standalone = _mental_simulation_standalone()
    identity = all(
        getattr(brain, name) is getattr(mental_simulation, name) is getattr(package, name)
        and getattr(mental_simulation, name).__module__ == "core.cognitivo.mental_simulation"
        for name in _MS_NAMES
    )
    grid = payload["grid"]
    ms1 = (
        grid["count"] == _MS_GRID_SIZE
        and grid["reference_matches"] == _MS_GRID_SIZE
        and grid["mutated_inputs"] == 0
        and grid["same_cached"] == _MS_GRID_SIZE
        and grid["invariant_rows"] == MS_INVARIANT_ROWS
        and grid["invariant_digest"] == MS_INVARIANT_SHA
    )
    credit = payload["credit"]
    ms7 = len(credit) == len(_MS_CREDIT_CASES) == 12 and all(
        _ms_close(credit[name]["quality"], quality) and credit[name]["success"] is success
        for name, (_state, _actions, _goal, quality, success) in _MS_CREDIT_CASES.items())
    table = payload["table"]
    table_hits = 0
    for name, (final, success, steps, quality, utility, risks, opps) in _MS_TABLE_EXPECTED.items():
        row = table[name]
        table_hits += (
            row["final_state"] == final and row["success"] is success and row["steps_taken"] == steps
            and _ms_close(row["outcome_quality"], quality) and _ms_close(row["predicted_utility"], utility)
            and row["risks"] == risks and row["opportunities"] == opps
        )
    ms2 = table_hits == len(_MS_TABLE_EXPECTED) == 14
    rates = {}
    for name, (_state, _actions, _goal, want, limit) in _MS_ROLLOUT_CASES.items():
        picks = payload["rollout"][name]["picks"]
        rates[name] = {"rate": picks.count(want) / len(picks), "limit": limit, "want": want}
    ms3 = all(len(payload["rollout"][name]["picks"]) == len(_MS_ROLLOUT_SEEDS) for name in rates) and all(
        item["rate"] >= item["limit"] for item in rates.values())
    cf = payload["counterfactual"]
    empty = cf["empty_actions"]
    ms4 = (
        cf["count"] == 120 and cf["reference_matches"] == 120 and cf["ids"] == 120
        and cf["cache_kept"] == 120 and cf["inputs_kept"] == 120
        and cf["empty_id"] == "counterfactual_e" and empty["steps_taken"] == 1
        and empty["final_state"] == {"position": 1}
    )
    gaps = payload["gaps"]
    compare = payload["compare"]
    collaborators = payload["collaborators"]
    small = gaps["rollout_small"]["stats"]
    ms5 = (
        gaps["stats_after_grid"] == {"total_simulations": 120, "successful_predictions": 0,
                                     "simulation_accuracy": 0.0, "cached_simulations": 120}
        and gaps["fresh_stats"] == {"total_simulations": 0, "successful_predictions": 0,
                                    "simulation_accuracy": 0.0, "cached_simulations": 0}
        and compare["win"]["scenario1_success"] is True and compare["win"]["scenario2_success"] is False
        and compare["win"]["preferred_scenario"] == 1 and _ms_close(compare["win"]["utility_difference"], 0.75)
        and compare["tie"]["preferred_scenario"] == 2 and compare["tie"]["utility_difference"] == 0.0
        and compare["risky"]["risk_comparison"] == {"scenario1_risks": 2, "scenario2_risks": 0}
        and compare["stats"]["total_simulations"] == 6 and compare["stats"]["cached_simulations"] == 6
        and small["total_simulations"] == 10 and small["cached_simulations"] == 10
        and collaborators["hook_calls"] == [({"position": 1}, "wait")]
        and collaborators["hook_state"]["final_state"] == {"position": 99}
        and collaborators["hook_state"]["success"] is True
        and collaborators["falsy_calls"] == []
        and collaborators["falsy_state"]["final_state"] == {"position": 2}
        and collaborators["plain_state"]["final_state"] == {"position": 2}
        and collaborators["wired"] == [True, True, True, True]
        and collaborators["untouched"] is True
        and payload["errors"] == _MS_ERRORS_EXPECTED
        and gaps["remove_moves"] == {"position": 1} and gaps["rebuild_builds"] == {"structures": 1}
        and gaps["upper_case_move"] == {"position": 1} and gaps["bool_position"] == 2
        and gaps["nan_position_type"] == "float" and gaps["nan_stays_valid"] == []
        and gaps["goal_none_empty_actions"] is False
    )
    rollout_counts = {name: (payload["rollout"][name]["totals"], payload["rollout"][name]["caches"])
                      for name in rates}
    ms5 = ms5 and all(
        totals == caches == [len(_MS_ROLLOUT_CASES[name][1]) * _MS_ROLLOUT_N]
        for name, (totals, caches) in rollout_counts.items())
    ms6 = (
        identity
        and standalone
        and source_sha == MS_SOURCE_SHA
        and sha_module == sha_brain == MS_BEHAVIOR_SHA
        and payload["clock"] == "none"
        and payload["init_params"] == ["self", "world_model", "planner", "reasoning", "causal"]
        and payload["banner"].endswith("Internal scenario testing active)\n")
    )
    return [
        _check("MS1", {"reference_matches": grid["reference_matches"], "count": grid["count"],
                       "mutated_inputs": grid["mutated_inputs"], "same_cached": grid["same_cached"],
                       "invariant_rows": grid["invariant_rows"], "invariant_digest": grid["invariant_digest"]},
               ms1, "120/120 iguais à referência; entrada intacta; objeto do cache; 93 linhas invariantes com SHA do código base"),
        _check("MS2", {"hits": table_hits, "cases": len(_MS_TABLE_EXPECTED)},
               ms2, "14/14 casos de borda de risco, oportunidade, objetivo e qualidade"),
        _check("MS3", rates, ms3,
               "reach ≥ 0.90, increase ≥ 0.80, increase_late ≥ 0.80, gather ≥ 0.90 e controles = acquire em 20 seeds"),
        _check("MS4", {"reference_matches": cf["reference_matches"], "ids": cf["ids"],
                       "cache_kept": cf["cache_kept"], "inputs_kept": cf["inputs_kept"]},
               ms4, "120/120 contrafactuais iguais à referência e isolados"),
        _check("MS5", {"stats_after_grid": gaps["stats_after_grid"], "errors": payload["errors"]},
               ms5, "contadores, colaboradores, erros e gaps conforme o fingerprint"),
        _check("MS6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": MS_SOURCE_SHA, "behavior_golden": MS_BEHAVIOR_SHA},
               ms6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
        _check("MS7", credit, ms7, "crédito parcial ciente do objetivo: 12 casos da tabela pré-registrada"),
    ]


# ── Analogical Reasoning (POST27) ───────────────────────────────────────────

AR_SOURCE_SHA = "64e1bd060e3cf267970165829bc71515d7ab232e888f24118804215f142966e1"
AR_BEHAVIOR_SHA = "997c1e59edbbce92e292c8c10b3f789c026c70bee039064241d9ff3ceff94201"  # re-fixado após o GREEN (detector; igual a MS6 em #119); 3.12 e 3.11
_AR_NAMES = ("AnalogySituation", "AnalogyMapping", "RealAnalogicalReasoningSystem")
_AR_MARKER = "# MÓDULO 27: ANALOGICAL REASONING SYSTEM (POST 27)"
_AR_END = '            "transfer_success_rate": self.successful_transfers / max(1, self.total_analogies_found)\n        }'
_AR_DOMAINS = ("physics", "biology", "economics", "computing", "music")
_AR_LABELS = ("CAUSES", "PART_OF", "SIMILAR", "ABOVE", "BEFORE")
_AR_GRID_SEED = 2027
_AR_GRID_SIZE = 120
_AR_SEEDS = tuple(range(20))
_AR_LIMIT = 0.90
_AR_PROP_SEED = 2028
_AR_PROP_SIZE = 200
_AR_TRANSFER_SEED = 2029
_AR_TRANSFER_SIZE = 100
_AR_MIN_UNAFFECTED = 52
_AR_BASE_ROWS_SHA = "22a2e7ba12ef6d9a50924ede00d168a80c5bc5e35ba5232ba33dc4c2e468a2c1"
# resumo (16 hex) por linha da grade de AR1 no código base 3eebbbb, capturado antes da correção (AR10)
_AR_BASE_ROWS = (
    "ab033db2a3bc7082",
    "59294d4d23bf7142",
    "6d446f7469af8ce0",
    "9d635636a6afbad0",
    "12b5f9679fc464f2",
    "0687dafba1b6ddec",
    "ed1a3dd2338c89ff",
    "9d635636a6afbad0",
    "9d635636a6afbad0",
    "b8d6156ff711b6f3",
    "8e329bc28dbc634e",
    "ab033db2a3bc7082",
    "52471d4cc071f5a5",
    "b4e8834cc712af59",
    "b8d6156ff711b6f3",
    "ae8684378e8ff7dd",
    "9d635636a6afbad0",
    "ab033db2a3bc7082",
    "edef78052727be1c",
    "3854134bdf8cab5d",
    "ab033db2a3bc7082",
    "ab033db2a3bc7082",
    "acad21785c655296",
    "ae8684378e8ff7dd",
    "293f55fbaaee8c32",
    "460ffd97fc6d1c13",
    "03c9eb646972f578",
    "7816736723831ded",
    "1d0ab6ec110e939a",
    "e4965fe9612a080a",
    "57ccd6321687d7f7",
    "40877eb627c578b6",
    "ab033db2a3bc7082",
    "9d635636a6afbad0",
    "15b49a3ea1fa81fe",
    "9d635636a6afbad0",
    "0c1fc54af90fc4f4",
    "b8d6156ff711b6f3",
    "d3a2dd10208e3db1",
    "ab033db2a3bc7082",
    "ad622245b382e22c",
    "54f4b8fe3b5be2a2",
    "6241d78e343d5e45",
    "ab033db2a3bc7082",
    "ab033db2a3bc7082",
    "ab033db2a3bc7082",
    "ff72438bad90c557",
    "d74e1f5383013736",
    "b8d6156ff711b6f3",
    "9d635636a6afbad0",
    "9d635636a6afbad0",
    "ab033db2a3bc7082",
    "ab033db2a3bc7082",
    "5e4566f33775b234",
    "460ffd97fc6d1c13",
    "b4c7d9d4223f6882",
    "ab033db2a3bc7082",
    "b19c6b4fa840256d",
    "b4c7d9d4223f6882",
    "8329ddfb74e5a9d9",
    "5e705321e9b1cb4b",
    "ab033db2a3bc7082",
    "d75e58c453801d34",
    "6ef9bd5d7e2a076f",
    "9d635636a6afbad0",
    "b8d6156ff711b6f3",
    "ab033db2a3bc7082",
    "ae8684378e8ff7dd",
    "9d635636a6afbad0",
    "460ffd97fc6d1c13",
    "c296602235fc1dbc",
    "460ffd97fc6d1c13",
    "28a5d5bf798917b3",
    "0dcd7cdc9e11f89c",
    "4317ab710ebf21d4",
    "9d635636a6afbad0",
    "ab033db2a3bc7082",
    "9d635636a6afbad0",
    "b4c7d9d4223f6882",
    "ab033db2a3bc7082",
    "d5ee350f8d04e8dc",
    "ae8684378e8ff7dd",
    "4d17656be3dcb7a6",
    "b4c7d9d4223f6882",
    "b8d6156ff711b6f3",
    "ab033db2a3bc7082",
    "678923c13b19c024",
    "9d635636a6afbad0",
    "b8d6156ff711b6f3",
    "bbed6dd3eab8a2b4",
    "9d635636a6afbad0",
    "b8d6156ff711b6f3",
    "dd510342e23194ba",
    "b8d6156ff711b6f3",
    "ab033db2a3bc7082",
    "ab033db2a3bc7082",
    "9cb7f1a0122ea9a9",
    "4a378262852c0d8b",
    "01c012ae930676cb",
    "b4c7d9d4223f6882",
    "59e71032311fefad",
    "b8d6156ff711b6f3",
    "9d635636a6afbad0",
    "9d635636a6afbad0",
    "d4c3384d132c0639",
    "9d635636a6afbad0",
    "ab033db2a3bc7082",
    "a863f2d57b6ef339",
    "898e77b964e4c559",
    "e8230f62d8fe0956",
    "1b841530c4f57fa0",
    "ae8684378e8ff7dd",
    "30b84a4b5080201d",
    "460ffd97fc6d1c13",
    "9d635636a6afbad0",
    "9d635636a6afbad0",
    "ab033db2a3bc7082",
    "b4c7d9d4223f6882",
    "9d635636a6afbad0",
    "1146c2063ca40058",
)
_AR_NAMES_POOL = ("Alpha", "Bravo", "Cedar", "Delta", "Echo", "Fable", "Gamma", "Harbor")
_AR_SRC_WORDS = ("sun", "moon", "cat", "dog", "red", "blue")
_AR_TGT_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")


class _ArBoom:
    def __getattr__(self, name):
        raise AssertionError("called " + name)


def _ar_err(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "no-error"


def _ar_grid():
    rng = random.Random(_AR_GRID_SEED)  # nosec B311 - dados sintéticos de avaliação
    rows = []
    for i in range(_AR_GRID_SIZE):
        known = []
        for k in range(rng.randint(0, 5)):
            ents = ["s%d_%d_%d" % (i, k, j) for j in range(rng.randint(2, 4))]
            rels = [(rng.choice(_AR_LABELS),) + tuple(rng.sample(ents, 2)) for _ in range(rng.randint(0, 4))]
            solution = rng.choice([None, "", "use %s with %s" % (ents[0], ents[1])])
            known.append((rng.choice(_AR_DOMAINS), ents, rels, solution))
        t_ents = ["t%d_%d" % (i, j) for j in range(rng.randint(2, 4))]
        t_rels = [(rng.choice(_AR_LABELS),) + tuple(rng.sample(t_ents, 2)) for _ in range(rng.randint(0, 4))]
        rows.append((known, rng.choice(_AR_DOMAINS), t_ents, t_rels))
    return rows


def _ar_reference_v1(known, domain, t_rels):
    """Referência v1 do #121 (mapeamento ingênuo e substituição em sequência), mantida só para o controle de AR10. Devolve (índice, mapa, qualidade, texto) ou None."""
    best = None
    for index, (s_domain, _ents, s_rels, solution) in enumerate(known):
        if s_domain == domain:
            continue
        first = {}
        for label, e1, e2 in s_rels:
            first.setdefault(label, (e1, e2))
        mapping, matched = {}, 0
        for label, e1, e2 in t_rels:
            if label in first:
                mapping[first[label][0]] = e1
                mapping[first[label][1]] = e2
                matched += 1
        if not matched:
            continue
        quality = matched / max(len(s_rels), len(t_rels))
        if best is None or quality > best[2]:
            text = None
            if solution:
                text = solution
                for src, dst in mapping.items():
                    text = text.replace(src, dst)
            best = (index, mapping, quality, text, bool(solution))
    return best


def _ar_reference(known, domain, t_rels):
    """Referência v2 (AR-v2): um-para-um, injetiva, substituição simultânea por palavra inteira.
    Devolve (índice, mapa, qualidade, texto, tem_solução) ou None."""
    import re
    best = None
    for index, (s_domain, _ents, s_rels, solution) in enumerate(known):
        if s_domain == domain:
            continue
        used, forward, backward, matched = set(), {}, {}, 0
        for label, t1, t2 in t_rels:
            for k, (s_label, s1, s2) in enumerate(s_rels):
                if k in used or s_label != label:
                    continue
                f, b, ok = dict(forward), dict(backward), True
                for src, dst in ((s1, t1), (s2, t2)):
                    if f.get(src, dst) != dst or b.get(dst, src) != src:
                        ok = False
                        break
                    f[src] = dst
                    b[dst] = src
                if ok:
                    forward, backward = f, b
                    used.add(k)
                    matched += 1
                    break
        if not matched:
            continue
        quality = matched / max(len(s_rels), len(t_rels))
        if best is None or quality > best[2]:
            text = None
            if solution:
                text = "".join(forward.get(tok, tok) for tok in re.split(r"(\w+)", solution))
            best = (index, forward, quality, text, bool(solution))
    return best


def _ar_system(cls, known, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        system = cls(**kwargs)
    for domain, ents, rels, solution in known:
        system.add_situation(domain, ents, rels, solution)
    return system


def _ar_find(cls, known, domain, ents, rels):
    system = _ar_system(cls, known)
    return system, system.find_analogy(domain, ents, rels)


def _ar_family_one_to_one(cls, seed):
    rng = random.Random(1000 + seed)  # nosec B311 - dados sintéticos de avaliação
    label = rng.choice(_AR_LABELS)
    known = [("src", ["a%d" % seed, "b%d" % seed], [(label, "a%d" % seed, "b%d" % seed)], None)]
    rels = [(label, "u%d" % seed, "v%d" % seed), (label, "w%d" % seed, "x%d" % seed)]
    _, mapping = _ar_find(cls, known, "tgt", ["u", "v", "w", "x"], rels)
    return mapping is not None and mapping.quality_score <= 0.5


def _ar_family_injective(cls, seed):
    rng = random.Random(2000 + seed)  # nosec B311 - dados sintéticos de avaliação
    l1, l2 = rng.sample(_AR_LABELS, 2)
    a, b, c, d = ("a%d" % seed, "b%d" % seed, "c%d" % seed, "d%d" % seed)
    known = [("src", [a, b, c, d], [(l1, a, b), (l2, c, d)], None)]
    x, y, z = ("x%d" % seed, "y%d" % seed, "z%d" % seed)
    _, mapping = _ar_find(cls, known, "tgt", [x, y, z], [(l1, x, y), (l2, x, z)])
    if mapping is None:
        return True
    values = list(mapping.entity_mappings.values())
    return len(set(values)) == len(values)


def _ar_family_control_isomorphic(cls, seed):
    rng = random.Random(3000 + seed)  # nosec B311 - dados sintéticos de avaliação
    l1, l2 = rng.sample(_AR_LABELS, 2)
    a, b, c = ("a%d" % seed, "b%d" % seed, "c%d" % seed)
    x, y, z = ("x%d" % seed, "y%d" % seed, "z%d" % seed)
    known = [("src", [a, b, c], [(l1, a, b), (l2, b, c)], None)]
    _, mapping = _ar_find(cls, known, "tgt", [x, y, z], [(l1, x, y), (l2, y, z)])
    return mapping is not None and mapping.quality_score == 1.0 and mapping.entity_mappings == {a: x, b: y, c: z}


def _ar_family_real_entities(cls, seed):
    rng = random.Random(4000 + seed)  # nosec B311 - dados sintéticos de avaliação
    c1, c2 = rng.sample(_AR_NAMES_POOL, 2)
    system = _ar_system(cls, [("src", ["Rain", "Flood"], [("CAUSES", "Rain", "Flood")], "avoid Rain before Flood")])
    return system.reason_by_analogy("%s causes %s." % (c1, c2), "tgt") == "avoid %s before %s" % (c1, c2)


def _ar_family_simultaneous(cls, seed):
    a, b, c = ("a%d" % seed, "b%d" % seed, "c%d" % seed)
    system = _ar_system(cls, [("src", [a, b], [("CAUSES", a, b)], "%s then %s" % (a, b))])
    mapping = system.find_analogy("tgt", [b, c], [("CAUSES", b, c)])
    return mapping is not None and system.transfer_solution(mapping) == "%s then %s" % (b, c)


def _ar_words(seed):
    rng = random.Random(5000 + seed)  # nosec B311 - dados sintéticos de avaliação
    p, q = rng.sample(_AR_SRC_WORDS, 2)
    tp, tq = rng.sample(_AR_TGT_WORDS, 2)
    return p, q, tp, tq


def _ar_family_whole_word(cls, seed):
    p, q, tp, tq = _ar_words(seed)
    solution = "%sny and %sish with %s and %s (case %d)" % (p, q, p, q, seed)
    system = _ar_system(cls, [("src", [p, q], [("USES", p, q)], solution)])
    mapping = system.find_analogy("tgt", [tp, tq], [("USES", tp, tq)])
    expected = "%sny and %sish with %s and %s (case %d)" % (p, q, tp, tq, seed)
    return mapping is not None and system.transfer_solution(mapping) == expected


def _ar_family_control_plain(cls, seed):
    p, q, tp, tq = _ar_words(seed)
    system = _ar_system(cls, [("src", [p, q], [("USES", p, q)], "move %s to %s" % (p, q))])
    mapping = system.find_analogy("tgt", [tp, tq], [("USES", tp, tq)])
    return mapping is not None and system.transfer_solution(mapping) == "move %s to %s" % (tp, tq)


_AR_FAMILIES = {
    "AR3": {"one_to_one": _ar_family_one_to_one, "injective": _ar_family_injective,
            "control_isomorphic": _ar_family_control_isomorphic},
    "AR4": {"real_entities": _ar_family_real_entities, "simultaneous": _ar_family_simultaneous,
            "whole_word": _ar_family_whole_word, "control_plain": _ar_family_control_plain},
}


def _ar_view(mapping):
    if mapping is None:
        return None
    return {"mappings": dict(mapping.entity_mappings), "quality": mapping.quality_score,
            "target_domain": mapping.target_situation.domain, "target_id": mapping.target_situation.situation_id,
            "source_domain": mapping.source_situation.domain}


_AR_CM = {"a": "x", "b": "y"}
# casos de borda esperados pela leitura do bloco (pré-registro AR2, 19 casos)
_AR_TABLE_EXPECTED = {
    "c01": [None, 0], "c02": [None, 0], "c03": [None, 0],
    "c04": [_AR_CM, 1.0, 1], "c05": [_AR_CM, 0.5, 1], "c06": [_AR_CM, 0.5, 1],
    "c07": [{"c": "p", "d": "q", "a": "x", "b": "y"}, 1.0, 1],
    "c08": ["S1", 1.0], "c09": ["S2", 1.0], "c10": [None, 0], "c11": [None, 0],
    "c12": ["move x to y", 1], "c13": [None, 0], "c14": [None, 0],
    "c15": ["Alice", "Bob", "Dave"], "c16": ["Aa", "Bb", "Cc", "Dd", "Ee"],
    "c17": [["SIMILAR", "entity1", "entity2"], ["CAUSES", "entity1", "entity2"], ["PART_OF", "entity1", "entity2"]],
    "c18": [[], [], []], "c19": [None, 0, 0],
}
_AR_GAPS_EXPECTED = {
    "repeated_transfer_stats": {"total_analogies_found": 1, "successful_transfers": 2, "known_situations": 1,
                                "transfer_success_rate": 2.0},
    "placeholder_transfer": "stop Storm",
    "the_cat_transfer": "stop The",
    "single_entity_placeholder": "stop entity1",
    "dissimilar": [["SIMILAR", "entity1", "entity2"]],
    "counterpart": [["PART_OF", "entity1", "entity2"]],
    "negation": [["CAUSES", "entity1", "entity2"]],
    "the_cat": ["The", "Harm"], "single_letter": ["A", "Bob"],
    "domain_case_found": True, "aliased_relations": True, "aliased_target_entities": True,
}
_AR_FRESH_STATS = {"total_analogies_found": 0, "successful_transfers": 0, "known_situations": 0,
                   "transfer_success_rate": 0.0}
_AR_BANNER = "🧠🔄 POST 27 - RealAnalogicalReasoningSystem initialized (Cross-domain transfer active)\n"
_AR_ERRORS_EXPECTED = {
    "add_all_none": "no-error", "find_rels_none_no_known": "no-error", "find_rels_none_with_known": "TypeError",
    "find_rel_two_items": "ValueError", "find_rel_int": "TypeError", "source_rels_none": "TypeError",
    "find_no_args": "TypeError", "transfer_none": "AttributeError", "transfer_solution_int": "AttributeError",
    "reason_none": "AttributeError", "reason_int": "AttributeError", "reason_empty": "no-error",
    "reason_blank": "no-error", "extract_entities_none": "AttributeError", "construct_extra": "TypeError",
}


_AR_C, _AR_P = "CAUSES", "PART_OF"
# AR7: tabela de mapeamento (pré-registro AR7, 14 casos): (fontes [(domínio, relações)], relações do alvo)
_AR_MAP_CASES = {
    "m01": ([("S", [(_AR_C, "a", "b")])], [(_AR_C, "x", "y"), (_AR_C, "u", "v")]),
    "m02": ([("S", [(_AR_C, "a", "b"), (_AR_C, "c", "d")])], [(_AR_C, "x", "y"), (_AR_C, "u", "v")]),
    "m03": ([("S", [(_AR_C, "a", "b"), (_AR_C, "c", "d")])], [(_AR_C, "x", "y")]),
    "m04": ([("S", [(_AR_C, "a", "b"), (_AR_P, "c", "d")])], [(_AR_C, "x", "y"), (_AR_P, "x", "z")]),
    "m05": ([("S", [(_AR_C, "a", "b"), (_AR_P, "b", "c")])], [(_AR_C, "x", "y"), (_AR_P, "y", "z")]),
    "m06": ([("S", [(_AR_C, "a", "b"), (_AR_P, "b", "c")])], [(_AR_C, "x", "y"), (_AR_P, "w", "z")]),
    "m07": ([("S", [(_AR_C, "a", "b")])], [(_AR_C, "x", "x")]),
    "m08": ([("S", [(_AR_C, "a", "a")])], [(_AR_C, "x", "y")]),
    "m09": ([("S", [(_AR_C, "a", "a")])], [(_AR_C, "x", "x")]),
    "m10": ([("S", [(_AR_C, "a", "b"), (_AR_C, "c", "d")])], [(_AR_C, "x", "y"), (_AR_C, "x", "z")]),
    "m11": ([("S", [(_AR_C, "a", "b"), (_AR_C, "b", "c"), (_AR_C, "d", "e")])],
            [(_AR_C, "x", "y"), (_AR_C, "p", "q")]),
    "m12": ([("S", [(_AR_C, "a", "b"), (_AR_C, "b", "c")])], [(_AR_C, "x", "y"), (_AR_C, "y", "z")]),
    "m13": ([("S", [(_AR_P, "a", "b"), (_AR_C, "c", "d")])], [(_AR_C, "x", "y"), (_AR_P, "p", "q")]),
    "m14": ([("S1", [(_AR_C, "a", "b")]), ("S2", [(_AR_C, "c", "d"), (_AR_C, "e", "f")])],
            [(_AR_C, "x", "y"), (_AR_C, "u", "v")]),
}
_AR_MAP_EXPECTED = {
    "m01": [{"a": "x", "b": "y"}, 0.5], "m02": [{"a": "x", "b": "y", "c": "u", "d": "v"}, 1.0],
    "m03": [{"a": "x", "b": "y"}, 0.5], "m04": [{"a": "x", "b": "y"}, 0.5],
    "m05": [{"a": "x", "b": "y", "c": "z"}, 1.0], "m06": [{"a": "x", "b": "y"}, 0.5],
    "m07": None, "m08": None, "m09": [{"a": "x"}, 1.0], "m10": [{"a": "x", "b": "y"}, 0.5],
    "m11": [{"a": "x", "b": "y", "d": "p", "e": "q"}, 2 / 3],
    "m12": [{"a": "x", "b": "y", "c": "z"}, 1.0],
    "m13": [{"c": "x", "d": "y", "a": "p", "b": "q"}, 1.0],
    "m14": ["S2", {"c": "x", "d": "y", "e": "u", "f": "v"}, 1.0],
}
# AR8: tabela de transferência (pré-registro AR8, 12 casos): (mapeamento, solução da fonte, esperado)
_AR_TRANSFER_CASES = {
    "t01": ({"a": "b", "b": "c"}, "a then b", "b then c"),
    "t02": ({"a": "b", "b": "a"}, "a and b", "b and a"),
    "t03": ({"sun": "alpha"}, "sunny sun sun's sun.", "sunny alpha alpha's alpha."),
    "t04": ({"cat": "dog"}, "concatenate cat", "concatenate dog"),
    "t05": ({"Rain": "X", "Rainfall": "Y"}, "Rainfall and Rain", "Y and X"),
    "t06": ({"New York": "Paris"}, "visit New York now, not New Yorker", "visit Paris now, not New Yorker"),
    "t07": ({"sun": "alpha"}, "Sun sun", "Sun alpha"),
    "t08": ({"a": "a"}, "a b", "a b"),
    "t09": ({"a.b": "x"}, "a.b and axb", "x and axb"),
    "t10": ({}, "keep it", "keep it"),
    "t11": ({"": "x", "a": "b"}, "ab a", "ab b"),
    "t12": ({"C++": "Rust"}, "use C++ now, C++11 later", "use Rust now, C++11 later"),
}
# AR9: reason_by_analogy (pré-registro AR9, 8 casos): (solução da fonte, texto, domínio alvo, esperado, [achadas, transferidas])
_AR_REASON_CASES = {
    "r1": ("stop Rain", "Storm causes Damage.", "T", "stop Storm", [1, 1]),
    "r2": ("avoid Rain before Flood", "Alpha causes Bravo.", "T", "avoid Alpha before Bravo", [1, 1]),
    "r3": ("avoid Rain before Flood", "Cc causes Dd Ee Ff", "T", "avoid Cc before Dd", [1, 1]),
    "r4": ("stop Rain", "it causes Harm.", "T", "stop entity1", [1, 1]),
    "r5": ("stop Rain", "the thing causes it.", "T", "stop entity1", [1, 1]),
    "r6": ("stop Rain", "The cat causes Harm.", "T", "stop The", [1, 1]),
    "r7": ("avoid Rain before Flood", "Alpha causes Bravo.", "S", None, [0, 0]),
    "r8": (None, "Alpha causes Bravo.", "T", None, [1, 0]),
}


def _ar_prop_instance(rng, i):
    labels = rng.sample(_AR_LABELS, rng.randint(1, 3))
    sources = []
    for k in range(rng.randint(1, 3)):
        ents = ["p%d_%d_%d" % (i, k, j) for j in range(rng.randint(2, 4))]
        rels = [(rng.choice(labels),) + tuple(rng.sample(ents, 2)) for _ in range(rng.randint(0, 6))]
        sources.append(("S%d" % k, ents, rels, None))
    t_ents = ["q%d_%d" % (i, j) for j in range(rng.randint(2, 5))]
    t_rels = [(rng.choice(labels),) + tuple(rng.sample(t_ents, 2)) for _ in range(rng.randint(0, 6))]
    return sources, t_ents, t_rels


def _ar_transfer_instance(rng):
    keys = ["w%d" % j for j in range(rng.randint(2, 5))]
    pool = keys + ["T%d" % j for j in range(5)]
    mapping = {key: rng.choice(pool) for key in keys}
    decoys = ["w0x", "xw0", "w0_", "1w0"]
    seps = [" ", ", ", ". ", "-", "'", "; "]
    tokens = [rng.choice(keys + decoys) for _ in range(rng.randint(6, 12))]
    text = tokens[0]
    for token in tokens[1:]:
        text += rng.choice(seps) + token
    return mapping, text


def analogical_reasoning_fingerprint(ns):
    import contextlib
    import copy
    import inspect
    import io
    import re
    if isinstance(ns, dict):
        cls, sit_cls, map_cls = ns["RealAnalogicalReasoningSystem"], ns["AnalogySituation"], ns["AnalogyMapping"]
    else:
        cls, sit_cls, map_cls = ns.RealAnalogicalReasoningSystem, ns.AnalogySituation, ns.AnalogyMapping

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh = cls()
    banner = buf.getvalue()
    fresh_stats = fresh.get_statistics()

    # grade (AR1)
    grid = _ar_grid()
    rows, mutated, reference_matches, counter_matches, found_sum, transfer_sum = [], 0, 0, 0, 0, 0
    ref_found_sum = ref_transfer_sum = 0
    row_digests, unaffected = [], []
    for known, domain, t_ents, t_rels in grid:
        known_before, ents_before, rels_before = copy.deepcopy(known), list(t_ents), list(t_rels)
        system = _ar_system(cls, known)
        mapping = system.find_analogy(domain, t_ents, t_rels)
        text = system.transfer_solution(mapping) if mapping is not None else None
        stats = system.get_statistics()
        sources = [(s.domain, s.entities, s.relations, s.solution) for s in system.known_situations]
        if sources != known_before or t_ents != ents_before or t_rels != rels_before:
            mutated += 1
        ref = _ar_reference(known, domain, t_rels)
        if ref is None:
            same = mapping is None and text is None
            counters = (0, 0)
        else:
            index, ref_map, ref_quality, ref_text, has_solution = ref
            same = (
                mapping is not None
                and mapping.source_situation is system.known_situations[index]
                and mapping.entity_mappings == ref_map and mapping.quality_score == ref_quality
                and mapping.target_situation.situation_id == "target"
                and mapping.target_situation.domain == domain
                and mapping.target_situation.entities == t_ents
                and mapping.target_situation.relations == t_rels
                and mapping.target_situation.solution is None
                and text == ref_text
            )
            counters = (1, 1 if has_solution else 0)
        ref_found_sum += counters[0]
        ref_transfer_sum += counters[1]
        reference_matches += bool(same)
        counter_matches += (
            stats["total_analogies_found"], stats["successful_transfers"], stats["known_situations"]
        ) == (counters[0], counters[1], len(known))
        found_sum += stats["total_analogies_found"]
        transfer_sum += stats["successful_transfers"]
        rows.append({"view": _ar_view(mapping), "text": text, "stats": stats})
        row_digests.append(_ms_sha(rows[-1])[:16])
        unaffected.append(_ar_reference_v1(known, domain, t_rels) == ref)
    grid_info = {"count": len(rows), "digest": _ms_sha(rows), "mutated_inputs": mutated,
                 "row_digests": row_digests, "unaffected": unaffected,
                 "reference_matches": reference_matches, "counter_matches": counter_matches,
                 "found_sum": found_sum, "transfer_sum": transfer_sum,
                 "ref_found_sum": ref_found_sum, "ref_transfer_sum": ref_transfer_sum}

    # tabela de borda (AR2): 19 casos
    c, p = "CAUSES", "PART_OF"
    table = {}

    def edge(name, known, rels, tgt_ents=("x", "y")):
        system, mapping = _ar_find(cls, known, "T", list(tgt_ents), rels)
        if name in ("c01", "c02", "c03", "c10", "c11"):
            table[name] = [_ar_view(mapping), system.total_analogies_found]
        elif name in ("c08", "c09"):
            table[name] = [mapping.source_situation.domain, mapping.quality_score]
        else:
            table[name] = [dict(mapping.entity_mappings), mapping.quality_score, system.total_analogies_found]

    edge("c01", [], [(c, "x", "y")])
    edge("c02", [("T", ["a", "b"], [(c, "a", "b")], None)], [(c, "x", "y")])
    edge("c03", [("S", ["a", "b"], [(p, "a", "b")], None)], [(c, "x", "y")])
    edge("c04", [("S", ["a", "b"], [(c, "a", "b")], None)], [(c, "x", "y")])
    edge("c05", [("S", ["a", "b"], [(c, "a", "b")], None)], [(c, "x", "y"), (p, "p", "q")])
    edge("c06", [("S", ["a", "b", "c", "d"], [(c, "a", "b"), (p, "c", "d")], None)], [(c, "x", "y")])
    edge("c07", [("S", ["a", "b", "c", "d"], [(c, "a", "b"), (p, "c", "d")], None)], [(p, "p", "q"), (c, "x", "y")])
    edge("c08", [("S1", ["a", "b"], [(c, "a", "b")], None), ("S2", ["c", "d"], [(c, "c", "d")], None)],
         [(c, "x", "y")])
    edge("c09", [("S1", ["a", "b", "e", "f"], [(c, "a", "b"), (p, "e", "f")], None),
                 ("S2", ["c", "d"], [(c, "c", "d")], None)], [(c, "x", "y")])
    edge("c10", [("S", ["a", "b"], [(c, "a", "b")], None)], [])
    edge("c11", [("S", [], [], None)], [(c, "x", "y")])

    def transfer_case(solution):
        known = [("S", ["a", "b"], [(c, "a", "b")], solution)]
        system, mapping = _ar_find(cls, known, "T", ["x", "y"], [(c, "x", "y")])
        return [system.transfer_solution(mapping), system.successful_transfers]

    table["c12"] = transfer_case("move a to b")
    table["c13"] = transfer_case(None)
    table["c14"] = transfer_case("")
    probe = _ar_system(cls, [])
    table["c15"] = probe._extract_entities("Alice meets Bob. carol Dave")
    table["c16"] = probe._extract_entities("Aa Bb Cc Dd Ee Ff Gg")
    table["c17"] = [list(r) for r in probe._extract_relations("x CAUSES y, part of z, SIMILAR TO w")]
    table["c18"] = [probe._extract_relations("nothing relational here"), probe._extract_relations(""),
                    probe._extract_entities("")]
    system = _ar_system(cls, [("S", ["a", "b"], [(c, "a", "b")], "sol")])
    table["c19"] = [system.reason_by_analogy("Nothing relational here.", "T"),
                    system.total_analogies_found, system.successful_transfers]

    # mapeamento um-para-um e injetivo (AR7)
    map_table = {}
    for name, (sources, rels) in _AR_MAP_CASES.items():
        known = [(domain, [], s_rels, None) for domain, s_rels in sources]
        _, mapping = _ar_find(cls, known, "T", [], rels)
        if mapping is None:
            map_table[name] = None
        elif name == "m14":
            map_table[name] = [mapping.source_situation.domain, dict(mapping.entity_mappings), mapping.quality_score]
        else:
            map_table[name] = [dict(mapping.entity_mappings), mapping.quality_score]
    prop_rng = random.Random(_AR_PROP_SEED)  # nosec B311 - dados sintéticos de avaliação
    prop_ok = prop_found = prop_mutated = 0
    for i in range(_AR_PROP_SIZE):
        sources, t_ents, t_rels = _ar_prop_instance(prop_rng, i)
        before = (copy.deepcopy(sources), list(t_ents), list(t_rels))
        system = _ar_system(cls, sources)
        mapping = system.find_analogy("T", t_ents, t_rels)
        if (copy.deepcopy([(x.domain, x.entities, x.relations, x.solution) for x in system.known_situations]),
                t_ents, t_rels) != (before[0], before[1], before[2]):
            prop_mutated += 1
        if mapping is None:
            prop_ok += 1
            continue
        prop_found += 1
        src = mapping.source_situation
        scale = max(len(src.relations), len(t_rels))
        matched = mapping.quality_score * scale
        s_args = {a for _l, e1, e2 in src.relations for a in (e1, e2)}
        t_args = {a for _l, e1, e2 in t_rels for a in (e1, e2)}
        values = list(mapping.entity_mappings.values())
        prop_ok += (
            len(set(values)) == len(values)
            and abs(matched - round(matched)) < 1e-9
            and 1 <= round(matched) <= min(len(src.relations), len(t_rels))
            and 0 < mapping.quality_score <= 1
            and set(mapping.entity_mappings) <= s_args and set(values) <= t_args
        )
    map_props = {"ok": prop_ok, "total": _AR_PROP_SIZE, "found": prop_found, "mutated": prop_mutated}

    # transferência simultânea e por palavra inteira (AR8)
    transfer_table = {}
    for name, (mapping_spec, solution, _expected) in _AR_TRANSFER_CASES.items():
        src = sit_cls("s", "S", list(mapping_spec), [], solution)
        system = _ar_system(cls, [])
        text = system.transfer_solution(map_cls("m", src, src, dict(mapping_spec), 1.0))
        transfer_table[name] = [text, system.successful_transfers]
    transfer_rng = random.Random(_AR_TRANSFER_SEED)  # nosec B311 - dados sintéticos de avaliação
    transfer_ok = 0
    for _ in range(_AR_TRANSFER_SIZE):
        mapping_spec, text = _ar_transfer_instance(transfer_rng)
        src = sit_cls("s", "S", list(mapping_spec), [], text)
        system = _ar_system(cls, [])
        out = system.transfer_solution(map_cls("m", src, src, dict(mapping_spec), 1.0))
        expected = "".join(mapping_spec.get(tok, tok) for tok in re.split(r"(\w+)", text))
        transfer_ok += out == expected and system.successful_transfers == 1
    transfer_props = {"ok": transfer_ok, "total": _AR_TRANSFER_SIZE}

    # reason_by_analogy usa as entidades extraídas (AR9)
    reason_table = {}
    for name, (solution, text, domain, _expected, _counts) in _AR_REASON_CASES.items():
        system = _ar_system(cls, [("S", ["Rain", "Flood"], [(_AR_C, "Rain", "Flood")], solution)])
        out = system.reason_by_analogy(text, domain)
        reason_table[name] = [out, [system.total_analogies_found, system.successful_transfers]]

    # famílias de propósito (AR3, AR4)
    families = {}
    for items in _AR_FAMILIES.values():
        for name, fn in items.items():
            outcomes = [bool(fn(cls, seed)) for seed in _AR_SEEDS]
            families[name] = {"hits": sum(outcomes), "total": len(outcomes), "rate": sum(outcomes) / len(outcomes),
                              "outcomes": outcomes}

    # estatísticas, colaboradores, gaps, erros (AR5)
    rate_system = _ar_system(cls, [("S", ["a", "b"], [(c, "a", "b")], "move a")])
    rate_map = rate_system.find_analogy("T", ["x", "y"], [(c, "x", "y")])
    rate_system.transfer_solution(rate_map)
    rate_system.transfer_solution(rate_map)
    repeated = rate_system.get_statistics()

    boom_args = {"reasoning": _ArBoom(), "memory": _ArBoom(), "semantic_networks": _ArBoom()}
    boom = _ar_system(cls, [], **boom_args)
    boom.add_situation("S", ["Rain", "Flood"], [(c, "Rain", "Flood")], "stop Rain")
    boom_map = boom.find_analogy("T", ["x", "y"], [(c, "x", "y")])
    boom.transfer_solution(boom_map)
    boom.reason_by_analogy("Storm causes Damage.", "T")
    boom.get_statistics()
    wired_objects = (object(), object(), object())
    wired = _ar_system(cls, [], reasoning=wired_objects[0], memory=wired_objects[1], semantic_networks=wired_objects[2])
    collaborators = {"untouched": True,
                     "wired": [wired.reasoning is wired_objects[0], wired.memory is wired_objects[1],
                               wired.semantic_networks is wired_objects[2]]}

    placeholder = _ar_system(cls, [("S", ["Rain", "Flood"], [(c, "Rain", "Flood")], "stop Rain")])
    case_domain = _ar_system(cls, [("Physics", ["a", "b"], [(c, "a", "b")], None)])
    alias_rels, alias_ents = [(c, "a", "b")], ["a", "b"]
    alias_system = _ar_system(cls, [])
    alias_system.add_situation("S", alias_ents, alias_rels, None)
    alias_rels.append((p, "a", "b"))
    tgt_ents = ["x", "y"]
    alias_map = _ar_system(cls, [("S", ["a", "b"], [(c, "a", "b")], None)]).find_analogy(
        "T", tgt_ents, [(c, "x", "y")])
    gaps = {
        "repeated_transfer_stats": repeated,
        "placeholder_transfer": placeholder.reason_by_analogy("Storm causes Damage.", "T"),
        "the_cat_transfer": placeholder.reason_by_analogy("The cat causes Harm.", "T"),
        "single_entity_placeholder": placeholder.reason_by_analogy("it causes Harm.", "T"),
        "dissimilar": [list(r) for r in probe._extract_relations("dissimilar to the rest")],
        "counterpart": [list(r) for r in probe._extract_relations("a counterpart of b")],
        "negation": [list(r) for r in probe._extract_relations("X never causes Y")],
        "the_cat": probe._extract_entities("The cat causes Harm"),
        "single_letter": probe._extract_entities("A. Bob"),
        "domain_case_found": case_domain.find_analogy("physics", ["x", "y"], [(c, "x", "y")]) is not None,
        "aliased_relations": alias_system.known_situations[0].relations is alias_rels
        and len(alias_system.known_situations[0].relations) == 2,
        "aliased_target_entities": alias_map.target_situation.entities is tgt_ents,
    }

    def mapping_with(solution):
        src = sit_cls("s", "S", ["a"], [(c, "a", "b")], solution)
        return map_cls("m", src, src, {"a": "x"}, 1.0)

    known_one = [("S", ["a", "b"], [(c, "a", "b")], None)]
    errors = {
        "add_all_none": _ar_err(lambda: _ar_system(cls, []).add_situation(None, None, None)),
        "find_rels_none_no_known": _ar_err(lambda: _ar_system(cls, []).find_analogy("T", ["x"], None)),
        "find_rels_none_with_known": _ar_err(lambda: _ar_system(cls, known_one).find_analogy("T", ["x"], None)),
        "find_rel_two_items": _ar_err(lambda: _ar_system(cls, known_one).find_analogy("T", ["x"], [(c, "x")])),
        "find_rel_int": _ar_err(lambda: _ar_system(cls, known_one).find_analogy("T", ["x"], [5])),
        "source_rels_none": _ar_err(lambda: _ar_system(cls, [("S", ["a"], None, None)]).find_analogy(
            "T", ["x", "y"], [(c, "x", "y")])),
        "find_no_args": _ar_err(lambda: _ar_system(cls, []).find_analogy()),
        "transfer_none": _ar_err(lambda: _ar_system(cls, []).transfer_solution(None)),
        "transfer_solution_int": _ar_err(lambda: _ar_system(cls, []).transfer_solution(mapping_with(5))),
        "reason_none": _ar_err(lambda: _ar_system(cls, []).reason_by_analogy(None, "T")),
        "reason_int": _ar_err(lambda: _ar_system(cls, []).reason_by_analogy(5, "T")),
        "reason_empty": _ar_err(lambda: _ar_system(cls, []).reason_by_analogy("", "T")),
        "reason_blank": _ar_err(lambda: _ar_system(cls, []).reason_by_analogy("   ", "T")),
        "extract_entities_none": _ar_err(lambda: _ar_system(cls, [])._extract_entities(None)),
        "construct_extra": _ar_err(lambda: cls(1, 2, 3, 4)),
    }
    return {
        "clock": "none",
        "randomness": "uuid4 ids only",
        "init_params": list(inspect.signature(cls.__init__).parameters),
        "banner": banner,
        "situation_fields": list(inspect.signature(sit_cls).parameters),
        "mapping_fields": list(inspect.signature(map_cls).parameters),
        "fresh_stats": fresh_stats,
        "grid": grid_info,
        "table": table,
        "map_table": map_table, "map_props": map_props,
        "transfer_table": transfer_table, "transfer_props": transfer_props,
        "reason_table": reason_table,
        "families": families,
        "collaborators": collaborators,
        "gaps": gaps,
        "errors": errors,
    }


def analogical_reasoning_source_text(src: str) -> str:
    idx = src.index(_AR_MARKER)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end = src.index(_AR_END, idx) + len(_AR_END)
    return src[line_start:end]


def _analogical_reasoning_source_sha(path="core/cognitivo/analogical_reasoning.py") -> str:
    import hashlib
    text = (ROOT / path).read_text(encoding="utf-8")
    return hashlib.sha256(analogical_reasoning_source_text(text).encode()).hexdigest()


def _analogical_reasoning_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha AR6."""
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/analogical_reasoning.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" \
                    or name.startswith("core.constitutional_brain."):
                return False
    return True


def _ar_rates(payload, ids):
    return {name: {"hits": payload["families"][name]["hits"], "total": payload["families"][name]["total"],
                   "rate": payload["families"][name]["rate"]} for name in ids}


def eval_analogical_reasoning(criteria="v1"):
    """AR1–AR10. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import core.cognitivo.analogical_reasoning as analogical_reasoning
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = analogical_reasoning_fingerprint(analogical_reasoning)
    sha_module = _ms_sha(payload)
    sha_brain = _ms_sha(analogical_reasoning_fingerprint(brain))
    source_sha = _analogical_reasoning_source_sha()
    standalone = _analogical_reasoning_standalone()
    identity = all(
        getattr(brain, name) is getattr(analogical_reasoning, name) is getattr(package, name)
        and getattr(analogical_reasoning, name).__module__ == "core.cognitivo.analogical_reasoning"
        for name in _AR_NAMES
    )
    grid = payload["grid"]
    ar1 = (
        grid["count"] == _AR_GRID_SIZE
        and grid["reference_matches"] == _AR_GRID_SIZE
        and grid["counter_matches"] == _AR_GRID_SIZE
        and grid["mutated_inputs"] == 0
        and grid["found_sum"] == grid["ref_found_sum"]
        and grid["transfer_sum"] == grid["ref_transfer_sum"]
    )
    table = payload["table"]
    table_hits = sum(1 for name, expected in _AR_TABLE_EXPECTED.items() if table.get(name) == expected)
    ar2 = len(table) == len(_AR_TABLE_EXPECTED) == 19 and table_hits == 19
    ar3_ids, ar4_ids = tuple(_AR_FAMILIES["AR3"]), tuple(_AR_FAMILIES["AR4"])
    rates3, rates4 = _ar_rates(payload, ar3_ids), _ar_rates(payload, ar4_ids)
    ar3 = all(item["rate"] >= _AR_LIMIT for item in rates3.values())
    ar4 = all(item["rate"] >= _AR_LIMIT for item in rates4.values())
    gaps = payload["gaps"]
    gap_hits = sum(1 for name, expected in _AR_GAPS_EXPECTED.items() if gaps.get(name) == expected)
    error_hits = sum(1 for name, expected in _AR_ERRORS_EXPECTED.items() if payload["errors"].get(name) == expected)
    ar5 = (
        payload["fresh_stats"] == _AR_FRESH_STATS
        and gap_hits == len(_AR_GAPS_EXPECTED) == len(gaps)
        and error_hits == len(_AR_ERRORS_EXPECTED) == len(payload["errors"])
        and payload["collaborators"] == {"untouched": True, "wired": [True, True, True]}
    )
    ar6 = (
        identity
        and standalone
        and source_sha == AR_SOURCE_SHA
        and sha_module == sha_brain == AR_BEHAVIOR_SHA
        and payload["clock"] == "none"
        and payload["init_params"] == ["self", "reasoning", "memory", "semantic_networks"]
        and payload["banner"] == _AR_BANNER
    )
    map_hits = sum(1 for name, expected in _AR_MAP_EXPECTED.items() if payload["map_table"].get(name) == expected)
    map_props = payload["map_props"]
    ar7 = (
        len(payload["map_table"]) == len(_AR_MAP_EXPECTED) == 14 and map_hits == 14
        and map_props["ok"] == map_props["total"] == _AR_PROP_SIZE and map_props["mutated"] == 0
    )
    transfer_hits = sum(1 for name, (_m, _s, expected) in _AR_TRANSFER_CASES.items()
                        if payload["transfer_table"].get(name) == [expected, 1])
    transfer_props = payload["transfer_props"]
    ar8 = (
        len(payload["transfer_table"]) == len(_AR_TRANSFER_CASES) == 12 and transfer_hits == 12
        and transfer_props["ok"] == transfer_props["total"] == _AR_TRANSFER_SIZE
    )
    reason_hits = sum(1 for name, (_sol, _text, _dom, expected, counts) in _AR_REASON_CASES.items()
                      if payload["reason_table"].get(name) == [expected, counts])
    ar9 = len(payload["reason_table"]) == len(_AR_REASON_CASES) == 8 and reason_hits == 8
    import hashlib
    base_rows_ok = (
        len(_AR_BASE_ROWS) == _AR_GRID_SIZE
        and hashlib.sha256(json.dumps(list(_AR_BASE_ROWS)).encode()).hexdigest() == _AR_BASE_ROWS_SHA
    )
    unaffected_rows = [i for i, flag in enumerate(grid["unaffected"]) if flag]
    base_diffs = [i for i in unaffected_rows if grid["row_digests"][i] != _AR_BASE_ROWS[i]] if base_rows_ok else []
    ar10_invalid = not base_rows_ok or len(unaffected_rows) < _AR_MIN_UNAFFECTED
    ar10 = not ar10_invalid and not base_diffs
    return [
        _check("AR1", {"reference_matches": grid["reference_matches"], "count": grid["count"],
                       "counter_matches": grid["counter_matches"], "mutated_inputs": grid["mutated_inputs"],
                       "found_sum": [grid["found_sum"], grid["ref_found_sum"]],
                       "transfer_sum": [grid["transfer_sum"], grid["ref_transfer_sum"]]},
               ar1, "120/120 iguais à referência independente v2; contadores iguais; 0 entradas alteradas"),
        _check("AR2", {"hits": table_hits, "cases": len(_AR_TABLE_EXPECTED)},
               ar2, "19/19 casos de borda de domínio, rótulo, qualidade, empate, transferência e extração"),
        _check("AR3", rates3, ar3, "one_to_one, injective e control_isomorphic ≥ 0.90 em 20 seeds"),
        _check("AR4", rates4, ar4,
               "real_entities, simultaneous, whole_word e control_plain ≥ 0.90 em 20 seeds"),
        _check("AR5", {"fresh_stats": payload["fresh_stats"], "gap_hits": [gap_hits, len(_AR_GAPS_EXPECTED)],
                       "error_hits": [error_hits, len(_AR_ERRORS_EXPECTED)],
                       "collaborators": payload["collaborators"], "gaps": gaps, "errors": payload["errors"]},
               ar5, "contadores, colaboradores, erros e gaps conforme o registrado"),
        _check("AR6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": AR_SOURCE_SHA, "behavior_golden": AR_BEHAVIOR_SHA},
               ar6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
        _check("AR7", {"table_hits": [map_hits, len(_AR_MAP_EXPECTED)], "props": map_props},
               ar7, "mapeamento um-para-um e injetivo: 14/14 casos e 200/200 instâncias da grade de propriedades"),
        _check("AR8", {"table_hits": [transfer_hits, len(_AR_TRANSFER_CASES)], "props": transfer_props},
               ar8, "transferência simultânea e por palavra inteira: 12/12 casos e 100/100 instâncias"),
        _check("AR9", {"hits": [reason_hits, len(_AR_REASON_CASES)], "cases": payload["reason_table"]},
               ar9, "reason_by_analogy usa as entidades extraídas: 8/8 casos"),
        _check("AR10", {"unaffected": len(unaffected_rows), "min_unaffected": _AR_MIN_UNAFFECTED,
                        "base_rows_sha_ok": base_rows_ok, "diffs": base_diffs},
               ar10, "linhas não afetadas iguais ao resumo congelado do código base (0 diferenças; ≥ 52 linhas)",
               **({"scenario_invalid": True} if ar10_invalid else {})),
    ]


# ── Contextual Memory (POST14) ──────────────────────────────────────────────

CM_SOURCE_SHA = "02b7fc0b23a6fed4a56ab9584524a7f87498e7ed21c5210ad41c673a2c5d8a64"
CM_BEHAVIOR_SHA = "91c326795ff80049b0c3b2f802227d5391d7076f3de3e84e31c30bb604a56ad2"  # capturado uma única vez no monólito antes do move; 3.11 e 3.12
_CM_NAMES = ("ContextualMemory", "ContextMatch", "ContextEncoder", "RealContextualMemorySystem")
_CM_MARKER = "# MÓDULO 14: CONTEXTUAL MEMORY SYSTEM (POST 14)"
_CM_END = '            "current_context_dimensions": len(self.current_context)\n        }'
_CM_BANNER = "🧠🔍 POST 14 - RealContextualMemorySystem initialized (Context-dependent recall active)\n"
_CM_POOLS = {
    "location": ("office", "home", "lab", "cafe", "gym", "park", "library", "train"),
    "time": ("morning", "noon", "evening", "night", "dawn", "weekend"),
    "mood": ("calm", "focused", "tense", "happy", "sad", "bored"),
    "task": ("coding", "reading", "review", "cooking", "running", "writing"),
    "social": ("alone", "team", "family", "friends", "client"),
}
_CM_WEATHER = ("rain", "sun", "fog")
_CM_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_CM_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
_CM_GRID_SEED = 2014
_CM_GRID_SIZE = 120
_CM_SEEDS = tuple(range(20))
_CM_LIMIT = 0.90
_CM_PRECISION = 0.80
_CM_CTX_SEED = 4014
_CM_HASH_SEEDS = (101, 101, 202)
_CM_BASE_ROWS_SHA = "580f978aae9c432701d3ff36aeac39377dca8815ce37c05d40e22ecc4c5cd728"
_CM_MIN_UNAFFECTED = 12
# resumo (16 hex) por linha da grade de CM1 no código base 0627ca4, capturado antes da correção (CM9)
_CM_BASE_ROWS = (
    "0140b931d1297d2c",
    "147e1ef664fa92b4",
    "41abad9a3f13ddf5",
    "cd8983f17c049dc7",
    "bde545be62a144b1",
    "f3489dde828db497",
    "4ec5bbda65757f88",
    "7f08fcb9c6789c42",
    "d9c4ecb6605aa941",
    "728939031f1b3863",
    "456946640f679b1e",
    "64a0c12c96b9d477",
    "e8fd0a9b73b71be6",
    "94f4799bfc734f40",
    "9dfbf6c6c397978a",
    "42c4e180e813155a",
    "a8bc24faf45e9ac8",
    "d27680e306c320a7",
    "332b6e8be1857df5",
    "62287686d2b3f629",
    "ec70d54050d0fa8f",
    "0414bcf9fc900df0",
    "80e6ea95628bb7db",
    "31419a4698894329",
    "6a435b7549cc26f9",
    "4503834f6fe44871",
    "41e576ae05fc94ff",
    "04fd67b9abb30f21",
    "6d8138e6deb1f63a",
    "b686f7a11464bfb0",
    "02973d5e21e0c2ed",
    "d94ff505c5724dd4",
    "63417aadaf1980d6",
    "39a84f890535f5a6",
    "ad749809741b7fc3",
    "3424b401d850d0e4",
    "1e5861315e1ecd53",
    "917c048daca7776c",
    "f1ab623fa774aa66",
    "631b2f2892fd69b3",
    "6de030c0f0c468f9",
    "30698a45bada91b4",
    "4fb66ea0972ac4f4",
    "cd0cf4f39009ca27",
    "1e6efc673ab158e3",
    "a4b61e24a42993a6",
    "797b362bf2137b13",
    "c833007fc7cd0689",
    "0bffea6a8eccf0c4",
    "dbe0e6a16c6fead2",
    "80e6ea95628bb7db",
    "6199a4bc5c756974",
    "4c5b66f4280d9918",
    "4283ce1579698164",
    "19174c73e4cbcc60",
    "a0e5a11e77c5925c",
    "47d27f5ae7b94706",
    "f3654f9728b43154",
    "03001de86ea8991f",
    "561548656717eb32",
    "e65a48a182ae504e",
    "a6f5b8abaaff3a7e",
    "c9fee29988746950",
    "9eee3561cf148e45",
    "413ec31ee4791c52",
    "5a3ecb81339e2aac",
    "a27091603e36e658",
    "5a979df47c05a372",
    "382f9fde7b1ece78",
    "7b3ea2df5603dfa3",
    "b4d02750bf44194a",
    "e6c3e5056581446a",
    "fa147f08f58ae1af",
    "2377b3bae6239e5b",
    "5d7e8a8bb2c1c3a0",
    "e5f3865d69c66f76",
    "0f5183a3b1bb86ce",
    "fb0f9199aa26f43a",
    "0b0b587b25566c0d",
    "858e27f18d61a91e",
    "df656edea8ce6889",
    "cc85a095f49bd8d2",
    "29fbe60f2b26cd24",
    "1fc3ac9c92816b65",
    "581b2158873b6184",
    "6f31d003db1435f1",
    "5c1f0a9dbe71f52c",
    "c0688c4a6bf43e15",
    "35f0b8e9005b04e5",
    "e499a627b9a53743",
    "315b5bd68cccb2eb",
    "34619a3c5ae8ff38",
    "926d6ce979101b7e",
    "201a777167b1f9a3",
    "cc11636a4a75702c",
    "0b974be6cee81848",
    "5b92065b80d0b9b2",
    "6e51b2a5230ce60b",
    "b029bdfaf54cb7ba",
    "bbf85a3f10b829fe",
    "8424872c7af63e7c",
    "0f8951412c49215e",
    "d34ede8c5018c0d5",
    "38b4db2e0fcbcaab",
    "b3e177b6e31afa61",
    "35baae12d0f2080a",
    "737e103c07d7ceca",
    "8e17c034a3fe2555",
    "4fbf1fdcb8fa2c1a",
    "fe4ff75177824807",
    "4b940cba17289615",
    "bec48b4fc70f43c5",
    "5480e3638937f637",
    "b65ea03cde898999",
    "4768dd0166090b89",
    "2e22b7467cc0fcdd",
    "2c8ef2c42a1a8d08",
    "be7cf56abc4a96d8",
    "28d58973e253a8fd",
    "8aadaf6642e0e5da",
)
_CM_STABILITY_SEED = 4016
_CM_STABILITY_SIZE = 200
_CM_STABILITY_VALUES = ("office", "café", "日本語", "😀", "\ud800", 1, 2.5, None, ("a", 1), True, "", "x" * 50)
_CM_GOLDEN = (
    ({}, [0.04, 0.04, 0.04, 0.04, 0.04]),
    ({"location": "office"}, [0.2, 0.04, 0.04, 0.04, 0.04]),
    ({"location": "office", "time": "morning", "mood": "calm", "task": "reading", "social": "alone"},
     [0.2, 0.32, 0.37, 0.78, 0.2]),
    ({"mood": "focused", "task": "review"}, [0.04, 0.04, 0.25, 0.2, 0.04]),
    ({"location": 1}, [0.19, 0.04, 0.04, 0.04, 0.04]),
    ({"social": "team", "weather": "rain"}, [0.04, 0.04, 0.04, 0.04, 0.48]),
    ({"mood": "\ud800"}, [0.04, 0.04, 0.05, 0.04, 0.04]),
)
_CM_M = {"location": "office", "time": "morning", "mood": "calm", "task": "reading", "social": "alone"}
_CM_SWAPS = (("location", "home"), ("time", "noon"), ("mood", "focused"), ("task", "review"), ("social", "team"))
_CM_CONTENT = "alpha beta"
# (nome, contexto da memória, consulta de texto, contexto de recuperação, esperado); M = _CM_M
_CM8_SCORES = (
    ("a1", "M", None, 0, 1.0), ("a2", "M", None, 1, 0.8), ("a3", "M", None, 2, 0.6), ("a4", "M", None, 3, 0.4),
    ("a5", "M", None, 4, 0.2), ("a6", "M", None, 5, 0.0),
    ("a7", "M", None, {"location": "office"}, 1.0),
    ("a8", "M", None, {"location": "office", "mood": "focused"}, 0.5),
    ("a9", "M", None, {}, 1.0),
    ("a10", "M", None, {"weather": "rain"}, 1.0),
    ("a11", {}, None, {"location": "office"}, 0.0),
    ("a12", {}, None, {"location": "unknown"}, 1.0),
    ("a13", {"mood": 1}, None, {"mood": "1"}, 1.0),
    ("a14", "M", "alpha", 0, 1.0), ("a15", "M", "zzz", 0, 0.65),
    ("a16", "M", "alpha", 5, 0.5), ("a17", "M", "zzz", 5, 0.15),
    ("a18", "M", "", 1, 0.8),
)
_CM8_BANDS = {"contents": ["d0", "d1", "d2"], "scores": [1.0, 0.8, 0.6], "features": [5, 4, 3], "retrievals": 1}
_CM8_MATCH = (
    ("c1", ([0.1, 0.2], [0.1, 0.3]), 0.5), ("c2", ([], [1.0]), 0.0), ("c3", (None, [1.0]), 0.0),
    ("c4", ([1.0], [1.0], []), 1.0), ("c5", ([0.1, 0.2, 0.3], [0.1]), 1.0),
    ("c6", ([0.1, 0.2], [0.1, 0.3], [1]), 0.0), ("c7", ([0.1, 0.2], [0.1, 0.3], [0]), 1.0),
)
_CM_FRESH_STATS = {"total_encodings": 0, "total_retrievals": 0, "context_matched_retrievals": 0,
                   "context_match_rate": 0.0, "stored_memories": 0, "current_context_dimensions": 0}
_CM_COUNTERS_EXPECTED = {
    "stats": {"total_encodings": 2, "total_retrievals": 2, "context_matched_retrievals": 1,
              "context_match_rate": 0.5, "stored_memories": 2, "current_context_dimensions": 1},
    "retrieval_counts": [1, 1],
}
_CM_COLLAB_EXPECTED = {"ltm_calls": [["c", "contextual", 0.7], ["5", "contextual", 0.7]],
                       "falsy_ltm_untouched": True, "wm_attention_untouched": True}
_CM_GAPS_EXPECTED = {
    "default_threshold_returns_all": 0,
    "unlisted_keys_ignored": True,
    "aliased_current_context": {"aliased": True, "stale_vector": True},
    "empty_query": {"same_score_as_none": True, "nonmatching_lowers": False},
    "specificity_side_effects": {"score": 1.0, "encodings": 2, "retrievals": 1, "stored": 2,
                                 "prior_retrieval_count": 1},
    "reinstate": {"unknown_none": True, "unknown_keeps_context": True, "known_same_object": True,
                  "vector_updated": True},
    "matched_features": ["mood", "location", "task"],
    "cues": ["location=office", "mood=focused"],
    "similarity_edges": {"empty": 0.0, "zero_vector": 0.0, "orthogonal": 0.0, "same": 1.0,
                         "length_mismatch": 0.447213595},
    "encode_defaults": {"dims": ["location", "time", "mood", "task", "social"], "all_unknown_equal": True},
    "str_coerced_values": True,
    "key_order_independent": True,
    "failed_set_leaves_none": True,
    "ltm_failure_after_append": {"stored": 1, "encodings": 1},
    "cue_subset_is_enough": 1.0,
    "query_hit_other_context_passes_default": [1, 0.5],
    "query_miss_same_context_passes_default": [1, 0.65],
    "bucket_collision": True,
}
_CM_ERRORS_EXPECTED = {
    "set_ctx_none": "AttributeError", "set_ctx_list": "AttributeError", "store_ctx_int": "AttributeError",
    "retrieve_ctx_str": "AttributeError", "encode_none": "AttributeError", "cues_none": "AttributeError",
    "retrieve_threshold_str_with_memory": "TypeError", "retrieve_threshold_none_with_memory": "TypeError",
    "store_no_args": "TypeError", "construct_extra": "TypeError", "similarity_str": "TypeError",
    "store_ltm_raises": "RuntimeError",
    "retrieve_threshold_str_empty": "no-error", "retrieve_query_int": "no-error",
    "reinstate_none": "no-error", "similarity_none": "no-error",
    "match_none": "no-error", "match_cue_out_of_range": "IndexError",
}
_CM_PROC_SCRIPT = (
    "import json, sys\n"
    "from {source} import ContextEncoder\n"
    "enc = ContextEncoder()\n"
    "ctxs = json.loads(sys.stdin.read())\n"
    "print('CMVEC:' + json.dumps([enc.encode_context(c) for c in ctxs]))\n"
)


def _cm_hash(value):
    import hashlib
    return int(hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).hexdigest()[:8], 16)


class _cm_det_hash:
    """Troca builtins.hash por uma função determinística (a original depende de PYTHONHASHSEED)."""

    def __enter__(self):
        import builtins
        self._original = builtins.hash
        builtins.hash = _cm_hash

    def __exit__(self, *exc):
        import builtins
        builtins.hash = self._original
        return False


def _cm_err(fn):
    try:
        fn()
    except Exception as exc:
        return type(exc).__name__
    return "no-error"


def _cm_ctx(rng):
    ctx = {}
    for dim, pool in _CM_POOLS.items():
        if rng.random() < 0.7:
            ctx[dim] = rng.choice(pool)
    if rng.random() < 0.3:
        ctx["weather"] = rng.choice(_CM_WEATHER)
    return ctx


def _cm_grid():
    rng = random.Random(_CM_GRID_SEED)  # nosec B311 - dados sintéticos de avaliação
    rows = []
    for _ in range(_CM_GRID_SIZE):
        set_ctx = _cm_ctx(rng) if rng.random() < 0.6 else None
        memories = []
        for j in range(rng.randint(0, 8)):
            content = f"{rng.choice(_CM_WORDS)} {rng.choice(_CM_WORDS)} {j}"
            memories.append((content, _cm_ctx(rng) if rng.random() < 0.8 else None))
        retrievals = []
        for _r in range(rng.randint(1, 3)):
            query = rng.choice([None, "", rng.choice(_CM_WORDS), "zzz-none"])
            context = None if rng.random() < 0.3 else _cm_ctx(rng)
            retrievals.append((query, context, rng.choice(_CM_THRESHOLDS)))
        reinstate = rng.randrange(len(memories)) if memories and rng.random() < 0.8 else None
        rows.append((set_ctx, memories, retrievals, reinstate))
    return rows


def _cm_ref_vec(ctx):
    return [_cm_hash(ctx.get(dim, "unknown")) % 100 / 100.0 for dim in _CM_POOLS]


def _cm_ref_cos(left, right):
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = sum(a * a for a in left) ** 0.5
    norm_r = sum(b * b for b in right) ** 0.5
    if norm_l == 0 or norm_r == 0:
        return 0.0
    return dot / (norm_l * norm_r)


def _cm_ref_match(item_vec, ctx):
    """v2: fração das dimensões consultadas (chave presente no contexto) cujo balde coincide."""
    vec = _cm_ref_vec(ctx)
    cued = [i for i, dim in enumerate(_CM_POOLS) if dim in ctx]
    if not cued:
        return 1.0
    return sum(1 for i in cued if item_vec[i] == vec[i]) / len(cued)


def _cm_reference(row, legacy=False):
    """Referência independente. `legacy=True` é a v1 (cosseno e content_match fixo), usada só em CM9."""
    set_ctx, memories, retrievals, reinstate = row
    current = set_ctx if set_ctx is not None else {}
    stored = []
    for content, ctx in memories:
        ctx = current if ctx is None else ctx
        stored.append({"content": content, "ctx": ctx, "vec": _cm_ref_vec(ctx), "n": 0})
    retrieved, matched_calls = [], 0
    for query, ctx, threshold in retrievals:
        ctx = current if ctx is None else ctx
        vec = _cm_ref_vec(ctx)
        found = []
        for item in stored:
            if legacy:
                content_match = 1.0
                if query:
                    content_match = 1.0 if str(query).lower() in str(item["content"]).lower() else 0.3
                score = (_cm_ref_cos(item["vec"], vec) + content_match) / 2.0
            else:
                score = _cm_ref_match(item["vec"], ctx)
                if query:
                    content_match = 1.0 if str(query).lower() in str(item["content"]).lower() else 0.3
                    score = (score + content_match) / 2.0
            if score >= threshold:
                item["n"] += 1
                feats = [k for k in item["ctx"] if k in ctx and item["ctx"][k] == ctx[k]]
                found.append((item["content"], score, feats, item["n"]))
        found = sorted(found, key=lambda t: t[1], reverse=True)
        matched_calls += 1 if found else 0
        retrieved.append(found)
    reinstated = None
    if reinstate is not None:
        reinstated = stored[reinstate]["ctx"]
        current = reinstated
    total = len(retrievals)
    return {
        "contents": [item["content"] for item in stored],
        "retrieved": retrieved,
        "reinstated": reinstated,
        "current": current,
        "cues": [f"{k}={v}" for k, v in stored[0]["ctx"].items()] if stored else None,
        "stats": {"total_encodings": len(stored), "total_retrievals": total,
                  "context_matched_retrievals": matched_calls,
                  "context_match_rate": matched_calls / total if total > 0 else 0.0,
                  "stored_memories": len(stored), "current_context_dimensions": len(current)},
    }


def _cm_system(cls, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(**kwargs)


def _cm_run(cls, row):
    set_ctx, memories, retrievals, reinstate = row
    system = _cm_system(cls)
    if set_ctx is not None:
        system.set_current_context(set_ctx)
    before = time.time()
    stored = [system.store_with_context(content, ctx) for content, ctx in memories]
    after = time.time()
    retrieved = []
    for query, ctx, threshold in retrievals:
        found = system.retrieve_by_context(query=query, context=ctx, threshold=threshold)
        retrieved.append([(m.memory.content, m.match_score, list(m.matched_features), m.memory.retrieval_count)
                          for m in found])
    reinstated = system.reinstate_context(stored[reinstate].memory_id if reinstate is not None else "missing")
    ids = [m.memory_id for m in stored]
    meta_ok = (
        len(set(ids)) == len(ids) and all(isinstance(i, str) and len(i) == 36 for i in ids)
        and all(isinstance(m.timestamp, float) and before <= m.timestamp <= after for m in stored)
    )
    return {
        "contents": [m.content for m in stored],
        "retrieved": retrieved,
        "reinstated": reinstated,
        "current": system.current_context,
        "cues": system.get_context_dependent_cues(stored[0]) if stored else None,
        "stats": system.get_statistics(),
    }, meta_ok


def _cm_same(view, ref):
    if view["contents"] != ref["contents"] or len(view["retrieved"]) != len(ref["retrieved"]):
        return False
    for got, want in zip(view["retrieved"], ref["retrieved"]):
        if len(got) != len(want):
            return False
        for (c1, s1, f1, n1), (c2, s2, f2, n2) in zip(got, want):
            if c1 != c2 or abs(s1 - s2) > 1e-12 or f1 != f2 or n1 != n2:
                return False
    stats_got, stats_want = view["stats"], ref["stats"]
    return (
        view["reinstated"] == ref["reinstated"] and view["current"] == ref["current"] and view["cues"] == ref["cues"]
        and set(stats_got) == set(stats_want)
        and all(abs(stats_got[k] - stats_want[k]) < 1e-12 for k in stats_want)
    )


def _cm_row_digest(view):
    rounded = [[(c, round(s, 12), f, n) for c, s, f, n in rows] for rows in view["retrieved"]]
    return _ms_sha({"contents": view["contents"], "retrieved": rounded, "reinstated": view["reinstated"],
                    "current": view["current"], "cues": view["cues"], "stats": view["stats"]})


def _cm_scenario(cls, seed):
    rng = random.Random(seed)  # nosec B311 - dados sintéticos de avaliação
    picks = {dim: rng.sample(pool, 5) for dim, pool in _CM_POOLS.items()}
    ctxs = [{dim: picks[dim][c] for dim in _CM_POOLS} for c in range(5)]
    system = _cm_system(cls)
    for c, ctx in enumerate(ctxs):
        for k in range(6):
            system.store_with_context(f"m{seed}_{c}_{k}", dict(ctx))
    vecs = [_cm_ref_vec(ctx) for ctx in ctxs]
    valid = len({tuple(v) for v in vecs}) == 5 and all(_cm_ref_cos(vecs[0], v) < 1 - 1e-9 for v in vecs[1:])
    return system, ctxs, valid


def _cm_selectivity_families(cls):
    ranked = precise = recalled = invalid = 0
    returned = []
    for seed in _CM_SEEDS:
        system, ctxs, valid = _cm_scenario(cls, seed)
        invalid += 0 if valid else 1
        want = {f"m{seed}_0_{k}" for k in range(6)}
        top = system.retrieve_by_context(context=ctxs[0], threshold=0.0)
        ranked += {m.memory.content for m in top[:6]} == want
        default = system.retrieve_by_context(context=ctxs[0])
        hits = sum(1 for m in default if m.memory.content in want)
        returned.append(len(default))
        precise += bool(default) and hits / len(default) >= _CM_PRECISION
        recalled += want <= {m.memory.content for m in default}
    total = len(_CM_SEEDS)
    return {"CM2": {"hits": ranked, "total": total, "rate": ranked / total, "invalid": invalid},
            "CM3": {"hits": precise, "total": total, "rate": precise / total, "returned": returned,
                    "recall_hits": recalled, "recall_rate": recalled / total}}


def _cm_process_vectors(source, ctxs, hash_seed):
    import subprocess  # nosec B404 - roda só o interpretador atual com script fixo
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(  # nosec B603 - interpretador atual e script fixo do harness
        [sys.executable, "-c", _CM_PROC_SCRIPT.format(source=source)],
        input=json.dumps(ctxs), capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("CMVEC:")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"subprocesso falhou ({source}, seed {hash_seed}): {proc.stderr[-400:]}")
    return json.loads(lines[-1][len("CMVEC:"):])


def _cm_determinism(source):
    rng = random.Random(_CM_CTX_SEED)  # nosec B311 - dados sintéticos de avaliação
    ctxs = [{dim: rng.choice(pool) for dim, pool in _CM_POOLS.items()} for _ in range(20)]
    first, again, other = (_cm_process_vectors(source, ctxs, seed) for seed in _CM_HASH_SEEDS)
    control = sum(1 for a, b in zip(first, again) if a == b)
    cross = sum(1 for a, b in zip(first, other) if a == b)
    return {"contexts": len(ctxs), "control_same": control, "cross_same": cross,
            "control_rate": control / len(ctxs), "cross_rate": cross / len(ctxs)}


def _cm_try(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - o tipo do erro vira o valor medido  # nosec B110
        return "erro:" + type(exc).__name__


def _cm_swapped(count):
    ctx = dict(_CM_M)
    for dim, value in _CM_SWAPS[:count]:
        ctx[dim] = value
    return ctx


def _cm_stability(enc_cls):
    """CM7: vetores dourados, sem chamar builtins.hash, e grade contra a referência de hashlib."""
    import builtins
    import copy
    enc = enc_cls()
    golden = sum(1 for ctx, vec in _CM_GOLDEN if _cm_try(lambda: enc.encode_context(ctx)) == vec)

    def boom(*_args, **_kwargs):
        raise AssertionError("builtins.hash chamado")

    original = builtins.hash
    builtins.hash = boom
    try:
        no_hash = sum(1 for ctx, vec in _CM_GOLDEN if _cm_try(lambda: enc.encode_context(ctx)) == vec)
    finally:
        builtins.hash = original
    rng = random.Random(_CM_STABILITY_SEED)  # nosec B311 - dados sintéticos de avaliação
    ok = mutated = 0
    for _ in range(_CM_STABILITY_SIZE):
        dims = rng.sample(list(_CM_POOLS), rng.randint(0, 5))
        ctx = {dim: rng.choice(_CM_STABILITY_VALUES) for dim in dims}
        if rng.random() < 0.3:
            ctx["weather"] = rng.choice(_CM_STABILITY_VALUES)
        snapshot = copy.deepcopy(ctx)
        vec = _cm_try(lambda: enc.encode_context(ctx))
        ok += vec == _cm_ref_vec(ctx) and len(vec) == len(_CM_POOLS) and all(0.0 <= x < 1.0 for x in vec)
        mutated += repr(ctx) != repr(snapshot)
    return {"golden": golden, "golden_total": len(_CM_GOLDEN), "no_hash": no_hash,
            "grid_ok": ok, "grid_total": _CM_STABILITY_SIZE, "mutated_inputs": mutated}


def _cm_scores(cls, enc_cls):
    """CM8: tabela de score e seletividade."""
    table = {}
    for name, mem_ctx, query, retr, _expected in _CM8_SCORES:
        memory = _cm_swapped(0) if mem_ctx == "M" else mem_ctx
        retrieval = _cm_swapped(retr) if isinstance(retr, int) else retr

        def run(memory=memory, query=query, retrieval=retrieval):
            system = _cm_system(cls)
            system.store_with_context(_CM_CONTENT, dict(memory))
            found = system.retrieve_by_context(query=query, context=dict(retrieval), threshold=0.0)
            return round(found[0].match_score, 6) if found else None

        table[name] = _cm_try(run)

    def bands():
        system = _cm_system(cls)
        for count in range(6):
            system.store_with_context(f"d{count}", _cm_swapped(count))
        found = system.retrieve_by_context(context=_cm_swapped(0))
        return {"contents": [m.memory.content for m in found],
                "scores": [round(m.match_score, 6) for m in found],
                "features": [len(m.matched_features) for m in found],
                "retrievals": system.get_statistics()["total_retrievals"]}

    table["b1"] = _cm_try(bands)
    for name, args, _expected in _CM8_MATCH:
        table[name] = _cm_try(lambda args=args: enc_cls().compute_match(*args))
    return table


def contextual_memory_fingerprint(ns):
    import copy
    import inspect
    if isinstance(ns, dict):
        get, source = ns.__getitem__, "core.constitutional_brain"
    else:
        def get(name):
            return getattr(ns, name)
        source = ns.__name__
    cls, mem_cls, match_cls, enc_cls = (get(name) for name in
                                        ("RealContextualMemorySystem", "ContextualMemory", "ContextMatch",
                                         "ContextEncoder"))
    stability = _cm_stability(enc_cls)  # fora do substituto: mede o hash real do módulo (CM7)
    with _cm_det_hash():
        return _cm_fingerprint_body(cls, mem_cls, match_cls, enc_cls, source, copy, inspect, stability)


def _cm_fingerprint_body(cls, mem_cls, match_cls, enc_cls, source, copy, inspect, stability):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fresh = cls()
    banner = buf.getvalue()
    fresh_stats = fresh.get_statistics()

    # grade (CM1)
    matches = mutated = meta_bad = 0
    digests, unaffected = [], []
    for row in _cm_grid():
        snapshot = copy.deepcopy(row)
        view, meta_ok = _cm_run(cls, row)
        ref = _cm_reference(row)
        matches += bool(_cm_same(view, ref))
        mutated += row != snapshot
        meta_bad += not meta_ok
        digests.append(_cm_row_digest(view))
        unaffected.append(_cm_row_digest(_cm_reference(row, legacy=True)) == _cm_row_digest(ref))
    grid_info = {"count": len(digests), "reference_matches": matches, "mutated_inputs": mutated,
                 "meta_bad": meta_bad, "rows_sha": _ms_sha(digests),
                 "row_digests": [d[:16] for d in digests], "unaffected": unaffected}

    families = _cm_selectivity_families(cls)
    determinism = _cm_determinism(source)

    # contadores
    counter_sys = _cm_system(cls)
    counter_sys.set_current_context({"location": "office"})
    counter_sys.store_with_context("alpha")
    counter_sys.store_with_context("beta", {"location": "home"})
    counter_sys.retrieve_by_context(threshold=2.0)
    counter_sys.retrieve_by_context(threshold=0.0)
    counters = {"stats": counter_sys.get_statistics(),
                "retrieval_counts": [m.retrieval_count for m in counter_sys.contextual_memories]}

    # colaboradores
    class Spy:
        def __init__(self):
            self.calls = []

        def store(self, *args, **kwargs):
            self.calls.append([args[0], args[1], kwargs.get("importance")])

    class Falsy:
        calls = 0

        def __bool__(self):
            return False

        def store(self, *args, **kwargs):
            Falsy.calls += 1

    class Boom:
        def __getattribute__(self, name):
            raise AssertionError(f"leitura de {name}")

    spy, falsy, boom = Spy(), Falsy(), Boom()
    spy_sys = _cm_system(cls, long_term_memory=spy, working_memory=boom, attention=boom)
    spy_sys.store_with_context("c", {"mood": "x"})
    spy_sys.store_with_context(5, {"mood": "y"})
    spy_sys.retrieve_by_context(context={"mood": "x"})
    spy_sys.get_statistics()
    falsy_sys = _cm_system(cls, long_term_memory=falsy)
    falsy_sys.store_with_context("z", {})
    collaborators = {"ltm_calls": spy.calls, "falsy_ltm_untouched": Falsy.calls == 0,
                     "wm_attention_untouched": spy_sys.wm is boom and spy_sys.attention is boom}

    # gaps
    enc = enc_cls()
    sys1 = _cm_system(cls)
    for i, ctx in enumerate(({"location": "a", "mood": "b"}, {"time": "c", "task": "d"},
                             {"social": "e", "location": "f"})):
        sys1.store_with_context(f"x{i}", ctx)
    default_returns = len(sys1.retrieve_by_context(context={"location": "zzz", "task": "q"}))

    alias_ctx = {"location": "office"}
    alias_sys = _cm_system(cls)
    alias_sys.set_current_context(alias_ctx)
    alias_mem = alias_sys.store_with_context("x")
    alias_ctx["location"] = "home"
    aliased = {
        "aliased": alias_mem.context is alias_ctx and alias_mem.context["location"] == "home"
        and alias_sys.current_context is alias_ctx,
        "stale_vector": alias_mem.context_vector == enc.encode_context({"location": "office"})
        and alias_mem.context_vector != enc.encode_context(alias_ctx),
    }

    q_sys = _cm_system(cls)
    q_sys.store_with_context("alpha", {"location": "a"})
    score_none = q_sys.retrieve_by_context(context={"location": "b"}, threshold=0.0)[0].match_score
    score_empty = q_sys.retrieve_by_context(query="", context={"location": "b"}, threshold=0.0)[0].match_score
    score_miss = q_sys.retrieve_by_context(query="zzz", context={"location": "b"}, threshold=0.0)[0].match_score

    spec_sys = _cm_system(cls)
    spec_sys.store_with_context("prior", {"location": "home"})
    spec_ctx = {"location": "lab", "task": "research"}
    spec_score = spec_sys.test_encoding_specificity("Python programming", dict(spec_ctx), dict(spec_ctx))
    spec_stats = spec_sys.get_statistics()

    re_sys = _cm_system(cls)
    re_mem = re_sys.store_with_context("x", {"location": "lab"})
    re_sys.set_current_context({"mood": "calm"})
    re_current = re_sys.current_context
    re_unknown = re_sys.reinstate_context("nope")
    re_keeps = re_sys.current_context is re_current
    re_known = re_sys.reinstate_context(re_mem.memory_id)
    reinstate = {
        "unknown_none": re_unknown is None, "unknown_keeps_context": re_keeps,
        "known_same_object": re_known is re_mem.context and re_sys.current_context is re_mem.context,
        "vector_updated": re_sys.current_context_vector == enc.encode_context({"location": "lab"}),
    }

    feat_sys = _cm_system(cls)
    feat_sys.store_with_context("x", {"mood": "x", "location": "a", "task": "t"})
    matched = feat_sys.retrieve_by_context(
        context={"location": "a", "task": "t", "mood": "x"}, threshold=0.0)[0].matched_features

    cue_sys = _cm_system(cls)
    cues = cue_sys.get_context_dependent_cues(cue_sys.store_with_context("x", {"location": "office", "mood": "focused"}))

    none_sys = _cm_system(cls)
    try:
        none_sys.set_current_context(None)
    except Exception:  # noqa: BLE001 - o registro do erro está em `errors`  # nosec B110
        pass

    class Raising:
        def store(self, *args, **kwargs):
            raise RuntimeError("ltm")

    ltm_sys = _cm_system(cls, long_term_memory=Raising())
    try:
        ltm_sys.store_with_context("x", {})
    except RuntimeError:
        pass

    sub_sys = _cm_system(cls)
    sub_sys.store_with_context("x", {"location": "office", "mood": "calm", "task": "reading"})
    sub_found = sub_sys.retrieve_by_context(context={"location": "office"})
    sub_score = round(sub_found[0].match_score, 9) if sub_found else None

    hit_sys = _cm_system(cls)
    hit_sys.store_with_context(_CM_CONTENT, {"location": "office"})
    hit_found = hit_sys.retrieve_by_context(query="alpha", context={"location": "home"})
    miss_found = hit_sys.retrieve_by_context(query="zzz", context={"location": "office"})

    gaps = {
        "default_threshold_returns_all": default_returns,
        "unlisted_keys_ignored": enc.encode_context({"weather": "rain"}) == enc.encode_context({"weather": "sun"})
        == enc.encode_context({}),
        "aliased_current_context": aliased,
        "empty_query": {"same_score_as_none": score_empty == score_none, "nonmatching_lowers": score_miss < score_none},
        "specificity_side_effects": {
            "score": round(spec_score, 9), "encodings": spec_stats["total_encodings"],
            "retrievals": spec_stats["total_retrievals"], "stored": spec_stats["stored_memories"],
            "prior_retrieval_count": spec_sys.contextual_memories[0].retrieval_count},
        "reinstate": reinstate,
        "matched_features": matched,
        "cues": cues,
        "similarity_edges": {
            "empty": enc.compute_similarity([], [1.0]), "zero_vector": enc.compute_similarity([0.0, 0.0], [1.0, 1.0]),
            "orthogonal": enc.compute_similarity([1.0, 0.0], [0.0, 1.0]),
            "same": round(enc.compute_similarity([0.5, 0.5], [0.5, 0.5]), 9),
            "length_mismatch": round(enc.compute_similarity([1.0, 2.0], [1.0]), 9)},
        "encode_defaults": {"dims": list(enc.context_dimensions),
                            "all_unknown_equal": len(set(enc.encode_context({}))) == 1},
        "str_coerced_values": enc.encode_context({"mood": 1}) == enc.encode_context({"mood": "1"}),
        "key_order_independent": enc.encode_context({"mood": "a", "task": "b"})
        == enc.encode_context({"task": "b", "mood": "a"}),
        "failed_set_leaves_none": none_sys.current_context is None,
        "ltm_failure_after_append": {"stored": len(ltm_sys.contextual_memories), "encodings": ltm_sys.total_encodings},
        "cue_subset_is_enough": sub_score,
        "query_hit_other_context_passes_default": [len(hit_found), round(hit_found[0].match_score, 9)
                                                   if hit_found else None],
        "query_miss_same_context_passes_default": [len(miss_found), round(miss_found[0].match_score, 9)
                                                   if miss_found else None],
        "bucket_collision": enc.encode_context({"time": "noon"}) == enc.encode_context({"time": "evening"}),
    }

    def loaded(threshold):
        system = _cm_system(cls)
        system.store_with_context("x", {})
        return system.retrieve_by_context(threshold=threshold)

    errors = {
        "set_ctx_none": _cm_err(lambda: _cm_system(cls).set_current_context(None)),
        "set_ctx_list": _cm_err(lambda: _cm_system(cls).set_current_context([])),
        "store_ctx_int": _cm_err(lambda: _cm_system(cls).store_with_context("x", 5)),
        "retrieve_ctx_str": _cm_err(lambda: _cm_system(cls).retrieve_by_context(context="a")),
        "encode_none": _cm_err(lambda: enc_cls().encode_context(None)),
        "cues_none": _cm_err(lambda: _cm_system(cls).get_context_dependent_cues(None)),
        "retrieve_threshold_str_with_memory": _cm_err(lambda: loaded("a")),
        "retrieve_threshold_none_with_memory": _cm_err(lambda: loaded(None)),
        "store_no_args": _cm_err(lambda: _cm_system(cls).store_with_context()),
        "construct_extra": _cm_err(lambda: cls(1, 2, 3, 4)),
        "similarity_str": _cm_err(lambda: enc_cls().compute_similarity("ab", "cd")),
        "store_ltm_raises": _cm_err(lambda: _cm_system(cls, long_term_memory=Raising()).store_with_context("x", {})),
        "retrieve_threshold_str_empty": _cm_err(lambda: _cm_system(cls).retrieve_by_context(threshold="a")),
        "retrieve_query_int": _cm_err(lambda: _cm_system(cls).retrieve_by_context(query=5)),
        "reinstate_none": _cm_err(lambda: _cm_system(cls).reinstate_context(None)),
        "similarity_none": _cm_err(lambda: enc_cls().compute_similarity(None, [1.0])),
        "match_none": _cm_err(lambda: enc_cls().compute_match(None, [1.0])),
        "match_cue_out_of_range": _cm_err(lambda: enc_cls().compute_match([1.0], [1.0], [3])),
    }
    return {
        "clock": "time.time",
        "randomness": "uuid4 ids; sha256 estável no codificador (hash() não é usado)",
        "init_params": list(inspect.signature(cls.__init__).parameters),
        "banner": banner,
        "memory_fields": list(inspect.signature(mem_cls).parameters),
        "match_fields": list(inspect.signature(match_cls).parameters),
        "fresh_stats": fresh_stats,
        "grid": grid_info,
        "families": families,
        "determinism": determinism,
        "counters": counters,
        "collaborators": collaborators,
        "gaps": gaps,
        "errors": errors,
        "stability": stability,
        "scores": _cm_scores(cls, enc_cls),
    }


def contextual_memory_source_text(src: str) -> str:
    idx = src.index(_CM_MARKER)
    line_start = src.rfind("\n", 0, idx - 1) + 1
    end = src.index(_CM_END, idx) + len(_CM_END)
    return src[line_start:end]


def _contextual_memory_source_sha(path="core/cognitivo/contextual_memory.py") -> str:
    import hashlib
    text = (ROOT / path).read_text(encoding="utf-8")
    return hashlib.sha256(contextual_memory_source_text(text).encode()).hexdigest()


def _contextual_memory_standalone() -> bool:
    """Árvore de imports: constitutional_brain, numpy ou vita no módulo extraído falha CM6."""
    import ast

    tree = ast.parse((ROOT / "core/cognitivo/contextual_memory.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in {"numpy", "vita"} or name == "core.constitutional_brain" \
                    or name.startswith("core.constitutional_brain."):
                return False
    return True


def eval_contextual_memory(criteria="v1"):
    """CM1–CM9. criteria é aceito e ignorado: há um único protocolo pré-registrado."""
    import core.cognitivo.contextual_memory as contextual_memory
    import core.constitutional_brain as brain
    import core.cognitivo as package

    payload = contextual_memory_fingerprint(contextual_memory)
    sha_module = _ms_sha(payload)
    sha_brain = _ms_sha(contextual_memory_fingerprint(brain))
    source_sha = _contextual_memory_source_sha()
    standalone = _contextual_memory_standalone()
    identity = all(
        getattr(brain, name) is getattr(contextual_memory, name) is getattr(package, name)
        and getattr(contextual_memory, name).__module__ == "core.cognitivo.contextual_memory"
        for name in _CM_NAMES
    )
    grid = payload["grid"]
    cm1 = (
        grid["count"] == _CM_GRID_SIZE and grid["reference_matches"] == _CM_GRID_SIZE
        and grid["mutated_inputs"] == 0 and grid["meta_bad"] == 0
    )
    fam2, fam3 = payload["families"]["CM2"], payload["families"]["CM3"]
    cm2_invalid = fam2["invalid"] > 0
    cm2 = not cm2_invalid and fam2["rate"] >= _CM_LIMIT
    cm3 = fam3["rate"] >= _CM_LIMIT and fam3["recall_hits"] == fam3["total"]
    det = payload["determinism"]
    cm4 = det["control_same"] == det["contexts"] and det["cross_rate"] >= _CM_LIMIT
    gaps = payload["gaps"]
    gap_hits = sum(1 for name, expected in _CM_GAPS_EXPECTED.items() if gaps.get(name) == expected)
    error_hits = sum(1 for name, expected in _CM_ERRORS_EXPECTED.items() if payload["errors"].get(name) == expected)
    cm5 = (
        payload["fresh_stats"] == _CM_FRESH_STATS
        and payload["counters"] == _CM_COUNTERS_EXPECTED
        and payload["collaborators"] == _CM_COLLAB_EXPECTED
        and gap_hits == len(_CM_GAPS_EXPECTED) == len(gaps)
        and error_hits == len(_CM_ERRORS_EXPECTED) == len(payload["errors"])
    )
    cm6 = (
        identity and standalone and source_sha == CM_SOURCE_SHA
        and sha_module == sha_brain == CM_BEHAVIOR_SHA
        and payload["clock"] == "time.time"
        and payload["init_params"] == ["self", "long_term_memory", "working_memory", "attention"]
        and payload["banner"] == _CM_BANNER
    )
    stab = payload["stability"]
    cm7 = (
        stab["golden"] == stab["golden_total"] == len(_CM_GOLDEN) == 7
        and stab["no_hash"] == len(_CM_GOLDEN)
        and stab["grid_ok"] == stab["grid_total"] == _CM_STABILITY_SIZE and stab["mutated_inputs"] == 0
    )
    scores = payload["scores"]
    expected8 = {name: expected for name, _m, _q, _r, expected in _CM8_SCORES}
    expected8["b1"] = _CM8_BANDS
    expected8.update({name: expected for name, _args, expected in _CM8_MATCH})
    cm8_hits = sum(1 for name, expected in expected8.items() if scores.get(name) == expected)
    cm8 = len(expected8) == len(scores) == 26 and cm8_hits == 26
    import hashlib
    base_rows_ok = (
        len(_CM_BASE_ROWS) == _CM_GRID_SIZE
        and hashlib.sha256(json.dumps(list(_CM_BASE_ROWS)).encode()).hexdigest() == _CM_BASE_ROWS_SHA
    )
    unaffected_rows = [i for i, flag in enumerate(grid["unaffected"]) if flag]
    base_diffs = [i for i in unaffected_rows if grid["row_digests"][i] != _CM_BASE_ROWS[i]] if base_rows_ok else []
    cm9_invalid = not base_rows_ok or len(unaffected_rows) < _CM_MIN_UNAFFECTED
    cm9 = not cm9_invalid and not base_diffs
    return [
        _check("CM1", {"reference_matches": grid["reference_matches"], "count": grid["count"],
                       "mutated_inputs": grid["mutated_inputs"], "meta_bad": grid["meta_bad"]},
               cm1, "120/120 iguais à referência independente; 0 entradas alteradas; ids e timestamps válidos"),
        _check("CM2", fam2, cm2, "contexto consultado ocupa as 6 primeiras posições em ≥ 0.90 das 20 seeds",
               **({"scenario_invalid": True} if cm2_invalid else {})),
        _check("CM3", fam3, cm3,
               "precisão ≥ 0.80 com o limiar padrão em ≥ 0.90 das 20 seeds e revocação das 6 memórias em 20/20"),
        _check("CM4", det, cm4, "controle 20/20 e vetor igual entre PYTHONHASHSEED 101 e 202 em ≥ 0.90 dos 20 contextos"),
        _check("CM5", {"fresh_stats": payload["fresh_stats"], "counters": payload["counters"],
                       "collaborators": payload["collaborators"], "gap_hits": [gap_hits, len(_CM_GAPS_EXPECTED)],
                       "error_hits": [error_hits, len(_CM_ERRORS_EXPECTED)], "gaps": gaps,
                       "errors": payload["errors"]},
               cm5, "contadores, colaboradores, erros e gaps conforme o registrado"),
        _check("CM6", {"source_sha": source_sha, "sha_module": sha_module, "sha_brain": sha_brain,
                       "identity": identity, "standalone": standalone, "clock": payload["clock"],
                       "source_golden": CM_SOURCE_SHA, "behavior_golden": CM_BEHAVIOR_SHA},
               cm6, "SHA de fonte e de comportamento pré-registrados; reexport idêntico"),
        _check("CM7", stab, cm7, "7/7 vetores dourados, 7/7 sem builtins.hash e 200/200 contra a referência de hashlib"),
        _check("CM8", {"hits": [cm8_hits, len(expected8)], "scores": scores},
               cm8, "26/26 itens de score e seletividade com valores exatos"),
        _check("CM9", {"unaffected": len(unaffected_rows), "min_unaffected": _CM_MIN_UNAFFECTED,
                       "base_rows_sha_ok": base_rows_ok, "diffs": base_diffs},
               cm9, "linhas não afetadas iguais ao resumo congelado do código base (0 diferenças; ≥ 12 linhas)",
               **({"scenario_invalid": True} if cm9_invalid else {})),
    ]


EVALUATORS = {"episodic": eval_episodic, "predictive": eval_predictive, "mcts": eval_mcts,
              "causal": eval_causal, "counterfactual": eval_counterfactual,
              "working_memory": eval_working_memory,
              "attention": eval_attention,
              "incremental": eval_incremental,
              "episodic_future": eval_episodic_future,
              "mental_simulation": eval_mental_simulation,
              "analogical_reasoning": eval_analogical_reasoning,
              "contextual_memory": eval_contextual_memory}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capability")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--criteria", default="v2")
    args = parser.parse_args(argv)
    cap = (args.capability or "").strip()
    if not cap:
        print("erro: --capability ausente ou vazio", file=sys.stderr)
        return EXIT_USAGE
    if cap != "all" and cap not in CAPABILITIES:
        print(f"erro: capacidade desconhecida: {cap}", file=sys.stderr)
        return EXIT_USAGE
    if args.criteria not in ("v1", "v2", "v3"):
        print(f"erro: versão de critérios desconhecida: {args.criteria}", file=sys.stderr)
        return EXIT_USAGE
    selected = CAPABILITIES if cap == "all" else (cap,)
    report = {name: (EVALUATORS[name](args.criteria) if name != "episodic" else EVALUATORS[name]())
              for name in selected}
    report["criteria"] = args.criteria
    failed = [c["id"] for name in selected for c in report[name] if not c["passed"] and not c.get("informative")]
    report["failed"] = failed
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    invalid = any(c.get("scenario_invalid") for name in selected for c in report[name])
    if invalid:
        return EXIT_INVALID_SCENARIO
    return EXIT_FAIL if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
