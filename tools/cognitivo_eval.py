"""Avaliação funcional das capacidades em core/cognitivo (critérios pré-registrados).

Critérios: /workspace/nexus/cognitivo/AUDITORIA_2026-10-22.md (sha256 3d24a63d…), escritos
antes da execução. Mede propósito, não só ausência de crash.

Uso: python tools/cognitivo_eval.py --capability {episodic,predictive,mcts,all} [--json PATH]
Saída: exit 0 se todo critério passa, 1 se algum falha, 2 para entrada ausente/vazia/desconhecida.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INVALID_SCENARIO = 0, 1, 2, 3
CAPABILITIES = ("episodic", "predictive", "mcts", "causal", "counterfactual")

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
                random.seed(seed)
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
                random.seed(seed)
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


EVALUATORS = {"episodic": eval_episodic, "predictive": eval_predictive, "mcts": eval_mcts, "causal": eval_causal, "counterfactual": eval_counterfactual}


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
