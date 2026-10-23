"""Avaliação funcional das capacidades em core/cognitivo (critérios pré-registrados).

Critérios: /workspace/nexus/cognitivo/AUDITORIA_2026-10-22.md (sha256 3d24a63d…), escritos
antes da execução. Mede propósito, não só ausência de crash.

Uso: python tools/cognitivo_eval.py --capability {episodic,predictive,mcts,causal,counterfactual,working_memory,attention,all} [--json PATH]
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
CAPABILITIES = ("episodic", "predictive", "mcts", "causal", "counterfactual", "working_memory", "attention")

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



EVALUATORS = {"episodic": eval_episodic, "predictive": eval_predictive, "mcts": eval_mcts,
              "causal": eval_causal, "counterfactual": eval_counterfactual,
              "working_memory": eval_working_memory,
              "attention": eval_attention}


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
