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

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CAPABILITIES = ("episodic", "predictive", "mcts")

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


def eval_predictive():
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
    return checks


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


def eval_mcts():
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


EVALUATORS = {"episodic": eval_episodic, "predictive": eval_predictive, "mcts": eval_mcts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capability")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)
    cap = (args.capability or "").strip()
    if not cap:
        print("erro: --capability ausente ou vazio", file=sys.stderr)
        return EXIT_USAGE
    if cap != "all" and cap not in CAPABILITIES:
        print(f"erro: capacidade desconhecida: {cap}", file=sys.stderr)
        return EXIT_USAGE
    selected = CAPABILITIES if cap == "all" else (cap,)
    report = {name: EVALUATORS[name]() for name in selected}
    failed = [c["id"] for checks in report.values() for c in checks if not c["passed"] and not c.get("informative")]
    report["failed"] = failed
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return EXIT_FAIL if failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
