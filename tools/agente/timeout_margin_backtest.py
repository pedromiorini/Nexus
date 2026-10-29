#!/usr/bin/env python3
"""Retroprojeta a guarda `timeout_margin.sh` sobre o historico real do job de CI.

Subcomandos (protocolo e criterios TM1-TM9 em docs/agentes/MARGEM_TIMEOUT_CI.md):
  coletar --saida ARQ   baixa o historico com `gh` e grava o instantaneo (so grava se valido)
  avaliar [--dados ARQ] [--json]   avalia TM1-TM5 e a tendencia (TM9) offline; 0 passa, 1 falha, 3 dados invalidos
  atual   [--dados ARQ]            relatorio da guarda sobre os ultimos sucessos do main; sempre 0 (so relatorio)

Cada decisao chama o script real da guarda; nenhum limiar e reimplementado aqui.
"""

import argparse
import bisect
import json
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "tools/agente/timeout_margin.sh"
DEFAULT_DATA = ROOT / "docs/agentes/timeout_margin_historico.json"
WORKFLOW = "nexus-contract-gate.yml"
JOB_NAME = "Contract and regression gate"
REPO = "pedromiorini/Nexus"
SCHEMA = 1

WINDOW = 5                      # N fixo do pre-registro
HEALTHY_RATIO = 0.70            # TM4/TM5
MIN_HEALTHY_POINTS = 30         # TM4
MIN_ALARMS = 5                  # TM5
MAX_FALSE_ALARM_RATE = 0.20     # TM5
MAX_UNKNOWN_TIMEOUT_SHARE = 0.10
KNOWN_INCIDENTS = ((40765682957, 1), (40783977991, 1), (40783977991, 2))
RECENT_JOB = (40923631957, 1)   # #129, 18 min 20 s com timeout 20
FIELDS = ("run", "attempt", "conclusion", "started", "completed", "sha", "event", "branch", "timeout")
EXIT_OK, EXIT_FAIL, EXIT_INVALID = 0, 1, 3


class InvalidData(Exception):
    pass


def parse_time(value, what):
    if not isinstance(value, str):
        raise InvalidData(f"{what} nao e texto")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise InvalidData(f"{what} invalido: {value!r}") from exc


def load_snapshot(path):
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InvalidData(f"instantaneo ilegivel ({path}): {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise InvalidData(f"JSON invalido ({path})") from exc
    return normalize(data)


def normalize(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise InvalidData("schema ausente ou diferente de 1")
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise InvalidData("lista de jobs ausente ou vazia")
    out = []
    for j in jobs:
        if not isinstance(j, dict) or any(f not in j for f in FIELDS):
            raise InvalidData(f"job sem campo obrigatorio: {j!r}")
        if not isinstance(j["run"], int) or not isinstance(j["attempt"], int):
            raise InvalidData(f"run/attempt nao inteiros: {j!r}")
        if j["timeout"] is not None and (not isinstance(j["timeout"], int) or j["timeout"] <= 0):
            raise InvalidData(f"timeout invalido: {j!r}")
        start = parse_time(j["started"], "started")
        end = parse_time(j["completed"], "completed")
        if end < start:
            raise InvalidData(f"completed anterior a started: {j!r}")
        out.append(dict(j, _start=start, _end=end, dur=int((end - start).total_seconds())))
    unknown = sum(j["timeout"] is None for j in out)
    if unknown / len(out) > MAX_UNKNOWN_TIMEOUT_SHARE:
        raise InvalidData(f"timeout desconhecido em {unknown}/{len(out)} jobs (> 10%)")
    return out


def guard(timeout_min, durations):
    """Chama o script real; devolve 0 (ok) ou 6 (bloqueio). Qualquer outro codigo e dado invalido."""
    with tempfile.TemporaryDirectory(prefix="tm-bt-") as tmp:
        path = Path(tmp) / "duracoes.txt"
        path.write_text("".join(f"{d}\n" for d in durations), encoding="utf-8")
        proc = subprocess.run(["bash", str(GUARD), str(timeout_min), str(path)], capture_output=True, text=True)
    if proc.returncode not in (0, 6):
        raise InvalidData(f"guarda saiu {proc.returncode}: {proc.stderr.strip()}")
    return proc.returncode


def job_key(j):
    return (j["_end"], j["run"], j["attempt"])


def is_success(j):
    return j["conclusion"] == "success"


def is_incident(j):
    return j["conclusion"] == "cancelled" and j["timeout"] is not None and j["dur"] >= j["timeout"] * 60


class History:
    def __init__(self, jobs):
        self.jobs = jobs
        self.successes = sorted((j for j in jobs if is_success(j)), key=job_key)
        self.success_keys = [job_key(j) for j in self.successes]
        self.success_starts = sorted(((j["_end"], j["run"], j["attempt"]) for j in self.successes))

    def window_until_end(self, job):
        """Ultimas N duracoes de sucesso com fim <= fim de `job` (inclui o proprio, se sucesso)."""
        idx = bisect.bisect_right(self.success_keys, job_key(job))
        return [j["dur"] for j in self.successes[max(0, idx - WINDOW):idx]]

    def window_before_start(self, job):
        """Ultimas N duracoes de sucesso com fim <= inicio de `job`."""
        idx = bisect.bisect_right([k[0] for k in self.success_keys], job["_start"])
        return [j["dur"] for j in self.successes[max(0, idx - WINDOW):idx]]


def evaluate(jobs):
    h = History(jobs)
    known = [j for j in jobs if j["timeout"] is not None]
    incidents = [j for j in known if is_incident(j)]
    present = {(j["run"], j["attempt"]) for j in incidents}
    known_present = all(k in present for k in KNOWN_INCIDENTS)
    if not known_present:
        raise InvalidData("incidente conhecido ausente do conjunto detectado (scenario_invalid)")

    # TM1
    flagged, missed, no_window = [], [], 0
    for j in incidents:
        w = h.window_before_start(j)
        if not w:
            no_window += 1
            continue
        (flagged if guard(j["timeout"], w) == 6 else missed).append((j["run"], j["attempt"]))
    tm1 = {"passou": not missed and no_window == 0 and len(flagged) == len(incidents),
           "incidentes": len(incidents), "sinalizados": len(flagged), "perdidos": missed,
           "sem_janela": no_window, "conhecidos_presentes": known_present,
           "lista": sorted((j["run"], j["attempt"], j["dur"], j["timeout"]) for j in incidents)}

    # pontos de avaliacao sobre sucessos com T conhecido
    points = []
    for j in h.successes:
        if j["timeout"] is None:
            continue
        w = h.window_until_end(j)
        points.append({"job": j, "window": w, "rc": guard(j["timeout"], w)})

    # TM2
    t12_alarms = [p["job"] for p in points if p["rc"] == 6 and p["job"]["timeout"] == 12]
    t12_incidents = [j for j in incidents if j["timeout"] == 12]
    first_alarm = min((j["_end"] for j in t12_alarms), default=None)
    first_incident = min((j["_start"] for j in t12_incidents), default=None)
    lead = (first_incident - first_alarm).total_seconds() if first_alarm and first_incident else None
    tm2 = {"passou": lead is not None and lead > 0,
           "primeiro_alarme": first_alarm.isoformat() if first_alarm else None,
           "primeiro_incidente": first_incident.isoformat() if first_incident else None,
           "antecedencia_s": lead}

    # TM3
    recent = next((p for p in points if (p["job"]["run"], p["job"]["attempt"]) == RECENT_JOB), None)
    if recent is None:
        raise InvalidData("job recente (#129) ausente do instantaneo (scenario_invalid)")
    tm3 = {"passou": recent["rc"] == 6, "duracao_s": recent["job"]["dur"], "timeout_min": recent["job"]["timeout"],
           "limite_s": recent["job"]["timeout"] * 60 * 85 // 100, "rc": recent["rc"]}

    # TM4
    healthy = [p for p in points if max(p["window"]) <= HEALTHY_RATIO * p["job"]["timeout"] * 60]
    if len(healthy) < MIN_HEALTHY_POINTS:
        raise InvalidData(f"controle saudavel com {len(healthy)} pontos (< {MIN_HEALTHY_POINTS}) (scenario_invalid)")
    wrong = [p["job"]["run"] for p in healthy if p["rc"] != 0]
    tm4 = {"passou": not wrong, "pontos_saudaveis": len(healthy), "sinalizados_indevidos": wrong}

    # TM5
    scoped = sorted((j for j in known if is_success(j) or is_incident(j)), key=lambda j: (j["_start"], j["run"], j["attempt"]))
    alarms = [p for p in points if p["rc"] == 6]
    classified = false_alarms = 0
    for p in alarms:
        j = p["job"]
        nxt = [x for x in scoped if x["timeout"] == j["timeout"] and (x["_start"], x["run"], x["attempt"]) > (j["_start"], j["run"], j["attempt"])][:WINDOW]
        if len(nxt) < WINDOW:
            continue
        classified += 1
        lim = HEALTHY_RATIO * j["timeout"] * 60
        if all(x["dur"] <= lim and not is_incident(x) for x in nxt):
            false_alarms += 1
    rate = false_alarms / classified if classified else None
    tm5 = {"passou": classified >= MIN_ALARMS and rate is not None and rate <= MAX_FALSE_ALARM_RATE,
           "alarmes": len(alarms), "classificaveis": classified, "falsos_alarmes": false_alarms,
           "taxa": rate, "inconclusivo": classified < MIN_ALARMS}

    return {"criterios": {"TM1": tm1, "TM2": tm2, "TM3": tm3, "TM4": tm4, "TM5": tm5},
            "tendencia": trend(h.successes),
            "escopo": {"jobs": len(jobs), "sucessos": len(h.successes), "incidentes": len(incidents),
                       "timeout_desconhecido": sum(j["timeout"] is None for j in jobs),
                       "pontos_de_avaliacao": len(points), "janela": WINDOW,
                       "cancelados_curtos_ou_falhas": sum(1 for j in jobs if not is_success(j) and not is_incident(j))}}


def trend(successes):
    days = {}
    for j in successes:
        days.setdefault(j["_end"].date().isoformat(), []).append(j)
    out = []
    for day in sorted(days):
        js = days[day]
        durs = [j["dur"] for j in js]
        ratios = [j["dur"] / (j["timeout"] * 60) for j in js if j["timeout"]]
        out.append({"dia": day, "sucessos": len(js), "mediana_s": int(statistics.median(durs)), "max_s": max(durs),
                    "razao_max_ao_timeout": round(max(ratios), 3) if ratios else None})
    return out


def report_text(res):
    lines = []
    for key, c in res["criterios"].items():
        lines.append(f"{key}: {'PASSOU' if c['passou'] else 'FALHOU'}  " + json.dumps({k: v for k, v in c.items() if k not in ('passou', 'lista')}, ensure_ascii=False))
    lines.append("escopo: " + json.dumps(res["escopo"], ensure_ascii=False))
    lines.append("tendencia (sucessos por dia UTC: n, mediana s, max s, max/timeout):")
    for t in res["tendencia"]:
        lines.append(f"  {t['dia']}  n={t['sucessos']:>3}  med={t['mediana_s']:>5}  max={t['max_s']:>5}  razao={t['razao_max_ao_timeout']}")
    return "\n".join(lines)


def cmd_avaliar(args):
    res = evaluate(load_snapshot(args.dados))
    res["passou"] = all(c["passou"] for c in res["criterios"].values())
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str) if args.json else report_text(res) + f"\nresultado: {'PASSOU' if res['passou'] else 'FALHOU'}")
    return EXIT_OK if res["passou"] else EXIT_FAIL


def cmd_atual(args):
    jobs = load_snapshot(args.dados)
    main = sorted((j for j in jobs if is_success(j) and j["branch"] == "main" and j["event"] == "push" and j["timeout"]), key=job_key)
    if not main:
        raise InvalidData("sem sucessos de push no main no instantaneo")
    last = main[-WINDOW:]
    t = last[-1]["timeout"]
    rc = guard(t, [j["dur"] for j in last])
    limit = t * 60 * 85 // 100
    print(f"guarda sobre os {len(last)} ultimos sucessos do main (timeout {t} min, limite 85% = {limit}s):")
    print("  duracoes (s): " + " ".join(str(j["dur"]) for j in last))
    print(f"  maximo {max(j['dur'] for j in last)}s -> rc={rc} ({'BLOQUEARIA' if rc == 6 else 'dentro da margem'}); modo relatorio: nao bloqueia")
    return EXIT_OK


def gh_json(*args):
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise InvalidData(f"gh {' '.join(args[:2])} falhou: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise InvalidData(f"gh {' '.join(args[:2])} nao devolveu JSON") from exc


def workflow_timeout(sha):
    import base64
    import re
    proc = subprocess.run(["gh", "api", f"repos/{REPO}/contents/.github/workflows/{WORKFLOW}?ref={sha}", "--jq", ".content"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        text = base64.b64decode(proc.stdout.strip()).decode("utf-8")
    except ValueError:
        return None
    m = re.search(r"contract-gate:.*?timeout-minutes:\s*(\d+)", text, re.S)
    return int(m.group(1)) if m else None


def cmd_coletar(args):
    runs = gh_json("run", "list", "--workflow", WORKFLOW, "--limit", "1000", "--json",
                   "databaseId,status,headSha,headBranch,event")
    if not isinstance(runs, list) or not runs:
        raise InvalidData("gh run list vazio")
    runs = [r for r in runs if r.get("status") == "completed"]
    shas = sorted({r["headSha"] for r in runs})
    with ThreadPoolExecutor(max_workers=8) as pool:
        timeouts = dict(zip(shas, pool.map(workflow_timeout, shas)))

    def jobs_of(run):
        out = []
        data = gh_json("api", f"repos/{REPO}/actions/runs/{run['databaseId']}/jobs?filter=all&per_page=100")
        for j in data.get("jobs", []):
            if j.get("name") != JOB_NAME or not j.get("started_at") or not j.get("completed_at"):
                continue
            out.append({"run": run["databaseId"], "attempt": j["run_attempt"], "conclusion": j.get("conclusion") or "",
                        "started": j["started_at"], "completed": j["completed_at"], "sha": run["headSha"],
                        "event": run["event"], "branch": run["headBranch"], "timeout": timeouts.get(run["headSha"])})
        return out

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [j for chunk in pool.map(jobs_of, runs) for j in chunk]
    jobs.sort(key=lambda j: (j["started"], j["run"], j["attempt"]))
    snap = {"schema": SCHEMA, "workflow": WORKFLOW, "job": JOB_NAME, "repo": REPO,
            "coletado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "execucoes_concluidas": len(runs), "jobs": jobs}
    normalize(snap)  # so grava se valido
    out = Path(args.saida)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"{len(jobs)} jobs de {len(runs)} execucoes gravados em {out}")
    return EXIT_OK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    a = sub.add_parser("avaliar")
    a.add_argument("--dados", default=str(DEFAULT_DATA))
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_avaliar)
    c = sub.add_parser("atual")
    c.add_argument("--dados", default=str(DEFAULT_DATA))
    c.set_defaults(fn=cmd_atual)
    k = sub.add_parser("coletar")
    k.add_argument("--saida", required=True)
    k.set_defaults(fn=cmd_coletar)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_INVALID if exc.code else EXIT_OK
    if not getattr(args, "fn", None):
        print("dados invalidos: subcomando ausente", file=sys.stderr)
        return EXIT_INVALID
    try:
        return args.fn(args)
    except InvalidData as exc:
        print(f"dados invalidos: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
