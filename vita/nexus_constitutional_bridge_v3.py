import json
import os
import sqlite3
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class NexusConstitutionalBridge:
    """
    Nexus Constitutional Bridge v3.0 (Fase 50 - Plenitude Eterna)
    
    Adapta a saída da NexusFederation (Vita v6.9) para o CompleteNexusBrain (Constitutional v3.94).
    Incorpora métricas avançadas de Inteligência Coletiva (CIS), Sucesso Referencial (SCI) 
    e Meta-Cognição (MCS).
    """
    def __init__(self, audit_db_path: Optional[str] = None):
        self.query_count = 0
        self.integration_history: List[dict] = []
        self.reprocessing_telemetry: List[dict] = []
        self.audit_db_path = audit_db_path or os.environ.get("NEXUS_AUDIT_DB", "nexus_audit.sqlite3")
        self._audit_db = sqlite3.connect(self.audit_db_path, check_same_thread=False)
        self._audit_db.execute("CREATE TABLE IF NOT EXISTS policy_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, timestamp REAL NOT NULL, payload TEXT NOT NULL)")
        self._audit_db.commit()
        self.reprocessing_thresholds = {
            "failure_rate": 0.50,
            "discard_rate": 0.10,
            "avg_latency_ms": 1000.0,
        }

    def record_reprocessing_telemetry(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Registra métricas do fluxo adiado e produz alertas operacionais determinísticos."""
        snapshot = dict(metrics)
        attempts = int(snapshot.get("attempts", 0))
        failed = int(snapshot.get("failed", 0))
        discarded = int(snapshot.get("discarded", 0))
        total_latency = float(snapshot.get("total_latency_ms", 0.0))
        avg_latency = total_latency / attempts if attempts else 0.0
        failure_rate = failed / attempts if attempts else 0.0
        discard_rate = discarded / attempts if attempts else 0.0
        alerts = []
        if attempts and failure_rate >= self.reprocessing_thresholds["failure_rate"]:
            alerts.append("reprocessing_failure_rate_high")
        if attempts and discard_rate >= self.reprocessing_thresholds["discard_rate"]:
            alerts.append("reprocessing_discard_rate_high")
        if avg_latency >= self.reprocessing_thresholds["avg_latency_ms"]:
            alerts.append("reprocessing_latency_high")
        record = {
            **snapshot,
            "avg_latency_ms": avg_latency,
            "failure_rate": failure_rate,
            "discard_rate": discard_rate,
            "alerts": alerts,
            "status": "alert" if alerts else "nominal",
        }
        self.reprocessing_telemetry.append(record)
        del self.reprocessing_telemetry[:-128]
        self._write_audit_event("telemetry", record)
        return record

    def record_policy_transition(self, previous_state: str, new_state: str, reason: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Persiste uma transição de política para auditoria longitudinal."""
        event = {
            "previous_state": previous_state,
            "new_state": new_state,
            "reason": reason,
            "policy": dict(policy),
        }
        self._write_audit_event("policy_transition", event)
        return event

    def _write_audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._audit_db.execute(
            "INSERT INTO policy_audit (event_type, timestamp, payload) VALUES (?, ?, ?)",
            (event_type, time.time(), json.dumps(payload, sort_keys=True)),
        )
        self._audit_db.commit()

    def get_policy_audit(self, limit: int = 128) -> List[Dict[str, Any]]:
        """Retorna eventos recentes de auditoria em ordem cronológica."""
        if limit < 1:
            raise ValueError("limit deve ser positivo")
        rows = self._audit_db.execute(
            "SELECT event_type, timestamp, payload FROM policy_audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"event_type": event_type, "timestamp": timestamp, "payload": json.loads(payload)} for event_type, timestamp, payload in reversed(rows)]

    def get_recovery_analysis(self, limit: int = 128, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Calcula recuperação em uma janela opcional e classifica a severidade dos eventos."""
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds deve ser positivo")
        events = self.get_policy_audit(limit)
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            events = [event for event in events if event["timestamp"] >= cutoff]
        paused_at = None
        pause_durations = []
        pauses = recoveries = critical_events = 0
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        latest_state = "active"
        for event in events:
            payload = event["payload"]
            severity = "info"
            if event["event_type"] == "policy_transition":
                previous = payload.get("previous_state")
                current = payload.get("new_state")
                latest_state = current or latest_state
                reason = payload.get("reason", "")
                if current == "paused" and previous != "paused":
                    pauses += 1
                    paused_at = event["timestamp"]
                    critical_events += 1
                    severity = "critical"
                elif current == "active" and previous == "paused":
                    recoveries += 1
                    if paused_at is not None:
                        pause_durations.append(max(0.0, event["timestamp"] - paused_at))
                    paused_at = None
                elif current == "degraded" or "latency" in reason:
                    severity = "warning"
            elif event["event_type"] == "telemetry":
                alerts = payload.get("alerts", [])
                if any(alert in alerts for alert in ("reprocessing_failure_rate_high", "reprocessing_discard_rate_high")):
                    critical_events += 1
                    severity = "critical"
                elif "reprocessing_latency_high" in alerts:
                    severity = "warning"
            severity_counts[severity] += 1
        if severity_counts["critical"]:
            overall_severity = "critical"
        elif severity_counts["warning"]:
            overall_severity = "warning"
        else:
            overall_severity = "info"
        return {
            "events_analyzed": len(events),
            "window_seconds": window_seconds,
            "latest_state": latest_state,
            "pauses": pauses,
            "recoveries": recoveries,
            "recovery_rate": recoveries / pauses if pauses else 0.0,
            "critical_events": critical_events,
            "severity": overall_severity,
            "severity_counts": severity_counts,
            "avg_pause_seconds": sum(pause_durations) / len(pause_durations) if pause_durations else 0.0,
            "total_pause_seconds": sum(pause_durations),
            "open_pause": paused_at is not None,
        }

    def export_recovery_diagnostics(self, limit: int = 128, window_seconds: Optional[float] = None, as_json: bool = False):
        """Exporta um snapshot versionado para consumidores externos de observabilidade."""
        payload = {
            "schema": "nexus.recovery.diagnostics.v1",
            "generated_at": time.time(),
            "source": "NexusConstitutionalBridge",
            "diagnostics": self.get_recovery_analysis(limit=limit, window_seconds=window_seconds),
            "events": self.get_policy_audit(limit=limit),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) if as_json else payload

    def get_brain_state(self, fed: Any, uci_global: float) -> dict:
        """Compatível com CompleteNexusBrain.get_status()."""
        self.query_count += 1
        all_inst = fed._all_instances()

        # Valores padrão (Fase 50 defaults)
        avg_fit = 0.0; avg_acc = 0.0; avg_rules = 0.0; avg_mcs = 0.0
        total_ltm = 0; total_coll = 0; avg_creat = 0.0; avg_anomaly = 0.0
        avg_rd_u_var = 0.0; avg_rd_u_mean = 0.0; lang_alignment = 0.0
        gw_active = False; shared_symbols = 0; stm_total = 0; sleep_cycles = 0
        rounds = fed._round; converged = fed.converged; resonating = False
        rule_emerged = False; novelty_boosts = 0; arch_mutations = 0
        planning_active = False; cis_score = 0.0; sci_score = 0.0; h0_rejected = False

        if all_inst:
            avg_fit = float(np.mean([inst.fitness for inst in all_inst]))
            avg_acc = float(np.mean([inst.predictor.recent_accuracy for inst in all_inst]))
            avg_rules = float(np.mean([inst.rules.n_rules() for inst in all_inst]))
            avg_mcs = float(np.mean([inst.meta_cognition for inst in all_inst]))
            total_ltm = sum(inst.ltm.size for inst in all_inst)
            total_coll = sum(len(inst.ltm.collective_episodes()) for inst in all_inst)
            avg_creat = float(np.mean([inst.creative_engine.creativity_score for inst in all_inst]))
            avg_anomaly = float(np.mean([inst.anomaly_detector.anomaly_count for inst in all_inst]))
            avg_rd_u_var = float(np.mean([float(inst.rd_u.var()) for inst in all_inst]))
            avg_rd_u_mean = float(np.mean([float(inst.rd_u.mean()) for inst in all_inst]))
            
            if hasattr(fed, 'lang_monitor') and fed.lang_monitor.alignment_history:
                lang_alignment = float(fed.lang_monitor.alignment_history[-1])
            gw_active = fed.gw.gw_colony >= 0
            shared_symbols = sum(inst.creative_engine.promoted_count for inst in all_inst)
            stm_total = sum(inst.stm.size for inst in all_inst)
            sleep_cycles = sum(inst.sleep_count for inst in all_inst)
            resonating = fed.resonance.resonating if hasattr(fed.resonance, 'resonating') else False
            rule_emerged = fed.rule_emergence.emerged if hasattr(fed.rule_emergence, 'emerged') else False
            novelty_boosts = sum(inst.novelty.boost_count for inst in all_inst)
            arch_mutations = sum(len(inst.arch_mut.mutation_history) for inst in all_inst)
            planning_active = any(inst.planner.plan_count > 0 for inst in all_inst)
            
            # Métricas de Alta Performance (Fase 50)
            # CIS (Collective Intelligence Score)
            if hasattr(fed, 'lang_monitor'):
                cis_score = (avg_mcs + lang_alignment + (1.0 if rule_emerged else 0.0)) / 3.0
            
            # SCI (Social Cohesion Index / Referential Success)
            sci_score = float(np.mean([inst.ref_monitor.recent_success_rate for inst in all_inst])) if hasattr(all_inst[0], 'ref_monitor') else 0.0
            h0_rejected = any(inst.ref_monitor.rejects_H0 for inst in all_inst) if hasattr(all_inst[0], 'ref_monitor') else False

        state = {
            "module": "NexusVita",
            "version": "6.9",
            "phase": 50,
            "emotions": {
                "valence": float(np.clip(avg_fit * 2 - 1, -1, 1)),
                "arousal": float(np.clip(avg_mcs * 10, 0, 1)),
                "curiosity": float(np.clip(avg_creat, 0, 1)),
                "surprise": float(avg_anomaly / max(rounds * 12, 1)),
                "collective_resonance": resonating
            },
            "cognition": {
                "uci": uci_global,
                "cis": cis_score,
                "sci": sci_score,
                "h0_rejected": h0_rejected,
                "predictive_acc": avg_acc,
                "meta_cognition": avg_mcs,
                "planning_active": planning_active,
            },
            "embodiment": {
                "pattern_variance": avg_rd_u_var,
                "avg_u": avg_rd_u_mean,
            },
            "social": {
                "n_instances": len(all_inst),
                "lang_alignment": lang_alignment,
                "shared_symbols": shared_symbols,
                "gw_active": gw_active
            },
            "memory": {
                "stm_total": stm_total,
                "ltm_total": total_ltm,
                "collective": total_coll,
                "sleep_cycles": sleep_cycles
            },
            "meta": {
                "rounds": rounds,
                "converged": converged,
                "rule_emerged": rule_emerged,
                "novelty_boosts": novelty_boosts,
                "arch_mutations": arch_mutations
            }
        }
        self.integration_history.append({"round": rounds, "uci": uci_global, "cis": cis_score})
        return state

    def generate_narrative(self, brain_state: dict) -> str:
        em = brain_state["emotions"]; cog = brain_state["cognition"]
        soc = brain_state["social"]; meta = brain_state["meta"]
        
        status = "SUPERINTELIGÊNCIA EMERGENTE" if cog["cis"] > 0.7 else "INTELIGÊNCIA COLETIVA PLENA" if cog["cis"] > 0.6 else "ESTÁVEL"
        
        return (
            f"🧬 NEXUS VITA v6.9 [FASE 50: PLENITUDE ETERNA]\n"
            f"   Status: {status} (CIS={cog['cis']:.3f})\n"
            f"   Consciência: UCI={cog['uci']:.4f} | MCS={cog['meta_cognition']:.3f}\n"
            f"   Emergência: SCI={cog['sci']:.3f} | H0 Rejeitada: {'SIM' if cog['h0_rejected'] else 'NÃO'}\n"
            f"   Social: Alinhamento={soc['lang_alignment']:.3f} | Símbolos={soc['shared_symbols']}\n"
            f"   Meta: Rounds={meta['rounds']} | Ressonância: {'SIM' if em['collective_resonance'] else 'NÃO'}"
        )
