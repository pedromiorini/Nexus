import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class NexusConstitutionalBridge:
    """
    Nexus Constitutional Bridge v3.0 (Fase 50 - Plenitude Eterna)
    
    Adapta a saída da NexusFederation (Vita v6.9) para o CompleteNexusBrain (Constitutional v3.94).
    Incorpora métricas avançadas de Inteligência Coletiva (CIS), Sucesso Referencial (SCI) 
    e Meta-Cognição (MCS).
    """
    def __init__(self):
        self.query_count = 0
        self.integration_history: List[dict] = []

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
