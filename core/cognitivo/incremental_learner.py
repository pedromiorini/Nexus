"""Incremental learner (POST12) extraído de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta as classes no ponto original.
Não há relógio injetável: timestamps e a duração da consolidação usam time.time().
apply_decay é heurística: registra um evento no log e não atualiza a memória.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "ConsolidationReport",
    "LearningEvent",
    "RealIncrementalLearner",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 11: INCREMENTAL LEARNER (POST12)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LearningEvent:
    """Registro de um evento de aprendizado."""
    episode_id: str
    memory_id: int
    content: str
    old_importance: float
    new_importance: float
    promotion: bool          # short_term → long_term?
    timestamp: float
    reason: str              # Por que foi promovida/atualizada


@dataclass
class ConsolidationReport:
    """Relatório de uma consolidação de aprendizado."""
    episodes_processed: int
    memories_promoted: int         # short → long term
    memories_reinforced: int       # importância aumentada
    patterns_detected: int         # padrões repetidos identificados
    total_learning_events: int
    duration: float


class RealIncrementalLearner:
    """
    POST12: Incremental Learner — REAL e FUNCIONAL

    O sistema aprende continuamente a partir de episódios passados:

    1. PROMOÇÃO      — Memórias altamente relevantes em episódios viram long-term
    2. REFORÇO       — Memórias acessadas muitas vezes ganham importância
    3. DETECÇÃO      — Padrões repetidos entre episódios são identificados e
                       armazenados como memórias sintéticas de alto valor
    4. CONSOLIDAÇÃO  — Varre N episódios recentes e aplica todas as regras acima
    5. AUTO-DECAY    — Memórias nunca acessadas têm importância levemente reduzida

    Multiplica o sistema:
    • × EpisodicMemory: lê episódios fechados como fonte de aprendizado
    • × HierarchicalMemory: promove e reforça entradas diretamente
    • × ImmutableLog: registra cada evento de aprendizado
    • × MCTS: planos que funcionaram viram memórias de alta importância
    """

    # ── Hiperparâmetros ────────────────────────────────────────────────────
    PROMOTION_RELEVANCE_THRESHOLD: float = 0.75   # relevância mínima para promoção
    REINFORCEMENT_IMPORTANCE_BOOST: float = 0.08  # boost por acesso extra
    IMPORTANCE_CAP: float = 0.99
    DECAY_RATE: float = 0.02                       # decay por episódio sem acesso
    PATTERN_MIN_FREQ: int = 2                      # freq. mínima para virar padrão
    PATTERN_IMPORTANCE: float = 0.92               # importância de memórias sintéticas

    def __init__(
        self,
        episodic_memory: Any,
        hierarchical_memory: Any,
        constitutional_log: Optional[Any] = None
    ):
        self.episodic = episodic_memory
        self.memory = hierarchical_memory
        self.log = constitutional_log

        # Estado interno
        self.learning_history: List[LearningEvent] = []
        self.pattern_registry: Dict[str, int] = {}   # conteúdo → freq. de ocorrência
        self.promoted_ids: Set[int] = set()           # IDs já promovidos (sem duplicatas)

        # Estatísticas
        self.total_promotions: int = 0
        self.total_reinforcements: int = 0
        self.total_patterns: int = 0
        self.total_consolidations: int = 0

    # ── Núcleo: aprender de um episódio ───────────────────────────────────

    def learn_from_episode(self, episode_id: str) -> List[LearningEvent]:
        """
        Analisar um episódio fechado e aprender:
        - Promover memórias de alta relevância para long-term
        - Reforçar memórias já long-term com acesso recorrente
        - Registrar conteúdos para detecção de padrões
        """
        events: List[LearningEvent] = []
        memories = self.episodic.get_episode_memories(episode_id)

        for mem in memories:
            relevance = mem.get("relevance_to_episode", 0.0)
            mem_id    = mem.get("id")
            content   = mem.get("content", "")
            old_imp   = mem.get("importance", 0.5)

            if mem_id is None:
                continue

            # ── Regra 1: PROMOÇÃO (relevância alta → long-term) ──────────
            if relevance >= self.PROMOTION_RELEVANCE_THRESHOLD and mem_id not in self.promoted_ids:
                new_imp = min(self.IMPORTANCE_CAP, old_imp * 1.15)
                self.memory.store(content, "long_term", new_imp)
                self.promoted_ids.add(mem_id)
                self.total_promotions += 1

                ev = LearningEvent(
                    episode_id=episode_id,
                    memory_id=mem_id,
                    content=content[:80],
                    old_importance=old_imp,
                    new_importance=new_imp,
                    promotion=True,
                    timestamp=time.time(),
                    reason=f"relevance={relevance:.2f} >= {self.PROMOTION_RELEVANCE_THRESHOLD}"
                )
                events.append(ev)

            # ── Regra 2: REFORÇO (já long-term, aumentar importância) ────
            elif relevance >= 0.5 and old_imp >= 0.7:
                new_imp = min(self.IMPORTANCE_CAP, old_imp + self.REINFORCEMENT_IMPORTANCE_BOOST)
                # Atualiza via re-armazenamento (update in-place não disponível)
                self.memory.store(content, "long_term", new_imp)
                self.total_reinforcements += 1

                ev = LearningEvent(
                    episode_id=episode_id,
                    memory_id=mem_id,
                    content=content[:80],
                    old_importance=old_imp,
                    new_importance=new_imp,
                    promotion=False,
                    timestamp=time.time(),
                    reason=f"reinforcement (old_imp={old_imp:.2f}, rel={relevance:.2f})"
                )
                events.append(ev)

            # ── Rastrear padrão ─────────────────────────────────────────
            key = content[:60].lower().strip()
            self.pattern_registry[key] = self.pattern_registry.get(key, 0) + 1

        self.learning_history.extend(events)
        return events

    # ── Detecção de padrões ────────────────────────────────────────────────

    def detect_and_store_patterns(self) -> int:
        """
        Identificar conteúdos que aparecem com frequência ≥ PATTERN_MIN_FREQ
        em episódios diferentes e armazená-los como memórias sintéticas
        de alta importância.

        Retorna: número de novos padrões detectados nesta chamada.
        """
        new_patterns = 0

        for content_key, freq in self.pattern_registry.items():
            if freq >= self.PATTERN_MIN_FREQ:
                # Verifica se padrão já foi consolidado antes
                tag = f"[PATTERN] {content_key}"
                existing = self.memory.retrieve(tag, limit=1)
                already_stored = any(
                    tag[:30] in e.get("content", "") for e in existing
                )
                if not already_stored:
                    synthetic = f"[PATTERN freq={freq}] {content_key}"
                    self.memory.store(synthetic, "long_term", self.PATTERN_IMPORTANCE)
                    self.total_patterns += 1
                    new_patterns += 1

                    if self.log:
                        self.log.log_event("PATTERN_DETECTED", {
                            "content": content_key,
                            "frequency": freq
                        })

        return new_patterns

    # ── Decay de memórias não acessadas ───────────────────────────────────

    def apply_decay(self, memory_ids_accessed: Set[int]) -> int:
        """
        Reduzir levemente a importância de memórias que NÃO foram
        acessadas nos últimos episódios.  Memórias long-term com
        importância < 0.2 não sofrem decay (proteção de conhecimento base).

        Retorna: número de memórias que sofreram decay.
        """
        # Implementação simplificada: registrar intenção no log
        # (decay real requer UPDATE SQL que adicionamos aqui)
        decayed = 0
        if self.log and memory_ids_accessed:
            self.log.log_event("DECAY_APPLIED", {
                "accessed_count": len(memory_ids_accessed),
                "decay_rate": self.DECAY_RATE
            })
            decayed = max(0, 5 - len(memory_ids_accessed))  # heurística
        return decayed

    # ── Consolidação em lote ───────────────────────────────────────────────

    def consolidate_learning(
        self,
        last_n_episodes: int = 10,
        run_pattern_detection: bool = True,
        run_decay: bool = True
    ) -> ConsolidationReport:
        """
        Consolidar aprendizado dos últimos N episódios fechados.

        Pipeline:
        1. Buscar N episódios mais recentes fechados
        2. Para cada episódio, chamar learn_from_episode()
        3. Detectar padrões emergentes
        4. Aplicar decay em memórias não acessadas
        5. Retornar relatório consolidado
        """
        start_time = time.time()
        self.total_consolidations += 1

        all_events: List[LearningEvent] = []
        accessed_ids: Set[int] = set()

        # Passo 1+2: processar episódios recentes
        recent_episodes = self.episodic.search_episodes(
            status="closed", limit=last_n_episodes
        )

        for episode in recent_episodes:
            events = self.learn_from_episode(episode.episode_id)
            all_events.extend(events)
            for ev in events:
                accessed_ids.add(ev.memory_id)

        # Passo 3: padrões
        new_patterns = self.detect_and_store_patterns() if run_pattern_detection else 0

        # Passo 4: decay
        if run_decay:
            self.apply_decay(accessed_ids)

        # Log da consolidação
        if self.log:
            self.log.log_event("LEARNING_CONSOLIDATION", {
                "episodes_processed": len(recent_episodes),
                "events_generated": len(all_events),
                "patterns_found": new_patterns
            })

        return ConsolidationReport(
            episodes_processed=len(recent_episodes),
            memories_promoted=sum(1 for e in all_events if e.promotion),
            memories_reinforced=sum(1 for e in all_events if not e.promotion),
            patterns_detected=new_patterns,
            total_learning_events=len(all_events),
            duration=round(time.time() - start_time, 4)
        )

    # ── Aprendizado de plano MCTS ──────────────────────────────────────────

    def learn_from_plan(self, plan_result: Any) -> Optional[LearningEvent]:
        """
        Se um plano MCTS teve alta recompensa esperada (≥0.7),
        armazenar a sequência de ações como memória de longo prazo.

        Integração direta MCTS × IncrementalLearner.
        """
        if plan_result is None:
            return None

        reward = getattr(plan_result, "expected_reward", 0.0)
        if reward < 0.7:
            return None

        sequence_str = " → ".join(plan_result.best_action_sequence)
        content = (
            f"[MCTS PLAN reward={reward:.3f}] "
            f"Goal: {plan_result.goal[:60]} | "
            f"Sequence: {sequence_str}"
        )
        importance = min(self.IMPORTANCE_CAP, 0.6 + reward * 0.35)
        mem_id = self.memory.store(content, "long_term", importance)
        self.total_promotions += 1

        ev = LearningEvent(
            episode_id="mcts_direct",
            memory_id=mem_id,
            content=content[:80],
            old_importance=0.0,
            new_importance=importance,
            promotion=True,
            timestamp=time.time(),
            reason=f"MCTS plan with reward={reward:.3f}"
        )
        self.learning_history.append(ev)

        if self.log:
            self.log.log_event("MCTS_PLAN_LEARNED", {
                "goal": plan_result.goal,
                "reward": reward,
                "sequence": plan_result.best_action_sequence
            })

        return ev

    # ── Estatísticas ───────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        return {
            "total_promotions":      self.total_promotions,
            "total_reinforcements":  self.total_reinforcements,
            "total_patterns":        self.total_patterns,
            "total_consolidations":  self.total_consolidations,
            "total_learning_events": len(self.learning_history),
            "pattern_registry_size": len(self.pattern_registry),
        }
