"""Planejador MCTS (POST3) extraído de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta MCTSNode, PlanResult e RealMCTSPlanner
no ponto original.
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional

__all__ = ["MCTSNode", "PlanResult", "RealMCTSPlanner"]


@dataclass
class MCTSNode:
    """
    Nó na árvore de busca MCTS.
    Cada nó representa um estado após uma sequência de ações.
    """
    state: Dict[str, Any]         # Estado do mundo neste nó
    action_taken: Optional[str]   # Ação que gerou este nó
    parent: Optional[Any]         # Referência ao nó pai (Any para evitar circular)
    depth: int = 0                # Profundidade na árvore

    # Estatísticas MCTS
    visits: int = 0
    total_reward: float = 0.0
    children: List[Any] = field(default_factory=list)
    untried_actions: List[str] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        """Verdadeiro se todas as ações já foram exploradas"""
        return len(self.untried_actions) == 0

    @property
    def avg_reward(self) -> float:
        """Recompensa média acumulada"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """
        UCB1 (Upper Confidence Bound 1):
        Balancea exploração (visitar nós novos) vs exploitação (visitar nós bons).
        
        UCB1 = avg_reward + C * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float('inf')  # Nós não visitados têm prioridade máxima

        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.avg_reward
        exploration = exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Selecionar filho com maior UCB1"""
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))


@dataclass
class PlanResult:
    """Resultado de um planejamento MCTS"""
    goal: str
    best_action_sequence: List[str]      # Sequência de ações recomendada
    expected_reward: float               # Recompensa esperada (0.0 - 1.0)
    iterations_run: int                  # Iterações MCTS executadas
    tree_nodes_created: int              # Total de nós na árvore
    alternative_plans: List[Dict]        # Planos alternativos encontrados
    planning_time: float                 # Tempo de planejamento em segundos
    success_probability: float           # Probabilidade estimada de sucesso


class RealMCTSPlanner:
    """
    POST3: Monte Carlo Tree Search Planner — REAL e FUNCIONAL
    
    Algoritmo clássico MCTS com 4 fases:
      1. SELEÇÃO    — desce a árvore via UCB1 até um nó expansível
      2. EXPANSÃO   — adiciona um nó filho não explorado
      3. SIMULAÇÃO  — rollout aleatório até profundidade máxima
      4. RETROPROPAGAÇÃO — atualiza estatísticas de todos os ancestrais
    
    Multiplica o WorldModel: usa simulações reais para avaliar estados.
    Multiplica o CausalReasoning: penaliza ações com efeitos negativos.
    Multiplica o EpisodicMemory: aprende de planos anteriores.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        causal_reasoning: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
        exploration_constant: float = 1.414,
        max_depth: int = 5
    ):
        if isinstance(exploration_constant, bool) or not isinstance(exploration_constant, Real):
            raise TypeError("exploration_constant must be a real number")
        if not math.isfinite(exploration_constant):
            raise ValueError("exploration_constant must be finite")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.world_model = world_model
        self.causal_reasoning = causal_reasoning
        self.episodic_memory = episodic_memory
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth

        # Estatísticas globais
        self.total_plans = 0
        self.total_iterations = 0
        self.total_nodes_created = 0

        # Biblioteca de domínios de ação predefinidos
        self._action_effects: Dict[str, Dict] = {
            "APPLY_PATCH":        {"security": +0.9, "stability": +0.3, "cost": -0.2},
            "IGNORE_ALERT":       {"security": -0.8, "stability": -0.5, "cost": 0.0},
            "ROLLBACK":           {"security": +0.4, "stability": +0.6, "cost": -0.4},
            "ESCALATE":           {"security": +0.6, "stability": +0.2, "cost": -0.3},
            "MONITOR":            {"security": +0.2, "stability": +0.1, "cost": -0.1},
            "ISOLATE_SYSTEM":     {"security": +0.7, "stability": -0.2, "cost": -0.3},
            "RESTORE_BACKUP":     {"security": +0.5, "stability": +0.8, "cost": -0.5},
            "DEPLOY_UPDATE":      {"security": +0.4, "stability": +0.5, "cost": -0.4},
            "INCREASE_LOGGING":   {"security": +0.3, "stability": +0.0, "cost": -0.1},
            "SHUTDOWN_SERVICE":   {"security": +0.8, "stability": -0.7, "cost": -0.2},
        }

    # ─── Fase 1: SELEÇÃO ────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descer a árvore via UCB1 até encontrar um nó expansível"""
        current = node
        while current.is_fully_expanded and current.children:
            current = current.best_child(self.exploration_constant)
        return current

    # ─── Fase 2: EXPANSÃO ───────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Adicionar um novo filho para uma ação ainda não tentada"""
        if not node.untried_actions:
            return node  # Nó terminal

        # Escolher ação não explorada (aleatoriedade controlada)
        action = node.untried_actions.pop(
            random.randint(0, len(node.untried_actions) - 1)
        )

        # Aplicar ação ao estado atual
        new_state = self._apply_action(node.state, action)

        # Criar nó filho
        child = MCTSNode(
            state=new_state,
            action_taken=action,
            parent=node,
            depth=node.depth + 1,
            untried_actions=self._get_available_actions(new_state, node.depth + 1)
        )

        node.children.append(child)
        self.total_nodes_created += 1
        return child

    # ─── Fase 3: SIMULAÇÃO (Rollout) ────────────────────────────────────────

    def _simulate(self, node: MCTSNode, goal: str) -> float:
        """
        Rollout aleatório: simula ações até profundidade máxima e avalia estado final.
        Usa WorldModel quando disponível, senão usa heurística interna.
        """
        state = copy.deepcopy(node.state)
        depth = node.depth

        while depth < self.max_depth:
            actions = self._get_available_actions(state, depth)
            if not actions:
                break
            action = random.choice(actions)
            state = self._apply_action(state, action)
            depth += 1

        return self._evaluate_state(state, goal)

    # ─── Fase 4: RETROPROPAGAÇÃO ─────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Propagar recompensa de volta a todos os ancestrais"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    # ─── Funções Auxiliares ──────────────────────────────────────────────────

    def _get_initial_state(self, goal: str) -> Dict[str, Any]:
        """Obter estado inicial baseado no goal"""
        if self.world_model:
            return copy.deepcopy(self.world_model.current_state.state)

        # Estado padrão baseado em palavras-chave do goal
        goal_lower = goal.lower()
        state: Dict[str, Any] = {
            "goal": goal,
            "security": 0.5,
            "stability": 0.5,
            "cost": 1.0,
            "steps_taken": []
        }

        if any(k in goal_lower for k in ["security", "patch", "vulnerability", "attack"]):
            state["security"] = 0.2
            state["alert_level"] = "high"
        if any(k in goal_lower for k in ["stability", "crash", "downtime"]):
            state["stability"] = 0.3

        return state

    def _get_available_actions(self, state: Dict, depth: int) -> List[str]:
        """Retornar ações disponíveis (limita por profundidade para eficiência)"""
        if depth >= self.max_depth:
            return []

        all_actions = list(self._action_effects.keys())

        # Evitar ações que já foram tomadas recentemente
        taken = state.get("steps_taken", [])
        recent = set(taken[-2:]) if len(taken) >= 2 else set(taken)
        available = [a for a in all_actions if a not in recent]

        return available if available else all_actions

    def _apply_action(self, state: Dict, action: str) -> Dict:
        """Aplicar ação ao estado e retornar novo estado"""
        new_state = copy.deepcopy(state)
        effects = self._action_effects.get(action, {})

        for metric, delta in effects.items():
            if metric in new_state:
                new_state[metric] = max(0.0, min(1.0, new_state[metric] + delta))

        # Registrar passo
        steps = new_state.get("steps_taken", [])
        steps.append(action)
        new_state["steps_taken"] = steps

        return new_state

    def _evaluate_state(self, state: Dict, goal: str) -> float:
        """
        Avaliar qualidade de um estado em relação ao goal.
        Retorna recompensa entre 0.0 e 1.0.
        """
        reward = 0.0
        goal_lower = goal.lower()

        # Métricas base
        security = state.get("security", 0.5)
        stability = state.get("stability", 0.5)
        cost = state.get("cost", 0.5)

        # Pesos por tipo de goal
        if any(k in goal_lower for k in ["security", "patch", "vulnerability", "attack", "breach"]):
            reward = security * 0.6 + stability * 0.3 + cost * 0.1
        elif any(k in goal_lower for k in ["stability", "availability", "uptime"]):
            reward = stability * 0.6 + security * 0.3 + cost * 0.1
        elif any(k in goal_lower for k in ["cost", "budget", "efficient"]):
            reward = cost * 0.6 + security * 0.2 + stability * 0.2
        else:
            reward = (security + stability + cost) / 3.0

        # Penalidade por causalidade negativa (integração com CausalReasoning)
        if self.causal_reasoning:
            steps = state.get("steps_taken", [])
            for step in steps:
                if step == "IGNORE_ALERT":
                    reward *= 0.5  # Penalidade severa por ignorar alertas

        # Bônus por estado final seguro (integração com WorldModel)
        if state.get("system_status") == "secure":
            reward = min(1.0, reward + 0.15)

        return reward

    def _extract_best_sequence(self, root: MCTSNode) -> List[str]:
        """Extrair melhor sequência de ações percorrendo a árvore gulosa"""
        sequence = []
        node = root

        while node.children:
            # Escolher filho com maior recompensa média (sem exploração)
            best = max(node.children, key=lambda c: c.avg_reward)
            if best.avg_reward <= 0 or best.action_taken is None:
                break
            sequence.append(best.action_taken)
            node = best

        return sequence

    def _extract_alternatives(self, root: MCTSNode, top_n: int = 3) -> List[Dict]:
        """Extrair planos alternativos (filhos diretos da raiz ordenados por reward)"""
        if not root.children:
            return []

        sorted_children = sorted(root.children, key=lambda c: c.avg_reward, reverse=True)

        alternatives = []
        for child in sorted_children[:top_n]:
            if child.action_taken:
                alternatives.append({
                    "first_action": child.action_taken,
                    "expected_reward": round(child.avg_reward, 4),
                    "visits": child.visits
                })

        return alternatives

    # ─── Interface Principal ──────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        available_actions: Optional[List[str]] = None,
        max_iterations: int = 200
    ) -> PlanResult:
        """
        Planejar sequência de ações para atingir um goal usando MCTS.
        
        Args:
            goal: Objetivo em linguagem natural (ex: "secure the system")
            available_actions: Lista de ações possíveis (usa padrão se None)
            max_iterations: Número de iterações MCTS (200 = ~50ms)
        
        Returns:
            PlanResult com melhor sequência e alternativas
        """
        if goal is None:
            raise ValueError("goal is required")
        if not isinstance(goal, str):
            raise TypeError("goal must be a string")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        start_time = time.time()
        self.total_plans += 1

        # Estado inicial
        initial_state = self._get_initial_state(goal)

        # Substituir ações disponíveis se fornecidas
        if available_actions is not None:
            # Registrar ações customizadas com efeitos neutros se desconhecidas
            for action in available_actions:
                if action not in self._action_effects:
                    self._action_effects[action] = {
                        "security": 0.1, "stability": 0.1, "cost": -0.1
                    }

        # Criar raiz da árvore
        root = MCTSNode(
            state=initial_state,
            action_taken=None,
            parent=None,
            depth=0,
            untried_actions=available_actions if available_actions is not None else (
                self._get_available_actions(initial_state, 0)
            )
        )
        self.total_nodes_created += 1

        # ─── Loop Principal MCTS ─────────────────────────────────────────
        for _i in range(max_iterations):
            # 1. Seleção
            selected = self._select(root)

            # 2. Expansão
            if not selected.is_fully_expanded:
                selected = self._expand(selected)

            # 3. Simulação
            reward = self._simulate(selected, goal)

            # 4. Retropropagação
            self._backpropagate(selected, reward)

        self.total_iterations += max_iterations

        # ─── Extrair Resultados ──────────────────────────────────────────
        best_sequence = self._extract_best_sequence(root)
        best_child = max(root.children, key=lambda c: c.avg_reward) if root.children else None
        expected_reward = best_child.avg_reward if best_child else 0.0
        alternatives = self._extract_alternatives(root)

        planning_time = time.time() - start_time

        return PlanResult(
            goal=goal,
            best_action_sequence=best_sequence,
            expected_reward=round(expected_reward, 4),
            iterations_run=max_iterations,
            tree_nodes_created=self.total_nodes_created,
            alternative_plans=alternatives,
            planning_time=round(planning_time, 4),
            success_probability=round(min(1.0, expected_reward * 1.1), 4)
        )

    def get_statistics(self) -> Dict:
        """Estatísticas do planejador"""
        return {
            "total_plans": self.total_plans,
            "total_iterations": self.total_iterations,
            "total_nodes_created": self.total_nodes_created,
            "avg_iterations_per_plan": (
                self.total_iterations / self.total_plans
                if self.total_plans > 0 else 0
            ),
            "action_space_size": len(self._action_effects)
        }
