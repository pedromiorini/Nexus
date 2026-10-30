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
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple

from core.cognitivo.action_effects import planner_deltas
from core.cognitivo.causal_world_model import STATUS_UTILITY
from core.cognitivo.counterfactual import _normalize_action_key

__all__ = ["DECLARED_SUCCESS_CAP", "DEFAULT_VETOES", "MCTSNode", "PlanResult", "RealMCTSPlanner",
           "compare_allowed_futures"]

# P1b: ações que o planejador não recomenda por padrão (motivo devolvido em PlanResult.vetoed_actions).
# Quem quiser planejar com elas passa ``vetoes={}`` (ou outro mapa) ao construir o planejador.
DEFAULT_VETOES: Dict[str, str] = {
    "IGNORE_ALERT": "ignorar um alerta de segurança não é uma resposta aceitável por padrão",
    "SHUTDOWN_SERVICE": "desligar o serviço é uma ação destrutiva e exige decisão humana explícita",
}

# P1c: ação declarada fora da tabela é hipótese do usuário, não verificação: a chance de sucesso nunca passa disto.
DECLARED_SUCCESS_CAP = 0.9


def _veto_key(action: Any) -> Optional[str]:
    """Nome normalizado como ``counterfactual._normalize_action_key``; ``None`` se não for texto."""
    return _normalize_action_key(action) if isinstance(action, str) else None


def _veto_table(vetoes: Mapping[Any, Any]) -> Dict[str, str]:
    """Mapa chave normalizada → motivo; com duas chaves iguais depois de normalizar, vence a primeira."""
    table: Dict[str, str] = {}
    for name, reason in vetoes.items():
        key = _veto_key(name)
        if key and key not in table:
            table[key] = reason
    return table


def compare_allowed_futures(world_model: Any, sequences: List[List[str]],
                            vetoes: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """``world_model.compare_futures`` só sobre as sequências sem ação vetada (P1c).

    ``best_future_index`` é o índice **original** em ``sequences`` (``None`` se todas forem vetadas ou a
    lista for vazia). ``results`` traz só as permitidas, na ordem de ``allowed_indices``; ``vetoed_futures``
    lista os índices das vetadas e ``vetoed_actions`` as ações vetadas (``[{"action", "reason"}]``, ordem de
    aparição, sem repetir o mesmo texto). ``vetoes=None`` usa ``DEFAULT_VETOES``; ``{}`` desliga.
    """
    if vetoes is not None and not isinstance(vetoes, Mapping):
        raise TypeError("vetoes must be a mapping of action -> reason")
    table = _veto_table(DEFAULT_VETOES if vetoes is None else vetoes)
    allowed: List[int] = []
    vetoed_futures: List[int] = []
    vetoed_actions: List[Dict[str, str]] = []
    for index, sequence in enumerate(sequences):
        steps = sequence if isinstance(sequence, (list, tuple)) else ()
        hits = [a for a in steps if _veto_key(a) in table]
        if not hits:
            allowed.append(index)
            continue
        vetoed_futures.append(index)
        for action in hits:
            if all(v["action"] != action for v in vetoed_actions):
                vetoed_actions.append({"action": action, "reason": table[_veto_key(action)]})
    if allowed:
        out = dict(world_model.compare_futures([sequences[i] for i in allowed]))
        best = out.get("best_future_index")  # um modelo sem esse campo (dublê) segue como veio
        if isinstance(best, int) and not isinstance(best, bool) and 0 <= best < len(allowed):
            out["best_future_index"] = allowed[best]
    else:
        out = {"results": [], "best_future_index": None, "best_future": None}
    out["allowed_indices"] = allowed
    out["vetoed_futures"] = vetoed_futures
    out["vetoed_actions"] = vetoed_actions
    return out


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
    """Resultado de um planejamento MCTS.

    ``vetoed_actions`` (P1b) é atributo da instância, não campo: ``copy.copy``, ``copy.deepcopy`` e ``pickle``
    o preservam, mas ``dataclasses.replace`` **o perde** (volta ao padrão ``()``). Promovê-lo a campo exige
    errata do contrato ER5 (conjunto de campos de ``PlanResult``).
    """
    goal: str
    best_action_sequence: List[str]      # Sequência de ações recomendada
    expected_reward: float               # Recompensa esperada (0.0 - 1.0)
    iterations_run: int                  # Iterações MCTS executadas
    tree_nodes_created: int              # Total de nós na árvore
    alternative_plans: List[Dict]        # Planos alternativos encontrados
    planning_time: float                 # Tempo de planejamento em segundos
    success_probability: float           # Probabilidade estimada de sucesso
    # P1b: ações vetadas e motivo. Atributo da instância, não campo do dataclass (``fields``/``asdict``
    # seguem iguais ao contrato ER5); o planejador o preenche a cada ``plan()``.
    vetoed_actions: ClassVar[Tuple[Dict[str, str], ...]] = ()


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
        max_depth: int = 5,
        vetoes: Optional[Mapping[str, str]] = None
    ):
        if isinstance(exploration_constant, bool) or not isinstance(exploration_constant, Real):
            raise TypeError("exploration_constant must be a real number")
        if not math.isfinite(exploration_constant):
            raise ValueError("exploration_constant must be finite")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if vetoes is not None:
            if not isinstance(vetoes, Mapping):
                raise TypeError("vetoes must be a mapping of action -> reason")
            for name, reason in vetoes.items():
                if not isinstance(name, str) or not isinstance(reason, str):
                    raise TypeError("vetoes must map action names (str) to reasons (str)")
                if not _veto_key(name):
                    raise ValueError("veto action name must not be empty")
        self.vetoes: Dict[str, str] = dict(DEFAULT_VETOES if vetoes is None else vetoes)
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
        self._action_effects: Dict[str, Dict] = planner_deltas()  # tabela única (core.cognitivo.action_effects)
        # Escopo de um plan(): ações permitidas apenas durante o plano e melhor
        # sequência completa encontrada nos rollouts (estratégia "top score").
        self._plan_actions: Optional[List[str]] = None
        self._plan_effects: Dict[str, Dict] = {}
        self._best_found: Optional[tuple] = None
        # Cache de recompensas do modo mundo (sequência -> recompensa); vale só dentro de um plan().
        self._world_rewards: Dict[tuple, float] = {}

    def _uses_world_effects(self) -> bool:
        """Modo mundo: o modelo injetado sabe simular sequências de ações."""
        return self.world_model is not None and callable(getattr(self.world_model, "simulate", None))

    def _evaluate_world(self, state: Dict) -> float:
        """Recompensa do estado = utilidade do status *verificado* simulado pelo modelo, em [0, 1].

        Afirmação causal de texto livre (``final_state``) é hipótese não confiável: não pontua nem escolhe.
        """
        steps = tuple(state.get("steps_taken", []))
        if steps not in self._world_rewards:
            simulated = self.world_model.simulate(list(steps))
            utility = STATUS_UTILITY.get(simulated.verified_status, 0)
            self._world_rewards[steps] = utility / 3.0
        return self._world_rewards[steps]

    def _is_unmodeled(self, action: str) -> bool:
        """Ação sem efeito conhecido, por estrutura: fora da tabela de ações e não declarada.

        Texto livre sobre a ação (afirmação causal, aviso do simulador) não decide: só a tabela e as
        intervenções declaradas. Um modelo sem ``is_modeled`` cai no aviso do simulador (compatibilidade).
        """
        if self._uses_world_effects():
            is_modeled = getattr(self.world_model, "is_modeled", None)
            if callable(is_modeled):
                return not is_modeled(action)
            return bool(getattr(self.world_model.simulate([action]), "warnings", []))
        return action not in self._action_effects

    def _is_declared_only(self, action: str) -> bool:
        """Modelada só por intervenção declarada (fora da tabela): vale para o simulador, mas não é verificada."""
        if not self._uses_world_effects() or action in self._action_effects:
            return False
        is_modeled = getattr(self.world_model, "is_modeled", None)
        return bool(callable(is_modeled) and is_modeled(action))

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
        if depth > 0:  # prefixo = nó da árvore; max_depth é horizonte, não obrigação
            self._record_found(self._evaluate_state(state, goal), state)

        while depth < self.max_depth:
            actions = self._get_available_actions(state, depth)
            if not actions:
                break
            action = random.choice(actions)
            state = self._apply_action(state, action)
            depth += 1

        reward = self._evaluate_state(state, goal)
        self._record_found(reward, state)
        return reward

    def _found_rank(self, steps: List[str]) -> Tuple[int, int, Tuple[str, ...]]:
        """Chave de desempate do modo mundo (menor vence): menos ações fora da tabela (declaradas ou
        desconhecidas), depois menos passos, depois ordem lexicográfica da sequência."""
        outside = sum(1 for action in steps if action not in self._action_effects)
        return (outside, len(steps), tuple(str(action) for action in steps))

    def _record_found(self, reward: float, state: Dict) -> None:
        if reward <= 0:
            return
        steps = list(state.get("steps_taken", []))
        if self._best_found is not None and reward == self._best_found[0] and self._uses_world_effects():
            # modo mundo: entre planos de mesma recompensa, decide a chave determinística (_found_rank)
            if self._found_rank(steps) < self._found_rank(self._best_found[1]):
                self._best_found = (reward, steps)
            return
        if self._best_found is None or reward > self._best_found[0]:
            self._best_found = (reward, steps)

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

        all_actions = (list(self._plan_actions) if self._plan_actions is not None
                       else list(self._action_effects.keys()))

        # Evitar ações que já foram tomadas recentemente
        taken = state.get("steps_taken", [])
        recent = set(taken[-2:]) if len(taken) >= 2 else set(taken)
        available = [a for a in all_actions if a not in recent]

        return available if available else all_actions

    def _apply_action(self, state: Dict, action: str) -> Dict:
        """Aplicar ação ao estado e retornar novo estado"""
        new_state = copy.deepcopy(state)
        effects = self._action_effects.get(action) or self._plan_effects.get(action, {})

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
        if self._uses_world_effects():
            return self._evaluate_world(state)

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
        """Extrair planos alternativos (filhos diretos da raiz ordenados por reward).

        A ordem é determinística só no empate de recompensa (decidido pelo nome da primeira ação). Fora do
        empate ela segue a recompensa média dos rollouts sorteados, que varia com a semente (P1c).
        """
        if not root.children:
            return []

        # recompensa (já arredondada) em ordem decrescente; empate decidido pela primeira ação (estável)
        sorted_children = sorted(
            (c for c in root.children if c.action_taken),
            key=lambda c: (-round(c.avg_reward, 4), str(c.action_taken)),
        )

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

        # Ações fornecidas valem em todas as profundidades, só neste plano; a lista do
        # chamador é copiada. Ações sem modelo deixam as métricas do estado intactas.
        allowed = list(available_actions) if available_actions is not None else None
        # P1b: veto padrão. Ações vetadas saem do espaço de busca e voltam em PlanResult.vetoed_actions.
        candidates = allowed if allowed is not None else list(self._action_effects)
        table = _veto_table(self.vetoes)
        vetoed: List[Dict[str, str]] = []
        for action in candidates:
            if _veto_key(action) in table and all(v["action"] != action for v in vetoed):
                vetoed.append({"action": action, "reason": table[_veto_key(action)]})
        if vetoed:
            allowed = [a for a in candidates if _veto_key(a) not in table]
        try:
            self._plan_actions = allowed
            self._plan_effects = {}
            self._best_found = None
            return self._run_plan(goal, initial_state, allowed, max_iterations, start_time, vetoed)
        finally:
            self._plan_actions = None
            self._plan_effects = {}
            self._best_found = None
            self._world_rewards = {}

    def _run_plan(self, goal: str, initial_state: Dict[str, Any], allowed: Optional[List[str]],
                  max_iterations: int, start_time: float,
                  vetoed: Optional[List[Dict[str, str]]] = None) -> PlanResult:
        # Criar raiz da árvore
        prefix = len(initial_state.get("steps_taken", []))
        root = MCTSNode(
            state=initial_state,
            action_taken=None,
            parent=None,
            depth=0,
            untried_actions=list(allowed) if allowed is not None else (
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
        best_sequence = (self._best_found[1][prefix:] if self._best_found is not None
                         else self._extract_best_sequence(root))
        best_child = max(root.children, key=lambda c: c.avg_reward) if root.children else None
        expected_reward = best_child.avg_reward if best_child else 0.0
        if self._uses_world_effects():
            # Modo mundo: a recompensa da sequência devolvida é determinística (simulate), não a média do filho.
            expected_reward = (
                self._evaluate_world({"steps_taken": list(initial_state.get("steps_taken", [])) + best_sequence})
                if best_sequence else 0.0
            )
        alternatives = self._extract_alternatives(root)

        planning_time = time.time() - start_time
        if any(self._is_unmodeled(a) for a in best_sequence):
            success_probability = 0.0  # ação sem efeito conhecido não ganha confiança
        elif any(self._is_declared_only(a) for a in best_sequence):
            success_probability = round(min(DECLARED_SUCCESS_CAP, expected_reward * 1.1), 4)
        else:
            success_probability = round(min(1.0, expected_reward * 1.1), 4)

        result = PlanResult(
            goal=goal,
            best_action_sequence=best_sequence,
            expected_reward=round(expected_reward, 4),
            iterations_run=max_iterations,
            tree_nodes_created=self.total_nodes_created,
            alternative_plans=alternatives,
            planning_time=round(planning_time, 4),
            success_probability=success_probability
        )
        result.vetoed_actions = [dict(item) for item in (vetoed or [])]
        return result

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
