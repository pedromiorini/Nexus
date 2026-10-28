"""Analogical reasoning (POST27) extraído de core/constitutional_brain.py.

Extração strangler: o código abaixo foi movido sem alteração de comportamento;
core.constitutional_brain reexporta as classes no ponto original.
Não há relógio nem aleatoriedade além de uuid.uuid4() para situation_id e
mapping_id. reasoning, memory e semantic_networks só são armazenados, nunca lidos.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "AnalogyMapping",
    "AnalogySituation",
    "RealAnalogicalReasoningSystem",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 27: ANALOGICAL REASONING SYSTEM (POST 27)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalogySituation:
    """Situação para raciocínio analógico"""
    situation_id: str
    domain: str
    entities: List[str]
    relations: List[Tuple[str, str, str]]  # (relation, entity1, entity2)
    solution: Optional[str]


@dataclass
class AnalogyMapping:
    """Mapeamento analógico entre domínios"""
    mapping_id: str
    source_situation: AnalogySituation
    target_situation: AnalogySituation
    entity_mappings: Dict[str, str]  # source -> target
    quality_score: float


class RealAnalogicalReasoningSystem:
    """
    POST27: Analogical Reasoning System — REAL Cross-Domain Transfer
    
    Raciocínio Analógico = transferir conhecimento entre domínios via similaridade estrutural.
    
    "A is to B as C is to D"
    
    Componentes:
    - Structure Mapping: Mapear estruturas relacionais
    - Analogy Finding: Encontrar analogias relevantes
    - Solution Transfer: Transferir soluções entre domínios
    - Evaluation: Avaliar qualidade da analogia
    
    Multiplica:
    • × POST2 (Reasoning): Raciocínio por analogia
    • × POST1 (Memory): Memória de casos analógicos
    • × POST26 (SemanticNetworks): Estrutura conceitual
    • × POST23 (Creativity): Analogias criativas
    
    Baseado em:
    - Gentner's Structure-Mapping Theory (1983)
    - Holyoak & Thagard's Multiconstraint Theory (1989)
    - Hummel & Holyoak's LISA Model (1997)
    - Falkenhainer et al.'s SME (1989)
    """

    def __init__(self, reasoning=None, memory=None, semantic_networks=None):
        self.reasoning = reasoning
        self.memory = memory
        self.semantic_networks = semantic_networks

        # Storage de situações conhecidas
        self.known_situations: List[AnalogySituation] = []

        # Estatísticas
        self.total_analogies_found = 0
        self.successful_transfers = 0

        print("🧠🔄 POST 27 - RealAnalogicalReasoningSystem initialized (Cross-domain transfer active)")

    def add_situation(self, domain: str, entities: List[str], 
                     relations: List[Tuple[str, str, str]], solution: str | None = None) -> AnalogySituation:
        """Adicionar situação conhecida"""
        situation = AnalogySituation(
            situation_id=str(uuid.uuid4()),
            domain=domain,
            entities=entities,
            relations=relations,
            solution=solution
        )

        self.known_situations.append(situation)
        return situation

    def find_analogy(self, target_domain: str, target_entities: List[str],
                    target_relations: List[Tuple[str, str, str]]) -> Optional[AnalogyMapping]:
        """
        Encontrar analogia para situação alvo.
        
        Args:
            target_domain: Domínio alvo
            target_entities: Entidades alvo
            target_relations: Relações alvo
        
        Returns:
            AnalogyMapping ou None
        """
        target_situation = AnalogySituation(
            situation_id="target",
            domain=target_domain,
            entities=target_entities,
            relations=target_relations,
            solution=None
        )

        # Buscar melhor analogia
        best_mapping = None
        best_score = 0.0

        for source in self.known_situations:
            if source.domain == target_domain:
                continue  # Skip mesmo domínio

            # Tentar mapear
            mapping = self._map_structures(source, target_situation)

            if mapping and mapping.quality_score > best_score:
                best_score = mapping.quality_score
                best_mapping = mapping

        if best_mapping:
            self.total_analogies_found += 1

        return best_mapping

    def _map_structures(self, source: AnalogySituation, 
                       target: AnalogySituation) -> Optional[AnalogyMapping]:
        """Structure mapping entre source e target"""
        # Simplificado: mapear entidades por posição nas relações
        entity_mappings = {}
        matched_relations = 0

        # Tentar mapear cada relação do target para source
        for t_rel, t_e1, t_e2 in target.relations:
            for s_rel, s_e1, s_e2 in source.relations:
                if t_rel == s_rel:  # Mesma relação
                    # Mapear entidades
                    entity_mappings[s_e1] = t_e1
                    entity_mappings[s_e2] = t_e2
                    matched_relations += 1
                    break

        if matched_relations == 0:
            return None

        # Quality score
        quality = matched_relations / max(len(source.relations), len(target.relations))

        mapping = AnalogyMapping(
            mapping_id=str(uuid.uuid4()),
            source_situation=source,
            target_situation=target,
            entity_mappings=entity_mappings,
            quality_score=quality
        )

        return mapping

    def transfer_solution(self, mapping: AnalogyMapping) -> Optional[str]:
        """
        Transferir solução do source para target via analogia.
        
        Args:
            mapping: Mapeamento analógico
        
        Returns:
            Solução transferida ou None
        """
        if not mapping.source_situation.solution:
            return None

        # Adaptar solução para target domain
        source_solution = mapping.source_situation.solution

        # Substituir entidades mapeadas
        adapted_solution = source_solution
        for source_entity, target_entity in mapping.entity_mappings.items():
            adapted_solution = adapted_solution.replace(source_entity, target_entity)

        self.successful_transfers += 1

        return adapted_solution

    def reason_by_analogy(self, problem_description: str, 
                         target_domain: str) -> Optional[str]:
        """
        Resolver problema por analogia.
        
        Args:
            problem_description: Descrição do problema
            target_domain: Domínio do problema
        
        Returns:
            Solução ou None
        """
        # Extrair estrutura do problema (simplificado)
        entities = self._extract_entities(problem_description)
        relations = self._extract_relations(problem_description)

        # Buscar analogia
        mapping = self.find_analogy(target_domain, entities, relations)

        if not mapping:
            return None

        # Transferir solução
        solution = self.transfer_solution(mapping)

        return solution

    def _extract_entities(self, text: str) -> List[str]:
        """Extrair entidades de texto (simplificado)"""
        # Palavras capitalizadas
        words = text.split()
        entities = [w.strip('.,!?') for w in words if w[0].isupper() and len(w) > 1]
        return entities[:5]  # Limit

    def _extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extrair relações de texto (simplificado)"""
        # Detectar padrões relacionais básicos
        relations = []

        if "similar to" in text.lower():
            relations.append(("SIMILAR", "entity1", "entity2"))
        if "causes" in text.lower():
            relations.append(("CAUSES", "entity1", "entity2"))
        if "part of" in text.lower():
            relations.append(("PART_OF", "entity1", "entity2"))

        return relations

    def get_statistics(self) -> Dict:
        """Estatísticas do Analogical Reasoning System"""
        return {
            "total_analogies_found": self.total_analogies_found,
            "successful_transfers": self.successful_transfers,
            "known_situations": len(self.known_situations),
            "transfer_success_rate": self.successful_transfers / max(1, self.total_analogies_found)
        }
