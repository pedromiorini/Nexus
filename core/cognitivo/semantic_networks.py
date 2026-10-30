"""Redes semânticas (POST26) extraídas de core/constitutional_brain.py.

Extração strangler: o código foi movido do monólito, move-only (byte-idêntico,
SHA-256 da fonte em tools/cognitivo_eval.py::SN_SOURCE_SHA); core.constitutional_brain
reexporta as classes no ponto original.
Dependências: apenas a biblioteca padrão. O relógio é time.time() (created_at de
cada conceito) e os ids vêm de uuid.uuid4(). knowledge_graph e language são
armazenados e nunca lidos. Lacunas declaradas e fora do escopo do move (pré-registro
SN1–SN6 em docs/cognitivo/AUDITORIA_2026-10-22.md): a ativação depende da ordem das
relações (SN3) e consultas IS-A em ciclo estouram a recursão (SN4).
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = [
    "ConceptNode",
    "RealSemanticNetworksSystem",
    "SemanticRelation",
]


# ═══════════════════════════════════════════════════════════════════════════
# MÓDULO 26: SEMANTIC NETWORKS SYSTEM (POST 26)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    """Nó representando conceito na rede semântica"""
    concept_id: str
    label: str
    properties: Dict[str, Any]
    activation: float  # Nível de ativação (0-1)
    created_at: float


@dataclass
class SemanticRelation:
    """Relação semântica entre conceitos"""
    relation_id: str
    relation_type: str  # IS-A, HAS-A, PART-OF, CAUSES, etc.
    source_id: str
    target_id: str
    strength: float  # Força da relação (0-1)


class RealSemanticNetworksSystem:
    """
    POST26: Semantic Networks System — REAL Conceptual Knowledge Graph
    
    Redes Semânticas = representação de conhecimento conceitual em grafo.
    
    Características:
    - Nós representam conceitos
    - Arestas representam relações semânticas
    - Spreading activation para recuperação
    - Herança de propriedades via IS-A
    
    Tipos de Relações:
    - IS-A: hierarquia taxonômica
    - HAS-A: propriedades/atributos
    - PART-OF: composição mereológica
    - CAUSES: relações causais
    - SIMILAR-TO: similaridade
    
    Componentes:
    - Concept Nodes: Conceitos com ativação
    - Semantic Relations: Relações tipadas
    - Spreading Activation: Propagação de ativação
    - Inheritance: Herança via IS-A
    
    Multiplica:
    • × POST4 (KnowledgeGraph): Complementa com semântica
    • × POST18 (Language): Semântica linguística
    • × POST1 (Memory): Memória conceitual
    • × POST2 (Reasoning): Raciocínio por redes
    
    Baseado em:
    - Collins & Quillian's Hierarchical Network (1969)
    - Collins & Loftus' Spreading Activation (1975)
    - Anderson's ACT-R Semantic Memory (1983)
    - Quillia

n's Semantic Networks (1968)
    """

    def __init__(self, knowledge_graph=None, language=None):
        self.kg = knowledge_graph
        self.language = language

        # Network storage
        self.concepts: Dict[str, ConceptNode] = {}
        self.relations: List[SemanticRelation] = []

        # Spreading activation parameters
        self.activation_decay = 0.1
        self.spread_factor = 0.7

        # Estatísticas
        self.total_concepts = 0
        self.total_relations = 0
        self.activations_performed = 0

        print("🧠🕸️ POST 26 - RealSemanticNetworksSystem initialized (Conceptual knowledge graph active)")

    def add_concept(self, label: str, properties: Dict[str, Any] | None = None) -> ConceptNode:
        """
        Adicionar conceito à rede.
        
        Args:
            label: Label do conceito
            properties: Propriedades do conceito
        
        Returns:
            ConceptNode criado
        """
        properties = properties or {}

        concept = ConceptNode(
            concept_id=str(uuid.uuid4()),
            label=label,
            properties=properties,
            activation=0.0,
            created_at=time.time()
        )

        self.concepts[concept.concept_id] = concept
        self.total_concepts += 1

        return concept

    def add_relation(self, 
                    source_label: str,
                    relation_type: str,
                    target_label: str,
                    strength: float = 1.0) -> SemanticRelation:
        """
        Adicionar relação semântica.
        
        Args:
            source_label: Conceito fonte
            relation_type: Tipo de relação
            target_label: Conceito alvo
            strength: Força da relação
        
        Returns:
            SemanticRelation criada
        """
        # Encontrar ou criar conceitos
        source = self._find_or_create_concept(source_label)
        target = self._find_or_create_concept(target_label)

        relation = SemanticRelation(
            relation_id=str(uuid.uuid4()),
            relation_type=relation_type,
            source_id=source.concept_id,
            target_id=target.concept_id,
            strength=strength
        )

        self.relations.append(relation)
        self.total_relations += 1

        return relation

    def _find_or_create_concept(self, label: str) -> ConceptNode:
        """Encontrar ou criar conceito por label"""
        # Buscar existente
        for concept in self.concepts.values():
            if concept.label.lower() == label.lower():
                return concept

        # Criar novo
        return self.add_concept(label)

    def activate_concept(self, label: str, activation: float = 1.0):
        """
        Ativar conceito.
        
        Args:
            label: Label do conceito
            activation: Nível de ativação inicial
        """
        concept = self._find_or_create_concept(label)
        concept.activation = min(1.0, activation)

        self.activations_performed += 1

    def spread_activation(self, iterations: int = 3) -> Dict[str, float]:
        """
        Propagar ativação pela rede.
        
        Spreading Activation: ativação se propaga de nós ativos
        para nós conectados, com decay.
        
        Args:
            iterations: Número de iterações
        
        Returns:
            Dict de conceitos ativados
        """
        for _ in range(iterations):
            # Para cada relação
            for relation in self.relations:
                source = self.concepts.get(relation.source_id)
                target = self.concepts.get(relation.target_id)

                if not source or not target:
                    continue

                # Propagar ativação de source para target
                if source.activation > 0.1:
                    spread_amount = source.activation * self.spread_factor * relation.strength
                    target.activation = min(1.0, target.activation + spread_amount)

            # Apply decay
            for concept in self.concepts.values():
                concept.activation = max(0.0, concept.activation - self.activation_decay)

        # Retornar conceitos ativados
        activated = {
            c.label: c.activation
            for c in self.concepts.values()
            if c.activation > 0.1
        }

        return activated

    def get_related_concepts(self, label: str, relation_type: str | None = None) -> List[ConceptNode]:
        """
        Obter conceitos relacionados.
        
        Args:
            label: Label do conceito
            relation_type: Tipo de relação (None = todos)
        
        Returns:
            Lista de conceitos relacionados
        """
        # Encontrar conceito
        source = None
        for c in self.concepts.values():
            if c.label.lower() == label.lower():
                source = c
                break

        if not source:
            return []

        # Buscar relações
        related_ids = []
        for relation in self.relations:
            if relation.source_id == source.concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    related_ids.append(relation.target_id)

        # Retornar conceitos
        related = [self.concepts[rid] for rid in related_ids if rid in self.concepts]

        return related

    def inherit_properties(self, concept_label: str) -> Dict[str, Any]:
        """
        Herdar propriedades via relações IS-A.
        
        Property Inheritance: conceitos herdam propriedades
        de conceitos superiores na hierarquia IS-A.
        
        Args:
            concept_label: Label do conceito
        
        Returns:
            Propriedades herdadas
        """
        # Encontrar conceito
        concept = self._find_or_create_concept(concept_label)

        # Propriedades próprias
        inherited = concept.properties.copy()

        # Buscar relações IS-A
        parent_concepts = self.get_related_concepts(concept_label, "IS-A")

        # Herdar recursivamente
        for parent in parent_concepts:
            parent_props = self.inherit_properties(parent.label)
            # Propriedades do pai são default (não sobrescrevem)
            for key, value in parent_props.items():
                if key not in inherited:
                    inherited[key] = value

        return inherited

    def verify_hierarchy(self, child_label: str, ancestor_label: str) -> bool:
        """
        Verificar se child é descendente de ancestor via IS-A.
        
        Args:
            child_label: Conceito filho
            ancestor_label: Conceito ancestral
        
        Returns:
            True se child IS-A ancestor
        """
        # Base case
        if child_label.lower() == ancestor_label.lower():
            return True

        # Buscar pais
        parents = self.get_related_concepts(child_label, "IS-A")

        # Recursão
        for parent in parents:
            if self.verify_hierarchy(parent.label, ancestor_label):
                return True

        return False

    def query_by_properties(self, required_properties: Dict[str, Any]) -> List[ConceptNode]:
        """
        Query conceitos por propriedades.
        
        Args:
            required_properties: Propriedades requeridas
        
        Returns:
            Conceitos matching
        """
        matches = []

        for concept in self.concepts.values():
            # Obter propriedades herdadas
            all_props = self.inherit_properties(concept.label)

            # Verificar match
            match = True
            for key, value in required_properties.items():
                if key not in all_props or all_props[key] != value:
                    match = False
                    break

            if match:
                matches.append(concept)

        return matches

    def get_statistics(self) -> Dict:
        """Estatísticas do Semantic Networks System"""
        activated_concepts = sum(1 for c in self.concepts.values() if c.activation > 0.1)

        return {
            "total_concepts": self.total_concepts,
            "total_relations": self.total_relations,
            "activations_performed": self.activations_performed,
            "activated_concepts": activated_concepts,
            "avg_activation": sum(c.activation for c in self.concepts.values()) / max(1, len(self.concepts))
        }
