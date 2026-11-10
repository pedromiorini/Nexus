"""System health diagnostics extracted from the legacy Nexus core."""
# NEXUS_EXTRACTED_CORE_MODULE

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class HealthIndicator:
    """Indicador de saúde"""
    indicator_id: str
    component: str
    status: str  # "healthy", "warning", "critical"
    health_score: float  # 0-1
    details: str

@dataclass
class DiagnosticReport:
    """Relatório de diagnóstico"""
    report_id: str
    timestamp: float
    overall_health: float  # 0-1
    indicators: List[HealthIndicator]
    recommendations: List[str]

@dataclass
class SystemAlert:
    """Alerta do sistema"""
    alert_id: str
    severity: str  # "info", "warning", "critical"
    component: str
    message: str
    timestamp: float

class RealSystemHealthDiagnosticsEngine:
    """
    POST62: System Health & Diagnostics Engine — REAL Comprehensive System Monitoring

    System Health & Diagnostics = monitoramento abrangente de saúde do sistema.
    "Prevention is better than cure" - Desiderius Erasmus

    Princípios Centrais:
    - Comprehensive monitoring: Monitoramento abrangente
    - Proactive diagnostics: Diagnósticos proativos
    - Health tracking: Rastreamento de saúde
    - Early warning system: Sistema de alerta precoce
    - Self-diagnosis: Auto-diagnóstico

    Componentes:
    1. Health Monitor: Monitora saúde de componentes
    2. Diagnostic Analyzer: Analisa e diagnostica problemas
    3. Alert Generator: Gera alertas
    4. Recommendation Engine: Gera recomendações
    5. System Inspector: Inspeciona sistema completo

    Multiplica:
    • × POST54 (Self-Monitoring): Self-monitoring + Diagnostics = saúde total
    • × POST50 (Error Correction): Error correction baseado em diagnósticos
    • × POST42 (Introspection): Introspection profunda do sistema
    • × POST55 (Integration): Integration monitora saúde global

    Baseado em:
    - System Health Monitoring (Vachtsevanos et al., 2006)
    - Fault Diagnosis (Isermann, 2006)
    - Prognostics and Health Management (PHM) (Jardine et al., 2006)
    - Self-Healing Systems (Ghosh et al., 2007)
    """

    def __init__(self, self_monitoring=None, error_correction=None,
                 introspection=None):
        self.self_monitoring = self_monitoring
        self.error_correction = error_correction
        self.introspection = introspection

        # Health indicators
        self.indicators: List[HealthIndicator] = []

        # Diagnostic reports
        self.reports: List[DiagnosticReport] = []

        # System alerts
        self.alerts: List[SystemAlert] = []

        # Health thresholds
        self.health_thresholds = {
            "healthy": 0.8,
            "warning": 0.5,
            "critical": 0.3
        }

        # Estatísticas
        self.total_checks = 0
        self.total_alerts = 0
        self.total_reports = 0
        self.avg_system_health = 1.0

        print("🏥📊 POST 62 - RealSystemHealthDiagnosticsEngine initialized (System diagnostics active)")

    def check_component_health(self, component: str,
                               metrics: Dict[str, float]) -> HealthIndicator:
        """
        Verificar saúde de componente.

        Args:
            component: Nome do componente
            metrics: Métricas do componente

        Returns:
            HealthIndicator
        """
        # Calcular score de saúde
        health_score = self._calculate_health_score(metrics)

        # Determinar status
        status = self._determine_status(health_score)

        # Gerar detalhes
        details = self._generate_health_details(metrics, health_score)

        indicator = HealthIndicator(
            indicator_id=str(uuid.uuid4()),
            component=component,
            status=status,
            health_score=health_score,
            details=details
        )

        self.indicators.append(indicator)
        self.total_checks += 1

        # Gerar alerta se necessário
        if status in ["warning", "critical"]:
            self._generate_alert(component, status, details)

        return indicator

    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calcular score de saúde"""
        if not metrics:
            return 1.0

        # Média das métricas (assumindo que métricas estão normalizadas 0-1)
        scores = []
        for metric_name, value in metrics.items():
            # Ajustar se métrica for invertida (ex: error_rate)
            if "error" in metric_name.lower() or "latency" in metric_name.lower():
                score = 1.0 - min(1.0, value)
            else:
                score = value
            scores.append(score)

        health_score = sum(scores) / len(scores) if scores else 1.0

        return health_score

    def _determine_status(self, health_score: float) -> str:
        """Determinar status baseado no score"""
        if health_score >= self.health_thresholds["healthy"]:
            return "healthy"
        elif health_score >= self.health_thresholds["warning"]:
            return "warning"
        else:
            return "critical"

    def _generate_health_details(self, metrics: Dict[str, float],
                                 health_score: float) -> str:
        """Gerar detalhes de saúde"""
        if health_score >= 0.8:
            return f"Component operating normally. Score: {health_score:.2f}"
        elif health_score >= 0.5:
            return f"Component showing degradation. Score: {health_score:.2f}"
        else:
            return f"Component critically degraded. Score: {health_score:.2f}"

    def _generate_alert(self, component: str, severity: str, message: str):
        """Gerar alerta do sistema"""
        alert = SystemAlert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            component=component,
            message=message,
            timestamp=time.time()
        )

        self.alerts.append(alert)
        self.total_alerts += 1

    def run_full_diagnostics(self,
                           component_metrics: Dict[str, Dict[str, float]]) -> DiagnosticReport:
        """
        Executar diagnóstico completo.

        Args:
            component_metrics: Métricas por componente

        Returns:
            DiagnosticReport
        """
        indicators = []

        # Verificar cada componente
        for component, metrics in component_metrics.items():
            indicator = self.check_component_health(component, metrics)
            indicators.append(indicator)

        # Calcular saúde geral
        overall_health = self._calculate_overall_health(indicators)

        # Gerar recomendações
        recommendations = self._generate_recommendations(indicators)

        report = DiagnosticReport(
            report_id=str(uuid.uuid4()),
            timestamp=time.time(),
            overall_health=overall_health,
            indicators=indicators,
            recommendations=recommendations
        )

        self.reports.append(report)
        self.total_reports += 1

        # Update statistics
        self._update_health_statistics(overall_health)

        return report

    def _calculate_overall_health(self, indicators: List[HealthIndicator]) -> float:
        """Calcular saúde geral do sistema"""
        if not indicators:
            return 1.0

        # Média ponderada por importância
        # Componentes críticos têm peso maior
        total_score = sum(i.health_score for i in indicators)
        overall = total_score / len(indicators)

        return overall

    def _generate_recommendations(self, indicators: List[HealthIndicator]) -> List[str]:
        """Gerar recomendações baseadas em indicadores"""
        recommendations = []

        for indicator in indicators:
            if indicator.status == "critical":
                recommendations.append(
                    f"URGENT: Investigate {indicator.component} immediately (score: {indicator.health_score:.2f})"
                )
            elif indicator.status == "warning":
                recommendations.append(
                    f"Monitor {indicator.component} closely (score: {indicator.health_score:.2f})"
                )

        # Recomendação geral se tudo OK
        if not recommendations:
            recommendations.append("System operating at optimal health")

        return recommendations

    def get_active_alerts(self, severity: Optional[str] = None) -> List[SystemAlert]:
        """Obter alertas ativos"""
        if severity:
            return [a for a in self.alerts if a.severity == severity]
        return self.alerts.copy()

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Obter resumo de saúde do sistema"""
        # Componentes por status
        healthy = sum(1 for i in self.indicators if i.status == "healthy")
        warning = sum(1 for i in self.indicators if i.status == "warning")
        critical = sum(1 for i in self.indicators if i.status == "critical")

        return {
            "overall_health": self.avg_system_health,
            "total_components_checked": len(self.indicators),
            "healthy_components": healthy,
            "warning_components": warning,
            "critical_components": critical,
            "active_alerts": len(self.alerts),
            "total_reports": len(self.reports)
        }

    def _update_health_statistics(self, health: float):
        """Atualizar estatísticas de saúde"""
        if self.avg_system_health == 1.0 and self.total_reports == 1:
            self.avg_system_health = health
        else:
            self.avg_system_health = (self.avg_system_health + health) / 2

    def get_statistics(self) -> Dict:
        """Estatísticas do System Health & Diagnostics Engine"""
        return {
            "total_health_checks": self.total_checks,
            "total_alerts": self.total_alerts,
            "total_diagnostic_reports": self.total_reports,
            "avg_system_health": self.avg_system_health,
            "active_indicators": len(self.indicators),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical"])
        }

# Preserve the historical reflection and pickle lookup path.
for _legacy_class in (
    DiagnosticReport,
    HealthIndicator,
    RealSystemHealthDiagnosticsEngine,
    SystemAlert,
):
    _legacy_class.__module__ = "core.constitutional_brain"
del _legacy_class
