"""Gate mínimo de omissão de evidência.

Uma chave exigida só conta quando o valor é exatamente True. Chave ausente
ou valor diferente de True devolve False, sem exceção. Forma inválida levanta
EvidenceGateRejection com código próprio, distinto da omissão.

execute_tool não consulta este helper. permission_level não entra aqui.
"""


class EvidenceGateRejection(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def evaluate_evidence_omission(required_evidence: object, tool_result: object) -> bool:
    if not isinstance(required_evidence, list) or any(
        not isinstance(key, str) for key in required_evidence
    ):
        raise EvidenceGateRejection("required_evidence_invalid")
    if not isinstance(tool_result, dict):
        raise EvidenceGateRejection("tool_result_not_dict")
    for key in required_evidence:
        if key not in tool_result or tool_result[key] is not True:
            return False
    return True
