"""Gate mínimo de omissão de evidência.

Uma chave exigida só conta quando o valor realmente armazenado é exatamente
True. Chave ausente ou valor diferente de True devolve False, sem exceção.
Forma inválida levanta EvidenceGateRejection com código próprio, distinto da
omissão. Subclasses de dict e str são lidas pelo storage/descritores base, sem
chamar seus métodos sobrescritos.

execute_tool não consulta este helper. permission_level não entra aqui.
"""


class EvidenceGateRejection(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def evaluate_evidence_omission(required_evidence: object, tool_result: object) -> bool:
    if not isinstance(required_evidence, list):
        raise EvidenceGateRejection("required_evidence_invalid")
    required_items = tuple(list.__iter__(required_evidence))
    if any(not isinstance(key, str) for key in required_items):
        raise EvidenceGateRejection("required_evidence_invalid")
    required_keys = tuple(str.__str__(key) for key in required_items)
    if not isinstance(tool_result, dict):
        raise EvidenceGateRejection("tool_result_not_dict")
    if not required_keys:
        return True

    required_set = set(required_keys)
    stored_values: dict[str, bool] = {}
    for raw_key, value in dict.items(tool_result):
        if not isinstance(raw_key, str):
            continue
        key = str.__str__(raw_key)
        if key not in required_set:
            continue
        value_is_true = value is True
        if key in stored_values:
            stored_values[key] = stored_values[key] and value_is_true
        else:
            stored_values[key] = value_is_true

    return all(stored_values.get(key, False) for key in required_keys)
