# Mutantes de `veto_key` retirados do catálogo `cognitivo_mutation` (auditáveis)

Origem: #157 (`veto_key`), revisão da Pearl e errata E-VK6 (`docs/cognitivo/AUDITORIA_2026-10-22.md`).
Cada entrada traz **arquivo, trecho (`needle`) e substituição exatos**, como o catálogo os aplicaria
(`tools/cognitivo_mutation.py`; o trecho tem de casar exatamente uma vez, senão o resultado é `not_applied`).
O teste `test_veto_key_retired_mutants_contract.py` confirma que cada trecho ainda aplica uma vez, que este documento o
cita, e que os equivalentes não divergem de `veto_key` num fuzz determinístico (`random.Random(11)`, 150 000 entradas
num alfabeto de separadores, pontuação, invisíveis e não-ASCII) nem, para os dois preenchedores, numa varredura de
**todos** os code points (`"A" + chr(i) + "B"`).

Critério para chamar de equivalente: não há entrada que distinga o mutante do original **e** o argumento abaixo
explica por quê. Lição de `dca56c4`: equivalência não se prova só pelo valor de retorno, também pelas chamadas a métodos
do chamador (`__eq__`, `__hash__`). Por isso cada argumento vem com a medição.

## Equivalentes (retirados; sem teste que possa matá-los)

### `vk_sep_not_greedy` (retirado em `dca56c4`)
- Arquivo: `core/cognitivo/action_effects.py`
- Trecho:
```
\u00b7]+")
```
- Substituição:
```
\u00b7]+?")
```
- Por quê: `+` guloso ou não cria runs de `_` diferentes, mas a compressão `re.sub(r"_+", "_", ...)` seguinte as iguala.
  Fuzz: 0 divergências.

### `vk_ascii_allows_dash` (retirado em `dca56c4`)
- Arquivo: `core/cognitivo/action_effects.py`
- Trecho:
```
_ASCII_KEY = re.compile(r"[A-Za-z0-9_]+")
```
- Substituição:
```
_ASCII_KEY = re.compile(r"[A-Za-z0-9_-]+")
```
- Por quê: o `-` já virou `_` no passo dos separadores (e o NFKC leva os hifens de largura total ao `-` antes), então
  nunca chega à checagem ASCII. Fuzz: 0 divergências.

### `vk_text_not_stripped_before_separators` (retirado na rodada 2, por S1)
- Arquivo: `core/cognitivo/action_effects.py`
- Trecho:
```
_SEPARATORS.sub("_", text.strip())
```
- Substituição:
```
_SEPARATORS.sub("_", text)
```
- Por quê: era **morto** em `dca56c4` (`NOME()  ` com espaços depois dos parênteses dava `None` sem o `strip()`).
  Com `_CALL_SYNTAX = (?:\(_*\))?[_;!?]*\Z` (S1) o `_` que sobra do espaço final é consumido pela cauda, e o `_` do
  início é tirado depois por `.strip("_")`. Passou a ser equivalente **por efeito de S1**. Fuzz: 0 divergências em
  600 000 entradas (medição de 2026-11-04) e 0 nas 150 000 do teste.

### `vk_filler_hangul_compat_kept` (rodada 2, S1)
- Arquivo: `core/cognitivo/action_effects.py`
- Trecho (a entrada U+3164 de `_FILLERS`):
```
\u3164
```
- Substituição: vazia (remove U+3164 de `_FILLERS`).
- Por quê: a NFKC (passo anterior) leva U+3164 a U+1160, que continua em `_FILLERS`; U+3164 nunca chega ao filtro.
  Varredura de todos os code points: 0 divergências. Mantida em `_FILLERS` por defesa em profundidade (a S1 aprovada pede
  as cinco).

### `vk_filler_halfwidth_hangul_kept` (rodada 2, S1)
- Arquivo: `core/cognitivo/action_effects.py`
- Trecho (a entrada U+FFA0 de `_FILLERS`):
```
\uffa0
```
- Substituição: vazia (remove U+FFA0 de `_FILLERS`).
- Por quê: idem, a NFKC leva U+FFA0 a U+1160. Varredura de todos os code points: 0 divergências.

## Não equivalente (reativado e morto)

### `vk_mcts_non_str_candidate_vetoed` (M3a; retirado por engano em `dca56c4`, reativado na rodada 2)
- Arquivo: `core/cognitivo/mcts_planner.py`
- Trecho:
```
if not isinstance(action, str) or any(v["action"] == action for v in vetoed):
```
- Substituição:
```
if any(v["action"] == action for v in vetoed):
```
- Por quê **não** é equivalente (Pearl): um candidato não-`str` cujo `__eq__` levanta exceção quebra `plan()` sem a
  guarda; o valor de retorno igual escondia a chamada ao método do chamador. Morto por
  `test_red_pr157_pearl.S3EntradaNaoStrNaoChamaMetodoDoChamador.test_plan_nao_chama_eq_de_candidato_nao_str`. O par
  `vk_lookup_non_str_hashed` (M3b, `if action in vetoes:` sem `isinstance`, `__hash__` hostil) foi adicionado e é morto
  pelo outro teste de S3.
