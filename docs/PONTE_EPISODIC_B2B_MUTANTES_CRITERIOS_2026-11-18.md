# Pré-registro — mutantes da B2b no gate direcionado

Status: critérios antes de qualquer edição de `tools/targeted_mutation.py`. Não registra os mutantes. Não é GREEN. Não altera produção.

Base: `origin/main` `b56ebf21354f944b4e294708d4cfac6e89e26757` (`B2b: o episódio aberto só entra na busca se a chamada pedir (#199)`).
Python da sonda: 3.12.3. `python3.11` não está neste host.

## Fato — o gate hoje

`tools/targeted_mutation.py` declara 85 `Mutation`. Cada um tem `name`, `target`, `needle`, `replacement` e `tests`. O gate copia o repositório para um diretório temporário, roda o baseline de cada tupla de testes e só então aplica o mutante.

A agulha tem de ocorrer uma vez. `count != 1` grava `not_applied` e o processo sai 1, porque a saída 0 exige `applicable == len(MUTATIONS)` e `survived == 0`. Mutante morto é teste com saída diferente de 0. Timeout de 180 s conta como morto. A substituição é `str.replace` da agulha pela primeira ocorrência.

`tools/cognitivo_mutation.py` declara 1137 mutantes. 1137 / 3 = 379. Esse catálogo não entra nesta fatia.

O código da B2b está em `core/cognitivo/episodic_memory.py:183-204`. O default é `include_current_open: bool = False`. O ramo que acrescenta o aberto é:

```python
        if include_current_open is not True:
            return found
        extra = self._open_current_ranked_match(query, status, limit, found)
```

## Hipótese

**H1.** Dois mutantes textuais, com agulha única nesse arquivo, são mortos pelos testes que já existem. O gate passa de 85 para 87 mortos, com 0 sobreviventes e 0 não aplicados. A produção não muda.

**H0.** Os mutantes não entram no catálogo. A lacuna declarada no pré-registro da B2b permanece. O gate continua 85/85.

Falsifica H1 se a agulha ocorrer mais de uma vez, se o teste do mutante continuar verde, ou se matar o mutante exigir editar produção ou um teste existente.

## M1 — o default passa a incluir o aberto

Nome previsto: `episodic_search_include_current_open_defaults_off`.

Arquivo: `core/cognitivo/episodic_memory.py`.

Âncora, uma ocorrência nesta base:

```text
include_current_open: bool = False
```

Substituição:

```text
include_current_open: bool = True
```

Quem deve matar, com estes nomes:

- `test_n2_percent_search_without_flag_keeps_the_open_out`
- `test_n3_default_search_does_not_append_the_current_open`
- `test_percent_matches_only_literal_percent`
- `test_open_episode_stays_out_of_the_default_status`
- `test_limit_still_bounds_literal_matches`

O default verdadeiro faz a chamada sem flag anexar `"100% open"`. N2 e os três testes de curinga exigem essa ausência, e `limit=1` deixa de ter comprimento 1. N3 exige lista vazia no cenário de C1 sem flag. `test_n1_like_literal_file_stays_and_passes` e `test_n10_like_literal_baseline_file_passes` também ficam vermelhos, porque reexecutam o arquivo de curinga.

Tupla prevista do gate: `test_episodic_b2b_open_current_contract.py` e `test_episodic_like_literal_contract.py`. Qualquer um dos dois, sozinho, já sai diferente de 0.

## M2 — o ramo do aberto corrente fica neutro

Nome previsto: `episodic_search_include_current_open_can_append`.

Arquivo: `core/cognitivo/episodic_memory.py`.

Âncora, uma ocorrência nesta base:

```text
        if include_current_open is not True:
            return found
```

Substituição:

```text
        if True:
            return found
```

O método devolve a lista fechada e não chama `_open_current_ranked_match`.

Quem deve matar:

- `test_c1_explicit_flag_returns_the_current_open_match`
- `test_c2_closed_stays_and_current_open_is_appended_when_asked`
- `test_c3_explicit_flag_appends_after_a_full_limit`

C1–C3 passam `include_current_open=True` e exigem o aberto no fim. Com o ramo sempre devolvendo a lista fechada, a lista não tem esse item.

Tupla prevista: `test_episodic_b2b_open_current_contract.py`.

## Fato — viabilidade, mutação à mão, sem commit

Nesta VM, cada substituição foi aplicada em `core/cognitivo/episodic_memory.py` e o arquivo foi restaurado. `git diff` depois ficou vazio. As duas agulhas tinham contagem 1 antes da troca.

Comando, `PYTHONPATH=.` e `NEXUS_AUDIT_DB=/tmp/nexus_audit_mut_probe.sqlite3`:

```sh
python3 -m unittest -v \
  test_episodic_b2b_open_current_contract.py \
  test_episodic_like_literal_contract.py \
  test_episodic_retrieval_contract.py \
  test_episodic_memory_extraction_contract.py
```

M1, default `True`: saída 1. `Ran 41 tests`, `FAILED (failures=7)`. Falharam N1, N2, N3, N10 e os três testes de curinga nomeados acima. C1–C3 não falharam: a chamada explícita continua `True`. Recuperação e extração não falharam.

M2, `if True`: saída 1. `Ran 41 tests`, `FAILED (failures=3)`. Falharam só C1, C2 e C3. N2, o arquivo de curinga, a recuperação e a extração não falharam.

## RED e GREEN previstos

RED, antes de registrar: a mutação à mão faz falhar os testes nomeados em M1 e em M2. Isso foi medido acima. O catálogo ainda tem 85 entradas. Este documento não o altera.

GREEN, quando uma tarefa futura autorizar o registro: `python tools/targeted_mutation.py` imprime 87 declarados, 87 aplicáveis, 87 mortos, 0 sobreviventes, 0 não aplicados, e sai 0. As 85 agulhas antigas continuam com uma ocorrência e continuam mortas.

## Não-regressão

- As 85 mutações já declaradas permanecem com a mesma agulha, o mesmo alvo e a mesma tupla de testes.
- `tools/cognitivo_mutation.py` permanece com 1137 mutantes, 379 por shard.
- A suíte `python -m unittest discover -p 'test_*.py'` passa no código sem mutante.
- Nenhum `test_*.py` existente é editado. O teste novo da B2b também não é editado: ele já mata M2 e, junto com o de curinga, mata M1.
- `core/cognitivo/episodic_memory.py` e `core/constitutional_brain.py` não mudam no commit que só registrar os mutantes. O mutante vive na cópia temporária do gate.

## O que invalida

- A agulha de M1 ou de M2 deixa de ter contagem 1.
- M1 fica verde em N2 ou nos três testes de curinga.
- M2 fica verde em C1, C2 ou C3.
- O gate sai com sobrevivente ou com `not_applied`.
- Registrar o mutante exige mudar a produção ou um teste existente.
- O default entregue no fonte passa a ser `True`, ou o ramo entregue passa a ser `if True`.
- O catálogo cognitivo muda de tamanho para acomodar estes dois.

## Escopo proibido

Este texto não edita `tools/targeted_mutation.py`, `tools/cognitivo_mutation.py`, produção, testes, workflows, timeouts, pisos de cobertura nem `docs/agentes/CLAIMS.yaml`. Não abre a B2c. Não faz push nem PR.
