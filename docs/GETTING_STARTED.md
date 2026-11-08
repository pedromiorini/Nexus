# Primeiros passos no Nexus

## O que você está executando

O Nexus é um protótipo experimental de arquitetura neuro-simbólica em Python, com componentes de roteamento, memória, governança, telemetria e auditoria. Heurísticas, fallbacks, demonstrações e testes internos verificam comportamentos delimitados; não demonstram AGI, ASI, consciência, autonomia geral nem prontidão para produção. Veja o [escopo real no README](../README.md#escopo-real) e a [Constituição do projeto](../NEXUS_CONSTITUTION.md).

## Preparar o ambiente

O CI do repositório usa Python 3.11. Os comandos abaixo pressupõem Linux ou macOS, a partir de um clone limpo e na raiz do repositório:

```bash
git clone https://github.com/pedromiorini/Nexus.git
cd Nexus
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export PYTHONPATH=.
```

`requirements-dev.txt` inclui as dependências de execução de `requirements.txt` e as ferramentas de desenvolvimento. O projeto não é distribuído como pacote Python instalável; execute os testes com a raiz no caminho de importação. Python 3.12 e 3.13 podem funcionar, mas não são atualmente verificadas pelo workflow.

## Primeiro teste

Ainda na raiz, rode o contrato focado de snapshots da fila de tarefas adiadas:

```bash
python -m unittest -v test_deferred_task_snapshot.py
```

Sucesso esperado: o processo termina com código 0 e `OK`. O teste cobre round-trip, rejeição de snapshots inválidos e ausência de mutação da fila nesses casos; não prova durabilidade em disco, execução distribuída ou entrega exactly-once. Consulte também o [contrato de snapshots no README](../README.md#snapshot-da-fila-de-tarefas-adiadas) e o [arquivo de teste](../test_deferred_task_snapshot.py).

## Suíte completa de testes `unittest`

Para descobrir e executar todos os arquivos `test_*.py`, a partir da raiz:

```bash
python -m unittest discover -v -p 'test_*.py'
```

Esse comando executa a suíte descoberta pelo `unittest`; não substitui os outros gates do CI, como suíte integrada, lint, mutation testing e auditorias.

## Referências para continuar

Leia a [arquitetura observada](../NEXUS_ARCHITECTURE.md), o [registro de decisões](../NEXUS_DECISIONS.md) e o [checkpoint de continuidade](../CONTINUATION.md). O [README](../README.md#como-executar) contém comandos e contratos adicionais; o [workflow de CI](../.github/workflows/nexus-contract-gate.yml) mostra os gates automatizados.
