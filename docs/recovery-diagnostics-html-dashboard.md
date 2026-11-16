# Painel HTML do snapshot de recuperação

Gere um arquivo HTML estático a partir de um snapshot JSON válido:

```sh
python -m tools.diagnostics_dashboard snapshot.json --format html > recovery-dashboard.html
```

Também é possível ler o snapshot pela entrada padrão:

```sh
cat snapshot.json | python -m tools.diagnostics_dashboard - --format html > recovery-dashboard.html
```

Sem `--format html`, o comando continua emitindo o resumo textual existente. O HTML inclui CSS embutido, gráficos em tabelas acessíveis e metadados do snapshot; não usa JavaScript, dependências adicionais, servidor ou rede. Mostra contagens e não inclui os payloads dos eventos. Se a validação falhar, o processo retorna código 2, informa o erro em `stderr` e não emite HTML em `stdout`.

Este MVP visualiza somente o snapshot de recuperação `nexus.recovery.diagnostics.v1` disponível no arquivo ou na entrada padrão. Não monitora nem representa um fluxo ao vivo de todo o Nexus.

## Rastro de eventos observáveis

O HTML inclui uma lista estática dos eventos permitidos do recorte, em ordem de registro: `policy_transition` (“Transição de política”) e `telemetry` (“Telemetria de reprocessamento”). Cada linha mostra somente o timestamp da ocorrência em Unix (segundos) e o rótulo traduzido; payloads, campos arbitrários e nomes de tipos não permitidos são omitidos. O rastro não é um feed ao vivo.

São exibidos no máximo 20 eventos permitidos mais recentes do array, preservando a ordem de registro. A página informa a quantidade omitida pelo limite e a quantidade de registros com tipo não permitido. Um array vazio e um array sem eventos permitidos apresentam estados textuais distintos. Isso descreve apenas os dados incluídos no snapshot e não indica o estado atual do sistema.
