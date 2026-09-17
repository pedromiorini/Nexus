# Evidências de pesquisa técnica

## Coverage.py

Fonte: https://coverage.readthedocs.io/en/latest/branch.html

A documentação oficial descreve cobertura de branches como pares de linhas de origem e destino observados durante a execução, comparados com as transições possíveis obtidas por análise estática. Isso confirma que cobertura de linhas ou branches mede caminhos exercitados, mas não prova por si só correção semântica, qualidade dos oráculos ou validade das capacidades declaradas.

## Hypothesis

Fonte: https://hypothesis.readthedocs.io/

A documentação oficial define Hypothesis como biblioteca de testes baseados em propriedades para Python. O desenvolvedor descreve propriedades que devem valer para uma faixa de entradas e a ferramenta gera casos, incluindo bordas não antecipadas, além de reduzir exemplos de falha. Isso é aplicável à fila, invariantes de prioridade, limites de retry e validação do schema.

## Direção preliminar

A auditoria deve combinar testes de contrato, propriedades, cobertura de branches, análise estática e testes negativos. A aprovação dos 29 testes embutidos não deve ser usada como prova de AGI, ASI, consciência, autonomia geral ou completude de produção; deve ser reportada como evidência de que um conjunto específico de demos e asserts executou sem falhar.

## Bandit

Fonte: https://bandit.readthedocs.io/en/latest/

A documentação oficial descreve Bandit como analisador que percorre arquivos Python, constrói uma árvore sintática e executa plugins para encontrar problemas comuns de segurança. Isso é adequado como uma camada independente de verificação no CI, sem confundir o resultado com prova de ausência de vulnerabilidades.

## Mutation testing

Fontes: https://github.com/boxed/mutmut e https://dl.acm.org/doi/10.1145/3701625.3701659

Mutmut é uma ferramenta de mutation testing para Python. A literatura recente trata mutation testing como avaliação da efetividade dos testes por meio da injeção de falhas controladas. A técnica é mais forte que cobertura isolada para avaliar se os asserts realmente detectam alterações comportamentais, mas deve ser aplicada primeiro aos módulos críticos e pequenos, não ao núcleo monolítico inteiro.

## Consequência para o Nexus

A auditoria deve adicionar: (a) uma matriz explícita de evidências e limitações no README; (b) análise estática e de segurança no CI; (c) propriedades para fila e contratos; e (d) mutation testing incremental nos componentes críticos. Nenhuma dessas ferramentas autoriza afirmar AGI, ASI, consciência ou autonomia geral; elas medem qualidade e verificabilidade do software.
