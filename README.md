# Classificação de mortalidade com rede neural MLP

Projeto de aprendizado supervisionado que usa uma rede **MLP (Multi-Layer Perceptron)** para classificar o desfecho `DEATH_EVENT` no `Heart Failure Clinical Records Dataset`. O foco é comparar um modelo base com uma versão ajustada por busca de hiperparâmetros e entender como o pré-processamento influencia uma rede neural em um conjunto clínico pequeno.

## Pergunta do projeto

É possível melhorar a classificação do desfecho de mortalidade ajustando arquitetura, regularização e taxa de aprendizado, sem alterar as variáveis clínicas disponíveis?

O dataset contém **299 pacientes** e 12 variáveis de entrada. A coluna `DEATH_EVENT` é o alvo binário: `0` indica que o paciente sobreviveu ao período observado e `1` indica ocorrência do evento.

## Variáveis e preparação

O notebook começa com análise exploratória, estatísticas descritivas, verificação de valores ausentes e matriz de correlação. Em seguida:

1. `X` recebe todas as colunas, exceto `DEATH_EVENT`.
2. `y` recebe `DEATH_EVENT`.
3. Os dados são divididos em 80% para treino e 20% para teste, com estratificação e `random_state=42`.
4. `StandardScaler` é ajustado somente no treino e aplicado ao teste.
5. A rede é treinada com as variáveis já padronizadas.

O uso do scaler apenas no treino evita que estatísticas do conjunto de teste vazem para o treinamento.

## Experimentos

### Modelo base

O baseline usa:

```text
MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)
```

Ele fornece uma referência simples para medir o ganho da otimização.

### Busca de hiperparâmetros

O `GridSearchCV` testa combinações de:

| Hiperparâmetro | Valores testados |
|---|---|
| `hidden_layer_sizes` | 16, 32, 64 neurônios |
| `activation` | `relu`, `tanh`, `logistic` |
| `alpha` | 0.0001, 0.001, 0.01 |
| `learning_rate_init` | 0.0005, 0.001, 0.01 |
| `batch_size` | 32, 64, 128 |

São 243 combinações avaliadas com `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`. O modelo usado na busca possui `early_stopping=True` e a métrica de seleção é acurácia.

## Resultados registrados

No notebook principal, usando o mesmo conjunto de teste de 20%:

| Modelo | Acurácia |
|---|---:|
| MLP base | **73,33%** |
| MLP otimizada | **78,33%** |

A melhor configuração registrada foi:

```text
activation=relu
alpha=0.01
batch_size=64
hidden_layer_sizes=32
learning_rate_init=0.01
```

Além da acurácia, o notebook gera matriz de confusão e relatório de classificação para mostrar como os erros se distribuem entre as duas classes.

## Como executar

O CSV já está incluído na raiz do projeto. Na pasta do repositório:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook treinamento.ipynb
```

Os notebooks `normal+indiv Felipe.ipynb` e `normal+indiv Lucas.ipynb` contêm análises complementares, incluindo experimentos com subconjuntos de variáveis mais correlacionadas ao evento.

## Estrutura

```text
Trabalho de IA 2/
├── treinamento.ipynb
├── normal+indiv Felipe.ipynb
├── normal+indiv Lucas.ipynb
├── heart_failure_clinical_records_dataset.csv
└── README.md
```

## Limitações e interpretação

- O conjunto tem apenas 299 registros; uma diferença de poucos casos altera bastante a acurácia.
- O resultado usa uma única divisão treino/teste, então não é uma estimativa definitiva de generalização.
- Acurácia, sozinha, não substitui sensibilidade, especificidade, F1 e análise do custo de cada erro.
- Correlação não implica causalidade e não significa que uma variável seja um biomarcador validado.
- O projeto é educacional: não fornece diagnóstico, prognóstico clínico nem recomendação médica.
