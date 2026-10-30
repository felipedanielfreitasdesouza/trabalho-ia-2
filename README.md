# Classificação de mortalidade com rede neural MLP

Projeto de aprendizado supervisionado que utiliza uma rede neural artificial do tipo MLP para estimar a ocorrência de morte em pacientes com insuficiência cardíaca.

## Objetivo

Treinar um classificador com o `Heart Failure Clinical Records Dataset`, comparar um modelo base com uma versão otimizada e investigar o impacto de hiperparâmetros como número de neurônios, tamanho do lote e taxa de aprendizado.

## Pipeline

1. Análise exploratória e verificação de valores ausentes.
2. Separação das variáveis de entrada e do alvo `DEATH_EVENT`.
3. Divisão estratificada entre treino e teste.
4. Normalização com `StandardScaler`.
5. Treinamento de um `MLPClassifier` base.
6. Otimização com `GridSearchCV` e `StratifiedKFold`.
7. Avaliação por acurácia, relatório de classificação e matriz de confusão.

No notebook principal, o modelo base alcança 73,33% de acurácia e o modelo otimizado alcança 78,33% no conjunto de teste. Os demais notebooks exploram análises individuais de hiperparâmetros.

## Tecnologias

- Python
- Jupyter Notebook
- Pandas e NumPy
- scikit-learn
- Matplotlib e Seaborn

## Como executar

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook treinamento.ipynb
```

Os notebooks `normal+indiv Felipe.ipynb` e `normal+indiv Lucas.ipynb` apresentam análises complementares.

## Dados

O dataset contém registros clínicos anonimizados e é usado exclusivamente para fins educacionais. O projeto não substitui avaliação médica nem fornece diagnóstico.
