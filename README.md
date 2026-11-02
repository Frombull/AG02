# AG02 - Marco Di Toro, 150, GES

### Vídeo da apresentação:

- **[https://youtu.be/wT5FEFiiKCI](https://youtu.be/wT5FEFiiKCI)**

## Como rodar

```bash
pip install -r requirements.txt
python main.py
```

O programa treina o modelo, mostra o `classification_report` e depois pergunta se você quer classificar uma amostra. A amostra vai em uma linha, separada por vírgula, na ordem `Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen`:

---

- **Dados:** 440 clientes (298 HoReCa e 142 Retail). `Channel` e `Region` vêm como texto e são convertidos com `replace` (HoReCa 0, Retail 1; Lisbon 0, Oporto 1, Other 2).
- **Colunas:** reordenadas com `reindex`, com `Channel` (o alvo) por último.
- **Divisão:** 80% treino e 20% teste (352 e 88 amostras), com `random_state=42` e `stratify`, já que as classes são desbalanceadas.
- **Modelo:** `DecisionTreeClassifier(max_depth=5)`. Sem limite de profundidade a árvore acerta 100% no treino e cai para cerca de 90% no teste, então a profundidade foi limitada. Não precisa normalizar os dados.
- **Resultado no teste:** acurácia de 0,93, com f1 de 0,95 para HoReCa e 0,89 para Retail.
- **Detalhe:** o `pd.set_option("future.no_silent_downcasting", True)` só evita um aviso do pandas no `replace`.

## Arquivos

- `main.py`: carga dos dados, treino, avaliação e classificação de amostras
- `wholesale.csv`: dataset
- `requirements.txt`: pandas e scikit-learn

:)
