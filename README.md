# Projeto Inteligência Artificial - Tema 7
## Comparação entre Perceptron e Adaline

Este projeto implementa e compara dois algoritmos fundamentais de aprendizado de máquina:
- **Perceptron**: Algoritmo de classificação linear com função de ativação discreta
- **Adaline**: Algoritmo de classificação linear com função de ativação identidade

### 📊 Características do Projeto

- **Dataset**: 1000 pontos (500 por classe) linearmente separáveis
- **Divisão**: 70% treino / 30% teste
- **Análise**: Convergência, acurácia e sensibilidade à taxa de aprendizado

### 🔧 Algoritmos Implementados

#### Perceptron
- Função de ativação: Discreta (step function)
- Taxas de aprendizado testadas: 0.1 e 0.5
- Critério de convergência: Erro de classificação = 0

#### Adaline
- Função de ativação: Identidade (linear)
- Taxas de aprendizado testadas: 0.0001 e 0.0005
- Critério de convergência: Erro médio quadrático estável

### 📈 Resultados

O projeto gera visualizações mostrando:
- Evolução da reta de decisão ao longo das épocas
- Gráficos de erro por época
- Comparação final de acurácias
- Análise de convergência

### 🛠️ Dependências

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

### 🚀 Como Executar

1. Certifique-se de ter as dependências instaladas:
```bash
pip install numpy matplotlib scikit-learn
```

2. Execute o arquivo principal:
```bash
python projeto_inteligência_artificial_tema_7.py
```

### 📁 Estrutura do Projeto

```
.
├── projeto_inteligência_artificial_tema_7.py  # Código principal
├── README.md                                  # Este arquivo
└── .gitignore                                # Arquivos ignorados pelo Git
```

### 👨‍💻 Autor

**Caio Nogueira**
- Email: caionogueirah@hotmail.com

### 📝 Observações

Este projeto foi desenvolvido como parte do curso de Inteligência Artificial, demonstrando a implementação prática de algoritmos fundamentais de machine learning e a importância da análise comparativa entre diferentes abordagens de aprendizado.
