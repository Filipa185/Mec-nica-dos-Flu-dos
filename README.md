# Mec-nica-dos-Flu-dos

# Operação Otimizada de Bombeamento em Sistemas de Distribuição de Água

## Visão Geral do Projeto
Este projeto Python implementa uma solução de otimização para operação de bombas em sistemas de distribuição de água, com o objetivo de minimizar os custos de operação enquanto respeita restrições hidráulicas e padrões de consumo variáveis. O sistema utiliza uma técnica de otimização para determinar a melhor programação de operação da bomba ao longo de um período de 24 horas.

## Objetivos
- Minimizar o custo total de operação da bomba ao longo de 24 horas.
- Assegurar níveis seguros no depósito (entre 0 m e 9 m), com nível inicial e final de 4 m.
- Penalizar operações que excedam os limites de segurança (abaixo de 2 m ou acima de 7 m), com um custo de 5 €/hora por hora violada.
  
## Descrição Técnica
A simulação considera diferentes curvas de consumo (mínimo/máximo e residencial). O funcionamento da bomba é otimizado em ciclos definidos pelo utilizador, e a vazão é ajustada dinamicamente com base no nível do depósito. O código leva em consideração tarifas de energia variáveis ao longo do dia, que são definidas em intervalos de 2h.

A otimização é realizada utilizando o algoritmo `differential_evolution` da biblioteca `scipy.optimize`, que busca a melhor programação de operação da bomba para minimizar os custos, ao mesmo tempo em que respeita as restrições hidráulicas.

## Estrutura do Projeto
- **OtimizadorBomba**: Classe principal que realiza a simulação e otimização da operação da bomba.
- **simular**: Função que simula o comportamento da bomba com base nos parâmetros dados.
- **func_objetivo**: Função de custo a ser minimizada, que inclui as penalizações por violações dos limites de nível do depósito.
- **main**: Executa a otimização para diferentes cenários de consumo (máximo, mínimo e residencial).
  
## Requisitos
### Python
- **Python 3.6 ou superior** (recomendado Python 3.11 ou superior).

### Bibliotecas Necessárias
Instale as seguintes bibliotecas Python usando o comando:

```bash
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.4.0


