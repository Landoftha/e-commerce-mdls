# Previsão de Vendas Semanais (MLDS) — Melhores Compras LTDA

## Objetivo
Estimar `Weekly_Sales` em horizonte semanal usando dados internos (histórico de vendas, feriados) e externos (temperatura, preço de combustível, CPI e desemprego).

## Sumário Executivo (não técnico)
- **O que fizemos**: analisamos os dados históricos, criamos indicadores de padrão temporal e memória (lags), comparamos 5 algoritmos e escolhemos o melhor por testes temporais.
- **Por que importa**: previsões mais estáveis ajudam a planejar estoque, logística e marketing com antecedência, reduzindo rupturas e custos.
- **Como medir sucesso**: usamos erros médios (MAE, RMSE, MAPE) e explicamos o quão perto as previsões ficam das vendas reais.

## Dados e Preparação
- `Date` convertido para data e ordenado.
- Ajustes conforme dicionário: `Fuel_Price` e `Unemployment` divididos por 1000.
- Criação de atributos temporais: ano, mês, trimestre, semana do ano; codificação cíclica (seno/cosseno) para sazonalidade semanal.
- Memória de vendas: `lags` (1,2,3,4,52 semanas) e médias/desvios móveis (4 e 12 semanas) para capturar tendência e sazonalidade.
- Tratamento de ausentes com imputação (mediana/moda) e padronização de variáveis numéricas.

## Metodologia de Validação
- Usamos validação de série temporal (`TimeSeriesSplit`) com 5 dobras e janela mínima de treino (50% inicial) para evitar vazamento de informação do futuro.
- Conjunto holdout final (~15% mais recente) para simular produção.

## Modelos Avaliados
- Regressão Linear, Ridge, Lasso, Random Forest e Gradient Boosting, todos em `Pipeline` com pré-processamento.
- Critério de seleção: menor RMSE médio na validação temporal.

## Justificativa da Escolha do Modelo
- Em séries com não linearidades, sazonalidade e interações entre variáveis (e.g., feriado x temperatura), **modelos de ensemble baseados em árvores** (RF/GB) costumam capturar relações complexas sem supor linearidade.
- O uso de `lags` e `rolling stats` fornece memória explícita, e árvores exploram esses padrões de forma robusta a outliers e escalas.
- Quando o vencedor for linear (Ridge/Lasso), a preferência decorre de interpretabilidade e estabilidade quando relações são aproximadamente lineares e a regularização reduz overfitting.
- A decisão final no notebook é orientada por desempenho empírico (RMSE/MAE) sob TSCV, garantindo generalização temporal.

## Importância de Variáveis
- Tipicamente, `Weekly_Sales_lag_1..4` e estatísticas móveis figuram entre as mais relevantes (capturam inércia e tendência).
- `Holiday_Flag` tende a explicar picos/vales; `Fuel_Price`, `CPI` e `Unemployment` agregam contexto macroeconômico.

## Métricas
- **MAE**: erro médio absoluto (fácil de interpretar em unidades de vendas).
- **RMSE**: penaliza mais erros grandes (sensível a picos inesperados).
- **MAPE**: erro percentual (ignora semanas com zero no denominador).
- **R²**: proporção da variância explicada (referência agregada, não usada isoladamente).

## Resultados e Leitura para o Board
- Tabelas e gráficos no notebook mostram o erro médio e a curva “real vs predito” nas semanas mais recentes.
- Diretriz: erros dentro de faixas históricas e tendência corretamente capturada indicam prontidão para uso tático (planejamento semanal), enquanto desvios persistentes sinalizam necessidade de novas variáveis.

## Riscos e Mitigações
- Mudanças estruturais (promoções atípicas, rupturas de estoque) degradam o modelo: mitigar com variáveis de calendário de promoções e dados operacionais.
- Sazonalidade móvel de feriados: usar calendários oficiais e janelas variáveis.
- Macroeconomia: atualizar séries externas com frequência e considerar defasagens (lags) específicas.

## Próximos Passos
1. Inclusão de variáveis: promoções/campanhas, estoque/disponibilidade, calendário regional de feriados, preços e concorrência.
2. Testes com modelos adicionais: XGBoost/LightGBM/CatBoost e regressão com componentes periódicos explícitos.
3. Ajuste de hiperparâmetros com `Optuna` (TSCV) e detecção de drift em produção.
4. Deploy: API de inferência com `FastAPI`, versionamento em `MLflow`, monitoramento de métricas e re-treino periódico.

## Reprodutibilidade e Operação
- Notebook `notebooks/sales_forecast_mlds.ipynb` contém todo o fluxo (EDA→features→TSCV→comparação→holdout→persistência em `models/best_model.joblib`).
- O pipeline salva pré-processamento + modelo, permitindo uso direto para previsões futuras sem vazamento.
