import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------- Data loading ----------
DATA_PATH = os.path.join('Asset', 'sales.csv')
MODEL_PATH = os.path.join('models', 'best_model.joblib')


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d['Date'] = pd.to_datetime(d['Date'], format='%d-%m-%Y')
    # scale adjustments per dictionary
    if 'Fuel_Price' in d.columns:
        d['Fuel_Price'] = d['Fuel_Price'] / 1000.0
    if 'Unemployment' in d.columns:
        d['Unemployment'] = d['Unemployment'] / 1000.0
    d = d.sort_values('Date').reset_index(drop=True)
    return d


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d['year'] = d['Date'].dt.year
    d['month'] = d['Date'].dt.month
    d['weekofyear'] = d['Date'].dt.isocalendar().week.astype(int)
    d['quarter'] = d['Date'].dt.quarter
    d['sin_woy'] = np.sin(2 * np.pi * d['weekofyear'] / 52.0)
    d['cos_woy'] = np.cos(2 * np.pi * d['weekofyear'] / 52.0)
    return d


def add_lag_rolling_features(frame: pd.DataFrame, target_col: str = 'Weekly_Sales') -> pd.DataFrame:
    d = frame.copy()
    lags = [1, 2, 3, 4, 52]
    for lag in lags:
        d[f'{target_col}_lag_{lag}'] = d[target_col].shift(lag)
    d[f'{target_col}_roll_mean_4'] = d[target_col].shift(1).rolling(window=4, min_periods=2).mean()
    d[f'{target_col}_roll_std_4'] = d[target_col].shift(1).rolling(window=4, min_periods=2).std()
    d[f'{target_col}_roll_mean_12'] = d[target_col].shift(1).rolling(window=12, min_periods=4).mean()
    d[f'{target_col}_roll_std_12'] = d[target_col].shift(1).rolling(window=12, min_periods=4).std()
    return d


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = add_time_features(frame)
    d = add_lag_rolling_features(d, target_col='Weekly_Sales')
    if 'Holiday_Flag' in d.columns:
        d['Holiday_Flag'] = d['Holiday_Flag'].astype(int)
    return d


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str):
    return load(model_path)


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def generate_future_features(df_historical: pd.DataFrame, n_weeks: int, target_col: str = 'Weekly_Sales') -> pd.DataFrame:
    """
    Gera features para n_weeks futuras baseado nos dados históricos.
    Para features externas, usa as últimas médias ou valores mais recentes.
    """
    df_future = pd.DataFrame()
    last_date = df_historical['Date'].max()
    
    # Gera datas futuras (semanas)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=n_weeks, freq='W')
    df_future['Date'] = future_dates
    
    # Copia últimas features externas (usa médias recentes ou últimos valores)
    last_row = df_historical.iloc[-1]
    recent_mean = df_historical.iloc[-12:].mean()  # Últimas 12 semanas
    
    # Features externas: usa média recente ou último valor
    for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']:
        if col in df_historical.columns:
            df_future[col] = recent_mean.get(col, last_row.get(col, 0))
    
    # Aplica features temporais
    df_future = add_time_features(df_future)
    
    # Adiciona coluna target temporária (será preenchida com NaN e depois substituída)
    df_future[target_col] = np.nan
    
    # Para lags e rolling stats, precisa dos valores anteriores
    # Prepara um DataFrame temporário com histórico + futuro
    df_extended = pd.concat([df_historical[[target_col, 'Date']], df_future[[target_col, 'Date']]], ignore_index=True)
    
    # Preenche temporariamente com 0 para previsões (será substituído depois)
    df_extended.loc[df_extended[target_col].isna(), target_col] = 0
    
    # Aplica lags e rolling (usando valores históricos onde disponível)
    df_extended = add_lag_rolling_features(df_extended, target_col=target_col)
    
    # Extrai apenas as linhas futuras
    future_start_idx = len(df_historical)
    df_future_features = df_extended.iloc[future_start_idx:].copy()
    
    # Substitui NaN em lags/rolling por últimos valores disponíveis
    for col in df_future_features.columns:
        if col.startswith(target_col + '_lag_') or col.startswith(target_col + '_roll_'):
            if df_future_features[col].isna().any():
                # Usa último valor não-nulo da coluna histórica
                last_val = df_historical[col].dropna().iloc[-1] if col in df_historical.columns and not df_historical[col].isna().all() else None
                if last_val is not None:
                    df_future_features[col] = df_future_features[col].fillna(last_val)
                else:
                    # Fallback: média recente
                    df_future_features[col] = df_future_features[col].fillna(df_historical[target_col].tail(4).mean())
    
    # Restaura colunas originais
    for col in df_future.columns:
        if col not in df_future_features.columns:
            df_future_features[col] = df_future[col]
    
    # Ajusta Holiday_Flag
    if 'Holiday_Flag' in df_future_features.columns:
        df_future_features['Holiday_Flag'] = df_future_features['Holiday_Flag'].astype(int)
    
    return df_future_features


def forecast_future(df_historical: pd.DataFrame, pipe, feature_cols: list, n_weeks: int = 12) -> pd.DataFrame:
    """
    Gera previsões futuras usando o modelo treinado.
    Retorna DataFrame com datas, previsões e features.
    """
    predictions = []
    df_working = df_historical.copy()
    
    # Gera datas futuras
    last_date = df_historical['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=n_weeks, freq='W')
    
    # Prepara última linha como base
    last_row = df_historical.iloc[-1]
    recent_mean = df_historical.iloc[-12:].mean()
    
    for i in range(n_weeks):
        # Cria DataFrame temporário com histórico + previsões anteriores + nova semana
        future_row_dict = {'Date': future_dates[i]}
        
        # Features externas (usa médias recentes)
        for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']:
            if col in df_historical.columns:
                future_row_dict[col] = recent_mean.get(col, last_row.get(col, 0))
        
        # Adiciona previsão temporária (0) para calcular lags/rolling
        future_row_dict['Weekly_Sales'] = 0
        
        # Converte para DataFrame
        future_row_df = pd.DataFrame([future_row_dict])
        future_row_df = add_time_features(future_row_df)
        
        # Concatena com histórico + previsões anteriores para calcular lags/rolling
        df_temp = pd.concat([df_working, future_row_df], ignore_index=True)
        df_temp = add_lag_rolling_features(df_temp, target_col='Weekly_Sales')
        
        # Extrai apenas a última linha (semana futura) com todas as features
        future_row = df_temp.iloc[-1:].copy()
        
        # Atualiza lags com previsões anteriores
        for lag in [1, 2, 3, 4, 52]:
            lag_col = f'Weekly_Sales_lag_{lag}'
            if lag_col in future_row.columns:
                if i >= lag:
                    # Usa previsão anterior
                    future_row[lag_col] = predictions[i - lag]
                else:
                    # Usa valor histórico
                    hist_idx = len(df_historical) - lag + i
                    if hist_idx >= 0 and hist_idx < len(df_historical):
                        future_row[lag_col] = df_historical['Weekly_Sales'].iloc[hist_idx]
        
        # Prepara features para predição (garante que todas as colunas necessárias estão presentes)
        X_future = future_row[feature_cols].copy()
        
        # Previsão
        pred = pipe.predict(X_future)[0]
        predictions.append(pred)
        
        # Atualiza DataFrame de trabalho para próximos lags
        future_row['Weekly_Sales'] = pred
        df_working = pd.concat([df_working, future_row], ignore_index=True)
    
    # Cria DataFrame final com previsões
    df_forecast = pd.DataFrame({
        'Date': future_dates,
        'Weekly_Sales_Predicted': predictions
    })
    
    # Adiciona features externas para referência
    for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']:
        if col in df_historical.columns:
            df_forecast[col] = recent_mean.get(col, last_row.get(col, 0))
    
    return df_forecast


def main():
    st.set_page_config(page_title='Previsão de Vendas Semanais', layout='wide')
    st.title('Inteligência Artificial no E-commerce: Previsão de Vendas Semanais')
    st.caption('Melhores Compras LTDA — Storytelling com dados ')

    # Sidebar controls
    with st.sidebar:
        st.header('Parâmetros')
        st.write('Selecione o intervalo para análise e a semana em foco:')

    # Load data and model
    df = load_data(DATA_PATH)

    # Build features and align with model
    df_feat = build_features(df)
    min_index = df_feat.dropna().index.min()
    df_model = df_feat.loc[min_index:].reset_index(drop=True)

    if not os.path.exists(MODEL_PATH):
        st.error('Modelo não encontrado. Execute o notebook para treinar e salvar o pipeline em models/best_model.joblib.')
        st.stop()

    bundle = load_pipeline(MODEL_PATH)
    pipe = bundle['pipeline']
    feature_cols = bundle['features']
    target = bundle['target']

    # Select window
    min_date, max_date = df_model['Date'].min(), df_model['Date'].max()
    date_options = sorted(list(df_model['Date'].dt.date.unique()))
    default_focus_date = date_options[-1] if date_options else max_date.date()
    
    with st.sidebar:
        date_range = st.date_input('Janela de análise', (min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
        focus_date = st.select_slider('Semana em foco', options=date_options, value=default_focus_date)

    # Handle date_range (can be single date or tuple)
    if isinstance(date_range, tuple):
        date_start, date_end = date_range
    else:
        date_start = date_end = date_range
    
    mask = (df_model['Date'].dt.date >= date_start) & (df_model['Date'].dt.date <= date_end)
    view = df_model.loc[mask].copy()

    # Predictions on selected window
    X_view = view[feature_cols]
    y_view = view[target]
    y_pred = pipe.predict(X_view)

    # Hero cards (slide 7 style)
    section_header('Semana em Foco')
    row = view.loc[view['Date'].dt.date == focus_date].tail(1)
    if not row.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('Data', row['Date'].dt.strftime('%d/%m/%Y').values[0])
            st.caption('semana de referência')
        with c2:
            st.metric('Temperatura média (°F)', f"{row['Temperature'].values[0]:.2f}")
            st.caption('na semana')
        with c3:
            st.metric('Desemprego (%)', f"{row['Unemployment'].values[0]:.2f}")
            st.caption('média nacional')
        with c4:
            st.metric('Feriado', 'Sim' if int(row['Holiday_Flag'].values[0]) == 1 else 'Não')

    # Story 1: Contexto macro (slide 8 — CPI)
    section_header('Nível de atividade econômica (CPI)', 'Crescimento/contração da atividade ao longo do tempo')
    cpi_year = (
        view.assign(year=view['Date'].dt.year)
            .groupby('year', as_index=False)['CPI'].mean()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cpi_year, x='year', y='CPI', color='#A08CFF', ax=ax)
    ax.set_xlabel('Ano')
    ax.set_ylabel('CPI médio')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown('- **Leitura**: variações no CPI deslocam a demanda agregada; o modelo captura esse contexto via feature `CPI`.')

    # Story 2: Combustível (melhorado - linha ao invés de scatter)
    section_header('Preço do combustível (semanal)', 'Relação com custos logísticos e poder de compra')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(view['Date'], view['Fuel_Price'], color='#7C83FD', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax2.fill_between(view['Date'], view['Fuel_Price'], alpha=0.2, color='#7C83FD')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Preço (R$)')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.markdown('- **Leitura**: períodos de alta sustentada de combustível podem reduzir a demanda e elevar custos de entrega.')

    # Story 3: Série alvo — verdade vs predito
    section_header('Vendas: Verdade vs Predito', 'Precisão do modelo na janela selecionada')
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(view['Date'], y_view.values, label='Verdade', linewidth=2, color='#4C72B0')
    ax3.plot(view['Date'], y_pred, label='Predito', linewidth=2, linestyle='--', color='#55A868')
    ax3.legend()
    ax3.set_xlabel('Data')
    ax3.set_ylabel('Weekly_Sales')
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    st.markdown(f"- **RMSE (janela)**: {rmse(y_view, y_pred):,.0f}")

    # Why this algorithm — didática
    section_header('Por que este algoritmo?')
    st.markdown(
        '- **Premissa**: as vendas apresentam sazonalidade e não linearidades (efeitos de feriado, clima e macro).\n'
        '  Modelos de árvores de decisão em ensemble (como RandomForest/GradientBoosting) capturam interações e padrões\n'
        '  complexos sem supor linearidade.\n'
        '- **Memória temporal**: `lags` e estatísticas móveis inserem inércia e tendência; as árvores exploram esses sinais.\n'
        '- **Validação temporal**: usamos `TimeSeriesSplit` para evitar vazamento do futuro e selecionar o melhor modelo por RMSE.'
    )

    # Feature importance / coeficientes
    section_header('Quais variáveis mais importam?')
    top_k = 15
    try:
        model = pipe.named_steps['model']
        pre = pipe.named_steps['pre']
        
        # Obtém as colunas numéricas e categóricas do preprocessor
        num_cols = []
        cat_cols = []
        # Acessa diretamente os transformers
        for name, transformer, cols in pre.transformers_:
            if name == 'num':
                num_cols = list(cols) if isinstance(cols, (list, tuple)) else list(cols)
            elif name == 'cat':
                cat_cols = list(cols) if isinstance(cols, (list, tuple)) else list(cols)
        
        # Tenta obter nomes das features categóricas após encoding
        cat_feature_names = []
        if cat_cols:
            try:
                # Acessa o transformer categórico e depois o OneHotEncoder
                cat_transformer = pre.named_transformers_.get('cat', None)
                if cat_transformer is not None:
                    if hasattr(cat_transformer, 'named_steps'):
                        # É um Pipeline
                        ohe = cat_transformer.named_steps.get('onehot', None)
                        if ohe is not None:
                            # Verifica se está fitted (tem o atributo categories_ que só existe após fit)
                            if hasattr(ohe, 'categories_'):
                                try:
                                    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
                                except:
                                    # Se falhar, usa nomes originais
                                    cat_feature_names = cat_cols
                            else:
                                # Encoder não foi fitted ainda, usa nomes originais
                                cat_feature_names = cat_cols
                        else:
                            cat_feature_names = cat_cols
                    else:
                        # Se não for um Pipeline, pode ser o OneHotEncoder diretamente
                        if hasattr(cat_transformer, 'categories_'):
                            try:
                                cat_feature_names = list(cat_transformer.get_feature_names_out(cat_cols))
                            except:
                                cat_feature_names = cat_cols
                        else:
                            # Transformer não foi fitted ainda, usa nomes originais
                            cat_feature_names = cat_cols
                else:
                    cat_feature_names = cat_cols
            except Exception as e:
                # Fallback: usa nomes originais
                cat_feature_names = cat_cols
        else:
            cat_feature_names = []
        
        feature_names = list(num_cols) + cat_feature_names

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Garante que o número de features corresponde
            if len(feature_names) != len(importances):
                # Tenta obter nomes diretamente do transform
                X_sample = X_view.iloc[:1]
                X_transformed = pre.transform(X_sample)
                n_features = X_transformed.shape[1]
                if n_features == len(importances):
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                else:
                    st.warning(f'Dimensão incompatível: {len(feature_names)} nomes vs {len(importances)} importâncias')
                    feature_names = feature_names[:len(importances)] if len(feature_names) > len(importances) else feature_names
            
            imp_df = (
                pd.DataFrame({'feature': feature_names[:len(importances)], 'importance': importances})
                  .sort_values('importance', ascending=False)
                  .head(top_k)
            )
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            sns.barplot(data=imp_df, x='importance', y='feature', color='#4C72B0', ax=ax_imp)
            ax_imp.set_title('Importância de atributos (top 15)', fontsize=12)
            ax_imp.set_xlabel('Importância', fontsize=10)
            ax_imp.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig_imp, use_container_width=True)
            
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            
            # Garante que o número de features corresponde
            if len(feature_names) != len(coefs):
                X_sample = X_view.iloc[:1]
                X_transformed = pre.transform(X_sample)
                n_features = X_transformed.shape[1]
                if n_features == len(coefs):
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                else:
                    feature_names = feature_names[:len(coefs)] if len(feature_names) > len(coefs) else feature_names
            
            coef_df = pd.DataFrame({
                'feature': feature_names[:len(coefs)], 
                'coef': coefs
            }).sort_values('coef', key=abs, ascending=False).head(top_k)
            
            fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
            sns.barplot(data=coef_df, x='coef', y='feature', color='#55A868', ax=ax_coef)
            ax_coef.set_title('Coeficientes (escala padronizada) - Top 15', fontsize=12)
            ax_coef.set_xlabel('Coeficiente', fontsize=10)
            ax_coef.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig_coef, use_container_width=True)
        else:
            st.info('O modelo não expõe importâncias diretamente; podemos usar Permutation Importance em uma evolução.')
    except Exception as e:
        import traceback
        st.warning(f'Não foi possível calcular importâncias: {e}')
        with st.expander('Detalhes do erro'):
            st.code(traceback.format_exc())

    # Previsão Futura
    section_header('Previsão Futura', 'Projeção de vendas para as próximas semanas')
    
    with st.sidebar:
        n_weeks_forecast = st.slider('Número de semanas para prever', min_value=4, max_value=52, value=12, step=4)
    
    try:
        df_forecast = forecast_future(df_model, pipe, feature_cols, n_weeks=n_weeks_forecast)
        
        # Gráfico: histórico + previsão
        fig_forecast, ax_forecast = plt.subplots(figsize=(8, 4))
        
        # Histórico (últimas 26 semanas)
        hist_window = min(26, len(view))
        hist_dates = view['Date'].iloc[-hist_window:]
        hist_sales = view[target].iloc[-hist_window:]
        
        ax_forecast.plot(hist_dates, hist_sales.values, label='Histórico (últimas 26 semanas)', 
                        color='#4C72B0', linewidth=2, alpha=0.7)
        ax_forecast.plot(df_forecast['Date'], df_forecast['Weekly_Sales_Predicted'], 
                        label=f'Previsão ({n_weeks_forecast} semanas)', 
                        color='#55A868', linewidth=2, linestyle='--', marker='o', markersize=3)
        
        # Conecta histórico com previsão
        if len(hist_dates) > 0:
            last_hist_date = hist_dates.iloc[-1]
            last_hist_value = hist_sales.iloc[-1]
            first_forecast_date = df_forecast['Date'].iloc[0]
            first_forecast_value = df_forecast['Weekly_Sales_Predicted'].iloc[0]
            
            ax_forecast.plot([last_hist_date, first_forecast_date], 
                           [last_hist_value, first_forecast_value], 
                           color='#55A868', linestyle='--', alpha=0.5)
        
        ax_forecast.axvline(x=hist_dates.iloc[-1] if len(hist_dates) > 0 else df_forecast['Date'].iloc[0], 
                           color='red', linestyle=':', alpha=0.5, label='Hoje')
        ax_forecast.set_xlabel('Data', fontsize=10)
        ax_forecast.set_ylabel('Vendas Semanais', fontsize=10)
        ax_forecast.set_title(f'Previsão de Vendas - Próximas {n_weeks_forecast} semanas', fontsize=12)
        ax_forecast.legend(fontsize=9)
        ax_forecast.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_forecast, use_container_width=True)
        
        # Tabela com previsões
        st.markdown('**Tabela de previsões:**')
        forecast_display = df_forecast[['Date', 'Weekly_Sales_Predicted']].copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d/%m/%Y')
        forecast_display['Weekly_Sales_Predicted'] = forecast_display['Weekly_Sales_Predicted'].apply(lambda x: f'R$ {x:,.0f}')
        forecast_display.columns = ['Data', 'Previsão de Vendas']
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)
        
        # Métricas agregadas
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_forecast = df_forecast['Weekly_Sales_Predicted'].mean()
            st.metric('Média Prevista', f'R$ {avg_forecast:,.0f}')
        with c2:
            max_forecast = df_forecast['Weekly_Sales_Predicted'].max()
            st.metric('Pico Previsto', f'R$ {max_forecast:,.0f}')
        with c3:
            min_forecast = df_forecast['Weekly_Sales_Predicted'].min()
            st.metric('Mínimo Previsto', f'R$ {min_forecast:,.0f}')
        
        st.markdown(
            '**Nota**: As previsões futuras usam médias recentes para variáveis externas (temperatura, combustível, CPI, desemprego). '
            'Para cenários mais precisos, considere usar projeções macroeconômicas reais ou permitir ajuste manual dessas variáveis.'
        )
        
    except Exception as e:
        st.error(f'Erro ao gerar previsões futuras: {e}')
        import traceback
        with st.expander('Detalhes do erro'):
            st.code(traceback.format_exc())

    # Conclusão
    section_header('Conclusão')
    st.markdown(
        '- **História**: combinando memória de vendas (lags), sazonalidade (semana/ano) e contexto (feriado, CPI, combustível, desemprego),\n'
        '  o modelo reproduz bem a tendência recente e minimiza erros fora da amostra.\n'
        '- **Previsão Futura**: o modelo pode projetar vendas futuras, ajudando no planejamento estratégico e operacional.\n'
        '- **Próximos passos**: incluir dados de promoções, estoque e calendário regional; avaliar XGBoost/LightGBM e otimização por `Optuna`.'
    )


if __name__ == '__main__':
    main()


