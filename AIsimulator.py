import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from data_provider import get_prices

# --- CONFIGURACIÓN DE INTELIGENCIA ARTIFICIAL ---

def prepare_advanced_data(prices):
    """Convierte lista de precios en tabla con indicadores técnicos."""
    df = pd.DataFrame(prices, columns=['Close'])
    # Indicadores que ayudan a la IA a entender el mercado
    df['Momentum'] = df['Close'].diff(1)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    return df.dropna()

def predict_future_range(df, days_to_predict=15):
    """Entrena el modelo y proyecta el precio futuro con rangos de error."""
    # Definimos nuestras variables de entrada (X) y salida (y)
    X = df[['Momentum', 'Volatility', 'MA10', 'MA30']].values
    y = df['Close'].values
    
    # Modelo Random Forest: El 'Bosque' que analiza escenarios
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # Preparamos la predicción paso a paso
    last_features = X[-1].reshape(1, -1)
    predictions = []
    current_features = last_features
    
    for _ in range(days_to_predict):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        # Simulación de evolución de indicadores (simplificada)
        current_features = current_features * 1.0002 
        
    # Cálculo de bandas de confianza (Incertidumbre)
    # Usamos la desviación estándar reciente para definir el rango
    error_margin = df['Volatility'].iloc[-1] * 2 
    lower_bound = [p - error_margin for p in predictions]
    upper_bound = [p + error_margin for p in predictions]
    
    return predictions, lower_bound, upper_bound

# --- FLUJO PRINCIPAL ---

if __name__ == "__main__":
    print("\n" + "="*40)
    print("   AI ASSET INVESTMENT SIMULATOR - V1.0")
    print("="*40)
    
    ticker = input("Introduce el ticker (ej: BTC-USD, NVDA, AAPL): ").upper() or "NVDA"
    
    # 1. Obtención de datos (2 años para entrenamiento sólido)
    hoy = datetime.now()
    inicio = (hoy - timedelta(days=730)).strftime('%Y-%m-%d')
    fin = hoy.strftime('%Y-%m-%d')

    print(f"\n[1/3] Descargando datos históricos para {ticker}...")
    prices = get_prices(ticker, inicio, fin)

    if prices and len(prices) > 60:
        # 2. Procesamiento e IA
        print(f"[2/3] Entrenando Random Forest y generando proyección...")
        data_df = prepare_advanced_data(prices)
        preds, lower, upper = predict_future_range(data_df, 15)
        
        # 3. Visualización de resultados
        print(f"[3/3] Renderizando gráfica de 6 meses + 15 días...")
        
        plt.figure(figsize=(14, 7))
        
        # --- PARTE HISTÓRICA (6 meses) ---
        past_limit = 180
        history_display = prices[-past_limit:]
        x_history = np.arange(len(history_display))
        plt.plot(x_history, history_display, label="Precio Histórico (Real)", color='#2c3e50', linewidth=2)
        
        # --- PARTE PREDICCIÓN (Ajuste de dimensiones exactas) ---
        # x_future debe tener 16 puntos (el último real + 15 nuevos)
        x_future = np.arange(len(history_display) - 1, len(history_display) + 15)
        
        # Unimos el último precio real con las listas de predicción
        full_preds = [history_display[-1]] + list(preds)
        full_lower = [history_display[-1]] + list(lower)
        full_upper = [history_display[-1]] + list(upper)

        # Graficar línea de predicción
        plt.plot(x_future, full_preds, 'r--', label="Predicción IA (Próximos 15d)", linewidth=2)
        
        # Sombreado de incertidumbre
        plt.fill_between(x_future, full_lower, full_upper, color='red', alpha=0.15, label="Rango de Probabilidad (95%)")

        # Estética de la gráfica
        plt.title(f"Análisis Predictivo Quant: {ticker}", fontsize=16, fontweight='bold')
        plt.xlabel("Días (Línea de tiempo)", fontsize=12)
        plt.ylabel("Precio en USD", fontsize=12)
        plt.axvline(x=len(history_display)-1, color='gray', linestyle='-', alpha=0.3) # Línea divisoria hoy
        plt.text(len(history_display)-1, plt.ylim()[0], ' HOY', color='gray', fontweight='bold')
        
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        print("\n--- PROYECCIÓN DE PRECIOS ---")
        print(f"Precio actual: ${history_display[-1]:,.2f}")
        print(f"Predicción a 15 días: ${preds[-1]:,.2f}")
        print("-" * 30)
        
        plt.show()
    else:
        print("\nError: No hay suficientes datos para procesar este activo.")