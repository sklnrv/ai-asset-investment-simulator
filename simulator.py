import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from data_provider import get_prices

def predict_prices(prices, days_to_predict=7):
    """Usa Machine Learning para predecir los precios de la próxima semana."""
    # Preparamos los datos (usamos los últimos 30 días para entrenar)
    y = np.array(prices[-30:]).reshape(-1, 1)
    X = np.array(range(len(y))).reshape(-1, 1)
    
    # Entrenamos el modelo de Regresión Lineal
    model = LinearRegression()
    model.fit(X, y)
    
    # Predecimos el futuro
    future_X = np.array(range(len(y), len(y) + days_to_predict)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    return predictions.flatten()

def plot_with_prediction(ticker, prices, predictions):
    """Grafica el pasado y la predicción futura."""
    plt.figure(figsize=(12, 6))
    
    # Precios pasados (últimos 30 días para que se vea claro)
    past_days = list(range(len(prices[-30:])))
    plt.plot(past_days, prices[-30:], label="Precio Real (Últimos 30d)", color="blue", linewidth=2)
    
    # Predicción futura
    future_days = list(range(len(past_days), len(past_days) + len(predictions)))
    # Unimos el último punto real con el primero de la predicción para que no haya hueco
    plt.plot([past_days[-1]] + future_days, [prices[-1]] + list(predictions), 
             label="Predicción IA (7 días)", color="red", linestyle="--", marker='o')
    
    plt.title(f"Predicción de Tendencia con Machine Learning: {ticker}")
    plt.xlabel("Días (Tiempo)")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    print("\n=== AI QUANT FORECASTER ===")
    ticker = input("Ticker para predecir (ej: BTC-USD, NVDA): ").upper() or "NVDA"
    
    # Obtenemos datos de los últimos meses
    hoy = datetime.now()
    inicio = (hoy - timedelta(days=60)).strftime('%Y-%m-%d')
    fin = hoy.strftime('%Y-%m-%d')
    
    prices = get_prices(ticker, inicio, fin)
    
    if prices:
        print(f"Entrenando modelo para {ticker}...")
        preds = predict_prices(prices)
        
        print("\n--- PREDICCIÓN PARA LA PRÓXIMA SEMANA ---")
        for i, p in enumerate(preds, 1):
            print(f"Día {i}: ${p:,.2f}")
            
        plot_with_prediction(ticker, prices, preds)
    else:
        print("No se pudieron obtener datos.")