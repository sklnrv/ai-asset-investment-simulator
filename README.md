# 📈 QuantVision IA (V2.1 - Interactive)

Un Analizador de inversiones avanzado que combina el poder de **Random Forest** con visualizaciones interactivas de grado profesional para proyectar tendencias de activos financieros a 15 días.

## 🧠 Documentación del Proyecto (Notion)
Puedes ver el desglose detallado de la lógica, los objetivos de negocio y el proceso de desarrollo en mi documentación técnica:
👉 **[Ver Documentación en Notion](https://www.notion.so/AI-Asset-Investment-Simulator-2f7227a9f96d80148631fcd65e328b6d)**

## ✨ Características Principales

* **Motor de IA Potenciado:** Entrenamiento automático con un modelo de **500 árboles de decisión** (Random Forest) para mayor estabilidad en la predicción.
* **Deep History:** El modelo se entrena internamente con un histórico de **48 meses** para capturar ciclos de mercado completos.
* **Gráfica Interactiva (Plotly):** Visualización dinámica que permite hacer zoom, paneo y consultar precios exactos día por día.
* **Filtro de Incertidumbre Realista:** Implementa un cono de probabilidad basado en la volatilidad real del activo ($\sigma \times \sqrt{t}$), ofreciendo un margen de error ajustado y profesional.
* **Dashboards Modernos:** Interfaz limpia creada con Streamlit, enfocada en métricas clave: Precio Actual, Objetivo a 15 días y Tendencia.

## 🛠️ Stack Tecnológico
* **Engine:** Python 3.x
* **Machine Learning:** `scikit-learn` (Random Forest Regressor)
* **Gráficos:** `Plotly` (Interactivo)
* **Web Framework:** `Streamlit`
* **Data API:** `yfinance` (Yahoo Finance)

## Resultados del proyecto

Aquí algunos resultados del análisis:

![Gráfico de ejemplo](screenshoot.png)

## 🚀 Cómo usar QuantVision IA

### Opción A: Acceso Web (Recomendado)
Accede instantáneamente sin instalar nada a través de nuestra plataforma en la nube:
👉 **[Ver App en Vivo (Streamlit Cloud)](https://quantvisionai.streamlit.app)**

---

### Opción B: Ejecución Local (Windows)
1. Descarga el proyecto desde **[Releases](https://github.com/sklnrv/QuantVisionAI/archive/refs/tags/1.0.0.zip)**.
2. Descomprime y ejecuta `INICIAR_ANALIZADOR.vbs`.
   * *Se requiere tener instalado Python 3.9 o posterior*

*Para usuarios de Mac/Linux:*
```bash
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
