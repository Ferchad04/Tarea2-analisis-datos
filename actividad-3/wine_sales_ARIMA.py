import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Cargar la serie con pandas, asegurando que el
# indice sea datetime
df = pd.read_excel('wine_sales_data_20182019.xlsx')
df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m')
df.set_index('日期', inplace=True)
# print(df.head())

# Agrupar ventas por fecha

sales_fecha = df.groupby('日期')['近30天销量（瓶）'].sum().sort_index()


meses = pd.date_range(start=sales_fecha.index.min(), end=sales_fecha.index.max(), freq='MS')
sf = sales_fecha.reindex(meses)
sf = sf.interpolate(method='linear')

# Descomposición clasica (tendencia, estacionalidad y residuo)
descomposicion = seasonal_decompose(sf, model='additive', period=2)
descomposicion.plot()
plt.tight_layout()
plt.savefig('decomposition.png')

# Modelo ARIMA
modelo_arima = ARIMA(sf, order=(1, 1, 1))
resultado_arima = modelo_arima.fit()
print(resultado_arima.summary())
# Predicción
prediccion = resultado_arima.get_forecast(steps=12)
prediccion_ci = prediccion.conf_int()
# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(sf, label='Ventas Reales')
plt.plot(prediccion.predicted_mean, label='Predicción ARIMA', color='red')
plt.fill_between(prediccion_ci.index, prediccion_ci.iloc[:, 0], prediccion_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('Predicción de Ventas de Vino con ARIMA')
plt.xlabel('Fecha')
plt.ylabel('Ventas (botellas)')
plt.savefig('arima_prediction.png')

# Pronostico para los próximos 12 meses 
pronostico_futuro = resultado_arima.get_forecast(steps=12)
pronostico_futuro_ci = pronostico_futuro.conf_int()
print("Pronóstico para los próximos 12 meses:")
print(pronostico_futuro.predicted_mean)

# Evaluar el modelo RMSE en un conjuntpo de prueba 12 meses
# Para esto separamos los datos en entrenamiento y prueba
train = sf[:-12]
test = sf[-12:]
modelo_arima_train = ARIMA(train, order=(1, 1, 1))
resultado_arima_train = modelo_arima_train.fit()
prediccion_test = resultado_arima_train.get_forecast(steps=12)
rmse = np.sqrt(mean_squared_error(test, prediccion_test.predicted_mean))
print(f'RMSE del modelo ARIMA: {rmse:.2f}')

