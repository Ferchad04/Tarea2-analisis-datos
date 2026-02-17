import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import folium as flm

def limpieza_bicis(bicis):
    partes = bicis.split('-')
    if len(partes) == 2:
        return int(partes[0])
    else:
        return int(bicis)
    return 0


# csv
df = pd.read_csv('actividad-1\cicloestaciones_ecobici.csv')
df = df.dropna()

"""
sns.boxplot(data=df['latitud'])
plt.title('Boxplot de Latitud')
plt.show()

sns.boxplot(data=df['longitud'])
plt.title('Boxplot de Longitud')   
plt.show()
"""

# Estraccion de cordenadas como una matriz de puntos
coor = df[['latitud', 'longitud']].values

# Crear mapa interativo con la matriz de coordenadas
mapa = flm.Map(location = coor[0], zoom_start=12)
for i, j in coor:
    flm.CircleMarker([i, j], radius=3).add_to(mapa)
mapa.save('actividad-1/mapa.html')
# Marcar cada estacion con un circulo proporcional
# al numero de bicicletas disponibles

df['num_cicloe'] = df['num_cicloe'].apply(limpieza_bicis)

coor_bicis = df[['latitud', 'longitud', 'num_cicloe']].values
mapa = flm.Map(location = coor[0], zoom_start=12)
for i, j, k in coor_bicis:
    flm.CircleMarker([i, j], radius=(k**0.5)*2+2, popup=f"Bicicletas: {int(k)}").add_to(mapa)
mapa.save('actividad-1/mapa_bicis.html')

# Clustering de las estaciones con KMeans

kmns = KMeans(n_clusters=5, random_state=0)
clustering = kmns.fit(coor_bicis)
# Agregar etiquetas de cluster al DataFrame
df['cluster'] = clustering.labels_
# Visualización de los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitud', y='latitud', hue='cluster', data=df, palette='Set1')
plt.title('Clustering de Cicloestaciones')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(title='Cluster')
plt.show()

# identificar las zonas con alta densidad de estaciones
centroides = kmns.cluster_centers_
mapa = flm.Map(location = coor[0], zoom_start=12)
for i in centroides:
    flm.Marker([i[0], i[1]], icon=flm.Icon(color='red')).add_to(mapa)
mapa.save('actividad-1/mapa_centroides.html')

# Superposicion de clusters con colores distintos en el mapa
mapa = flm.Map(location = coor[0], zoom_start=12)
colores = ['blue', 'green', 'orange', 'purple', 'cyan']
for i, j, k in coor_bicis:
    cluster_id = df[(df['latitud'] == i) & (df['longitud'] == j)]['cluster'].values[0]
    flm.CircleMarker([i, j], radius=(k**0.5)*2+2, popup=f"Bicicletas: {int(k)}", color=colores[cluster_id]).add_to(mapa)
mapa.save('actividad-1/mapa_clusters.html')
