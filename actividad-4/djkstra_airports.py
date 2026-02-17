import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

# ==========================================
# 1. FUNCIÓN AUXILIAR: FÓRMULA DE HAVERSINE
# ==========================================
def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula la distancia en kilómetros entre dos puntos geográficos 
    (latitud/longitud) para estimar la 'duración' o peso del vuelo.
    """
    # Convertir grados decimales a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Fórmula de haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radio de la Tierra en km
    return c * r

# ==========================================
# 2. CARGA Y LIMPIEZA DE DATOS
# ==========================================
print("Cargando datos...")

# Cargar Aeropuertos (Sin encabezados)
# Cols: 0=ID, 1=Name, 2=City, 3=Country, 4=IATA, 6=Lat, 7=Lon
df_airports = pd.read_csv('airports.csv', header=None, encoding='utf-8')
df_airports = df_airports[[0, 1, 2, 3, 4, 6, 7]].copy()
df_airports.columns = ['ID', 'Name', 'City', 'Country', 'IATA', 'Lat', 'Lon']

# Cargar Rutas (Sin encabezados, manejar \N como nulo)
# Cols: 3=SourceID, 5=DestID
df_routes = pd.read_csv('routes.csv', header=None, na_values='\\N')
df_routes = df_routes[[3, 5]].dropna().astype(int)
df_routes.columns = ['SourceID', 'DestID']

# ==========================================
# 3. CONSTRUCCIÓN DEL GRAFO (REQ 1 y 3)
# ==========================================
print("Construyendo el grafo...")
G = nx.DiGraph() # Grafo Dirigido

# Diccionario rápido para buscar coordenadas por ID
coords = df_airports.set_index('ID')[['Lat', 'Lon', 'IATA', 'City']].to_dict('index')

# Agregar nodos (aeropuertos) con sus atributos
for id_aeropuerto, datos in coords.items():
    G.add_node(id_aeropuerto, 
               pos=(datos['Lon'], datos['Lat']), # (x, y) para matplotlib
               label=datos['IATA'],
               city=datos['City'])

# Agregar aristas (vuelos) calculando el peso (distancia)
# Iteramos sobre las rutas válidas
for row in df_routes.itertuples(index=False):
    src, dst = row.SourceID, row.DestID
    
    # Solo agregamos la ruta si ambos aeropuertos existen en el archivo airports.dat
    if src in coords and dst in coords:
        # Calcular distancia real
        dist = haversine(coords[src]['Lon'], coords[src]['Lat'], 
                         coords[dst]['Lon'], coords[dst]['Lat'])
        
        # Agregamos la arista con peso 'weight' (distancia)
        G.add_edge(src, dst, weight=dist)

print(f"Grafo construido: {G.number_of_nodes()} aeropuertos, {G.number_of_edges()} rutas.")

# ==========================================
# 4. IMPLEMENTACIÓN ALGORITMOS (REQ 2 y 3)
# ==========================================

def encontrar_ruta(origen_id, destino_id):
    try:
        # REQ 2: Ruta más corta en ESCALAS (BFS / Dijkstra sin pesos)
        ruta_escalas = nx.shortest_path(G, source=origen_id, target=destino_id)
        print(f"\n--- Ruta con menos escalas ({len(ruta_escalas)-1} vuelos) ---")
        print(" -> ".join([G.nodes[n]['label'] for n in ruta_escalas]))

        # REQ 3: Ruta más rápida/corta en DISTANCIA (Dijkstra con pesos)
        ruta_distancia = nx.dijkstra_path(G, source=origen_id, target=destino_id, weight='weight')
        distancia_total = nx.dijkstra_path_length(G, source=origen_id, target=destino_id, weight='weight')
        print(f"\n--- Ruta más corta por distancia ({int(distancia_total)} km) ---")
        print(" -> ".join([G.nodes[n]['label'] for n in ruta_distancia]))
        
        return ruta_distancia # Retornamos esta para graficarla
        
    except nx.NetworkXNoPath:
        print("\nNo existe una ruta entre estos dos aeropuertos.")
        return None
    except KeyError:
        print("\nUno de los IDs ingresados no existe.")
        return None

# Ejemplo: De JFK (Nueva York) a SYD (Sydney)
# Nota: Debes buscar los IDs en tu archivo airports.csv. 
# Usaremos IDs de ejemplo comunes: JFK=3797, Heathrow=507, Sydney=3361
ruta_visualizar = encontrar_ruta(3797, 3361) 

# ==========================================
# 5. ANÁLISIS DE CONECTIVIDAD (REQ 5)
# ==========================================
print("\n--- Análisis de Conectividad ---")
es_fuertemente_conexo = nx.is_strongly_connected(G)
print(f"¿El grafo es fuertemente conexo? {'SÍ' if es_fuertemente_conexo else 'NO'}")

if not es_fuertemente_conexo:
    # Obtenemos los componentes fuertemente conexos
    scc = list(nx.strongly_connected_components(G))
    largest_scc = max(scc, key=len)
    print(f"  -> El grafo tiene {len(scc)} subgrupos aislados.")
    print(f"  -> El componente más grande tiene {len(largest_scc)} aeropuertos.")

# ==========================================
# 6. DIÁMETRO DE LA RED (REQ 4)
# ==========================================
# NOTA: Calcular el diámetro de todo el grafo es computacionalmente costoso (horas).
# Lo calcularemos solo sobre el componente conexo más grande para demostrarlo.
print("\n--- Calculando Diámetro (Componente Principal) ---")
subgrafo_principal = G.subgraph(largest_scc)

# El radio y diámetro pueden tardar mucho en grafos grandes. 
# Usamos approximation.diameter o calculamos exacto si el grafo es pequeño (<500 nodos).
# Para este ejemplo, usaremos una aproximación rápida para no congelar tu PC.
diametro_aprox = nx.approximation.diameter(subgrafo_principal)
print(f"Diámetro aproximado del cluster principal: {diametro_aprox} saltos.")

# ==========================================
# 7. VISUALIZACIÓN (REQ 6)
# ==========================================
print("\nGenerando visualización...")
plt.figure(figsize=(12, 8))

# Dibujar todos los aeropuertos (puntos pequeños grises)
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightgray', alpha=0.5)

# Si encontramos una ruta, la resaltamos
if ruta_visualizar:
    # Dibujar nodos de la ruta
    path_edges = list(zip(ruta_visualizar, ruta_visualizar[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=ruta_visualizar, node_size=50, node_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    
    # Etiquetas de inicio y fin
    nx.draw_networkx_labels(G, pos, labels={ruta_visualizar[0]: 'INICIO', ruta_visualizar[-1]: 'FIN'}, font_size=10, font_weight='bold')

plt.title("Red de Aeropuertos y Ruta Óptima (Proyección Plana)")
plt.axis('off')
plt.show()