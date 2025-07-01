import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random as ra
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

'''
preguntas: 
por que deben ser poblaciones impares?
el mejor de cada generacion debe ser padre o pasar directamente?
p_corte aleatorio
'''

def getDatos(path):
  '''obtener los datos del json'''
  # cargar el archivo JSON
  with open(path) as f:
    data = json.load(f)

  # inicializar arrays
  ids = []         # identificador
  points = []      # lista de puntos [x, y]

  # extraer datos relevantes
  for rollcall in data['rollcalls']:
    for vote in rollcall['votes']:
      ids.append(vote['icpsr'])
      points.append([vote['x'], vote['y']])  # guardar [x,y] 

  n_votes = len(ids)
  return np.array(ids), np.array(points), n_votes  # convertir a array np


def generarCromosoma(total, quorum):
  '''genera un cromosoma que cumple con la coalicion minima ganadora'''
  # vector ceros
  cromosoma = np.zeros(total, dtype=np.uint8)
  # seleccionar posiciones aleatorias para los unos
  posicion = np.random.choice(total, size=quorum, replace=False)
  # asignar unos a las posiciones
  cromosoma[posicion] = 1

  return cromosoma


def graf_polig_convexo(cromosoma, pts):
  # filtrar puntos
  pts_select = pts[cromosoma == 1]

  fig, ax = plt.subplots() # crear figura
  
  # graficar puntos
  ax.scatter(pts[:, 0], pts[:, 1])
  ax.scatter(pts_select[:, 0], pts_select[:, 1]) #ejes x e y
  
  # calcular y graficar el poligono
  if len(pts_select) >= 3:  #minimo 3 puntos para crearlo
    hull = ConvexHull(pts_select)
    # dibujar las aristas
    for simplex in hull.simplices:
      ax.plot(pts_select[simplex, 0], pts_select[simplex, 1], 'c-')
    # dibujar los vertices
    ax.plot(pts_select[hull.vertices, 0], pts_select[hull.vertices, 1], 'o',
            mec='orange', color='none', lw=1, markersize=8)
  else:
      print("No hay puntos suficientes para formar el poligono convexo")
    
  ax.set_xlabel('Posicion Politica')
  ax.set_ylabel('Posicion Politica (otro)')
  ax.set_title('Algoritmo Genetico para MWC')
  ax.grid(True)
  plt.axis([-1, 1, -1, 1]) #limites del grafico
  plt.show()


def generarPoblacion(n_pob, n_votes, quorum):
  poblacion = np.empty((n_pob, n_votes), dtype=np.uint8) # crear vacio
  for i in range(n_pob):
    poblacion[i] = generarCromosoma(n_votes, quorum)
  return poblacion


#utiliza scipy pdist y normaliza el resultado de 0 a 1
def calcularFitness(cromosoma, puntos, poblacion):
    """
    fitness normalizado 0 a 1 para puntos en el plano
    1 Minima distancia; 0 Maxima distancia posible.
    """
    selected_points = puntos[cromosoma == 1]

    p = len(selected_points) #216 puntos
    
    #suma de distancia total
    distancia_total = np.sum(pdist(selected_points, 'euclidean'))

    #print(f'distancia total:{distancia_total}')
    
    #distancia maxima en el plano -1,-1 a 1,1
    d_max = np.sqrt((1-(-1))**2+((1-(-1))**2)) #sqrt(8)
    max_possible_dist = (p * (p - 1) / 2) * d_max

    #fitness normalizado para minimizacion
    fitness = 1 - (distancia_total / max_possible_dist)

    #print("fitness",fitness)
    
    return fitness


def mutacion(poblacion, p_mut):
  '''cambiar un dato aleatorio de un cromosoma mediante una probabilidad p_mut'''
  #for cromosoma in poblacion:
    #for voter in cromosoma:
      #cromosoma[voter] ^= cromosoma[voter]
    #probalidad de mutar en una posicion aleatoria

  poblacion_mutada = np.copy(poblacion)
  n_cromosomas, n_votantes = poblacion.shape
  
  for cromosoma in range(n_cromosomas):
    if ra.random() < p_mut:  # Decide si mutar este cromosoma
      posicion = ra.randint(0, n_votantes -1)  # posicion aleatoria
      poblacion_mutada[cromosoma, posicion] ^= 1  # XOR cambia el bit
  
  return poblacion_mutada


def cuantile(prob_acumulada, poblacion):
  '''selecciona un cromosma ganador de acuerdo a la probabilidad acumulada de cada uno'''
  #p = ra.uniform()

  #print(p)
  
  ...


def validarCromosoma(poblacion, quorum):
  '''revisar si el cromosoma cumple con el quorum, si no cumple se cambia 
  un votante hasta cumplir'''

  for cromosoma in poblacion:
    #si el numero de votos es menor al quorum
    # if n_votos < quorum:
    #  aplicar XOR a un 0 para agregar un voto aleatorio

    #elif n_votos > quorum:
    # aplicar XOR a un 1 para eliminar un voto aleatorio

    #else break
    ...
  ...


def pcorte_validos(padre1, padre2, unos):
  '''calcular puntajes de corte validos para cumplir el quorum'''
  n = len(padre1)
  valid_points = []
  

  for i in range(1, n):
    # Contar los unos en la parte superior de cada padre
    unos_padre1_superior = np.sum(padre1[:i])
    unos_padre2_superior = np.sum(padre2[:i])
    
    # Verificar si el intercambio mantiene el balance
    if (unos_padre1_superior + (unos - unos_padre2_superior)) == unos:
      valid_points.append(i)
  
  return valid_points


def crossover(poblacion):
  '''crea dos cromosomas hijos a partir de dos cromosomas padres aleatorios
  necesita de un punto de corte para realizar el entrecruzamiento
  TODO: cambiar p_corte aleatorio por encontrar un punto de corte 
  valido???'''

  #separar los cromosomas en pares, y realizar el cruzamiento de acuerdo al punto de corte
  n_cromosomas, longitud = poblacion.shape
  nueva_poblacion = np.empty_like(poblacion)

  #p_corte aleatorio entre 0 a longitud
  p_corte = ra.randrange(longitud)

  #print("punto de corte:",p_corte)
    
  # mezclar indices para emparejamiento radomico
  indices = np.random.permutation(n_cromosomas)

  # se guardar el mejor fitness:
  nueva_poblacion[0] = poblacion[0]

  for i in range(1, n_cromosomas, 2): #cada 2 elementos TODO: sin contar el mejor?
    # Tomar dos padres
    padre1 = poblacion[indices[i]]
    padre2 = poblacion[indices[i+1]]      

    # crear hijos
    hijo1 = np.concatenate((padre1[:p_corte], padre2[p_corte:])) #:N hasta; N: desde (incluye)
    hijo2 = np.concatenate((padre2[:p_corte], padre1[p_corte:]))

    nueva_poblacion[i] = hijo1
    nueva_poblacion[i+1] = hijo2

  print('nueva:\n',nueva_poblacion)
  
  return nueva_poblacion



def prob_posicion(n_cromosomas, prob):
  '''calcular la probabilidad de elegir al ganador segun su posicion
  retorna el arreglo de probabilidades por posicion 
  TODO: devolver probabilidades por posicion o acumulada?'''
  sum = 0
  probabilidades = []
  probab_acumulada = 0

  #calcula la sumatoria de todas las probabilidades
  for j in range(1, n_cromosomas+1):
    sum += prob*(1-prob)**(j-1)

  #calcula la probabilidad por posicion
  for i in range(1, n_cromosomas+1):
    # cuantil = prob*(1-prob)**(i-1)/sum
    probab_acumulada += prob*(1-prob)**(i-1)/sum

    probabilidades.append(probab_acumulada)

    #probabilidades.append(prob*(1-prob)**(i-1)/sum)
  

  print("probabilidades",probabilidades)

  return probabilidades


def ordenarPoblacion(poblacion, puntos):
  '''Se selecciona al mejor individuo de acuerdo a la probabilidad del fitness
  TODO:falta aplicar la probabilidad'''

  ################### 2.-Evaluar cada cromosoma de acuerdo a Z(fitness) ###################
  fitness_values = [calcularFitness(cromosoma, puntos, poblacion) for cromosoma in poblacion]
  #print('fit',fitness_values)
  fitness_array = np.array(fitness_values)

  ########## 3.-Ordenar la poblacion de manera decendente de acuerdo al fitness ##########
  sorted_indices = np.argsort(fitness_array)[::-1]
  # poblacion ordenada
  poblacion_ordenada =  poblacion[sorted_indices] 

  return poblacion_ordenada




np.set_printoptions(edgeitems=10, threshold=100, linewidth=150)


########################################---INICIO---########################################

ids, puntos, n_votes = getDatos('./votes.json')    #id y posiciones
quorum_min = math.floor(n_votes/2)+1      #quorum minimo
n_pobl = 11     # poblacion inicial
p_sel = 0.6     # prob seleccion para el mejor
p_mut = 0.4    # probailidad de mutacion


#repetir hasta cuando? 

########## 1.- Generar la poblacion inicial ##########
poblacion = generarPoblacion(n_pobl, n_votes, quorum_min) # generar la poblacion inicial

########## 2.-Evaluar cada cromosoma de acuerdo a Z(fitness) y ordenar ##########
poblacion_sorted = ordenarPoblacion(poblacion, puntos)


#print("normal",poblacion) #imprimir poblacion normal
#print("ordena",poblacion_sorted) #imprimir poblacion ordenadaordena
nueva_poblacion = crossover(poblacion_sorted)
poblacion_mutada = mutacion(nueva_poblacion, p_mut)


mejor = poblacion_sorted[0]
### graficar el poligono
graf_polig_convexo(mejor, puntos)

prob_posicion(n_pobl, p_sel)

#fitness(crom, puntos)

#poblacion es una matriz con filas correspondientes a los cromosomas



#PASOS:
#1.-Crear la poblacion inicial tamano pobl maxima -> siempre debe ser impar
#2.-Evaluar cada cromosoma de acuerdo a Z(fitness) -> que tan bueno es 
#3.-Ordenar la poblacion de manera decendente de acuerdo al fitness
#4.-Seleccionar los padres (Dos cromosomas) #
#5.-Seleccionar punto de corte aleatorio
#6.-Cruzar padres
#7.-Generar mutaciones con probabilidad P_mut  #una o mas mutaciones? probabilidad aleatoria?
#8.-Validar hijos (se refiere a que cumplan con el quorum si no cumple se agregan o quitan bits)
#9.-Poblar la nueva poblacion con los hijos supervivientes
#9 -> 2 pasar el mejor de la poblacion anterior
#9 -> 4 hasta llenar la poblacion





#torneo p -> probabilidad de seleccionar al mejor
#combinacion convexa todo suma 1
#p
#p(1-p) / sumatoria p(1-p)**i-1
#p(1-p)**2 / sumatoria p(1-p)**i-1
#p(1-p)i-1 / sumatoria p(1-p)**i-1
#si no suma 1 se debe normalizar


#P/sum p(1-p)**i-1
#P+p(1-p) / sum p(1-p)i-1
#p+p(1-p)+p(1-p)**2 / sum p(1-p)i-1