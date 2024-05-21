
###################################################################################
#                    Algoritmo de Clasificación K-nn                              #
#                               -----------                                       #
#                  Por: Kathie Malti Franco Gómez                                 #
###################################################################################

#Librerías
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
from statistics import mode

# Use raw string literal to avoid escape characters (esto porque mi ruta tiene un numero)
file_path = r'C:\\Users\\kathi\\OneDrive - Universidad de Guadalajara\\Documentos\\9no Semestre\\CID\\iris.csv'

#lectura de datos
df = pd.read_csv(file_path)
datos = df.values.tolist()

dataTraining = []# lista de datos que vamos a usar para el entrenamiento
dataVerify = []# lista de los "puntos de prueba" que usaremos para verificar el funcionamiento de nuestro algoritmo
classCollection = []#lista de clasificaciones realizadas

k=3 #Valor de "K"--> numero de los vecinos más cercanos

aux = 0
for i in datos:
    aux += 1
    if aux<=120:# toma los primeros 120 registros 80%
        dataTraining.append(i)
    else:#toma los 30 registros posteriores 20%
        dataVerify.append(i)


#FUNCIÓN KNN 
def knn (xprueba):
    aux=0 #indice de iteración
    disCollection = [] #lista de las distancias asignadas a un punto

    for i in dataTraining:

        # formula de la distancia euclidiana
        distancia = math.sqrt((dataTraining[aux][0] - xprueba[0])**2 + (dataTraining[aux][1] - xprueba[1])**2 + (dataTraining[aux][2] - xprueba[2])**2 + (dataTraining[aux][3] - xprueba[3])**2)

        nuevosRegistro = [distancia,dataTraining[aux][4]]#toma la distancia euclidiana obtenida y su salida
        disCollection.append(nuevosRegistro)#agrega los datos obtenidos a una collecion de distancias euclidianas con respecto al punto asigando
        aux += 1

    disCollection = sorted(disCollection)#ordena la lista de las distancias

    firstK = disCollection [:k]# toma los primeros "k" registros
    firstK = [fila[1] for fila in firstK]#toma la segunda fila
    nuevoRegistro = [xprueba[0],xprueba[1],xprueba[2],xprueba[3],mode(firstK)]#variable auxiliar para ser ingresada a la colleción se las clasificaciones, Mode() saca la moda de una lista
    return nuevoRegistro#retorna el nuevo punto ya clasificado en base al algoritmo
    
# MAIN LOOP K-NN    
inter = 0
while inter < len(dataVerify):

    xPrueba = dataVerify[inter] #punto de prueba para verificar el funcionamiento del algoritmo
    classCollection.append(knn(xPrueba)) # se añade el registro del punto xPrueba a la lista de puntos ya clasificados 
    inter+=1

#muestra los resultados de la clasificación
Aciertos = 0
inter=1
print ("n  |  Registro Original             |            Registro Clasificado        |       Clasificación Correcta?")
print ("__________________________________________________________________________________________")
for i in range(len(classCollection)):
    if dataVerify[i][4]==classCollection[i][4]:
        Aciertos +=1

    print (f"{inter} | {dataVerify[i]}  |   {classCollection[i]}    |   {dataVerify[i][4]==classCollection[i][4]}")
    inter +=1

print ("")
print ("##############")
print ("# RESULTADOS #")
print ("################################################################################")
print (f"# Total de registros: {len(classCollection)}  # Aciertos: {Aciertos}  # Fallos: {len(classCollection)-Aciertos}  # margen de error: {((len(classCollection)-Aciertos)*100)/len(classCollection)}% #")
print ("################################################################################")
