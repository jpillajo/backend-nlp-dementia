import json
from flask import Flask, request
from flask_cors import CORS
import re
import nltk
import pandas as pd
from pandas import *
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
import numpy as np
import math
import itertools
import os

stemmer = SnowballStemmer('spanish')
bolsaStopwords = stopwords.words("spanish")


# IMPORTAR DATOS DE UN ARCHIVO CSV SEGÚN SU ENFOQUE
def importarDatosColumna(columna, path):
    archivoCSV = read_csv(path, sep=',')
    columna = archivoCSV[columna].tolist()
    return columna


def eliminarFilasVacias(columna):
    columna = [fila for fila in columna if pd.isnull(fila) == False]
    return columna


# NORMALIZACIÓN DE LOS DATOS
def convertirMayusculasEnMinusculas(lista):
    listaEnMinusculas = []
    for token in lista:
        listaEnMinusculas.append(token.lower())
    return listaEnMinusculas


def eliminarCaracteresEspeciales(lista):
    listaSinCaracteresEspeciales = []
    for token in lista:
        listaSinCaracteresEspeciales.append(re.sub('[^A-Za-záéíóúñ]+', ' ', token))
    return listaSinCaracteresEspeciales


def tokenizacion(lista):
    cadenaTokenizada = []
    for token in lista:
        cadenaTokenizada.append(token.split())
    return cadenaTokenizada


# Stopwords
def comprobarStopwords(lista):
    for cadena in lista:
        for word in cadena:
            if word in bolsaStopwords:
                return True
    return False


def eliminarStopwords(lista):
    while comprobarStopwords(lista):
        for cadena in lista:
            for word in cadena:
                if word in bolsaStopwords:
                    cadena.remove(word)
    return lista


# Stemming
def stemming(lista):
    cadenaConStemming = []
    for cadena in lista:
        palabraBase = []
        for word in cadena:
            palabraBase.append(stemmer.stem(word))
        cadenaConStemming.append(palabraBase)
    return cadenaConStemming


def eliminarPalabrasRepetidas(lista):
    listaSinPalabrasRepetidas = []
    for cadena in lista:
        if cadena not in listaSinPalabrasRepetidas:
            listaSinPalabrasRepetidas.append(cadena)
    return listaSinPalabrasRepetidas


# ALGORITMOS DE MACHINE LEARNING
## COEFICIENTE DE JACCARD
def unionConjuntos(lista1, lista2):
    resultadoUnion = list(lista1.union(lista2))
    return resultadoUnion


def interseccionConjuntos(lista1, lista2):
    resultadoInterseccion = list(lista1.intersection(lista2))
    return resultadoInterseccion


def metodoJaccard(bolsaDePalabrasCurado, documentos):
    matrizJaccard = []
    for i in range(len(bolsaDePalabrasCurado)):
        vectorInterseccion = []  # tenemos todas las intersecciones
        vectorUnion = []
        for documento in documentos:
            interseccion = interseccionConjuntos(set(documento), set(bolsaDePalabrasCurado[i]))
            union = unionConjuntos(set(documento), set(bolsaDePalabrasCurado[i]))
            vectorUnion.append(len(union))
            vectorInterseccion.append(len(interseccion))
        resultadoJaccard = np.array(vectorInterseccion) / np.array(vectorUnion)
        resultadoJaccard = list(np.around(np.array(resultadoJaccard), 2))
        matrizJaccard.append(resultadoJaccard)
    matrizSimilitudJaccard = np.array(matrizJaccard)
    return matrizSimilitudJaccard


def calcularTF(vocabulario, dataset, matrizTF):
    listaContadorFrecuencia = []
    for lista in dataset:
        for palabra in vocabulario:
            listaContadorFrecuencia.append(lista.count(palabra))
        matrizTF.append(listaContadorFrecuencia)
        listaContadorFrecuencia = []


def calcularWTF(matrizTF, matrizWTF):
    listaPesadoTF = []
    for listaFrecuencia in matrizTF:
        for dato in listaFrecuencia:
            if dato > 0:
                # listaPesadoTF.append(round((math.log(dato, 10)) + 1, 2))
                listaPesadoTF.append(1 + (math.log10(dato)))
            else:
                listaPesadoTF.append(0)
        matrizWTF.append(listaPesadoTF)
        listaPesadoTF = []


def calcularDF(matrizWTF, matrizDF, vocabulario):
    cont = 0
    index = 0
    for rep in range(len(vocabulario)):
        for lista in matrizWTF:
            if lista[index] > 0:
                cont += 1
        index += 1
        matrizDF.append(cont)
        cont = 0


def calcularIDF(matrizDF, dataset, matrizIDF):
    for dato in matrizDF:
        if dato == 0:
            dato = 1
        matrizIDF.append(math.log10(len(dataset) / dato))


def calcularWTFxIDF(matrizIDF, matrizWTF, matrizWTFxIDF):
    for lista in matrizWTF:
        matrizWTFxIDF.append(np.multiply(lista, matrizIDF))


def redondearMatriz(matrizNormalizada):
    lista = []
    lista_aux = []
    for i in range(len(matrizNormalizada)):
        for j in range(len(matrizNormalizada[i])):
            lista_aux.append(round(matrizNormalizada[i][j], 2))
        lista.append(lista_aux)
        lista_aux = []
    return lista


def calcularModulo(matrizWTFxIDF, matrizModulo):
    acum = 0
    for lista in matrizWTFxIDF:
        for dato in lista:
            if dato > 0:
                acum = acum + pow(dato, 2)
        matrizModulo.append(math.sqrt(acum))
        acum = 0


def normalizacionMatriz(matrizWTF, matrizModulo, matrizNormalizada):
    indice = 0
    for lista in matrizWTF:
        if matrizModulo[indice] == 0:
            matrizModulo[indice] = 1
        matrizNormalizada.append(list(map(lambda x: x / matrizModulo[indice], lista)))
        indice += 1


def metodoCoseno(bolsaDePalabrasCuradoSR, dataset):
    listaSimilitudCoseno = []
    for i in range(len(bolsaDePalabrasCuradoSR)):
        listaDocumentos = []
        matrizTF = []
        matrizDF = []
        matrizWTF = []
        matrizIDF = []
        matrizWTFxIDF = []
        # TF
        calcularTF(bolsaDePalabrasCuradoSR[i], dataset, matrizTF)
        # WTF
        calcularWTF(matrizTF, matrizWTF)
        # DF
        calcularDF(matrizWTF, matrizDF, bolsaDePalabrasCuradoSR[i])
        # IDF
        calcularIDF(matrizDF, dataset, matrizIDF)
        # WTF-IDF
        calcularWTFxIDF(matrizIDF, matrizWTF, matrizWTFxIDF)
        matrizModulo = []
        matrizNormalizada = []
        # if sum(sum(matrizWTFxIDF)) == 0:
        calcularModulo(matrizWTF, matrizModulo)
        normalizacionMatriz(matrizWTF, matrizModulo, matrizNormalizada)
        # else:
        #     calcularModulo(matrizWTFxIDF, matrizModulo)
        #     normalizacionMatriz(matrizWTFxIDF, matrizModulo, matrizNormalizada)
        # modulo_raiz(lista_wtf, lista_modulo, bolsa_de_palabras[i])
        # lista_normalizada(lista_wtf, lista_modulo, lista_normal)
        # lista_normal = redondear(lista_normal)
        matrizNormalizada = redondearMatriz(matrizNormalizada)

        for lista in matrizNormalizada:
            listaDocumentos.append(round((sum(lista) / len(bolsaDePalabrasCuradoSR[i])), 2))
        listaSimilitudCoseno.append(listaDocumentos)
    return listaSimilitudCoseno


app = Flask(__name__)
CORS(app)

# CARGA DE LA BOLSA DE PALABRAS (SEGÚN ENFOQUE)
urlBolsaDePalabras = "https://raw.githubusercontent.com/jpillajo/Documentos/main/BOLSA%20DE%20PALABRAS%203%20MODELOS%20PARA%20CATIA.csv"
enfoqueBiomedico = importarDatosColumna("A. MODELO BIO MEDICO", urlBolsaDePalabras)
columnaEnfoquePsicosocial = importarDatosColumna("B. ENFOQUE PSICOSOCIAL - COMUNITARIO", urlBolsaDePalabras)
enfoquePsicosocial = eliminarFilasVacias(columnaEnfoquePsicosocial)
columnaEnfoqueCotidiano = importarDatosColumna("C. ENFOQUE COTIDIANO", urlBolsaDePalabras)
enfoqueCotidiano = eliminarFilasVacias(columnaEnfoqueCotidiano)

# NORMALIZACIÓN
enfoqueBiomedico = convertirMayusculasEnMinusculas(enfoqueBiomedico)
enfoqueBiomedico = eliminarCaracteresEspeciales(enfoqueBiomedico)
enfoquePsicosocial = convertirMayusculasEnMinusculas(enfoquePsicosocial)
enfoquePsicosocial = eliminarCaracteresEspeciales(enfoquePsicosocial)
enfoqueCotidiano = convertirMayusculasEnMinusculas(enfoqueCotidiano)
enfoqueCotidiano = eliminarCaracteresEspeciales(enfoqueCotidiano)

# TOKENIZACIÓN
enfoqueBiomedico = tokenizacion(enfoqueBiomedico)
enfoqueBiomedico = eliminarStopwords(enfoqueBiomedico)
enfoquePsicosocial = tokenizacion(enfoquePsicosocial)
enfoquePsicosocial = eliminarStopwords(enfoquePsicosocial)
enfoqueCotidiano = tokenizacion(enfoqueCotidiano)
enfoqueCotidiano = eliminarStopwords(enfoqueCotidiano)

# STEMMING
enfoqueBiomedico = stemming(enfoqueBiomedico)
enfoqueBiomedico = list(itertools.chain(*enfoqueBiomedico))
enfoquePsicosocial = stemming(enfoquePsicosocial)
enfoquePsicosocial = list(itertools.chain(*enfoquePsicosocial))
enfoqueCotidiano = stemming(enfoqueCotidiano)
enfoqueCotidiano = list(itertools.chain(*enfoqueCotidiano))
bolsaDePalabrasCurado = [enfoqueBiomedico, enfoquePsicosocial, enfoqueCotidiano]

# ELIMINAR PALABRAS REPETIDAS
enfoqueBiomedicoSR = eliminarPalabrasRepetidas(enfoqueBiomedico)
enfoquePsicosocialSR = eliminarPalabrasRepetidas(enfoquePsicosocial)
enfoqueCotidianoSR = eliminarPalabrasRepetidas(enfoqueCotidiano)
bolsaDePalabrasCuradoSR = [enfoqueBiomedicoSR, enfoquePsicosocialSR, enfoqueCotidianoSR]


def analizarSimilitud(activador, documentos = ''):
    dataset = []
    if activador == 0:
        dataset.append(documentos)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = eliminarCaracteresEspeciales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminarStopwords(dataset)
        dataset = stemming(dataset)
        similitudJaccard = metodoJaccard(bolsaDePalabrasCurado, dataset)
        similitudCoseno = metodoCoseno(bolsaDePalabrasCuradoSR, dataset)
    if activador == 1:
        urlDatasetDefinicionesDemencia = "https://raw.githubusercontent.com/jpillajo/Documentos/main/EXCEL%20DE%20VACIADO%20COMPLETO%20DE%20ENTREVISTAS%20PROFESIONALES%20MAYO2021.csv"
        dataset = importarDatosColumna("P7. ¿Qué entiende por demencia?", urlDatasetDefinicionesDemencia)
        dataset = eliminarFilasVacias(dataset)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = eliminarCaracteresEspeciales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminarStopwords(dataset)
        dataset = stemming(dataset)
        similitudJaccard = metodoJaccard(bolsaDePalabrasCurado, dataset)
        similitudCoseno = metodoCoseno(bolsaDePalabrasCuradoSR, dataset)
    if activador == 2:
        dataset = importarDatosColumna(documentos, "static/archivos/server_interno.csv")
        dataset = eliminarFilasVacias(dataset)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = eliminarCaracteresEspeciales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminarStopwords(dataset)
        dataset = stemming(dataset)
        similitudJaccard = metodoJaccard(bolsaDePalabrasCurado, dataset)
        similitudCoseno = metodoCoseno(bolsaDePalabrasCuradoSR, dataset)
    return [similitudJaccard, similitudCoseno]


def normalizacionDatosSimilitud(matriz):
    matrizDatosNormalizados = []
    sumaTotal = ((matriz[0])[0]) + ((matriz[1])[0]) + ((matriz[2])[0])
    if (matriz[0])[0] != 0:
        dato0normalizado = (((matriz[0])[0]) * 100) / sumaTotal
    else:
        dato0normalizado = 0
    if (matriz[1])[0] != 0:
        dato1normalizado = (((matriz[1])[0]) * 100) / sumaTotal
    else:
        dato1normalizado = 0
    if (matriz[2])[0] != 0:
        dato2normalizado = (((matriz[2])[0]) * 100) / sumaTotal
    else:
        dato2normalizado = 0
    matrizDatosNormalizados.append(dato0normalizado)
    matrizDatosNormalizados.append(dato1normalizado)
    matrizDatosNormalizados.append(dato2normalizado)
    return matrizDatosNormalizados


# APIs
@app.route('/api/consultar-definicion', methods=['POST'])
def consultarDefinicion():
    dataSend = json.loads(request.data.decode())
    definicionIngresada = dataSend["definicion"]
    activador = 0
    matrizSimilitud = analizarSimilitud(activador, definicionIngresada)

    similitudJaccardNormalizado = normalizacionDatosSimilitud(matrizSimilitud[0])
    similitudCosenoNormalizado = normalizacionDatosSimilitud(matrizSimilitud[1])

    dto = json.dumps({'jaccard': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': similitudJaccardNormalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': similitudJaccardNormalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': similitudJaccardNormalizado[2]}
    ], 'coseno': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': similitudCosenoNormalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': similitudCosenoNormalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': similitudCosenoNormalizado[2]}
    ]})
    return dto


@app.route('/api/obtener-dataset', methods=['POST'])
def obtenerDataset():
    dataSend = json.loads(request.data.decode())
    enfoque = dataSend["valor"]
    activador = 1
    matrizSimilitud = analizarSimilitud(activador)
    matrizJaccardPrevia = matrizSimilitud[0].tolist()
    matrizJaccard = matrizJaccardPrevia[enfoque]
    matrizCosenoPrevia = matrizSimilitud[1]
    matrizCoseno = matrizCosenoPrevia[enfoque]
    jsonJaccard = []
    jsonCoseno = []

    for i in range(len(matrizJaccard)):
        jsonJaccard.append({
            'id': i + 1, 'porcentaje': matrizJaccard[i]
        })

    for i in range(len(matrizCoseno)):
        jsonCoseno.append({
            'id': i + 1, 'porcentaje': matrizCoseno[i]
        })

    dto = json.dumps({'jaccard': jsonJaccard, 'coseno': jsonCoseno})
    return dto


@app.route('/api/subir-dataset', methods=['POST'])
def subirArchivoCSV():
    archivoEnviadoCSV = request.files['file']
    archivoEnviadoCSV.save('assets/dataset.csv')
    archivoAlmacenadoCSV = pd.read_csv('assets/dataset.csv')
    vectorAutores = []
    for i in range(len(archivoAlmacenadoCSV)):
        vectorAutores.append({'id': i, 'valor': archivoAlmacenadoCSV.loc[i]['Autor']})
    dto = json.dumps(vectorAutores)
    return dto


@app.route('/api/consultar-similitud-dataset', methods=['POST'])
def consultarSimilitudDataset():
    dataSend = json.loads(request.data.decode())
    autor = dataSend["valor"]
    archivoAlmacenadoCSV = pd.read_csv('assets/dataset.csv')

    definicion = archivoAlmacenadoCSV.loc[autor, 'Definición']
    activador = 0
    matrizSimilitud = analizarSimilitud(activador, definicion)

    similitudJaccardNormalizado = normalizacionDatosSimilitud(matrizSimilitud[0])
    similitudCosenoNormalizado = normalizacionDatosSimilitud(matrizSimilitud[1])

    dto = json.dumps({'jaccard': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': similitudJaccardNormalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': similitudJaccardNormalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': similitudJaccardNormalizado[2]}
    ], 'coseno': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': similitudCosenoNormalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': similitudCosenoNormalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': similitudCosenoNormalizado[2]}
    ]})
    return dto


@app.route('/api/eliminar-archivo-dataset', methods=['GET'])
def eliminarArchivoDataset():
    if os.path.exists('assets/dataset.csv'):
        os.remove('assets/dataset.csv')
    return 'Eliminación exitosa'
