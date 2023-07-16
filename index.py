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
encabezadosEnfoques = ['BIOMÉDICO', 'PSICOSOCIAL - COMUNITARIO', 'COTIDIANO']


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


# UNIFICAR BOLSAS DE PALABRAS
def crearBolsaUnificada(dataset):
    bolsaGeneral = []
    for enfoque in dataset:
        for termino in enfoque:
            bolsaGeneral.append(termino)
    bolsaGeneralSR = eliminarPalabrasRepetidas(bolsaGeneral)
    return bolsaGeneralSR


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


def calcularTF(dataset, vocabulario, matrizTF):
    for lista in dataset:
        listaContadorFrecuencia = []
        for palabra in vocabulario:
            listaContadorFrecuencia.append(lista.count(palabra))
        matrizTF.append(listaContadorFrecuencia)


def calcularWTF(matrizTF, matrizWTF):
    for listaFrecuencia in matrizTF:
        listaPesadoTF = []
        for dato in listaFrecuencia:
            if dato > 0:
                # listaPesadoTF.append(round((math.log(dato, 10)) + 1, 2))
                listaPesadoTF.append(1 + (math.log10(dato)))
            else:
                listaPesadoTF.append(0)
        matrizWTF.append(listaPesadoTF)


def calcularDF(matrizWTF, matrizDF):
    transpuestaWTF = np.array(matrizWTF).transpose()
    for fila in transpuestaWTF:
        cont = 0
        for columna in fila:
            if columna > 0:
                cont += 1
        matrizDF.append(cont)


def calcularIDF(matrizDF, dataset, matrizIDF):
    for dato in matrizDF:
        if dato == 0:
            dato = 1
        matrizIDF.append(math.log10(len(dataset) / dato))


def calcularWTFxIDF(matrizIDF, matrizWTF, matrizWTFxIDF):
    for lista in matrizWTF:
        matrizWTFxIDF.append(np.multiply(lista, matrizIDF))


def calcularModulo(matrizWTFxIDF, matrizModulo):
    for lista in matrizWTFxIDF:
        acum = 0
        for dato in lista:
            if dato > 0:
                acum = acum + pow(dato, 2)
        matrizModulo.append(math.sqrt(acum))


def normalizacionMatriz(matrizWTFxIDF, matrizModulo, matrizNormalizada):
    indice = 0
    for lista in matrizWTFxIDF:
        if matrizModulo[indice] == 0:
            matrizModulo[indice] = 1
        matrizNormalizada.append(list(map(lambda x: x / matrizModulo[indice], lista)))
        indice += 1


def metodoCoseno(dataset, bolsaDePalabrasSR, definiciones):
    matrizTF = []
    matrizWTF = []
    matrizDF = []
    matrizIDF = []
    matrizWTFxIDF = []
    # TF
    calcularTF(dataset, bolsaDePalabrasSR, matrizTF)
    # WTF
    calcularWTF(matrizTF, matrizWTF)
    # DF
    calcularDF(matrizWTF, matrizDF)
    # IDF
    calcularIDF(matrizDF, dataset, matrizIDF)
    # WTF-IDF
    calcularWTFxIDF(matrizIDF, matrizWTF, matrizWTFxIDF)
    ## NORMALIZACION DE VECTORES
    matrizModulo = []
    matrizNormalizada = []
    # CALCULAR MODULO
    calcularModulo(matrizWTFxIDF, matrizModulo)
    # NORMALIZAR VECTORES
    normalizacionMatriz(matrizWTFxIDF, matrizModulo, matrizNormalizada)

    ## COEFICIENTE COSENO
    matrizNormalizada = np.array(matrizNormalizada)
    coeficienteCoseno = []
    for i in range(0, 3):
        for j in range(3, len(matrizNormalizada)):
            factor = round(matrizNormalizada[i].dot(matrizNormalizada[j]), 3)
            coeficienteCoseno.append(factor)
    matrizCoseno = np.array(coeficienteCoseno).reshape(3, len(definiciones))
    return matrizCoseno


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

# UNIFICAR BOLSAS DE PALABRAS
bolsaGeneralSR = crearBolsaUnificada(bolsaDePalabrasCurado)

# ELIMINAR PALABRAS REPETIDAS
enfoqueBiomedicoSR = eliminarPalabrasRepetidas(enfoqueBiomedico)
enfoquePsicosocialSR = eliminarPalabrasRepetidas(enfoquePsicosocial)
enfoqueCotidianoSR = eliminarPalabrasRepetidas(enfoqueCotidiano)
bolsaDePalabrasCuradoSR = [enfoqueBiomedicoSR, enfoquePsicosocialSR, enfoqueCotidianoSR]


def analizarSimilitud(activador, bolsaGeneralSR, documentos=''):
    dataset = []
    similitudJaccard = []
    similitudCoseno = []
    bolsaDePalabrasCuradoSR = [enfoqueBiomedicoSR, enfoquePsicosocialSR, enfoqueCotidianoSR]
    if activador == 0:
        dataset.append(documentos)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = eliminarCaracteresEspeciales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminarStopwords(dataset)
        dataset = stemming(dataset)
        for lista in dataset:
            bolsaDePalabrasCuradoSR.append(lista)
            for termino in lista:
                bolsaGeneralSR.append(termino)
        bolsaGeneralSR = eliminarPalabrasRepetidas(bolsaGeneralSR)
        similitudJaccard = metodoJaccard(bolsaDePalabrasCurado, dataset)
        similitudCoseno = metodoCoseno(bolsaDePalabrasCuradoSR, bolsaGeneralSR, dataset)
    if activador == 1:
        urlDatasetDefinicionesDemencia = "https://raw.githubusercontent.com/jpillajo/Documentos/main/EXCEL%20DE%20VACIADO%20COMPLETO%20DE%20ENTREVISTAS%20PROFESIONALES%20MAYO2021.csv"
        dataset = importarDatosColumna("P7. ¿Qué entiende por demencia?", urlDatasetDefinicionesDemencia)
        dataset = eliminarFilasVacias(dataset)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = eliminarCaracteresEspeciales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminarStopwords(dataset)
        dataset = stemming(dataset)
        enfBiomedico = []
        enfPsicosocial = []
        enfCotidiano = []
        for lista in dataset:
            temp = []
            bolsaDePalabrasCuradoSR = [enfoqueBiomedicoSR, enfoquePsicosocialSR, enfoqueCotidianoSR]
            bolsaGeneralSR = crearBolsaUnificada(bolsaDePalabrasCurado)
            bolsaDePalabrasCuradoSR.append(lista)
            for termino in lista:
                bolsaGeneralSR.append(termino)
            bolsaGeneralSR = eliminarPalabrasRepetidas(bolsaGeneralSR)
            temp = metodoCoseno(bolsaDePalabrasCuradoSR, bolsaGeneralSR, [""])
            enfBiomedico.append(float(temp[0]))
            enfPsicosocial.append(float(temp[1]))
            enfCotidiano.append(float(temp[2]))
        similitudCoseno = [enfBiomedico, enfPsicosocial, enfCotidiano]
        similitudJaccard = metodoJaccard(bolsaDePalabrasCurado, dataset)
    return [similitudJaccard, similitudCoseno]


def normalizacionDatosSimilitud(matriz):
    vectorSumaEnfoque = []
    matrizDatosNormalizados = []
    for enfoque in matriz:
        acum = 0
        for i in enfoque:
            acum = acum + i
        vectorSumaEnfoque.append(acum)

    cont = 0
    for enfoque in matriz:
        vectorDatosNormalizados = []
        for i in enfoque:
            if i != 0 or vectorSumaEnfoque[cont] != 0:
                n1 = (i * 100) / vectorSumaEnfoque[cont]
            else:
                n1 = 0
            vectorDatosNormalizados.append(n1)
        cont += 1
        matrizDatosNormalizados.append(vectorDatosNormalizados)
    return matrizDatosNormalizados


# APIs
@app.route('/api/consultar-definicion', methods=['POST'])
def consultarDefinicion():
    dataSend = json.loads(request.data.decode())
    definicionIngresada = dataSend["definicion"]
    activador = 0
    matrizSimilitud = analizarSimilitud(activador, bolsaGeneralSR, definicionIngresada)
    m1 = matrizSimilitud[0].transpose()
    m2 = np.array(matrizSimilitud[1]).transpose()
    matrizJaccardPrevia = normalizacionDatosSimilitud(m1)
    matrizCosenoPrevia = normalizacionDatosSimilitud(m2)
    similitudJaccardNormalizado = np.array(matrizJaccardPrevia).transpose()
    similitudCosenoNormalizado = np.array(matrizCosenoPrevia).transpose()
    jsonJaccard = []
    jsonCoseno = []

    for i in range(len(similitudJaccardNormalizado)):
        jsonJaccard.append({
            'enfoque': encabezadosEnfoques[i], 'porcentaje': float(similitudJaccardNormalizado[i][0])
        })

    for i in range(len(similitudCosenoNormalizado)):
        jsonCoseno.append({
            'enfoque': encabezadosEnfoques[i], 'porcentaje': float(similitudCosenoNormalizado[i][0])
        })
    dto = json.dumps({'jaccard': jsonJaccard, 'coseno': jsonCoseno})
    return dto


@app.route('/api/obtener-dataset', methods=['POST'])
def obtenerDataset():
    dataSend = json.loads(request.data.decode())
    enfoque = dataSend["valor"]
    activador = 1
    matrizSimilitud = analizarSimilitud(activador, bolsaGeneralSR)
    m1 = matrizSimilitud[0].transpose()
    m2 = np.array(matrizSimilitud[1]).transpose()
    matrizJaccardPrevia = normalizacionDatosSimilitud(m1)
    matrizCosenoPrevia = normalizacionDatosSimilitud(m2)
    s1 = np.array(matrizJaccardPrevia).transpose()
    s2 = np.array(matrizCosenoPrevia).transpose()

    matrizJaccard = s1[enfoque]
    matrizCoseno = s2[enfoque]
    jsonJaccard = []
    jsonCoseno = []

    for i in range(len(matrizJaccard)):
        jsonJaccard.append({
            'id': i + 1, 'porcentaje': float(matrizJaccard[i])
        })

    for i in range(len(matrizCoseno)):
        jsonCoseno.append({
            'id': i + 1, 'porcentaje': float(matrizCoseno[i])
        })

    dto = json.dumps({'jaccard': jsonJaccard, 'coseno': jsonCoseno})
    return dto


@app.route('/api/subir-dataset', methods=['POST'])
def subirArchivoCSV():
    archivoEnviadoCSV = request.files['file']
    if archivoEnviadoCSV.mimetype == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        archivoEnviadoCSV.save('assets/dataset.xlsx')
        archivoAlmacenadoCSV = pd.read_excel('assets/dataset.xlsx')
        archivoAlmacenadoCSV.to_csv('assets/dataset.csv')
        if os.path.exists('assets/dataset.xlsx'):
            os.remove('assets/dataset.xlsx')
    else:
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
    matrizSimilitud = analizarSimilitud(activador, bolsaGeneralSR, definicion)
    m1 = matrizSimilitud[0].transpose()
    m2 = np.array(matrizSimilitud[1]).transpose()
    matrizJaccardPrevia = normalizacionDatosSimilitud(m1)
    matrizCosenoPrevia = normalizacionDatosSimilitud(m2)
    similitudJaccardNormalizado = np.array(matrizJaccardPrevia).transpose()
    similitudCosenoNormalizado = np.array(matrizCosenoPrevia).transpose()
    jsonJaccard = []
    jsonCoseno = []

    for i in range(len(similitudJaccardNormalizado)):
        jsonJaccard.append({
            'enfoque': encabezadosEnfoques[i], 'porcentaje': float(similitudJaccardNormalizado[i][0])
        })

    for i in range(len(similitudCosenoNormalizado)):
        jsonCoseno.append({
            'enfoque': encabezadosEnfoques[i], 'porcentaje': float(similitudCosenoNormalizado[i][0])
        })

    dto = json.dumps({'jaccard': jsonJaccard, 'coseno': jsonCoseno, 'definicion': definicion})
    return dto


@app.route('/api/eliminar-archivo-dataset', methods=['GET'])
def eliminarArchivoDataset():
    if os.path.exists('assets/dataset.csv'):
        os.remove('assets/dataset.csv')
    return 'Eliminación exitosa'
