import json
from flask import Flask, request, render_template
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
n = stopwords.words("spanish")

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


def caracter_especiales(lista):
    tit = []
    for token in lista:
        tit.append(re.sub('[^A-Za-záéíóúñ]+', ' ', token))
    return tit


def tokenizacion(lista):
    tit = []
    aux = []
    for token in lista:
        aux.append(token.split())
    tit = aux
    return tit


# Stopwords
def comprobar_stop_words(lista):
    global n
    for cadena in lista:
        for word in cadena:
            if (word in n):
                return True
    return False


def eliminar_stop_words(lista):
    global n
    while (comprobar_stop_words(lista)):
        for cadena in lista:
            for word in cadena:
                if (word in n):
                    cadena.remove(word)
    return lista


# Stemming
def stemming(lista):
    global stemmer
    tit = []
    aux = []
    for cadena in lista:
        aux = []
        for token in cadena:
            aux.append(stemmer.stem(token))
        tit.append(aux)
    return tit


def eliminar_palabras_repetidas(lista):
    vector = []
    for palabra in lista:
        if palabra not in vector:
            vector.append(palabra)
    return vector

# ALGORITMOS DE MACHINE LEARNING
## COEFICIENTE DE JACCARD
def unionConjuntos(list1, list2):
    result = list(list1.union(list2))
    return result

def interseccionConjuntos(list1, list2):
    result = list(list1.intersection(list2))
    return result


def jacard_res(bolsa_de_palabras, documentos):
    union = []
    vector_interseccion = []  # tenemos todas las intersecciones
    vector_union = []
    interseccion = []
    resultado = []
    m = []
    for i in range(len(bolsa_de_palabras)):
        vector_interseccion = []  # tenemos todas las intersecciones
        vector_union = []
        for documento in documentos:
            interseccion = interseccionConjuntos(set(documento), set(bolsa_de_palabras[i]))
            union = unionConjuntos(set(documento), set(bolsa_de_palabras[i]))
            vector_union.append(len(union))
            vector_interseccion.append(len(interseccion))
        resultado = np.array(vector_interseccion) / np.array(vector_union)
        resultado = list(np.around(np.array(resultado), 2))

        m.append(resultado)
    matriz = np.array(m)
    return matriz


def frecuencias(vocabulario, abstract, frecuencia):
    lista_aux = []

    for lista in abstract:
        for palabra in vocabulario:
            lista_aux.append(lista.count(palabra))
        frecuencia.append(lista_aux)
        lista_aux = []


def calcular_wtf(frecuencia, lista_wtf):
    lista_aux = []
    for lista_frecuencia in frecuencia:
        for dato in lista_frecuencia:
            if (dato > 0):
                lista_aux.append(round((math.log(dato, 10)) + 1, 2))
            else:
                lista_aux.append(0)
        lista_wtf.append(lista_aux)
        lista_aux = []


def calcular_df(lista_wtf, lista_df, vocabulario):
    cont = 0
    index = 0
    for rep in range(len(vocabulario)):
        for lista in lista_wtf:
            if (lista[index] > 0):
                cont += 1
        index += 1
        lista_df.append(cont)
        cont = 0


def calcular_idf(lista_df, abstract, lista_idf):
    for dato in lista_df:
        # lista_idf.append(round(math.log(3/dato,10),2))
        if (dato == 0):
            dato = 1
        lista_idf.append(round(math.log(len(abstract) / dato, 10), 2))


def calcular_Tf_Idf(lista_idf, lista_wtf, lista_tf_idf):
    for lista in lista_wtf:
        lista_tf_idf.append(np.multiply(lista, lista_idf))


def redondear(lista_tf_idf):
    lista = []
    lista_aux = []
    for i in range(len(lista_tf_idf)):
        for j in range(len(lista_tf_idf[i])):
            lista_aux.append(round(lista_tf_idf[i][j], 2))
        lista.append(lista_aux)
        lista_aux = []
    return lista


def modulo_raiz(lista_wtf, lista_modulo, vocabulario):
    acum = 0

    for lista in lista_wtf:
        for dato in lista:
            if (dato > 0):
                acum = acum + dato ** 2
        lista_modulo.append(round(math.sqrt(acum), 2))
        acum = 0


def lista_normalizada(lista_wtf, lista_modulo, lista_normal):
    indice = 0
    for lista in lista_wtf:
        if (lista_modulo[indice] == 0):
            lista_modulo[indice] = 1
        lista_normal.append(list(map(lambda x: x / lista_modulo[indice], lista)))

        indice += 1


def coseno_vectorial(bolsa_de_palabras, dataset):
    lista_similitud_cat = []
    for i in range(len(bolsa_de_palabras)):
        lista_doc = []
        lista_enfoque = []
        lista_df = []
        lista_wtf = []
        lista_idf = []
        lista_tf_idf = []
        # Tf
        frecuencias(bolsa_de_palabras[i], dataset, lista_enfoque)
        # WTF
        calcular_wtf(lista_enfoque, lista_wtf)
        # DF
        calcular_df(lista_wtf, lista_df, bolsa_de_palabras[i])
        # IDF
        calcular_idf(lista_df, dataset, lista_idf)
        # TF-IDF
        calcular_Tf_Idf(lista_idf, lista_wtf, lista_tf_idf)
        lista_modulo = []
        lista_normal = []
        modulo_raiz(lista_wtf, lista_modulo, bolsa_de_palabras[i])
        lista_normalizada(lista_wtf, lista_modulo, lista_normal)
        lista_normal = redondear(lista_normal)

        for lista in lista_normal:
            lista_doc.append(round((sum(lista) / len(bolsa_de_palabras[i])), 2))
        lista_similitud_cat.append(lista_doc)
    return lista_similitud_cat


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/archivos'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
archivo_selec = False
url_archivo = "static/archivos/server_interno.csv"

activador = 0

bolsa_enfoque = ["BIO MEDICO", "PSICOSOCIAL - COMUNITARIO", "COTIDIANO"];
tam_enfoque = len(bolsa_enfoque)

link_bolsa = "https://raw.githubusercontent.com/Freddy8-C/Proyecto_MachineLearning2/master/BOLSA%20DE%20PALABRAS%203%20MODELOS%20PARA%20CATIA.csv?token=GHSAT0AAAAAABWHCJVIARVPDV22QAW7UQ3EYWZ7Z2A"
matriz = [0, 0]
bolsa_de_palabras = []
bio_medico = importarDatosColumna("A. MODELO BIO MEDICO", link_bolsa)
psicosocial_com = importarDatosColumna("B. ENFOQUE PSICOSOCIAL - COMUNITARIO", link_bolsa)
psicosocial_com = eliminarFilasVacias(psicosocial_com)
cotidiano = importarDatosColumna("C. ENFOQUE COTIDIANO", link_bolsa)
cotidiano = eliminarFilasVacias(cotidiano)

# Normalizacion
bio_medico = convertirMayusculasEnMinusculas(bio_medico)
bio_medico = caracter_especiales(bio_medico)
psicosocial_com = convertirMayusculasEnMinusculas(psicosocial_com)
psicosocial_com = caracter_especiales(psicosocial_com)
cotidiano = convertirMayusculasEnMinusculas(cotidiano)
cotidiano = caracter_especiales(cotidiano)

# Tokenizacion
bio_medico = tokenizacion(bio_medico)
bio_medico = eliminar_stop_words(bio_medico)
psicosocial_com = tokenizacion(psicosocial_com)
psicosocial_com = eliminar_stop_words(psicosocial_com)
cotidiano = tokenizacion(cotidiano)
cotidiano = eliminar_stop_words(cotidiano)

# Stemming
bio_medico = stemming(bio_medico)
psicosocial_com = stemming(psicosocial_com)
cotidiano = stemming(cotidiano)
bio_medico = list(itertools.chain(*bio_medico))
bio_medico = eliminar_palabras_repetidas(bio_medico)
psicosocial_com = list(itertools.chain(*psicosocial_com))
psicosocial_com = eliminar_palabras_repetidas(psicosocial_com)
cotidiano = list(itertools.chain(*cotidiano))
cotidiano = eliminar_palabras_repetidas(cotidiano)
bolsa_de_palabras = [bio_medico, psicosocial_com, cotidiano]


def datos_listos_bolsa(documentos):
    global bolsa_de_palabras
    global matriz_jaccard
    global activador
    lista_similitud_cat = []
    dataset = []
    tam = 0
    if (activador == 0):
        # link_dataset = "https://raw.githubusercontent.com/Freddy8-C/Proyecto_MachineLearning2/master/EXCEL%20DE%20VACIADO%20COMPLETO%20DE%20ENTREVISTAS%20PROFESIONALES%20MAYO2021.csv?token=GHSAT0AAAAAABWHCJVIIRXVIMY3XQOTUTWSYWZ7Z3Q"
        # dataset= importarDatosColumna("P7. ¿Qué entiende por demencia?", link_dataset)
        # dataset =  eliminarFilasVacias(dataset)
        dataset.append(documentos)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = caracter_especiales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminar_stop_words(dataset)
        dataset = stemming(dataset)
        matriz_jacard = np.zeros((3, len(dataset)))
        matriz_jacard = jacard_res(bolsa_de_palabras, dataset)
        lista_similitud_cat = coseno_vectorial(bolsa_de_palabras, dataset)

    if (activador == 1):
        link_dataset = "https://raw.githubusercontent.com/Freddy8-C/Proyecto_MachineLearning2/master/EXCEL%20DE%20VACIADO%20COMPLETO%20DE%20ENTREVISTAS%20PROFESIONALES%20MAYO2021.csv?token=GHSAT0AAAAAABWHCJVIIRXVIMY3XQOTUTWSYWZ7Z3Q"
        dataset = importarDatosColumna("P7. ¿Qué entiende por demencia?", link_dataset)
        dataset = eliminarFilasVacias(dataset)
        # dataset.append(documentos)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = caracter_especiales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminar_stop_words(dataset)
        dataset = stemming(dataset)
        matriz_jacard = np.zeros((3, len(dataset)))
        matriz_jacard = jacard_res(bolsa_de_palabras, dataset)
        lista_similitud_cat = coseno_vectorial(bolsa_de_palabras, dataset)
    if (activador == 2):
        # link_dataset = "https://raw.githubusercontent.com/Freddy8-C/Proyecto_MachineLearning2/master/EXCEL%20DE%20VACIADO%20COMPLETO%20DE%20ENTREVISTAS%20PROFESIONALES%20MAYO2021.csv?token=GHSAT0AAAAAABWHCJVIIRXVIMY3XQOTUTWSYWZ7Z3Q"
        dataset = importarDatosColumna(documentos, "static/archivos/server_interno.csv")
        dataset = eliminarFilasVacias(dataset)
        # dataset.append(documentos)
        dataset = convertirMayusculasEnMinusculas(dataset)
        dataset = caracter_especiales(dataset)
        dataset = tokenizacion(dataset)
        dataset = eliminar_stop_words(dataset)
        dataset = stemming(dataset)
        matriz_jacard = np.zeros((3, len(dataset)))
        matriz_jacard = jacard_res(bolsa_de_palabras, dataset)
        tam = len(dataset)
        lista_similitud_cat = coseno_vectorial(bolsa_de_palabras, dataset)
    return [matriz_jacard, lista_similitud_cat, tam]

lista = []

def normalizacionDatosSimilitud(matriz):
    print('matriz',matriz)
    matrizDatosNormalizados = []
    sumaTotal = ((matriz[0])[0])+((matriz[1])[0])+((matriz[2])[0])
    print('suma total',sumaTotal)
    if (matriz[0])[0] != 0:
        dato0normalizado = (((matriz[0])[0])*100)/sumaTotal
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
    data = json.loads(request.data.decode())
    url = data["definicion"]
    global activador
    activador = 0
    global matriz
    matriz = datos_listos_bolsa(url)

    matriz1normalizado = normalizacionDatosSimilitud(matriz[0])
    matriz2normalizado = normalizacionDatosSimilitud(matriz[1])

    dto = json.dumps({'jaccard': [
        {'enfoque':'BIO MEDICO', 'porcentaje': matriz1normalizado[0]},
        {'enfoque':'PSICOSOCIAL - COMUNITARIO', 'porcentaje': matriz1normalizado[1]},
        {'enfoque':'COTIDIANO', 'porcentaje': matriz1normalizado[2]}
    ], 'coseno': [
        {'enfoque':'BIO MEDICO', 'porcentaje': matriz2normalizado[0]},
        {'enfoque':'PSICOSOCIAL - COMUNITARIO', 'porcentaje': matriz2normalizado[1]},
        {'enfoque':'COTIDIANO', 'porcentaje': matriz2normalizado[2]}
    ]})
    return dto

@app.route('/api/obtener-dataset', methods=['POST'])
def obtenerDataset():
    data = json.loads(request.data.decode())
    url = data["valor"]
    global activador
    activador = 1
    matriz = datos_listos_bolsa("")
    dto_jaccard = [];
    dto_coseno = [];
    matriz_jaccard_previa = matriz[0].tolist()
    matriz_jaccard = matriz_jaccard_previa[url]
    matriz_coseno_previa = matriz[1]
    matriz_coseno = matriz_coseno_previa[url]

    for i in range(len(matriz_jaccard)):
        dto_jaccard.append({
            'id': i+1, 'porcentaje': matriz_jaccard[i]
        })

    for i in range(len(matriz_coseno)):
        dto_coseno.append({
            'id': i+1, 'porcentaje': matriz_coseno[i]
        })

    dto = json.dumps({'jaccard': dto_jaccard, 'coseno': dto_coseno})
    return dto

@app.route('/api/subir-dataset', methods=['POST'])
def subirArchivoCSV():
    f = request.files['file']
    f.save('assets/dataset.csv')
    uploaded_df = pd.read_csv('assets/dataset.csv')
    headers = uploaded_df.columns.tolist()
    vectorColumnas = []
    for i in range(len(headers)):
        vectorColumnas.append({'id': i, 'valor': headers[i]})
    jsonData = json.dumps(vectorColumnas)
    return jsonData

@app.route('/api/consultar-similitud-dataset', methods=['POST'])
def consultarSimilitudDataset():
    data = json.loads(request.data.decode())
    columna = data["valor"]
    uploaded_df = pd.read_csv('assets/dataset.csv')
    definicion = uploaded_df.iat[0, columna]
    matriz = datos_listos_bolsa(definicion)
    print(matriz)

    matriz1normalizado = normalizacionDatosSimilitud(matriz[0])
    matriz2normalizado = normalizacionDatosSimilitud(matriz[1])

    dto = json.dumps({'jaccard': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': matriz1normalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': matriz1normalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': matriz1normalizado[2]}
    ], 'coseno': [
        {'enfoque': 'BIO MEDICO', 'porcentaje': matriz2normalizado[0]},
        {'enfoque': 'PSICOSOCIAL - COMUNITARIO', 'porcentaje': matriz2normalizado[1]},
        {'enfoque': 'COTIDIANO', 'porcentaje': matriz2normalizado[2]}
    ]})
    os.remove('assets/dataset.csv')
    return dto

###############################
@app.route('/Subir_demencia.php')
def demencia():
    global lista
    global activador
    global matriz
    global bolsa_enfoque
    act = 0;

    activador = 2
    return render_template("Subir_demencia.php", activador=activador, tam_enfoque=tam_enfoque,
                           bolsa_enfoque=bolsa_enfoque, matriz_jaccard=matriz[0], lista_tf_idf=matriz[1], lista=lista,
                           act=act)


def mostrar_cabeceras():
    df = pd.read_csv("static/archivos/server_interno.csv")  # ===> Incluir las cabeceras
    import_headers = df.axes[1]  # ==> 1 es para identificar las columnas
    return list(import_headers)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    global bolsa_de_palabras
    global matriz_jaccard
    global activador
    activador = 2
    global matriz
    global archivo_selec
    global lista
    act = 0;
    if (request.method == "POST"):
        f = request.files["upfile"]
        archivo_selec = True
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], "server_interno.csv"))
        lista = mostrar_cabeceras()
        return render_template("Subir_demencia.php", activador=activador, tam_enfoque=tam_enfoque,
                               bolsa_enfoque=bolsa_enfoque, matriz_jaccard=matriz[0], lista_tf_idf=matriz[1],
                               lista=lista, act=act)


@app.route('/Subir_demencia.php', methods=["GET", "POST"])
def subir():
    global activador

    activador = 2
    global matriz
    act = 1;
    columna = request.form['combo']
    matriz = datos_listos_bolsa(columna)
    tamanio = matriz[2]
    return render_template("Subir_demencia.php", activador=activador, tam_enfoque=tam_enfoque,
                           bolsa_enfoque=bolsa_enfoque, matriz_jaccard=matriz[0], lista_tf_idf=matriz[1], act=act,
                           tamanio=tamanio)
