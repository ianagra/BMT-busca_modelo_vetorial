import configparser
import csv
import logging
import pandas as pd
import numpy as np
import time
import re
import nltk
import unicodedata
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

#Função para verificar se uma string contém dígito
def contem_digitos(string):
    for char in string:
        if char.isdigit():
            return True
    return False

#Configurar o LOG
inicio_geral = time.time()
logging.basicConfig(filename="../LOG/BUSCA.LOG", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Início do programa.")

nltk.download("stopwords")
stopwords = set(stopwords.words("english"))

#Ler o arquivo BUSCA.CFG
logging.info("Lendo arquivo de configuração...")
config = configparser.ConfigParser()
config.read("BUSCA.CFG")
logging.info("Arquivo de configuração lido com sucesso.")

#Ler a matriz termo-documento e as listas de termos e documentos
logging.info("Lendo arquivo do modelo...")
inicio_ler_modelo = time.time()
modelo = np.load(config["INSTRUCOES"]["MODELO"])
matriz_tfidf = np.array(modelo["matriz_tfidf"]).T
termos = modelo["termos"]
documentos = modelo["documentos"]
tempo_ler_modelo = time.time() - inicio_ler_modelo
qtd_termos = len(termos)
qtd_docs = len(documentos)
logging.info(f"Arquivo do modelo lido com sucesso. Processados {qtd_termos} termos e {qtd_docs} documentos em {tempo_ler_modelo} segundos")

#Ler arquivo com as consultas e criar um dicionário
logging.info("Obtendo os vetores de termos dos documentos e das consultas")
inicio_obter_vetores = time.time()
consultas = {}
df_consultas = pd.read_csv(config["INSTRUCOES"]["CONSULTAS"], header=0, sep=";")
termos = termos.tolist()
for index, linha in df_consultas.iterrows():
    texto_consulta = str(linha["QueryText"])
    #Tratar o texto da consulta
    texto_consulta = texto_consulta.replace("\"","")
    texto_consulta = texto_consulta.replace("\'","")
    texto_consulta = texto_consulta.replace("(","")
    texto_consulta = texto_consulta.replace(")","")
    texto_consulta = texto_consulta.replace("\n"," ")
    texto_consulta = texto_consulta.replace(".","")
    texto_consulta = texto_consulta.replace(";","")
    texto_consulta = texto_consulta.replace("-"," ")
    texto_consulta = texto_consulta.replace(":","")
    texto_consulta = texto_consulta.replace(",","")
    texto_consulta = texto_consulta.replace("!","")
    texto_consulta = texto_consulta.replace("?","")
    texto_consulta = texto_consulta.replace("/","")
    texto_consulta = texto_consulta.replace("[","")
    texto_consulta = texto_consulta.replace("]","")
    texto_consulta = texto_consulta.replace(">","")
    texto_consulta = texto_consulta.replace("<","")
    texto_consulta = texto_consulta.replace("+","")
    texto_consulta = texto_consulta.replace("%","")
    texto_consulta = texto_consulta.upper()
    tokens = word_tokenize(texto_consulta)
    tokens_filtrados = [w for w in tokens if not contem_digitos(w) and not w.lower() in stopwords and len(w) > 1]
    texto_consulta = " ".join(tokens_filtrados)
    "".join(c for c in unicodedata.normalize("NFD", texto_consulta) if unicodedata.category(c) != "Mn")
    texto_consulta = unidecode(texto_consulta)
    consultas[linha["QueryNumber"]] = [0]*(len(termos))
    for palavra in texto_consulta.split():
        #Adicionar a palavra ao dicionário ou atualizar a lista de documentos em que ela aparece
        if palavra in termos:
            pos = termos.index(palavra)
            consultas[linha["QueryNumber"]][pos] = 1
        else:
            termos.append(palavra)
            coluna_zeros = np.zeros((matriz_tfidf.shape[0], 1))
            matriz_tfidf = np.column_stack((matriz_tfidf, coluna_zeros))
            consultas[linha["QueryNumber"]].append(1)

#Igualar o número de itens nos vetores das consultas
tamanho_maximo = max(len(lista) for lista in consultas.values())
for chave, lista in consultas.items():
    while len(lista) < tamanho_maximo:
        lista.append(0)
tempo_obter_vetores = time.time() - inicio_obter_vetores
logging.info(f"Processados {len(termos)} termos de {matriz_tfidf.shape[0]} documentos e {len(consultas.keys())} consultas em {tempo_obter_vetores} segundos")

#Calcular a distância entre cada vetor de consulta e cada vetor de documento pelo cosseno do ângulo entre eles
lista_consultas = consultas.keys()
qtd_consultas = len(lista_consultas)
matriz_distancias = np.zeros((qtd_consultas, qtd_docs), dtype=float)
indice_consulta = 0
for consulta in consultas.keys():
    q = np.array(consultas[consulta])
    for i in range(matriz_tfidf.shape[0]):
        d = matriz_tfidf[i, :]
        produto_escalar = np.dot(d, q)
        modulo_q = np.linalg.norm(q)
        modulo_d = np.linalg.norm(d)
        cos = produto_escalar / (modulo_q * modulo_d)
        if cos > 0:
            matriz_distancias[indice_consulta, i] = cos
    indice_consulta += 1

#Criar o arquivo CSV
logging.info("Início da gravação dos dados.")
inicio_grav = time.time()
with open(config["INSTRUCOES"]["RESULTADOS"], "w", newline="") as arquivo_saida:
    writer = csv.writer(arquivo_saida, delimiter=";")
    linha = 0
    for consulta in lista_consultas:
        ranking = 1
        ordenada = sorted(matriz_distancias[linha], reverse=True)
        for dist in ordenada:
            if dist > 0:
                lista_resultado = []
                posicao = np.where(matriz_distancias[linha] == dist)[0][0]
                lista_resultado.append(ranking)
                lista_resultado.append(documentos[posicao])
                lista_resultado.append(dist)
                writer.writerow([consulta,lista_resultado])
                ranking += 1
tempo_grav = time.time() - inicio_grav
logging.info("Fim da gravação dos dados. Gravados {} registros em {tempo_grav} segundos (tempo médio por registro {tempo_medio_grav} segundos).")
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")

