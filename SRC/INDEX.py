import configparser
import csv
import logging
import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict

#Função para normalização dos termos

#Configurar o LOG
inicio_geral = time.time()
logging.basicConfig(filename="../LOG/INDEX.LOG", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Início do programa.")

#Ler o arquivo INDEX.CFG
logging.info("Lendo arquivo de configuração...")
config = configparser.ConfigParser()
config.read("INDEX.CFG")
logging.info("Arquivo de configuração lido com sucesso.")

#Leitura do arquivo CSV com a lista invertida de termos
logging.info("Início da leitura da lista invertida.")
inicio_li = time.time()
df = pd.read_csv(config["INSTRUCOES"]["LEIA"], header=None, names=["termo", "documento"], sep=";")
        
#Criar um dicionário para armazenar os termos e seus documentos
dicionario_termos = {}

#Percorrer todas as linhas do dataframe
documentos = []
count_rows = 0
for index, row in df.iterrows():
    count_rows += 1
    termo = str(row["termo"])
    documento = eval(row["documento"])
    documentos.extend(documento)
    #Adicionar o documento à lista de documentos do termo
    if termo not in dicionario_termos:
        dicionario_termos[termo] = []
    dicionario_termos[termo].extend(documento)
documentos = sorted(list(set(documentos)))

#Obter a lista de termos
termos = sorted(list(dicionario_termos.keys()))
tempo_li = time.time() - inicio_li
tempo_medio_li = tempo_li / count_rows
logging.info(f"Lista invertida lida com sucesso. Processados {count_rows} termos - tempo médio de {tempo_medio_li} segundos.")

#Construir a matriz termo-documento
logging.info("Início da criação da matriz termo_documento com tf-idf.")
inicio_mtd = time.time()
matriz_termo_documento = []
for termo in termos:
    linha = []
    for documento in documentos:
        linha.append(dicionario_termos[termo].count(documento))
    matriz_termo_documento.append(linha)

#Cálculo do TFmax
tfmax = dict.fromkeys(documentos)
matriz_termo_documento = np.array(matriz_termo_documento)
for coluna in range(matriz_termo_documento.shape[1]):
    valor_max = np.amax(matriz_termo_documento[:, coluna])
    tfmax[documentos[coluna]] = valor_max

#Cálculo do tf-idf
matriz_termo_documento = matriz_termo_documento.tolist()
N = len(documentos)
matriz_tfidf = []
for i in range(len(matriz_termo_documento)):
    matriz_tfidf.append([])
    n_k = len(matriz_termo_documento[i]) - matriz_termo_documento[i].count(0)
    for j in range(len(matriz_termo_documento[i])):
        tf = matriz_termo_documento[i][j]
        tf_max = int(tfmax[documentos[j]])
        tfidf = (tf / tf_max) * np.log10(N / n_k)
        tfidf = tfidf.round(2)
        matriz_tfidf[i].append(tfidf)
tempo_mtd = time.time() - inicio_mtd
tempo_medio_mtd = tempo_mtd / ((i + 1) * (j + 1))
logging.info(f"Matriz termo_documento criada com sucesso. Processadas {i+1} linhas e {j+1} colunas em {tempo_mtd} segundos - tempo médio por item de {tempo_medio_mtd} segundos.")

#Salvar tfidf, lista de termos e lista de documentos em arquivos
logging.info("Início da gravação dos dados do modelo em arquivo.")
inicio_grav = time.time()
matriz_tfidf = np.array(matriz_tfidf)
np.savez(config["INSTRUCOES"]["ESCREVA"], matriz_tfidf=matriz_tfidf, termos=termos, documentos=documentos)
tempo_grav = time.time() - inicio_grav
conta_linha = matriz_tfidf.shape[0] + 2
tempo_medio_grav = tempo_grav / conta_linha
logging.info(f"Gravação dos dados do modelo concluída. Gravadas {conta_linha} linhas em {tempo_grav} segundos - tempo médio de {tempo_medio_grav} segundos.")
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")
