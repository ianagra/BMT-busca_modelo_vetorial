import configparser
import csv
import logging
import nltk
import time
import unicodedata
import xml.etree.ElementTree as ET
from unidecode import unidecode

#Configurar o LOG
inicio_geral = time.time()
logging.basicConfig(filename="../LOG/PC.LOG", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Início do programa.")

#Ler o arquivo PC.CFG
logging.info("Lendo arquivo de configuração...")
config = configparser.ConfigParser()
config.read("PC.CFG")
logging.info("Arquivo de configuração lido com sucesso.")

#Carregar o arquivo XML indicado
logging.info("Lendo arquivo de dados...")
arvore = ET.parse(config["INSTRUCOES"]["LEIA"])
raiz = arvore.getroot()
logging.info("Arquivo de dados lido com sucesso.")

#Criar arquivo CSV para CONSULTAS
with open(config["INSTRUCOES"]["CONSULTAS"], "w", newline="") as consultas:
    writer = csv.writer(consultas, delimiter=";")
    writer.writerow(["QueryNumber", "QueryText"])

    #Extrair o número da consulta e o texto do arquivo XML
    cont_consultas = 1
    logging.info("Início do processamento para geração do arquivo CONSULTAS.")
    inicio_1 = time.time()
    for query in raiz.findall("QUERY"):
        cont_consultas += 1
        query_number = query.find("QueryNumber").text.lstrip("0")
        query_text = query.find("QueryText").text

        #Tratar o texto
        query_text = query_text.replace("\"","")
        query_text = query_text.replace("\n","")
        query_text = query_text.replace(";","")
        palavras = query_text.split()
        query_text = " ".join(palavras)
        "".join(c for c in unicodedata.normalize("NFD", query_text) if unicodedata.category(c) != "Mn")
        query_text = unidecode(query_text.upper())
                 
        #Escrever a linha no arquivo CSV
        writer.writerow([query_number, query_text])
    tempo_1 = time.time() - inicio_1
    tempo_medio_1 = tempo_1 / cont_consultas
    logging.info(f"Fim do processamento de CONSULTAS - processadas {cont_consultas} consultas em {tempo_1} segundos (tempo médio por consulta {tempo_medio_1} segundos.")

#Criar arquivo CSV para ESPERADOS
with open(config["INSTRUCOES"]["ESPERADOS"], "w", newline="") as esperados:
    writer = csv.writer(esperados, delimiter=";")
    writer.writerow(["QueryNumber", "DocNumber", "DocVotes"])

    #Extrair o número da consulta, o número dos documentos e a quantidade de votos
    cont_esperados = 1
    logging.info("Início do processamento para geração do arquivo ESPERADOS.")
    inicio_2 = time.time()
    for query in raiz.findall("QUERY"):
        query_number = query.find("QueryNumber").text.lstrip("0")
        for item in query.findall('./Records/Item'):
            cont_esperados += 1
            doc_number = item.text.lstrip("0")
            score = str(item.get("score"))
            doc_votes = 0
            for char in score:
                if char != "0":
                    doc_votes += 1
                    
            #Escrever a linha no arquivo CSV
            writer.writerow([query_number, doc_number, doc_votes])
            
    tempo_2 = time.time() - inicio_2
    tempo_medio_2 = tempo_2 / cont_esperados
    logging.info(f"Fim do processamento de ESPERADOS - processados {cont_esperados} registros em {tempo_2} segundos (tempo médio por registro {tempo_medio_2} segundos.")
            
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")
