import csv
import logging
import time
import unicodedata
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

#Função que verifica se uma string contém dígito
def contem_digitos(string):
    for char in string:
        if char.isdigit():
            return True
    return False

#Configurar o LOG
inicio_geral = time.time()
logging.basicConfig(filename="../LOG/GLI.LOG", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Início do programa.")

#Baixar lista de stopwords do NLTK
nltk.download("stopwords")
stopwords = set(stopwords.words("english"))

#Ler o arquivo GLI.CFG
logging.info("Lendo arquivo de configuração...")
with open("GLI.CFG", "r") as f:  
    conteudo = f.read()
    linhas = conteudo.split("\n")
    arquivos_leia = []
    for linha in linhas:
        if linha.startswith("LEIA"):
            while True:
                linha = linhas.pop(0)
                if linha == "" or linha.startswith("ESCREVA"):
                    break
                linha = linha.replace("LEIA=","")
                arquivos_leia.append(linha)
        if linha.startswith("ESCREVA"):
            arquivo_saida = linha.replace("ESCREVA=","")
logging.info("Arquivo de configuração lido com sucesso.")

logging.info("Início da leitura e processamento dos dados.")
inicio_proc = time.time()
dicionario = {}
cont_arq = 0
cont_doc = 0
for arquivo in arquivos_leia:
    cont_arq += 1
    arvore = ET.parse(arquivo)
    raiz = arvore.getroot()

    #Extrair as informações dos documentos
    for documento in raiz.findall("RECORD"):
        cont_doc += 1
        record_num = documento.find("RECORDNUM").text
        record_num = record_num.replace(" ","").lstrip("0")
        if documento.find("ABSTRACT") is not None:
            texto_base = documento.find("ABSTRACT").text
        elif documento.find("EXTRACT") is not None:
            texto_base = documento.find("EXTRACT").text
        else:
            texto_base = ""
                
        #Tratar o texto base
        texto_base = texto_base.replace("\"","")
        texto_base = texto_base.replace("\'","")
        texto_base = texto_base.replace("(","")
        texto_base = texto_base.replace(")","")
        texto_base = texto_base.replace("\n"," ")
        texto_base = texto_base.replace(".","")
        texto_base = texto_base.replace(";","")
        texto_base = texto_base.replace("-"," ")
        texto_base = texto_base.replace(":","")
        texto_base = texto_base.replace(",","")
        texto_base = texto_base.replace("!","")
        texto_base = texto_base.replace("?","")
        texto_base = texto_base.replace("/","")
        texto_base = texto_base.replace("[","")
        texto_base = texto_base.replace("]","")
        texto_base = texto_base.replace(">","")
        texto_base = texto_base.replace("<","")
        texto_base = texto_base.replace("+","")
        texto_base = texto_base.replace("%","")
        texto_base = texto_base.upper()
        tokens = word_tokenize(texto_base)
        tokens_filtrados = [w for w in tokens if not contem_digitos(w) and not w.lower() in stopwords and len(w) > 1]
        texto_base = " ".join(tokens_filtrados)
        "".join(c for c in unicodedata.normalize("NFD", texto_base) if unicodedata.category(c) != "Mn")
        texto_base = unidecode(texto_base)
        for palavra in texto_base.split():
            
            #Adicionar a palavra ao dicionário ou atualizar a lista de documentos em que ela aparece
            if palavra in dicionario:
                dicionario[palavra].append(int(record_num))
            else:
                dicionario[palavra] = [int(record_num)]
tempo_proc = time.time() - inicio_proc
tempo_medio = tempo_proc / cont_doc
logging.info(f"Fim do processamento dos dados. Processados {cont_arq} arquivos, {cont_doc} documentos e extraídos {len(dicionario)} termos em {tempo_proc} segundos (tempo médio por documento {tempo_medio} segundos).")

#Criar o arquivo CSV
logging.info("Início da gravação dos dados.")
inicio_grav = time.time()
with open(arquivo_saida, "w", newline="") as arquivo_csv_saida:
    writer = csv.writer(arquivo_csv_saida, delimiter=";")

    #Ler o dicionário de palavras e escrever cada linha no arquivo CSV
    for palavra, record_num in dicionario.items():
        writer.writerow([palavra, record_num])
tempo_grav = time.time() - inicio_grav
tempo_medio_grav = tempo_grav / len(dicionario)
logging.info(f"Fim da gravação dos dados. Gravados {len(dicionario)} registros em {tempo_grav} segundos (tempo médio por registro {tempo_medio_grav} segundos).")
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")
