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

#Stemmer de Porter importado de http://tartarus.org/martin/PorterStemmer/
class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]

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
stemmer = False
if config.has_section("STEMMER"):
    stemmer = True
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
        palavra = palavra.upper()
        if stemmer:
                palavra = palavra.lower()
                ult_char = (len(palavra) - 1)
                palavra = PorterStemmer().stem(palavra, 0, ult_char)
                palavra = palavra.upper()
        #Adicionar a palavra ao dicionário ou atualizar a lista de documentos em que ela aparece
        if palavra in termos:
            pos = termos.index(palavra)
            consultas[linha["QueryNumber"]][pos] = 1
        elif palavra not in termos:
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
res = "RESULTADOS-NOSTEMMER"
if stemmer:
    res = "RESULTADOS-STEMMER"
with open(config["INSTRUCOES"][res], "w", newline="") as arquivo_saida:
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
        linha += 1
tempo_grav = time.time() - inicio_grav
logging.info("Fim da gravação dos dados. Gravados {} registros em {tempo_grav} segundos (tempo médio por registro {tempo_medio_grav} segundos).")
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")

