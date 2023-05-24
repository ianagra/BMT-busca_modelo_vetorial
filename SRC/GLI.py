import csv
import logging
import time
import unicodedata
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

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
    stemmer = False
    nostemmer = False
    for linha in linhas:
        if linha == "STEMMER":
            stemmer = True
        elif linha == "NOSTEMMER":
            nostemmer = True
        elif linha.startswith("LEIA"):
            linha = linha.replace("LEIA=","")
            arquivos_leia.append(linha)
        elif linha.startswith("ESCREVA-STEMMER"):
            if stemmer:
                arquivo_saida = linha.replace("ESCREVA-STEMMER=","")
                break
            else:
                continue
        elif linha.startswith("ESCREVA-NOSTEMMER"):
            if not stemmer:
                arquivo_saida = linha.replace("ESCREVA-NOSTEMMER=","")
                break
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
            if stemmer:
                palavra = palavra.lower()
                ult_char = (len(palavra) - 1)
                palavra = PorterStemmer().stem(palavra, 0, ult_char)
                
            #Adicionar a palavra ao dicionário ou atualizar a lista de documentos em que ela aparece
            palavra = palavra.upper()
            if palavra in dicionario:
                dicionario[palavra].append(int(record_num))
            else:
                dicionario[palavra] = [int(record_num)]
        
tempo_proc = time.time() - inicio_proc
tempo_medio = tempo_proc / cont_doc
logging.info(f"Fim do processamento dos dados. Processados {cont_arq} arquivos, {cont_doc} documentos e extraídos {len(dicionario)} termos em {tempo_proc} segundos (tempo médio por documento {tempo_medio} segundos).")

#Criar os arquivos CSV
logging.info("Início da gravação dos dados.")
inicio_grav = time.time()
with open(arquivo_saida, "w", newline="") as arquivo_csv_saida:
    writer = csv.writer(arquivo_csv_saida, delimiter=";")

    #Ler o dicionário de palavras e escrever cada linha no arquivo CSV
    for palavra, record_num in dicionario.items():
        writer.writerow([palavra, record_num])

tempo_grav = time.time() - inicio_grav
tempo_medio_grav = tempo_grav / (len(dicionario))
logging.info(f"Fim da gravação dos dados. Gravados {len(dicionario)} registros em {tempo_grav} segundos (tempo médio por registro {tempo_medio_grav} segundos).")
tempo_total = time.time() - inicio_geral
logging.info(f"Fim do programa. Tempo total {tempo_total} segundos.")
