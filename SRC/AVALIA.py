import time
import logging
import configparser
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

#Configurar o LOG
inicio_geral = time.time()
logging.basicConfig(filename="../LOG/AVALIA.LOG", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Início do programa.")

#Ler o arquivo PC.CFG
logging.info("Lendo arquivo de configuração...")
config = configparser.ConfigParser()
config.read("AVALIA.CFG")
logging.info("Arquivo de configuração lido com sucesso.")

# Leitura dos arquivos de resultados com e sem o uso do stemmer
df_resultados_stemmer = pd.read_csv(config["INSTRUCOES"]["RESULTADOS-STEMMER"], header=None, sep=";")
df_resultados_stemmer[["DocRanking", "DocNumber", "Cos"]] = df_resultados_stemmer[1].str.split(', ', expand=True)
df_resultados_stemmer["DocRanking"] = df_resultados_stemmer["DocRanking"].str[1]
df_resultados_stemmer.columns = ["QueryNumber", "Result", "DocRanking", "DocNumber", "Cos"]
df_resultados_stemmer['QueryNumber'] = df_resultados_stemmer['QueryNumber'].astype(int)
df_resultados_stemmer['DocRanking'] = df_resultados_stemmer['DocRanking'].astype(int)
df_resultados_stemmer['DocNumber'] = df_resultados_stemmer['DocNumber'].astype(int)

df_resultados_nostemmer = pd.read_csv(config["INSTRUCOES"]["RESULTADOS-NOSTEMMER"], header=None, sep=";")
df_resultados_nostemmer[["DocRanking", "DocNumber", "Cos"]] = df_resultados_nostemmer[1].str.split(', ', expand=True)
df_resultados_nostemmer["DocRanking"] = df_resultados_nostemmer["DocRanking"].str[1]
df_resultados_nostemmer.columns = ["QueryNumber", "Result", "DocRanking", "DocNumber", "Cos"]
df_resultados_nostemmer['QueryNumber'] = df_resultados_nostemmer['QueryNumber'].astype(int)
df_resultados_nostemmer['DocRanking'] = df_resultados_nostemmer['DocRanking'].astype(int)
df_resultados_nostemmer['DocNumber'] = df_resultados_nostemmer['DocNumber'].astype(int)

# Leitura do arquivo de resultados esperados
df_esperados = pd.read_csv(config["INSTRUCOES"]["ESPERADOS"], header=0, sep=";")
df_esperados.columns = ["QueryNumber", "DocNumber", "DocVotes"]
df_esperados['QueryNumber'] = df_esperados['QueryNumber'].astype(int)
df_esperados['DocVotes'] = df_esperados['DocVotes'].astype(int)
df_esperados['DocNumber'] = df_esperados['DocNumber'].astype(int)

# Obtenção de uma lista de códigos de consultas
consultas = list(df_resultados_nostemmer["QueryNumber"].unique())


### Avaliação do algoritmo sem stemmer
# Inicialização de variaveis
precisao_revocacao = []
precision_at_5 = 0
precision_at_10 = 0
f_1 = 0.0
map_score = 0
mrr_score = 0
dcg_score = 0.0
ndcg_score = 0.0
matriz_precisoes = []
revocacoes = []
r_precision = []

# Iterando as consultas
for id_consulta in consultas:

    # Filtrar resultados e resultados esperados para a consulta atual
    resultados = df_resultados_nostemmer[df_resultados_nostemmer["QueryNumber"] == id_consulta]
    esperados = df_esperados[df_esperados["QueryNumber"] == id_consulta]
    
    # Obter os IDs de documentos relevantes e recuperados para a consulta atual
    docs_relevantes = list(map(int, esperados["DocNumber"].to_list()))
    num_relevantes = len(docs_relevantes)
    docs_recuperados = list(map(int, resultados["DocNumber"].to_list()))
    num_recuperados = len(docs_recuperados)
    
    # Precision@5 e Precision@10
    lista_5 = list(map(int, resultados["DocNumber"].head(5).to_list()))
    lista_10 = list(map(int, resultados["DocNumber"].head(10).to_list()))
    precision_at_5 += len([doc_id for doc_id in lista_5 if doc_id in docs_relevantes])
    precision_at_10 += len([doc_id for doc_id in lista_10 if doc_id in docs_relevantes])

    # R-Precision
    lista_r = list(map(int, resultados["DocNumber"].head(num_relevantes).to_list()))
    r_precision.append(len([doc_id for doc_id in lista_r if doc_id in docs_relevantes]))
    
    # DCG e NDCG
    dcg = 0.0
    ideal_dcg = 0.0
    df_filtered = resultados[resultados['DocRanking'] <= 10].copy()

    # Lista para armazenar os valores de DCG e NDCG para cada consulta
    relevantes = esperados[esperados["QueryNumber"] == id_consulta]
    relevant_dict = dict(zip(esperados['DocNumber'], esperados['DocVotes']))
        
    rankings = df_filtered[df_filtered['QueryNumber'] == id_consulta]['DocNumber'].tolist()
    ideal_rankings = sorted(rankings, key=lambda doc: relevant_dict.get(doc, 0), reverse=True)
    
    for i, doc in enumerate(rankings):
        relevance = relevant_dict.get(doc, 0)
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
        ideal_relevance = relevant_dict.get(ideal_rankings[i], 0)
        ideal_dcg += (2 ** ideal_relevance - 1) / np.log2(i + 2)
        
    ndcg = dcg / ideal_dcg if ideal_dcg != 0 else 0
    dcg_score += dcg
    ndcg_score += ndcg

    vp = 0
    precisao_soma = 0
    rank_reciproco = 0.0
    rank_limite = 10
    i = 0   

    # Iterando os documentos recuperados
    for doc_id in docs_recuperados:
        if doc_id in docs_relevantes:
            vp += 1
            precisao = vp / (i + 1)
            revocacao = vp / num_relevantes
            precisao_soma += precisao
            par = (revocacao, precisao)
            precisao_revocacao.append(par)
            
            # Cálculo de F_1
            rank = int(resultados.loc[resultados["DocNumber"] == doc_id, "DocRanking"].values[0])
            if rank == 1:
                f_1 += (2 * precisao * revocacao) / (precisao + revocacao)
                
            # Cálculo de MRR
            if rank_limite >= rank:
                rank_reciproco = 1.0 / rank
                rank_limite = rank
                mrr_score += rank_reciproco
        i += 1
        if vp == num_relevantes:
            break

    # Ordenar a lista de valores pela ordem crescente de recall
    precisao_revocacao.sort(key=lambda x: x[0])

    # Extrair os valores de recall e precisão em listas separadas
    recalls = [t[0] for t in precisao_revocacao]
    precisions = [t[1] for t in precisao_revocacao]

    # Definir os valores de recall desejados
    valores_recall_desejados = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Calcular a precisão interpolada para cada valor de recall desejado
    precisoes_interpoladas = np.interp(valores_recall_desejados, recalls, precisions)
    
    # Adicionar os valores de precisao calculados para a consulta na matriz da coleção
    matriz_precisoes.append(list(precisoes_interpoladas))

    # MAP, DCG e nDCG
    map_score += precisao_soma / num_relevantes

# Cálculo das métricas finais
precision_at_5 /= len(consultas)
precision_at_10 /= len(consultas)
f_1 /= len(consultas)
map_score /= len(consultas)
mrr_score /= len(consultas)
dcg_score /= len(consultas)
ndcg_score /= len(consultas)

# Salvando os resultados no arquivo RELATORIO.MD
with open("../AVALIACAO/RELATORIO.MD", "w") as arquivo_md:
    arquivo_md.write("# Relatório da avaliação do algoritmo de busca\n\n")
    arquivo_md.write("## Resultados obtidos sem o uso do stemmer de Porter:\n\n")
    arquivo_md.write(f"- Precision@5: {precision_at_5}\n")
    arquivo_md.write(f"- Precision@10: {precision_at_10}\n")
    arquivo_md.write(f"- MAP: {map_score}\n")
    arquivo_md.write(f"- MRR: {mrr_score}\n")
    arquivo_md.write(f"- DCG: {dcg_score}\n")
    arquivo_md.write(f"- NDCG: {ndcg_score}\n\n")

# Plotagem do gráfico de 11 pontos de precisão e revocação
matriz_precisoes = np.array(matriz_precisoes)
media_colunas = np.mean(matriz_precisoes, axis=0)

plt.plot(valores_recall_desejados, media_colunas, marker='o')
plt.xlabel("Revocação")
plt.ylabel("Precisão")
plt.title("Gráfico de 11 pontos de precisão e revocação")
plt.savefig('../AVALIACAO/11pontos-nostemmer-1.pdf', format='pdf')
dados = list(zip(valores_recall_desejados, media_colunas))
with open("../AVALIACAO/11pontos-nostemmer-1.csv", "w", newline="") as pontos_11_nostemmer_csv:
    writer = csv.writer(pontos_11_nostemmer_csv, delimiter=";")
    writer.writerow(["Revocacao", "Precisao"])  # Escreve o cabeçalho
    writer.writerows(dados)  # Escreve os dados
plt.show()


### Avaliação do algoritmo com o uso do stemmer
# Inicialização de variaveis
precisao_revocacao_2 = []
precision_at_5_2 = 0
precision_at_10_2 = 0
f_1_2 = 0.0
map_score_2 = 0
mrr_score_2 = 0
dcg_score_2 = 0.0
ndcg_score_2 = 0.0
matriz_precisoes_2 = []
revocacoes_2 = []
r_precision_2 = []

# Iterando as consultas
for id_consulta_2 in consultas:

    # Filtrar resultados e resultados esperados para a consulta atual
    resultados_2 = df_resultados_stemmer[df_resultados_stemmer["QueryNumber"] == id_consulta]
    esperados_2 = df_esperados[df_esperados["QueryNumber"] == id_consulta]
    
    # Obter os IDs de documentos relevantes e recuperados para a consulta atual
    docs_relevantes_2 = list(map(int, esperados_2["DocNumber"].to_list()))
    num_relevantes_2 = len(docs_relevantes_2)
    docs_recuperados_2 = list(map(int, resultados_2["DocNumber"].to_list()))
    num_recuperados_2 = len(docs_recuperados_2)
    
    # Precision@5 e Precision@10
    lista_5_2 = list(map(int, resultados_2["DocNumber"].head(5).to_list()))
    lista_10_2 = list(map(int, resultados_2["DocNumber"].head(10).to_list()))
    precision_at_5_2 += len([doc_id_2 for doc_id_2 in lista_5_2 if doc_id_2 in docs_relevantes_2])
    precision_at_10_2 += len([doc_id_2 for doc_id_2 in lista_10_2 if doc_id_2 in docs_relevantes_2])

    # R-Precision
    lista_r_2 = list(map(int, resultados_2["DocNumber"].head(num_relevantes_2).to_list()))
    r_precision_2.append(len([doc_id_2 for doc_id_2 in lista_r_2 if doc_id_2 in docs_relevantes_2]))
    
    # DCG e NDCG
    dcg_2 = 0.0
    ideal_dcg_2 = 0.0
    df_filtered_2 = resultados_2[resultados_2['DocRanking'] <= 10].copy()

    # Lista para armazenar os valores de DCG e NDCG para cada consulta
    relevantes_2 = esperados_2[esperados_2["QueryNumber"] == id_consulta_2]
    relevant_dict_2 = dict(zip(esperados_2['DocNumber'], esperados_2['DocVotes']))
        
    rankings_2 = df_filtered_2[df_filtered_2['QueryNumber'] == id_consulta_2]['DocNumber'].tolist()
    ideal_rankings_2 = sorted(rankings_2, key=lambda doc_2: relevant_dict_2.get(doc_2, 0), reverse=True)
    
    for i_2, doc_2 in enumerate(rankings_2):
        relevance_2 = relevant_dict_2.get(doc_2, 0)
        dcg_2 += (2 ** relevance_2 - 1) / np.log2(i_2 + 2)
        ideal_relevance_2 = relevant_dict_2.get(ideal_rankings_2[i_2], 0)
        ideal_dcg_2 += (2 ** ideal_relevance_2 - 1) / np.log2(i_2 + 2)
        
    ndcg_2 = dcg_2 / ideal_dcg_2 if ideal_dcg_2 != 0 else 0
    dcg_score_2 += dcg_2
    ndcg_score_2 += ndcg_2

    vp_2 = 0
    precisao_soma_2 = 0
    rank_reciproco_2 = 0.0
    rank_limite_2 = 10
    i_2 = 0   

    # Iterando os documentos recuperados
    for doc_id_2 in docs_recuperados_2:
        if doc_id_2 in docs_relevantes_2:
            vp_2 += 1
            precisao_2 = vp / (i_2 + 1)
            revocacao_2 = vp_2 / num_relevantes_2
            precisao_soma_2 += precisao_2
            par_2 = (revocacao_2, precisao_2)
            precisao_revocacao_2.append(par_2)
            
            # Cálculo de F_1
            rank_2 = int(resultados_2.loc[resultados_2["DocNumber"] == doc_id_2, "DocRanking"].values[0])
            if rank_2 == 1:
                f_1_2 += (2 * precisao_2 * revocacao_2) / (precisao_2 + revocacao_2)
                
            # Cálculo de MRR
            if rank_limite_2 >= rank_2:
                rank_reciproco_2 = 1.0 / rank_2
                rank_limite_2 = rank_2
                mrr_score_2 += rank_reciproco_2
        i_2 += 1
        if vp_2 == num_relevantes_2:
            break

    # Ordenar a lista de valores pela ordem crescente de recall
    precisao_revocacao_2.sort(key=lambda x_2: x_2[0])

    # Extrair os valores de recall e precisão em listas separadas
    recalls_2 = [t_2[0] for t_2 in precisao_revocacao_2]
    precisions_2 = [t_2[1] for t_2 in precisao_revocacao_2]

    # Definir os valores de recall desejados
    valores_recall_desejados_2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Calcular a precisão interpolada para cada valor de recall desejado
    precisoes_interpoladas_2 = np.interp(valores_recall_desejados_2, recalls_2, precisions_2)
    
    # Adicionar os valores de precisao calculados para a consulta na matriz da coleção
    matriz_precisoes_2.append(list(precisoes_interpoladas_2))

    # MAP, DCG e nDCG
    map_score_2 += precisao_soma_2 / num_relevantes_2

# Cálculo das métricas finais
precision_at_5_2 /= len(consultas)
precision_at_10_2 /= len(consultas)
f_1_2 /= len(consultas)
map_score_2 /= len(consultas)
mrr_score_2 /= len(consultas)
dcg_score_2 /= len(consultas)
ndcg_score_2 /= len(consultas)

# Salvando os resultados no arquivo RELATORIO.MD
with open("../AVALIACAO/RELATORIO.MD", "a") as arquivo_md:
    arquivo_md.write("## Resultados obtidos com o uso do stemmer de Porter:\n\n")
    arquivo_md.write(f"- Precision@5: {precision_at_5_2}\n")
    arquivo_md.write(f"- Precision@10: {precision_at_10_2}\n")
    arquivo_md.write(f"- MAP: {map_score_2}\n")
    arquivo_md.write(f"- MRR: {mrr_score_2}\n")
    arquivo_md.write(f"- DCG: {dcg_score_2}\n")
    arquivo_md.write(f"- NDCG: {ndcg_score_2}\n\n")

# Plotagem do gráfico de 11 pontos de precisão e revocação
matriz_precisoes_2 = np.array(matriz_precisoes_2)
media_colunas_2 = np.mean(matriz_precisoes_2, axis=0)

plt.plot(valores_recall_desejados_2, media_colunas_2, marker='o')
plt.xlabel("Revocação")
plt.ylabel("Precisão")
plt.title("Gráfico de 11 pontos de precisão e revocação")
plt.savefig('../AVALIACAO/11pontos-stemmer-1.pdf', format='pdf')
dados_2 = list(zip(valores_recall_desejados_2, media_colunas_2))
with open("../AVALIACAO/11pontos-stemmer-1.csv", "w", newline="") as pontos_11_stemmer_csv:
    writer = csv.writer(pontos_11_stemmer_csv, delimiter=";")
    writer.writerow(["Revocacao", "Precisao"])
    writer.writerows(dados_2)
plt.show()

# Histograma de R-Precision
r_precisions = [pair[1] for pair in precisao_revocacao if pair[0] > 0]
plt.hist((r_precisions - r_precisions_2), bins=100)
plt.xlabel("R-Precision")
plt.ylabel("Frequência")
plt.title("Histograma de R-Precision")
plt.savefig('../AVALIACAO/R-precision-comparativo-1.pdf', format='pdf')
dados_3 = list(zip(r_precisions - r_precisions_2))
with open("../AVALIACAO/R-precision-comparativo-1.csv", "w", newline="") as pontos_11_stemmer_csv:
    writer = csv.writer(pontos_11_stemmer_csv, delimiter=";")
    writer.writerow(["R-precision-stemmer - R-precision-nostemmer"])
    writer.writerows(dados_3)

plt.show()
