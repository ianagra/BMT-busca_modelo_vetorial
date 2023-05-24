# Implementação de Um Sistema de Recuperação Em Memória Segundo o Modelo Vetorial

Conforme orientações, o sistema foi atualizado e conta com 5 módulos:  

* **PC:** Processador de Consultas;
* **GLI:** Gerador de Lista Invertida;
* **INDEX:** Indexador;
* **BUSCA:** Buscador;
* **AVALIA:** Avaliador do algoritmo.

O código de cada módulo e seu respectivo arquivo de configuração encontram-se no diretório 'SRC'.  
Os arquivos .log gerados encontram-se no diretório 'LOG'.  
Os arquivos da coleção utilizada encontram-se na pasta 'data'.  
Os arquivos gerados pelo algoritmo de busca encontram-se na pasta "RESULT".  
O relatório gerado, RELATORIO.MD, encontra-se na pasta 'AVALIACAO', bem como os gráficos gerados, que foram salvos em formato PDF e CSV.  

## Uso do stemmer de Porter

Os módulos GLI e BUSCA contam com a opção de utilizar ou não o stemmer de Porter.  
Para utilizá-lo, basta acrescentar na primeira linha do arquivo GLI.CFG a palavra STEMMER e acrescentar uma seção [STEMMER] no arquivo BUSCA.CFG.
