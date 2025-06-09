import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import pickle

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (f1_score, silhouette_score, adjusted_rand_score, 
                             adjusted_mutual_info_score, homogeneity_completeness_v_measure,
                             fowlkes_mallows_score, calinski_harabasz_score, davies_bouldin_score)
import matplotlib.pyplot as plt


# Função para extrair classificação hierárquica da descrição
def extract_classification(description):
    match = re.search(r'(\w+\.\d+\.\d+\.\d+)', description)
    if match:
        return match.group(0)
    return "Desconhecido"


# Função para processar arquivo FASTA e extrair dados
def process_fasta(fasta_path):
    print(f"Processando arquivo FASTA: {fasta_path}")
    data = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        classification = extract_classification(record.description)
        data.append({
            "ID": record.id,
            "Descrição": record.description,
            "Classificação": classification,
            "Sequência": str(record.seq)
        })
    print(f"Total de sequências processadas: {len(data)}")
    return data


# Função para agrupar dados pela classificação e mostrar exemplo
def agrupar_e_exibir(sequences, n_classes=2, n_exemplos=2):
    grouped_data = defaultdict(list)
    for seq in sequences:
        grouped_data[seq["Classificação"]].append(seq)

    print(f"\nTotal de classificações únicas: {len(grouped_data)}\n")
    for classification, group in list(grouped_data.items())[:n_classes]:
        print(f"Classificação: {classification}")
        for seq in group[:n_exemplos]:
            print(f"  ID: {seq['ID']}")
            print(f"  Descrição: {seq['Descrição']}")
        print("-" * 60)


# Variáveis globais para subprocessos
projecao = None
y_true = None

def init_globals(proj, y):
    global projecao
    global y_true
    projecao = proj
    y_true = y


# Função para avaliar o AgglomerativeClustering com as métricas fornecidas
def avaliar_agglomerative(n_clusters):
    try:
        pid = os.getpid()
        print(f"[PID {pid}] Processando k={n_clusters} clusters...")

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        y_pred = clustering.fit_predict(projecao)

        if len(set(y_pred)) < 2:
            print(f"[PID {pid}] ⚠️ Apenas 1 cluster formado para k={n_clusters}, pulando métricas.")
            return None

        # Mapeamento para os rótulos reais mais comuns dentro de cada cluster
        y_pred_mapeado = np.zeros_like(y_pred, dtype=object)
        for cluster_id in np.unique(y_pred):
            mask = y_pred == cluster_id
            mais_comum = Counter(y_true[mask]).most_common(1)[0][0]
            y_pred_mapeado[mask] = mais_comum

        # Calcular as métricas de avaliação
        f1 = f1_score(y_true, y_pred_mapeado, average='micro')
        silhouette = silhouette_score(projecao, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
        fms = fowlkes_mallows_score(y_true, y_pred)
        calinski = calinski_harabasz_score(projecao, y_pred)
        davies = davies_bouldin_score(projecao, y_pred)

        return {
            'k': n_clusters,
            'f1_score': f1,
            'silhouette_score': silhouette,
            'adjusted_rand_score': ari,
            'adjusted_mutual_info_score': ami,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'fowlkes_mallows_score': fms,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies
        }

    except Exception as e:
        print(f"[PID {pid}] Erro ao processar k={n_clusters}: {e}")
        return None


# Função para retomar a execução do processamento a partir do último cluster
def main():
    fasta_path = os.path.join("data", "raw", "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
    proj_path = os.path.join("notebooks", "all_vectors_reduced_svd_k500.npy")

    # 1. Processar FASTA para extrair dados
    sequences = process_fasta(fasta_path)

    # 2. Exibir agrupamento exemplo
    agrupar_e_exibir(sequences)

    # 3. Carregar projeção e classes para clusterização
    print(f"\nCarregando projeção: {proj_path}")
    proj = np.load(proj_path)

    realClasses = [seq['Classificação'] for seq in sequences]
    y = np.array(realClasses)

    print(f"Shape da projeção: {proj.shape}")
    print(f"Número de classes distintas: {len(set(y))}")

    # Carregar resultados parciais (se houver)
    resultados_parciais = []
    if os.path.exists('resultados_parciais.pkl'):
        with open('resultados_parciais.pkl', 'rb') as f:
            resultados_parciais = pickle.load(f)
        print(f"\nResultados parciais carregados, {len(resultados_parciais)} clusters processados até agora.")
    
    # Continuar a partir do último cluster processado
    if resultados_parciais:
        ultimo_cluster = resultados_parciais[-1]['k']
        intervalo_inicio = ultimo_cluster + 1
        print(f"Iniciando processamento a partir do cluster {intervalo_inicio}")
    else:
        intervalo_inicio = 2

    # Definir intervalos de clusters a serem processados
    max_clusters = min(15000, proj.shape[0] - 1)
    step = 100
    intervalos = list(range(intervalo_inicio, max_clusters + 1, step))

    print(f"Total de clusters a testar a partir de {intervalo_inicio}: {len(intervalos)}")

    resultados = resultados_parciais

    # Paralelizar clusterização com ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6, initializer=init_globals, initargs=(proj, y)) as executor:
        count = 0
        batch_size = 10  # Definir o tamanho do lote a ser salvo
        for resultado in executor.map(avaliar_agglomerative, intervalos):
            count += 1
            if resultado:
                resultados.append(resultado)
            if count % batch_size == 0:
                # Salvar resultados parciais a cada 100 clusters processados
                with open('resultados_parciais.pkl', 'wb') as f:
                    pickle.dump(resultados, f)
                print(f"{count} clusters processados e resultados salvos até agora...")
        
        # Salvar os resultados finais após todos os clusters serem processados
        with open('resultados_parciais.pkl', 'wb') as f:
            pickle.dump(resultados, f)
        print(f"Resultados parciais finais salvos em 'resultados_parciais.pkl'")

    # Criar DataFrame com os resultados finais
    df = pd.DataFrame(resultados)
    print("\n✅ Resultados encontrados:")
    print(df)

    # Plotagem dos resultados
    plt.figure(figsize=(18, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df['k'], df['f1_score'], marker='o', label='F1 Score')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df['k'], df['silhouette_score'], marker='s', color='orange', label='Silhouette Score')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(df['k'], df['calinski_harabasz_score'], marker='^', color='green', label='Calinski-Harabasz Score')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Calinski-Harabasz Score")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(df['k'], df['davies_bouldin_score'], marker='d', color='red', label='Davies-Bouldin Score')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Davies-Bouldin Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
