import os
import re
import pickle
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, silhouette_score, adjusted_rand_score,
    adjusted_mutual_info_score, homogeneity_completeness_v_measure,
    fowlkes_mallows_score, calinski_harabasz_score, davies_bouldin_score
)
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message=".*unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
    module="sklearn.metrics.cluster._supervised"
)

# globals para subprocessos
projecao = None
y_true = None

def init_globals(proj, y):
    global projecao, y_true
    projecao, y_true = proj, y

def extract_classification(description):
    m = re.search(r'(\w+\.\d+\.\d+\.\d+)', description)
    return m.group(0) if m else "Desconhecido"

def process_fasta(fasta_path):
    print(f"Processando arquivo FASTA: {fasta_path}")
    data = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        data.append({
            "ID": rec.id,
            "Descrição": rec.description,
            "Classificação": extract_classification(rec.description)
        })
    print(f"Total de sequências processadas: {len(data)}")
    return data

def agrupar_e_exibir(sequences, n_classes=2, n_exemplos=2):
    grouped = defaultdict(list)
    for seq in sequences:
        grouped[seq["Classificação"]].append(seq)
    print(f"\nTotal de classificações únicas: {len(grouped)}\n")
    for cls, group in list(grouped.items())[:n_classes]:
        print(f"Classificação: {cls}")
        for seq in group[:n_exemplos]:
            print(f"  ID: {seq['ID']}")
            print(f"  Descrição: {seq['Descrição']}")
        print("-" * 60)

def avaliar_kmeans(n_clusters):
    try:
        km = KMeans(
            n_clusters=n_clusters,
            init='random',
            n_init=1,
            max_iter=100,
            random_state=42,
            verbose=0
        )
        y_pred = km.fit_predict(projecao)
        if len(np.unique(y_pred)) < 2:
            return None

        # mapear cada cluster ao rótulo mais comum
        mapped = np.empty_like(y_pred, dtype=object)
        for cid in np.unique(y_pred):
            mask = (y_pred == cid)
            mapped[mask] = Counter(y_true[mask]).most_common(1)[0][0]

        # calcular métricas
        f1   = f1_score(y_true, mapped, average='micro')
        sil  = silhouette_score(projecao, y_pred)
        ari  = adjusted_rand_score(y_true, y_pred)
        ami  = adjusted_mutual_info_score(y_true, y_pred)
        h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
        fms  = fowlkes_mallows_score(y_true, y_pred)
        ch   = calinski_harabasz_score(projecao, y_pred)
        db   = davies_bouldin_score(projecao, y_pred)

        return {
            'k': n_clusters,
            'f1_score': f1,
            'silhouette_score': sil,
            'adjusted_rand_score': ari,
            'adjusted_mutual_info_score': ami,
            'homogeneity': h,
            'completeness': c,
            'v_measure': v,
            'fowlkes_mallows_score': fms,
            'calinski_harabasz_score': ch,
            'davies_bouldin_score': db
        }
    except Exception:
        return None

def main():
    fasta_path = os.path.join("data", "raw", "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
    proj_path  = os.path.join("notebooks", "all_vectors_reduced_svd_k500.npy")

    # 1. Extrai sequências e classes
    sequences = process_fasta(fasta_path)
    agrupar_e_exibir(sequences)

    # 2. Carrega projeção e vetor de rótulos
    proj = np.load(proj_path)
    y = np.array([seq["Classificação"] for seq in sequences])

    # 3. Carregar resultados parciais
    resultados = []
    if os.path.exists('resultados_parciais.pkl'):
        with open('resultados_parciais.pkl', 'rb') as f:
            resultados = pickle.load(f)

    # 4. Definir ks de 2 em 100 até max
    inicio = resultados[-1]['k'] + 100 if resultados else 2
    max_k  = min(15000, proj.shape[0] - 1)
    step   = 100
    ks     = list(range(inicio, max_k + 1, step))
    total_intervals = len(ks)
    print(f"Testando ks {inicio}, {inicio+step}, … até {max_k} → {total_intervals} intervalos")

    # 5. Paralelização com logs
    batch_save = 10
    with ProcessPoolExecutor(max_workers=6, initializer=init_globals, initargs=(proj, y)) as exe:
        for idx, (k, res) in enumerate(zip(ks, exe.map(avaliar_kmeans, ks)), start=1):
            # cada intervalo corresponde a `step` clusters
            clusters_done = idx * step

            if res:
                resultados.append(res)

            # log de progresso em clusters (100, 200, 300…)
            print(f"{clusters_done} clusters processados até agora…")

            # a cada 10 intervalos (= 10 * 100 clusters), salva e printa
            if idx % batch_save == 0:
                with open('resultados_parciais.pkl', 'wb') as f:
                    pickle.dump(resultados, f)
                print(f"{clusters_done} clusters processados e salvos parcialmente.")

    # 6. salvar finais e plotar…
    with open('resultados_parciais.pkl', 'wb') as f:
        pickle.dump(resultados, f)
    print("Resultados finais salvos em 'resultados_parciais.pkl'")

    df = pd.DataFrame(resultados)
    print("\n✅ Resultados encontrados:")
    print(df)
    # … (plotagem igual)

if __name__ == "__main__":
    main()
