import os
import re
import pickle
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from Bio import SeqIO
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    f1_score, silhouette_score, adjusted_rand_score,
    adjusted_mutual_info_score, homogeneity_completeness_v_measure,
    fowlkes_mallows_score, calinski_harabasz_score, davies_bouldin_score
)
import matplotlib.pyplot as plt

# Suprime warnings de métricas com classes >50% do número de amostras
warnings.filterwarnings(
    "ignore",
    message=".*unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
    module="sklearn.metrics.cluster._supervised"
)

# Variáveis globais para subprocessos
projecao = None
y_true = None

def init_globals(proj, y):
    global projecao, y_true
    projecao, y_true = proj, y

# Extrai classificação hierárquica da descrição FASTA
def extract_classification(description):
    match = re.search(r'(\w+\.\d+\.\d+\.\d+)', description)
    if match:
        return match.group(0)
    return "Desconhecido"

# Processa FASTA para extrair IDs e classes
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

# Exibe exemplos de agrupamento por classe
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

# Avalia GMM para um dado número de clusters
def avaliar_gmm(n_clusters):
    try:
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        y_pred = model.fit_predict(projecao)
        if len(np.unique(y_pred)) < 2:
            return None

        # Mapeia cada cluster ao rótulo mais comum
        mapped = np.empty_like(y_pred, dtype=object)
        for cid in np.unique(y_pred):
            mask = (y_pred == cid)
            mapped[mask] = Counter(y_true[mask]).most_common(1)[0][0]

        # Calcula métricas
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

# Função principal
def main():
    fasta_path = os.path.join("data/raw/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
    proj_path  = os.path.join("notebooks/all_vectors_random_orth_proj_k500.npy")

    # 1. Processa FASTA
    sequences = process_fasta(fasta_path)
    agrupar_e_exibir(sequences)

    # 2. Carrega projeção e rótulos
    print(f"\nCarregando projeção: {proj_path}")
    proj = np.load(proj_path)
    y = np.array([seq['Classificação'] for seq in sequences])

    # 3. Carrega resultados parciais
    resultados = []
    if os.path.exists('resultados_gmm.pkl'):
        with open('resultados_gmm.pkl','rb') as f:
            resultados = pickle.load(f)
        print(f"\nResultados parciais carregados: {len(resultados)} clusters processados.")

    # 4. Define ks de 2 em 100 até o máximo
    max_k = min(15000, proj.shape[0] - 1)
    start_k = resultados[-1]['k'] + 100 if resultados else 2
    ks = list(range(start_k, max_k+1, 100))
    total = len(ks)
    print(f"Testando ks de {start_k} a {max_k}, step=100 (total {total})")

    # 5. Paraleliza com logs e salvamentos
    batch = 10
    with ProcessPoolExecutor(max_workers=6, initializer=init_globals, initargs=(proj, y)) as exe:
        for idx, k in enumerate(ks, start=1):
            res = exe.submit(avaliar_gmm, k).result()
            if res: resultados.append(res)

            # Log cada 100 clusters
            if idx % 1 == 0:
                print(f"{idx*100}/{total*100} clusters processados...")
            # Salva a cada batch
            if idx % batch == 0:
                with open('resultados_gmm.pkl','wb') as f:
                    pickle.dump(resultados, f)
                print(f"{idx*100} clusters processados e salvos.")

    # 6. Salva finais e imprime
    with open('resultados_gmm.pkl','wb') as f:
        pickle.dump(resultados, f)
    print("Resultados finais salvos em 'resultados_gmm.pkl'")

    df = pd.DataFrame(resultados)
    print("\n✅ Resultados GMM encontrados:")
    print(df)

    # 7. Plotagem
    plt.figure(figsize=(18,8))
    plt.subplot(2,2,1); plt.plot(df['k'],df['f1_score'], 'o-'); plt.title('F1 Score'); plt.grid(True)
    plt.subplot(2,2,2); plt.plot(df['k'],df['silhouette_score'], 's-'); plt.title('Silhouette'); plt.grid(True)
    plt.subplot(2,2,3); plt.plot(df['k'],df['calinski_harabasz_score'], '^-'); plt.title('Calinski-Harabasz'); plt.grid(True)
    plt.subplot(2,2,4); plt.plot(df['k'],df['davies_bouldin_score'], 'd-'); plt.title('Davies-Bouldin'); plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
