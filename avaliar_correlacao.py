import pickle
import pandas as pd

def main():
    # 1. Carrega resultados parciais
    with open('resultados_birch.pkl', 'rb') as f:
        resultados = pickle.load(f)
    
    df = pd.DataFrame(resultados)
    
    # 2. Seleciona métricas internas (numéricas, exceto 'k' e 'f1_score')
    métricas = df.select_dtypes(include='number').columns.drop(['k', 'f1_score'])
    
    # 3. Calcula correlações de Pearson entre f1_score e cada métrica
    corrs = df[['f1_score']].join(df[métricas]).corr()['f1_score'].drop('f1_score')
    abs_corrs = corrs.abs()
    
    # Identifica métrica de maior correlação absoluta
    melhor_metrica = abs_corrs.idxmax()
    valor_corr = corrs[melhor_metrica]
    
    # 4. Encontra maior f1_score e número de clusters onde ocorreu
    idx_max_f1 = df['f1_score'].idxmax()
    max_f1 = df.loc[idx_max_f1, 'f1_score']
    k_max_f1 = df.loc[idx_max_f1, 'k']
    
    # 5. Número ideal de clusters segundo a métrica mais correlacionada
    idx_ideal = df[melhor_metrica].idxmax()
    k_ideal = df.loc[idx_ideal, 'k']
    valor_ideal = df.loc[idx_ideal, melhor_metrica]
    
    # 6. Impressão dos resultados
    print("=== Correlações entre f1_score e métricas internas ===")
    for metr, corr in corrs.items():
        print(f"  {metr}: {corr:.4f}")
    print(f"\nMétrica com MAIOR correlação absoluta com f1_score: {melhor_metrica} ({valor_corr:.4f})\n")
    
    print(f"MAIOR f1_score: {max_f1:.4f}  →  ocorreu em k = {k_max_f1}")
    print(f"Número IDEAL de clusters (maximiza '{melhor_metrica}'): k = {k_ideal}  →  {melhor_metrica} = {valor_ideal:.4f}")

if __name__ == '__main__':
    main()
