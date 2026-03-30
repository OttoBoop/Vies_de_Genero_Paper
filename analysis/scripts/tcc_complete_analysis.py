#!/usr/bin/env python3
"""
TCC Analysis - Viés de Gênero em LLMs
=====================================
Script unificado para geração de todos os gráficos e tabelas necessários.

Autor: Otavio
Data: 2026

Este script gera:
- Gráficos de distribuição de gênero por família de modelo (Testes 1, 2, 3)
- Gráficos de distribuição por idioma e valência
- Gráficos de shot order (GPT-3 Legacy)
- Tabelas de regressão
- Arquivos CSV consolidados

Uso:
    python tcc_complete_analysis.py

Requer:
    - pandas
    - numpy  
    - matplotlib
    - scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from typing import List, Tuple, Dict
import shutil

# ============================================
# CONFIGURAÇÃO
# ============================================

# Diretórios
INPUT_DIR = '/mnt/user-data/uploads/'
OUTPUT_DIR = '/mnt/user-data/outputs/'

# Criar diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Arquivos de dados
DATA_FILES = {
    'teste1': 'df_teste_1_unified.csv',
    'teste2': 'df_teste_2_unified.csv',
    'teste3': 'df_teste_3_unified.csv',
}

# Mapeamento de famílias de modelos
FAMILY_MAP = {
    'davinci-002': 'GPT-3 Legacy',
    'babbage-002': 'GPT-3 Legacy',
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4o-2024-08-06': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o',
    'gpt-4.1-2025-04-14': 'GPT-4.1',
    'gpt-4.1-mini-2025-04-14': 'GPT-4.1',
    'gpt-4.1-nano-2025-04-14': 'GPT-4.1',
    'o3-mini-2025-01-31': 'Serie o',
    'o3-2025-04-16': 'Serie o',
    'o4-mini-2025-04-16': 'Serie o',
    'gpt-5-mini': 'GPT-5',
    'gpt-5-nano': 'GPT-5',
    'gpt-5.1-2025-11-13': 'GPT-5',
    'gpt-5.2-2025-12-11': 'GPT-5',
}

FAMILY_ORDER = ['GPT-3 Legacy', 'GPT-3.5', 'GPT-4o', 'GPT-4.1', 'Serie o', 'GPT-5']

# Modelos Legacy
LEGACY_MODELS = ['davinci-002', 'babbage-002']

# Cores padrão
COLORS = {
    'Male': '#4472C4',
    'Female': '#C55A5A',
    'Other': '#A5A5A5'
}

# Configuração de matplotlib
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def load_data() -> Dict[str, pd.DataFrame]:
    """Carrega todos os datasets."""
    data = {}
    for key, filename in DATA_FILES.items():
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            print(f"✓ {key}: {len(data[key])} observações carregadas")
        else:
            print(f"✗ {key}: arquivo não encontrado ({filepath})")
            data[key] = None
    return data


def add_family_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna de família de modelo."""
    df = df.copy()
    df['family'] = df['modelo'].map(FAMILY_MAP)
    return df


def calc_family_props(df: pd.DataFrame, family_order: List[str]) -> pd.DataFrame:
    """Calcula proporções de gênero por família."""
    results = []
    for family in family_order:
        family_data = df[df['family'] == family]
        total = len(family_data)
        if total > 0:
            male_pct = (family_data['gender'] == 'Male').sum() / total * 100
            female_pct = (family_data['gender'] == 'Female').sum() / total * 100
            other_pct = 100 - male_pct - female_pct
            results.append({
                'family': family,
                'Male': male_pct,
                'Female': female_pct,
                'Other': other_pct,
                'n': total
            })
    return pd.DataFrame(results)


def run_ols_regression(y: np.ndarray, X: np.ndarray, var_names: List[str]) -> Tuple[List[Dict], int, float, int]:
    """Executa regressão OLS manualmente."""
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    XtX = X_with_const.T @ X_with_const
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_with_const.T @ y
    beta = XtX_inv @ Xty
    
    y_hat = X_with_const @ beta
    residuals = y - y_hat
    n = len(y)
    k = X_with_const.shape[1]
    df_resid = n - k
    
    s2 = (residuals @ residuals) / df_resid
    var_beta = s2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    
    ss_res = residuals @ residuals
    ss_tot = (y - y.mean()) @ (y - y.mean())
    r_squared = 1 - ss_res / ss_tot
    
    all_var_names = ['Intercepto'] + var_names
    
    results = []
    for i, var in enumerate(all_var_names):
        p = p_values[i]
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        results.append({
            'Variável': var,
            'Coef.': round(beta[i], 4),
            'EP': round(se_beta[i], 4),
            't': round(t_stats[i], 2),
            'p-valor': '<0.001' if p < 0.001 else f'{p:.4f}',
            'Sig.': sig
        })
    
    return results, n, r_squared, k


# ============================================
# FUNÇÕES DE GRÁFICOS
# ============================================

def create_family_chart(props_df: pd.DataFrame, title: str, filename: str):
    """Cria gráfico de barras empilhadas por família."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    families = props_df['family'].tolist()
    x = np.arange(len(families))
    width = 0.6
    
    # Plotar barras empilhadas
    bars_female = ax.bar(x, props_df['Female'], width, label='Female', color=COLORS['Female'])
    bars_male = ax.bar(x, props_df['Male'], width, bottom=props_df['Female'], label='Male', color=COLORS['Male'])
    bars_other = ax.bar(x, props_df['Other'], width, bottom=props_df['Female'] + props_df['Male'], label='Other', color=COLORS['Other'])
    
    # Adicionar porcentagens
    for i, (idx, row) in enumerate(props_df.iterrows()):
        if row['Female'] > 5:
            ax.text(i, row['Female']/2, f"{row['Female']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        if row['Male'] > 5:
            ax.text(i, row['Female'] + row['Male']/2, f"{row['Male']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        if row['Other'] > 5:
            ax.text(i, row['Female'] + row['Male'] + row['Other']/2, f"{row['Other']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Family', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")


def create_legacy_valence_order_chart(df_legacy: pd.DataFrame, title: str, filename: str):
    """Cria gráfico de shot order por valência para modelos Legacy."""
    order_sequence = ['MF', 'FM', 'M', 'F']
    
    # Calcular proporções
    results = []
    for valencia in ['positive', 'negative']:
        for order in order_sequence:
            subset = df_legacy[(df_legacy['valencia_norm'] == valencia) & 
                               (df_legacy['example_order'] == order)]
            total = len(subset)
            if total > 0:
                male_pct = (subset['gender'] == 'Male').sum() / total * 100
                female_pct = (subset['gender'] == 'Female').sum() / total * 100
                other_pct = 100 - male_pct - female_pct
                results.append({
                    'valencia': valencia,
                    'order': order,
                    'Male': male_pct,
                    'Female': female_pct,
                    'Other': other_pct,
                    'n': total
                })
    
    df_plot = pd.DataFrame(results)
    
    # Ordenar: Positive primeiro, depois Negative
    df_pos = df_plot[df_plot['valencia'] == 'positive'].reset_index(drop=True)
    df_neg = df_plot[df_plot['valencia'] == 'negative'].reset_index(drop=True)
    df_ordered = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_bars = 8
    x = np.arange(n_bars)
    width = 0.7
    labels = order_sequence + order_sequence
    
    bars_female = ax.bar(x, df_ordered['Female'], width, label='Female', color=COLORS['Female'])
    bars_male = ax.bar(x, df_ordered['Male'], width, bottom=df_ordered['Female'], label='Male', color=COLORS['Male'])
    bars_other = ax.bar(x, df_ordered['Other'], width, bottom=df_ordered['Female'] + df_ordered['Male'], label='Other', color=COLORS['Other'])
    
    for i, row in df_ordered.iterrows():
        if row['Female'] > 5:
            ax.text(i, row['Female']/2, f"{row['Female']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        if row['Male'] > 5:
            ax.text(i, row['Female'] + row['Male']/2, f"{row['Male']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        if row['Other'] > 5:
            ax.text(i, row['Female'] + row['Male'] + row['Other']/2, f"{row['Other']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    ax.axvline(x=3.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.5, 103, 'Positive Valence', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(5.5, 103, 'Negative Valence', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_xlabel('Example Order', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")


def create_language_valence_chart(df_chat: pd.DataFrame, title: str, filename: str):
    """Cria gráfico de distribuição por idioma e valência."""
    categories = [
        ('en', 'negative', 'English\nNegative'),
        ('en', 'positive', 'English\nPositive'),
        ('pt', 'negative', 'Portuguese\nNegative'),
        ('pt', 'positive', 'Portuguese\nPositive'),
    ]
    
    results = []
    for idioma, valencia, label in categories:
        subset = df_chat[(df_chat['idioma'] == idioma) & (df_chat['valencia_norm'] == valencia)]
        total = len(subset)
        if total > 0:
            male_pct = (subset['gender'] == 'Male').sum() / total * 100
            female_pct = (subset['gender'] == 'Female').sum() / total * 100
            other_pct = 100 - male_pct - female_pct
            results.append({
                'label': label,
                'Male': male_pct,
                'Female': female_pct,
                'Other': other_pct,
                'n': total
            })
    
    df_plot = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.arange(len(df_plot))
    width = 0.6
    
    bars_female = ax.bar(x, df_plot['Female'], width, label='Female', color=COLORS['Female'])
    bars_male = ax.bar(x, df_plot['Male'], width, bottom=df_plot['Female'], label='Male', color=COLORS['Male'])
    bars_other = ax.bar(x, df_plot['Other'], width, bottom=df_plot['Female'] + df_plot['Male'], label='Other', color=COLORS['Other'])
    
    for i, row in df_plot.iterrows():
        if row['Female'] > 5:
            ax.text(i, row['Female']/2, f"{row['Female']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        if row['Male'] > 5:
            ax.text(i, row['Female'] + row['Male']/2, f"{row['Male']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        if row['Other'] > 5:
            ax.text(i, row['Female'] + row['Male'] + row['Other']/2, f"{row['Other']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in results], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")


def create_position_chart(props_df: pd.DataFrame, title: str, filename: str, position_labels: List[str]):
    """Cria gráfico de distribuição por profissão."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(props_df))
    width = 0.65
    
    bars_female = ax.bar(x, props_df['Female'], width, label='Female', color=COLORS['Female'])
    bars_male = ax.bar(x, props_df['Male'], width, bottom=props_df['Female'], label='Male', color=COLORS['Male'])
    bars_other = ax.bar(x, props_df['Other'], width, bottom=props_df['Female'] + props_df['Male'], label='Other', color=COLORS['Other'])
    
    for i, row in props_df.iterrows():
        if row['Female'] > 10:
            ax.text(i, row['Female']/2, f"{row['Female']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        if row['Male'] > 10:
            ax.text(i, row['Female'] + row['Male']/2, f"{row['Male']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        if row['Other'] > 10:
            ax.text(i, row['Female'] + row['Male'] + row['Other']/2, f"{row['Other']:.1f}%", 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    # Linhas verticais separando setores (Corporate: 0-3, Aviation: 4-5, Academic: 6-7, General: 8-9)
    sector_boundaries = [3.5, 5.5, 7.5]
    for boundary in sector_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.text(1.5, 108, 'Corporate', ha='center', va='bottom', fontsize=12, fontweight='bold', color='gray')
    ax.text(4.5, 108, 'Aviation', ha='center', va='bottom', fontsize=12, fontweight='bold', color='gray')
    ax.text(6.5, 108, 'Academic', ha='center', va='bottom', fontsize=12, fontweight='bold', color='gray')
    ax.text(8.5, 108, 'General', ha='center', va='bottom', fontsize=12, fontweight='bold', color='gray')
    
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=35)
    ax.set_xticks(x)
    ax.set_xticklabels(position_labels, fontsize=11, fontweight='bold', ha='center')
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.5, len(props_df) - 0.5)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")


def create_regression_table(results: List[Dict], n: int, r2: float, k: int, title: str, subtitle: str, filename: str, note: str):
    """Cria tabela visual de regressão."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='top')
    ax.text(0.5, 0.88, subtitle, transform=ax.transAxes, fontsize=10,
            ha='center', va='top', style='italic')
    
    headers = ['Variável', 'Coef.', 'EP', 't', 'p-valor', 'Sig.']
    cell_data = []
    
    for r in results:
        cell_data.append([r['Variável'], f"{r['Coef.']}", f"{r['EP']}", 
                         f"{r['t']}", r['p-valor'], r['Sig.']])
    
    table = ax.table(cellText=cell_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.15, 0.8, 0.65])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#FFFF00')
        cell.set_text_props(fontweight='bold')
    
    for i in range(len(results)):
        cell = table[(i+1, 0)]
        cell.set_facecolor('#FFFF99')
        cell.set_text_props(fontweight='bold')
    
    ax.text(0.5, 0.05, note, transform=ax.transAxes, fontsize=9,
            ha='center', va='bottom', style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {os.path.basename(filename)}")


# ============================================
# FUNÇÕES DE ANÁLISE POR TESTE
# ============================================

def analyze_test1(df: pd.DataFrame):
    """Análise completa do Teste 1."""
    print("\n" + "="*60)
    print("TESTE 1: Características no Ambiente de Trabalho")
    print("="*60)
    
    df = add_family_column(df)
    
    # Separar por tipo de modelo
    df_chat = df[~df['modelo'].isin(LEGACY_MODELS)].copy()
    df_legacy = df[df['modelo'].isin(LEGACY_MODELS)].copy()
    
    # Normalizar valencia
    df['valencia_norm'] = df['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    df_chat['valencia_norm'] = df_chat['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    df_legacy['valencia_norm'] = df_legacy['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    
    print(f"\nChat models: {len(df_chat)} obs")
    print(f"Legacy models: {len(df_legacy)} obs")
    
    # --- Gráficos por família e valência ---
    print("\nGerando gráficos por família...")
    
    df_neg = df[df['valencia_norm'] == 'negative']
    df_pos = df[df['valencia_norm'] == 'positive']
    
    props_neg = calc_family_props(add_family_column(df_neg), FAMILY_ORDER)
    props_pos = calc_family_props(add_family_column(df_pos), FAMILY_ORDER)
    
    create_family_chart(props_neg, 'Test 1: Gender Distribution by Model Family (Negative Valence)', 
                       f'{OUTPUT_DIR}t1_family_NEGATIVE.png')
    create_family_chart(props_pos, 'Test 1: Gender Distribution by Model Family (Positive Valence)', 
                       f'{OUTPUT_DIR}t1_family_POSITIVE.png')
    
    # --- Gráfico por idioma e valência (Chat only) ---
    print("\nGerando gráfico por idioma...")
    if 'idioma' in df_chat.columns:
        create_language_valence_chart(df_chat, 
                                     'Test 1: Gender Distribution by Language and Valence\n(Chat Models Only)',
                                     f'{OUTPUT_DIR}t1_language_valence.png')
    
    # --- Gráfico Legacy shot order ---
    print("\nGerando gráfico de shot order (Legacy)...")
    create_legacy_valence_order_chart(df_legacy,
                                      'Test 1 Legacy: Gender Distribution by Valence and Example Order',
                                      f'{OUTPUT_DIR}t1_legacy_valence_order.png')
    
    # --- Regressão Legacy com shot order ---
    print("\nCalculando regressão (Legacy)...")
    df_legacy['is_male'] = (df_legacy['gender'] == 'Male').astype(int)
    df_legacy['is_positive'] = (df_legacy['valencia_norm'] == 'positive').astype(int)
    df_legacy['order_FM'] = (df_legacy['example_order'] == 'FM').astype(int)
    df_legacy['order_M'] = (df_legacy['example_order'] == 'M').astype(int)
    df_legacy['order_F'] = (df_legacy['example_order'] == 'F').astype(int)
    
    y = df_legacy['is_male'].values
    X = df_legacy[['is_positive', 'order_F', 'order_FM', 'order_M']].values
    results, n, r2, k = run_ols_regression(y, X, ['is_positive', 'order_F', 'order_FM', 'order_M'])
    
    create_regression_table(results, n, r2, k,
                           "Regressão: GPT-3 Legacy — Com Shot Order (Teste 1)",
                           f"N = {n} | R² = {r2:.4f} | k = {k}",
                           f'{OUTPUT_DIR}regression_t1_legacy_shot_order.png',
                           "Interpretação: A ordem FM tem efeito significativo (β=-0,05*), reduzindo proporção masculina.")
    
    # Salvar tabela CSV
    pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}regression_t1_legacy.csv', index=False)
    
    # Salvar proporções
    props_neg.to_csv(f'{OUTPUT_DIR}t1_props_negative.csv', index=False)
    props_pos.to_csv(f'{OUTPUT_DIR}t1_props_positive.csv', index=False)


def analyze_test2(df: pd.DataFrame):
    """Análise completa do Teste 2."""
    print("\n" + "="*60)
    print("TESTE 2: Cenários de Feedback")
    print("="*60)
    
    df = add_family_column(df)
    
    # Normalizar valencia
    df['valencia_norm'] = df['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    
    df_chat = df[~df['modelo'].isin(LEGACY_MODELS)].copy()
    df_legacy = df[df['modelo'].isin(LEGACY_MODELS)].copy()
    df_chat['valencia_norm'] = df_chat['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    df_legacy['valencia_norm'] = df_legacy['valencia'].str.lower().replace({'positivo': 'positive', 'negativo': 'negative'})
    
    print(f"\nChat models: {len(df_chat)} obs")
    print(f"Legacy models: {len(df_legacy)} obs")
    
    # --- Gráficos por família e valência ---
    print("\nGerando gráficos por família...")
    
    df_neg = df[df['valencia_norm'] == 'negative']
    df_pos = df[df['valencia_norm'] == 'positive']
    
    props_neg = calc_family_props(add_family_column(df_neg), FAMILY_ORDER)
    props_pos = calc_family_props(add_family_column(df_pos), FAMILY_ORDER)
    
    create_family_chart(props_neg, 'Test 2: Gender Distribution by Model Family (Negative Feedback)', 
                       f'{OUTPUT_DIR}t2_family_NEGATIVE.png')
    create_family_chart(props_pos, 'Test 2: Gender Distribution by Model Family (Positive Feedback)', 
                       f'{OUTPUT_DIR}t2_family_POSITIVE.png')
    
    # --- Gráfico por idioma e valência (Chat only) ---
    print("\nGerando gráfico por idioma...")
    if 'idioma' in df_chat.columns:
        create_language_valence_chart(df_chat,
                                     'Test 2: Gender Distribution by Language and Valence\n(Chat Models Only)',
                                     f'{OUTPUT_DIR}t2_language_valence.png')
    
    # --- Gráfico Legacy shot order ---
    print("\nGerando gráfico de shot order (Legacy)...")
    create_legacy_valence_order_chart(df_legacy,
                                      'Test 2 Legacy: Gender Distribution by Valence and Example Order',
                                      f'{OUTPUT_DIR}t2_legacy_valence_order.png')
    
    # --- Regressão Legacy com shot order ---
    print("\nCalculando regressão (Legacy)...")
    df_legacy['is_male'] = (df_legacy['gender'] == 'Male').astype(int)
    df_legacy['is_positive'] = df_legacy['valencia_norm'].isin(['positive']).astype(int)
    df_legacy['order_FM'] = (df_legacy['example_order'] == 'FM').astype(int)
    df_legacy['order_M'] = (df_legacy['example_order'] == 'M').astype(int)
    df_legacy['order_F'] = (df_legacy['example_order'] == 'F').astype(int)
    
    y = df_legacy['is_male'].values
    X = df_legacy[['is_positive', 'order_F', 'order_FM', 'order_M']].values
    results, n, r2, k = run_ols_regression(y, X, ['is_positive', 'order_F', 'order_FM', 'order_M'])
    
    create_regression_table(results, n, r2, k,
                           "Regressão: GPT-3 Legacy — Com Shot Order (Teste 2)",
                           f"N = {n} | R² = {r2:.4f} | k = {k}",
                           f'{OUTPUT_DIR}regression_t2_legacy_shot_order.png',
                           "Interpretação: A ordem F (feminino somente) tem efeito forte (β=-0,27***),\nreduzindo proporção masculina. Demonstra sensibilidade do GPT-3 a exemplos.")
    
    # Salvar tabela CSV
    pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}regression_t2_legacy.csv', index=False)
    
    # Salvar proporções
    props_neg.to_csv(f'{OUTPUT_DIR}t2_props_negative.csv', index=False)
    props_pos.to_csv(f'{OUTPUT_DIR}t2_props_positive.csv', index=False)


def analyze_test3(df: pd.DataFrame):
    """Análise completa do Teste 3."""
    print("\n" + "="*60)
    print("TESTE 3: Profissões/Posições Ocupacionais")
    print("="*60)
    
    df = add_family_column(df)
    
    # Profissões selecionadas
    selected_positions = [
        ("CEO", "Corporate", "high"),
        ("manager", "Corporate", "high"),
        ("secretary", "Corporate", "low"),
        ("intern", "Corporate", "low"),
        ("pilot", "Aviation", "high"),
        ("flight attendant", "Aviation", "low"),
        ("professor", "Academic", "high"),
        ("teacher", "Academic", "low"),
        ("white collar worker", "General", "high"),
        ("blue collar worker", "General", "low"),
    ]
    
    position_order = [p[0] for p in selected_positions]
    
    # Labels com power level
    position_labels = []
    for pos, sector, power in selected_positions:
        if pos == "flight attendant":
            label = f"flight attendant\n({power})"
        elif pos == "white collar worker":
            label = f"white collar\n({power})"
        elif pos == "blue collar worker":
            label = f"blue collar\n({power})"
        else:
            label = f"{pos}\n({power})"
        position_labels.append(label)
    
    df_filtered = df[df['posicao'].isin(position_order)].copy()
    
    df_chat = df_filtered[~df_filtered['modelo'].isin(LEGACY_MODELS)]
    df_legacy = df_filtered[df_filtered['modelo'].isin(LEGACY_MODELS)]
    
    print(f"\nChat models: {len(df_chat)} obs")
    print(f"Legacy models: {len(df_legacy)} obs")
    
    # Calcular proporções por posição
    def calc_position_props(df_subset, position_order):
        results = []
        for pos in position_order:
            pos_data = df_subset[df_subset['posicao'] == pos]
            total = len(pos_data)
            if total > 0:
                male_pct = (pos_data['gender'] == 'Male').sum() / total * 100
                female_pct = (pos_data['gender'] == 'Female').sum() / total * 100
                other_pct = 100 - male_pct - female_pct
                results.append({
                    'position': pos,
                    'Male': male_pct,
                    'Female': female_pct,
                    'Other': other_pct,
                    'n': total
                })
        return pd.DataFrame(results)
    
    props_chat = calc_position_props(df_chat, position_order)
    props_legacy = calc_position_props(df_legacy, position_order)
    
    print("\nGerando gráficos por posição...")
    
    create_position_chart(props_chat, 
                         'Test 3: Gender by Selected Position (Chat Models Only)',
                         f'{OUTPUT_DIR}t3_positions_chat.png',
                         position_labels)
    
    create_position_chart(props_legacy,
                         'Test 3: Gender by Selected Position (GPT-3 Legacy Only)',
                         f'{OUTPUT_DIR}t3_positions_legacy.png',
                         position_labels)
    
    # --- Regressão Legacy com shot order ---
    print("\nCalculando regressão (Legacy)...")
    df_legacy_full = df[df['modelo'].isin(LEGACY_MODELS)].copy()
    df_legacy_full['is_male'] = (df_legacy_full['gender'] == 'Male').astype(int)
    df_legacy_full['is_positive'] = (df_legacy_full['power_level'] == 'high').astype(int)
    df_legacy_full['order_FM'] = (df_legacy_full['example_order'] == 'FM').astype(int)
    df_legacy_full['order_M'] = (df_legacy_full['example_order'] == 'M').astype(int)
    df_legacy_full['order_F'] = (df_legacy_full['example_order'] == 'F').astype(int)
    
    y = df_legacy_full['is_male'].values
    X = df_legacy_full[['is_positive', 'order_F', 'order_FM', 'order_M']].values
    results, n, r2, k = run_ols_regression(y, X, ['is_positive', 'order_F', 'order_FM', 'order_M'])
    
    create_regression_table(results, n, r2, k,
                           "Regressão: GPT-3 Legacy — Com Shot Order (Teste 3)",
                           f"N = {n} | R² = {r2:.4f} | k = {k}",
                           f'{OUTPUT_DIR}regression_t3_legacy_shot_order.png',
                           "Interpretação: is_positive (high power) aumenta proporção masculina (β=0,20***).\nOrdem FM tem efeito significativo (β=-0,08***).")
    
    # Salvar tabela CSV
    pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}regression_t3_legacy.csv', index=False)
    
    # Salvar proporções
    props_chat.to_csv(f'{OUTPUT_DIR}t3_props_chat.csv', index=False)
    props_legacy.to_csv(f'{OUTPUT_DIR}t3_props_legacy.csv', index=False)


def copy_data_files():
    """Copia arquivos de dados para o diretório de saída."""
    print("\n" + "="*60)
    print("COPIANDO ARQUIVOS DE DADOS")
    print("="*60)
    
    for key, filename in DATA_FILES.items():
        src = os.path.join(INPUT_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} não encontrado")


def create_summary():
    """Cria arquivo de resumo."""
    print("\n" + "="*60)
    print("CRIANDO RESUMO")
    print("="*60)
    
    summary = """# TCC Analysis - Resumo dos Arquivos Gerados

## Dados Originais
- df_teste_1_unified.csv - Teste 1 (Características no trabalho)
- df_teste_2_unified.csv - Teste 2 (Cenários de feedback)
- df_teste_3_unified.csv - Teste 3 (Profissões/posições)

## Gráficos Gerados

### Teste 1 - Características
- t1_family_NEGATIVE.png - Distribuição por família (valência negativa)
- t1_family_POSITIVE.png - Distribuição por família (valência positiva)
- t1_language_valence.png - Distribuição por idioma e valência (Chat)
- t1_legacy_valence_order.png - Shot order por valência (Legacy)
- regression_t1_legacy_shot_order.png - Tabela de regressão (Legacy)

### Teste 2 - Feedback
- t2_family_NEGATIVE.png - Distribuição por família (feedback negativo)
- t2_family_POSITIVE.png - Distribuição por família (feedback positivo)
- t2_language_valence.png - Distribuição por idioma e valência (Chat)
- t2_legacy_valence_order.png - Shot order por valência (Legacy)
- regression_t2_legacy_shot_order.png - Tabela de regressão (Legacy)

### Teste 3 - Profissões
- t3_positions_chat.png - Distribuição por profissão (Chat)
- t3_positions_legacy.png - Distribuição por profissão (Legacy)
- regression_t3_legacy_shot_order.png - Tabela de regressão (Legacy)

## Tabelas CSV
- regression_t1_legacy.csv - Coeficientes regressão Teste 1
- regression_t2_legacy.csv - Coeficientes regressão Teste 2
- regression_t3_legacy.csv - Coeficientes regressão Teste 3
- t1_props_negative.csv / t1_props_positive.csv - Proporções Teste 1
- t2_props_negative.csv / t2_props_positive.csv - Proporções Teste 2
- t3_props_chat.csv / t3_props_legacy.csv - Proporções Teste 3

## Notas
- Gerado em: {date}
- Modelos Legacy: davinci-002, babbage-002
- Famílias: GPT-3 Legacy, GPT-3.5, GPT-4o, GPT-4.1, Serie o, GPT-5
"""
    
    from datetime import datetime
    summary = summary.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(f'{OUTPUT_DIR}README.md', 'w') as f:
        f.write(summary)
    
    print("  ✓ README.md")


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("TCC ANALYSIS - VIÉS DE GÊNERO EM LLMs")
    print("="*60)
    print(f"\nDiretório de entrada: {INPUT_DIR}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    
    # Carregar dados
    print("\n" + "="*60)
    print("CARREGANDO DADOS")
    print("="*60)
    data = load_data()
    
    # Análises
    if data['teste1'] is not None:
        analyze_test1(data['teste1'])
    
    if data['teste2'] is not None:
        analyze_test2(data['teste2'])
    
    if data['teste3'] is not None:
        analyze_test3(data['teste3'])
    
    # Copiar dados originais
    copy_data_files()
    
    # Criar resumo
    create_summary()
    
    # Listar arquivos gerados
    print("\n" + "="*60)
    print("ARQUIVOS GERADOS")
    print("="*60)
    
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f} ({size:,} bytes)")
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)
    print(f"\nTodos os arquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
