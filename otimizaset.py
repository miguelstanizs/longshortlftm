import re
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, value
from datetime import datetime
import time
from tqdm import tqdm

# ---------------------------
# Variáveis de configuração
# ---------------------------
rest = 50   # Limite da exposição setorial (|Long - Short| <= rest)

# Frequência de otimização: 'd' (diário), 's' (semanal), 'q' (quinzenal),
# 'm' (mensal), 't' (trimestral), 'a' (anual)
frequencia_otimizacao = 'q'

# Lista de tickers na ordem desejada para as colunas dos arquivos de saída
tickers = [
    'TGMA3', 'ARML3', 'VAMO3', 'ALPA4', 'AZZA3', 'ASAI3', 'CEAB3', 'CRFB3',
    'ESPA3', 'GGPS3', 'GMAT3', 'GRND3', 'GUAR3', 'LJQQ3', 'LREN3', 'MGLU3',
    'NTCO3', 'PETZ3', 'PNVL3', 'RADL3', 'SBFG3', 'SMFT3', 'VIVA3', 'VULC3',
    'DXCO3', 'FRAS3', 'INTB3', 'LEVE3', 'MILS3', 'MYPK3', 'POMO4', 'PRNR3',
    'PTBL3', 'RAPT4', 'TUPY3', 'WEGE3', 'CBAV3', 'CMIN3', 'CSNA3', 'GGBR4',
    'KLBN11', 'RANI3', 'SUZB3', 'USIM5', 'VALE3', 'ALOS3', 'CURY3', 'CYRE3',
    'DIRR3', 'EZTC3', 'IGTI11', 'JHSF3', 'MRVE3', 'MULT3', 'PLPL3', 'TEND3',
    'BMOB3', 'LWSA3', 'NGRD3', 'POSI3', 'TOTS3', 'VLID3', 'HYPE3', 'BRIT3',
    'DESK3', 'FIQE3', 'TIMS3', 'VIVT3', 'MOVI3', 'RENT3', 'RAIZ4', 'UGPA3',
    'VBBR3', 'ENAT3', 'OPCT3', 'PETR4', 'PRIO3', 'RECV3', 'BRAV3', 'CCRO3',
    'ECOR3', 'HBSA3', 'PORT3', 'RAIL3', 'STBP3', 'AESB3', 'AURE3', 'CMIG4',
    'CPFE3', 'CPLE6', 'CSMG3', 'EGIE3', 'ELET6', 'ENEV3', 'ENGI11', 'EQTL3',
    'NEOE3', 'SAPR11', 'SBSP3', 'SRNA3', 'ISAE4', 'ABCB4', 'B3SA3', 'BBAS3',
    'BBDC4', 'BBSE3', 'BPAC11', 'BRBI11', 'CSUD3', 'CXSE3', 'IRBR3', 'ITUB4',
    'QUAL3', 'PSSA3', 'SANB11', 'WIZC3', 'ABEV3', 'ZAMP3', 'BEEF3', 'BRFS3',
    'CAML3', 'JBSS3', 'MDIA3', 'MRFG3', 'JALL3', 'KEPL3', 'SLCE3', 'SMTO3',
    'SOJA3', 'TTEN3', 'ANIM3', 'COGN3', 'YDUQ3', 'FLRY3', 'HAPV3', 'ODPV3',
    'ONCO3', 'RDOR3', 'XPBR31', 'MLAS3', 'CIEL3', 'SOMA3', 'BPAN4', 'BRSR6',
    'LOGG3', 'EVEN3', 'MDNE3', 'LAVV3', 'DASA3', 'MATD3', 'VVEO3', 'CLSA3',
    'CASH3', 'TRIS3', 'BLAU3', 'GFSA3', 'MTRE3', 'IFCM3', 'MBLY3', 'AMAR3',
    'BRML3', 'TAEE11', 'ALUP11', 'AZUL4', 'UNIP6', 'ORVR3', 'AMBP3', 'CVCB3',
    'TASA4', 'GOLL4', 'EMBR3'
]

def clean_sheet_name(name):
    return re.sub(r'[:\\/*?\[\]]', '_', name)

def deve_otimizar(data_atual, data_ultima_otimizacao, frequencia):
    """
    Retorna True se, a partir da última otimização, 
    deve otimizar novamente segundo a frequência configurada.
    """
    if frequencia == 'd':
        return True
    if data_ultima_otimizacao is None:
        return True
    # converter strings para datetime
    if isinstance(data_atual, str):
        data_atual = pd.to_datetime(data_atual)
    if isinstance(data_ultima_otimizacao, str):
        data_ultima_otimizacao = pd.to_datetime(data_ultima_otimizacao)
    # checar frequência
    if frequencia == 's':
        return (data_atual.isocalendar()[1] != data_ultima_otimizacao.isocalendar()[1] or
                data_atual.year != data_ultima_otimizacao.year)
    if frequencia == 'q':
        return (data_atual - data_ultima_otimizacao).days >= 15
    if frequencia == 'm':
        return (data_atual.month != data_ultima_otimizacao.month or
                data_atual.year != data_ultima_otimizacao.year)
    if frequencia == 't':
        return ((data_atual.month - 1)//3 != (data_ultima_otimizacao.month - 1)//3 or
                data_atual.year != data_ultima_otimizacao.year)
    if frequencia == 'a':
        return data_atual.year != data_ultima_otimizacao.year
    return True

def resolver_otimizacao(df_dia_long, df_dia_short, n_assets, dia_label):
    """
    Função modificada para resolver o problema de otimização 
    com DataFrame separados para long e short
    """
    # Verifica se há ativos suficientes em ambos os DataFrames
    if len(df_dia_long) < n_assets or len(df_dia_short) < n_assets:
        return None

    # ----- Modelo de otimização -----
    modelo = LpProblem(f"Otim_{dia_label}", LpMaximize)
    
    # Índices separados para long e short
    indices_long = df_dia_long.index.tolist()
    indices_short = df_dia_short.index.tolist()
    
    # Pré-calcular índices por setor para evitar recálculos repetidos
    setores_unicos = set(df_dia_long["Setor"].unique()).union(set(df_dia_short["Setor"].unique()))
    indices_long_por_setor = {s: df_dia_long.index[df_dia_long["Setor"] == s].tolist() for s in setores_unicos}
    indices_short_por_setor = {s: df_dia_short.index[df_dia_short["Setor"] == s].tolist() for s in setores_unicos}
    
    # Criar variáveis em lotes
    x_long = {j: LpVariable(f"Long_{j}", cat=LpBinary) for j in indices_long}
    x_short = {j: LpVariable(f"Short_{j}", cat=LpBinary) for j in indices_short}

    # Objetivo - usando vetorização para criar termos
    # Maximizamos as notas long e minimizamos as notas short (colocando sinal negativo)
    notas_long = df_dia_long["Nota"].to_dict()
    notas_short = df_dia_short["Nota"].to_dict()
    
    # O objetivo é maximizar a soma das notas long e minimizar a soma das notas short
    modelo += lpSum(notas_long[j] * x_long[j] for j in indices_long) - lpSum(notas_short[j] * x_short[j] for j in indices_short)

    # Restrições para selecionar n_assets para cada conjunto
    modelo += lpSum(x_long[j] for j in indices_long) == n_assets
    modelo += lpSum(x_short[j] for j in indices_short) == n_assets

    # Restrição setorial - otimizada
    for s in setores_unicos:
        idx_long_s = indices_long_por_setor.get(s, [])
        idx_short_s = indices_short_por_setor.get(s, [])
        
        # Verificamos se existem elementos para evitar erro
        long_sum = lpSum(x_long[j] for j in idx_long_s) if idx_long_s else 0
        short_sum = lpSum(x_short[j] for j in idx_short_s) if idx_short_s else 0
        
        modelo += long_sum - short_sum <= rest
        modelo += long_sum - short_sum >= -rest

    # Resolve
    status = modelo.solve()
    if LpStatus[modelo.status] != "Optimal":
        return None

    # Extrai resultados - otimizado com listas de compreensão
    long_positions = [
        {"Posicao": "Long", "Empresa": df_dia_long.loc[j, "Empresa"], "Nota": df_dia_long.loc[j, "Nota"], "Setor": df_dia_long.loc[j, "Setor"]}
        for j in indices_long if x_long[j].varValue == 1
    ]
    
    short_positions = [
        {"Posicao": "Short", "Empresa": df_dia_short.loc[j, "Empresa"], "Nota": df_dia_short.loc[j, "Nota"], "Setor": df_dia_short.loc[j, "Setor"]}
        for j in indices_short if x_short[j].varValue == 1
    ]
    
    df_result = pd.DataFrame(long_positions + short_positions)
    return df_result

# ---------------------------
# Leitura e extração das tabelas
# ---------------------------
print("Carregando dados do arquivo Excel...")
arquivo_excel = "notas_ls.xlsx"

# Lê as duas abas do arquivo Excel
df_long = pd.read_excel(arquivo_excel, sheet_name="notas_long", header=None)
df_short = pd.read_excel(arquivo_excel, sheet_name="notas_short", header=None)

# Extrair trimestres (coluna A), setores e tickers para cada aba
trimestres_long = df_long.iloc[3:, 0].reset_index(drop=True)
setores_row_long = df_long.iloc[1, 3:].reset_index(drop=True)
empresas_row_long = df_long.iloc[2, 3:].reset_index(drop=True)

trimestres_short = df_short.iloc[3:, 0].reset_index(drop=True)
setores_row_short = df_short.iloc[1, 3:].reset_index(drop=True)
empresas_row_short = df_short.iloc[2, 3:].reset_index(drop=True)

# Verificamos se os trimestres e empresas são iguais nas duas abas
if not trimestres_long.equals(trimestres_short):
    print("AVISO: Os trimestres nas abas long e short não são idênticos!")

# Dados diários: linhas a partir do índice 3
df_notas_long = df_long.iloc[3:].reset_index(drop=True)
df_notas_short = df_short.iloc[3:].reset_index(drop=True)

# Pré-alocação dos DataFrames de saída
total_dias = len(df_notas_long)  # Assumimos que as duas abas têm o mesmo número de dias
df_long_all = pd.DataFrame(columns=['Data'] + tickers)
df_short_all = pd.DataFrame(columns=['Data'] + tickers)

# Converter tickers para conjunto para pesquisa rápida
tickers_set = set(tickers)

# Preparar estruturas de saída
resultados_por_dia = {}
portfolio_anterior = None
ultima_data_otimizacao = None
trimestre_anterior = None
datas_otimizacao = []

# Pré-processamento: filtrar e preparar dados
print("Pré-processando dados...")
notas_long_anteriores = None
notas_short_anteriores = None
dia_label_anterior = None

# ---------------------------
# Loop principal com barra de progresso
# ---------------------------
print("Iniciando otimização de portfólio...")
long_rows = []
short_rows = []

for i, (row_long, row_short) in tqdm(enumerate(zip(df_notas_long.iterrows(), df_notas_short.iterrows())), 
                                     total=len(df_notas_long), desc="Processando dias"):
    _, row_long_data = row_long
    _, row_short_data = row_short
    
    # Extrai trimestre, n_assets e dia_label
    trimestre_atual = trimestres_long[i]
    n_assets = row_long_data[1]
    try:
        n_assets = int(n_assets)
    except:
        # se n_assets inválido, repete portfólio anterior
        dia_label = row_long_data[2] if not pd.isna(row_long_data[2]) else f"Dia_{i+4}"
        resultados_por_dia[dia_label] = portfolio_anterior.copy() if portfolio_anterior is not None else None
        continue

    dia_raw = row_long_data[2]
    dia_label = str(dia_raw) if not pd.isna(dia_raw) else f"Dia_{i+4}"

    # Checa se precisa otimizar: primeiro dia, frequência, OU mudança de trimestre
    primeiro_dia = (i == 0)
    mudou_trimestre = (trimestre_atual != trimestre_anterior and trimestre_anterior is not None)
    por_frequencia = deve_otimizar(dia_label, ultima_data_otimizacao, frequencia_otimizacao)
    precisa_otimizar = primeiro_dia or mudou_trimestre or por_frequencia

    if not precisa_otimizar and portfolio_anterior is not None:
        # Repetir portfólio anterior - otimizado para não usar concat
        resultados_por_dia[dia_label] = portfolio_anterior.copy()
        
        # Montar linhas binárias - otimizado
        long_empresas = set(portfolio_anterior.query("Posicao=='Long'")["Empresa"].tolist())
        short_empresas = set(portfolio_anterior.query("Posicao=='Short'")["Empresa"].tolist())
        
        long_row = {"Data": dia_label}
        short_row = {"Data": dia_label}
        
        for ticker in tickers:
            long_row[ticker] = 1 if ticker in long_empresas else 0
            short_row[ticker] = 1 if ticker in short_empresas else 0
            
        long_rows.append(long_row)
        short_rows.append(short_row)
        
        notas_long_anteriores = row_long_data[3:]
        notas_short_anteriores = row_short_data[3:]
        trimestre_anterior = trimestre_atual
        continue

    # Extrai notas (primeiro dia usa próprio dia, depois usa notas anteriores)
    if primeiro_dia or notas_long_anteriores is None or notas_short_anteriores is None:
        notas_long = row_long_data[3:]
        notas_short = row_short_data[3:]
    else:
        notas_long = notas_long_anteriores
        notas_short = notas_short_anteriores

    # Monta DataFrame do dia e filtra notas -999 para long e short
    df_dia_long = pd.DataFrame({
        "Empresa": empresas_row_long,
        "Setor": setores_row_long,
        "Nota": notas_long.reset_index(drop=True)
    }).query("Nota != -999").reset_index(drop=True)
    
    df_dia_short = pd.DataFrame({
        "Empresa": empresas_row_short,
        "Setor": setores_row_short,
        "Nota": notas_short.reset_index(drop=True)
    }).query("Nota != -999").reset_index(drop=True)

    # Resolve otimização usando função separada
    df_result = resolver_otimizacao(df_dia_long, df_dia_short, n_assets, dia_label)
    
    if df_result is None:
        resultados_por_dia[dia_label] = portfolio_anterior.copy() if portfolio_anterior is not None else None
        continue
    
    resultados_por_dia[dia_label] = df_result.copy()
    portfolio_anterior = df_result.copy()
    
    # Atualiza controles
    ultima_data_otimizacao = dia_label
    trimestre_anterior = trimestre_atual
    datas_otimizacao.append(dia_label)
    notas_long_anteriores = row_long_data[3:]
    notas_short_anteriores = row_short_data[3:]

    # Constrói linhas long/short binárias - otimizado
    long_empresas = set(df_result.query("Posicao=='Long'")["Empresa"].tolist())
    short_empresas = set(df_result.query("Posicao=='Short'")["Empresa"].tolist())
    
    long_row = {"Data": dia_label}
    short_row = {"Data": dia_label}
    
    for ticker in tickers:
        long_row[ticker] = 1 if ticker in long_empresas else 0
        short_row[ticker] = 1 if ticker in short_empresas else 0
        
    long_rows.append(long_row)
    short_rows.append(short_row)

# Cria DataFrames a partir das listas coletadas - mais eficiente que concat repetido
print("Preparando resultados para exportação...")
df_long_all = pd.DataFrame(long_rows)
df_short_all = pd.DataFrame(short_rows)

# ---------------------------
# Exportação de resultados
# ---------------------------
print("Exportando resultados para arquivos Excel...")
df_long_all.to_excel("port_comp.xlsx", index=False)
df_short_all.to_excel("bottom_comp.xlsx", index=False)

# Novo: exporta lista de datas de otimização
df_datas = pd.DataFrame({"Data_otimizacao": datas_otimizacao})
df_datas.to_excel("datas_otimizacao.xlsx", index=False)

print("Processo concluído com sucesso!")
print("Exportados:")
print(" - port_comp.xlsx")
print(" - bottom_comp.xlsx")
print(" - datas_otimizacao.xlsx")