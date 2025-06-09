import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import csv

from collections import defaultdict
import os
from datetime import datetime


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.patches as patches

TIMEOUT = 3600


def carregar_dados(caminho_arquivo):

    dados = {
        'parametros': {},
        'veiculos': [],
        'ums': [],
        'clientes': []
    }

    with open(caminho_arquivo, mode='r', encoding='utf-8') as file:

        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            tipo = row['tipo']

            if tipo == 'parametro':
                dados['parametros'][row['descricao']] = float(row['valor'])

            elif tipo == 'cliente':
                dados['clientes'].append({
                    'id': int(row['id']),
                    'nome': row['descricao'],
                    'destino': row['destino']
                })

            elif tipo == 'veiculo':
                dados['veiculos'].append({
                    'id': int(row['id']),
                    'tipo': row['descricao'].replace('Veiculo_', ''),
                    'capacidade_peso': float(row['capacidade_peso']),
                    'capacidade_volume': float(row['capacidade_vol']),
                    'custo': float(row['custo']),
                    'carga_minima': float(row['carga_minima']),
                    'destino': row['destino'] if 'destino' in row else None
                })

            elif tipo == 'um':
                cliente_id = int(row['cliente'])

                destino = next(
                    (c['destino'] for c in dados['clientes'] if c['id'] == cliente_id), '')

                compatibilidade = row['compatibilidade'].strip()
                if not compatibilidade:
                    compatibilidade = ",".join(
                        str(v['tipo']) for v in dados['veiculos'])

                dados['ums'].append({
                    'id': int(row['id']),
                    'tipo': row['descricao'],
                    'peso': float(row['peso']),
                    'volume': float(row['volume']),
                    'destino': destino,
                    'cliente': cliente_id,
                    'compatibilidade': row['compatibilidade'] or ",".join(str(v['tipo']) for v in dados['veiculos']),
                    'restricao': row['restricao'],
                    'penalidade': float(row['penalidade'])






                })

    return dados


def criar_instancia(tipo_instancia):

    dados = carregar_dados(tipo_instancia)

    return {
        "veiculos": dados['veiculos'],
        "ums": dados['ums'],
        "clientes": dados['clientes'],

    }


def criar_modelo(instancia):

    model = gp.Model("AlocacaoCargas")

    veiculos = instancia["veiculos"]
    ums = instancia["ums"]
    clientes = instancia["clientes"]

    destino_para_clientes = defaultdict(list)
    for cliente in clientes:
        destino_para_clientes[cliente['destino']].append(cliente['nome'])

    delta = {}
    for v in veiculos:
        for c in clientes:

            delta[(c['id'], v['id'])] = 1 if v['destino'] == c['destino'] else 0

    x = {}
    y = {}
    alpha = {}

    beta_v = 1

    frete_morto_por_veiculo = {}

    for i in ums:
        for v in veiculos:
            for c in clientes:
                x[(i["id"], v["id"], c["id"])] = model.addVar(vtype=GRB.BINARY,

                                                              name=f"x_{i['id']}_{v['id']}_{c['id']}")

    for v in veiculos:

        alpha[v["id"]] = model.addVar(
            vtype=GRB.BINARY, name=f"alpha_{v['id']}")

        for c in clientes:

            y[(v["id"], c["id"])] = model.addVar(
                vtype=GRB.BINARY, name=f"y_{v['id']}_{c['id']}")

    custo_nao_alocacao = gp.quicksum(

        i["peso"] * i["penalidade"] *

        (1 - gp.quicksum(x[(i["id"], v["id"], c["id"])]
         for v in veiculos for c in clientes))
        for i in ums
    )

    custo_frete_morto = gp.quicksum(beta_v * (v["capacidade_peso"] * alpha[v["id"]] -
                                              gp.quicksum(i["peso"] * x[(i["id"], v["id"], c["id"])]
                                                          for i in ums for c in clientes))
                                    for v in veiculos
                                    )

    custo_transporte = gp.quicksum(
        v["custo"] * alpha[v["id"]]
        for v in veiculos
    )

    model.setObjective(custo_nao_alocacao +
                       custo_frete_morto + custo_transporte, GRB.MINIMIZE)

    for v in veiculos:

        model.addConstr(
            gp.quicksum(i["peso"] * x[(i["id"], v["id"], c["id"])]
                        for i in ums for c in clientes) <= v["capacidade_peso"],
            name=f"cap_peso_{v['id']}"
        )

        model.addConstr(
            gp.quicksum(i["volume"] * x[(i["id"], v["id"], c["id"])]
                        for i in ums for c in clientes) <= v["capacidade_volume"],
            name=f"cap_vol_{v['id']}"
        )

        model.addConstr(gp.quicksum(i["peso"] * x[(i["id"], v["id"], c["id"])]
                                    for i in ums for c in clientes) >= alpha[v["id"]] * v["carga_minima"],
                        name=f"frete_morto_minimo_{v['id']}"
                        )

        for c in clientes:
            model.addConstr(
                alpha[v["id"]] >= y[(v["id"], c["id"])],
                name=f"ativacao_{v['id']}_{c['id']}"
            )

    for i in ums:

        model.addConstr(
            gp.quicksum(x[(i["id"], v["id"], c["id"])]
                        for v in veiculos for c in clientes) <= 1,
            name=f"alocacao_unica_{i['id']}"
        )

        for v in veiculos:
            for c in clientes:

                veiculos_compatíveis = [vc.strip()
                                        for vc in i['compatibilidade'].split(',')]

                gamma = 1 if v['tipo'] in veiculos_compatíveis else 0
                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= gamma,
                    name=f"compat_{i['id']}_{v['id']}_{c['id']}"
                )

                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= y[(
                        v["id"], c["id"])],
                    name=f"aloc_uso_{i['id']}_{v['id']}_{c['id']}"
                )

                model.addConstr(
                    x[(i["id"], v["id"], c["id"])] <= delta.get(
                        (c["id"], v["id"]), 0),
                    name=f"destino_{i['id']}_{v['id']}_{c['id']}"
                )

            for v in veiculos:
                model.addConstr(
                    alpha[v["id"]] <= gp.quicksum(
                        y[(v["id"], c["id"])] for c in clientes),
                    name=f"ativacao_max_{v['id']}"
                )

    return model, x, y, alpha


def gerar_visualizacoes(resultados, instancia, pasta_saida):

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = resultados['tipo_instancia']

    plot_tempo_execucao(resultados, pasta_saida, nome_base)
    plot_gap_otimizacao(resultados, pasta_saida, nome_base)
    plot_status_solucao(resultados, pasta_saida, nome_base)

    plot_utilizacao_veiculos(resultados, pasta_saida, nome_base)
    plot_distribuicao_utilizacao(resultados, pasta_saida, nome_base)
    plot_ums_por_veiculo(resultados, pasta_saida, nome_base)

    plot_distribuicao_alocacao(resultados, instancia, pasta_saida, nome_base)

    plot_composicao_custos(resultados, pasta_saida, nome_base)
    plot_custo_por_componente(resultados, pasta_saida, nome_base)
    plot_penalidades_nao_alocacao(resultados, pasta_saida, nome_base)

    if resultados['ums_nao_alocadas'] > 0:
        plot_heatmap_compatibilidade(instancia, pasta_saida, nome_base)
        plot_distribuicao_ums_nao_alocadas(
            instancia, resultados, pasta_saida, nome_base)


def plot_distribuicao_alocacao(resultados, instancia, pasta_saida, nome_base):

    plt.figure(figsize=(16, 12))
    ax = plt.gca()

    um_width = 0.8
    um_height = 0.9
    espacamento_vertical = 1.2
    margin_left = 2.0
    ums_por_linha = 8
    altura_por_linha = 1.2

    cores_veiculos = plt.cm.tab20.colors
    cores_ums = plt.cm.Set3.colors

    tipos_veiculos = sorted(
        list(set(v['tipo'] for v in instancia['veiculos'])))
    tipos_ums = sorted(list(set(um['tipo'] for um in instancia['ums'])))

    cor_veiculo = {tipo: cores_veiculos[i % len(cores_veiculos)]
                   for i, tipo in enumerate(tipos_veiculos)}
    cor_um = {tipo: cores_ums[i % len(cores_ums)]
              for i, tipo in enumerate(tipos_ums)}

    y_pos = 0
    ums_alocadas = set()

    for aloc in resultados['alocacoes']:
        veic_id = aloc['veiculo_id']
        veic_tipo = aloc['veiculo_tipo']
        ums = aloc['cargas']
        tipos_um = aloc['tipos_um']

        num_linhas = (len(ums) + ums_por_linha - 1) // ums_por_linha
        altura_veiculo = 1.0 + (num_linhas * altura_por_linha)

        ax.add_patch(patches.Rectangle(
            (margin_left, y_pos - altura_veiculo/2),
            width=ums_por_linha,
            height=altura_veiculo,
            facecolor=cor_veiculo[veic_tipo],
            alpha=0.2,
            edgecolor='black',
            linewidth=1.5
        ))

        ax.text(margin_left - 0.5, y_pos,
                f'V{veic_id} ({veic_tipo})\n{len(ums)} UMs',
                ha='right', va='center', fontsize=10)

        for i, (um_id, um_tipo) in enumerate(zip(ums, tipos_um)):
            linha = i // ums_por_linha
            coluna = i % ums_por_linha

            x_pos = margin_left + coluna
            y_um = y_pos - altura_veiculo/2 + (linha + 0.7) * altura_por_linha

            ax.add_patch(patches.Rectangle(
                (x_pos, y_um),
                width=um_width,
                height=um_height,
                facecolor=cor_um[um_tipo],
                edgecolor='black',
                linewidth=0.8
            ))
            ax.text(x_pos + um_width/2, y_um + um_height/2,
                    f'UM{um_id}',
                    ha='center', va='center', fontsize=6)

            ums_alocadas.add(um_id)

        y_pos -= (altura_veiculo + espacamento_vertical)

    ums_nao_alocadas = [um for um in instancia['ums']
                        if um['id'] not in ums_alocadas]

    if ums_nao_alocadas:
        y_pos -= espacamento_vertical
        ax.text(margin_left - 0.5, y_pos,
                f'UMs Não Alocadas: {len(ums_nao_alocadas)}',
                ha='right', va='center', fontsize=10)

        num_linhas_na = (len(ums_nao_alocadas) +
                         ums_por_linha - 1) // ums_por_linha

        for i, um in enumerate(ums_nao_alocadas):
            linha = i // ums_por_linha
            coluna = i % ums_por_linha

            x_pos = margin_left + coluna
            y_um = y_pos - (linha * (um_height + 0.2))

            ax.add_patch(patches.Rectangle(
                (x_pos, y_um),
                width=um_width,
                height=um_height,
                facecolor=cor_um[um['tipo']],
                edgecolor='black',
                linestyle='dashed',
                linewidth=0.8
            ))
            ax.text(x_pos + um_width/2, y_um + um_height/2,
                    f'UM{um["id"]}',
                    ha='center', va='center', fontsize=6)

        y_min = y_pos - (num_linhas_na * (um_height + 0.2)) - \
            espacamento_vertical
    else:
        y_min = y_pos

    ax.set_xlim(0, margin_left + ums_por_linha + 1)
    ax.set_ylim(y_min, 2)
    ax.axis('off')

    legend_elements = []
    for tipo, cor in cor_veiculo.items():
        legend_elements.append(patches.Patch(
            facecolor=cor, alpha=0.2, edgecolor='black',
            label=f'Veículo {tipo}'))

    for tipo, cor in cor_um.items():
        legend_elements.append(patches.Patch(
            facecolor=cor, edgecolor='black',
            label=f'UM {tipo}'))

    ax.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5),
              fontsize=9)

    plt.title(f'Distribuição de Cargas - {nome_base}\n', fontsize=12)
    plt.tight_layout()

    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, f"{nome_base}_alocacao_organizada.png")
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho


def plot_tempo_execucao(resultados, pasta_saida, nome_base):
    plt.figure(figsize=(10, 6))
    plt.bar(nome_base, resultados['tempo_execucao'], color='skyblue')
    plt.axhline(y=TIMEOUT, color='r', linestyle='--', label='Timeout')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo de Execução')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_tempo_execucao.png"), dpi=300)
    plt.close()


def plot_gap_otimizacao(resultados, pasta_saida, nome_base):
    if resultados['gap_otimizacao'] is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(nome_base, resultados['gap_otimizacao'], color='orange')
        plt.ylabel('GAP (%)')
        plt.title('GAP de Otimização')
        plt.tight_layout()
        plt.savefig(os.path.join(
            pasta_saida, f"{nome_base}_gap_otimizacao.png"), dpi=300)
        plt.close()


def plot_status_solucao(resultados, pasta_saida, nome_base):
    status_map = {
        GRB.OPTIMAL: "Ótimo",
        GRB.TIME_LIMIT: "Timeout",
        GRB.INFEASIBLE: "Inviável",
        GRB.INF_OR_UNBD: "Infinito/Ilimitado",
        GRB.UNBOUNDED: "Ilimitado"
    }
    status = status_map.get(resultados['status'], "Desconhecido")

    plt.figure(figsize=(6, 6))
    plt.pie([1], labels=[status], autopct='%1.0f%%', colors=['lightgreen'])
    plt.title('Status da Solução')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_status_solucao.png"), dpi=300)
    plt.close()


def plot_utilizacao_veiculos(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])
    df = df.sort_values('veiculo_id')

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(df))

    bars1 = ax.bar(x - bar_width/2,
                   df['peso_total'], bar_width, label='Peso Real')
    bars2 = ax.bar(x + bar_width/2,
                   df['peso_minimo'], bar_width, label='Peso Mínimo')

    ax.set_xlabel('Veículos')
    ax.set_ylabel('Peso (kg)')
    ax.set_title('Comparação: Peso Real vs Peso Mínimo')
    ax.set_xticks(x)
    ax.set_xticklabels(df['veiculo_id'])
    ax.legend()

    for i, cap in enumerate(df['capacidade_peso']):
        ax.axhline(y=cap, xmin=(i - 0.5)/len(x), xmax=(i + 0.5)/len(x),
                   color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_utilizacao_veiculos.png"), dpi=300)
    plt.close()


def plot_distribuicao_utilizacao(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='taxa_utilizacao_peso',
                 bins=10, kde=True, color='skyblue')
    plt.xlabel('Taxa de Utilização de Peso (%)')
    plt.ylabel('Número de Veículos')
    plt.title('Distribuição das Taxas de Utilização de Peso')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_distribuicao_utilizacao.png"), dpi=300)
    plt.close()


def plot_ums_por_veiculo(resultados, pasta_saida, nome_base):
    if not resultados['alocacoes']:
        return

    df = pd.DataFrame(resultados['alocacoes'])
    df['num_cargas'] = df['cargas'].apply(len)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='veiculo_id', y='num_cargas', hue='veiculo_tipo', dodge=False)
    plt.xlabel('ID do Veículo')
    plt.ylabel('Número de UMs Transportadas')
    plt.title('Distribuição de UMs por Veículo')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_ums_por_veiculo.png"), dpi=300)
    plt.close()


def plot_composicao_custos(resultados, pasta_saida, nome_base):
    componentes = ['Transporte', 'Frete Morto', 'Não Alocação']
    valores = [
        resultados['custo_transporte'],
        resultados['frete_morto_total'],
        resultados['custo_nao_alocacao']
    ]

    plt.figure(figsize=(8, 8))
    plt.pie(valores, labels=componentes, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.title('Composição do Custo Total')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f"{nome_base}_composicao_custos.png"), dpi=300)
    plt.close()


def plot_custo_por_componente(resultados, pasta_saida, nome_base):
    componentes = ['Transporte', 'Frete Morto', 'Não Alocação']
    valores = [
        resultados['custo_transporte'],
        resultados['frete_morto_total'],
        resultados['custo_nao_alocacao']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(componentes, valores, color=['blue', 'red', 'green'])
    plt.ylabel('Custo (R$)')
    plt.title('Custo por Componente')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'R${height:,.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_custo_por_componente.png"), dpi=300)
    plt.close()


def plot_penalidades_nao_alocacao(resultados, pasta_saida, nome_base):
    if resultados['ums_nao_alocadas'] == 0:
        return

    dados = {
        'Peso Não Alocado': resultados['peso_nao_alocado'],
        'Volume Não Alocado': resultados['volume_nao_alocado']
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(dados.keys(), dados.values(), color=['orange', 'purple'])
    plt.ylabel('Valor Total')
    plt.title('Recursos Não Alocados')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:,.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_penalidades_nao_alocacao.png"), dpi=300)
    plt.close()


def plot_heatmap_compatibilidade(instancia, pasta_saida, nome_base):

    compat_data = []
    for um in instancia['ums']:
        compat_veiculos = []
        for veiculo in instancia['veiculos']:
            compat = 1 if veiculo['tipo'] in um['compatibilidade'].split(
                ',') else 0
            compat_veiculos.append(compat)
        compat_data.append(compat_veiculos)

    df = pd.DataFrame(
        compat_data,
        index=[f"UM_{um['id']}" for um in instancia['ums']],
        columns=[f"V_{v['id']}({v['tipo']})" for v in instancia['veiculos']]
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap="Blues", cbar=False)
    plt.title('Matriz de Compatibilidade UMs x Veículos')
    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_heatmap_compatibilidade.png"), dpi=300)
    plt.close()


def plot_distribuicao_ums_nao_alocadas(instancia, resultados, pasta_saida, nome_base):
    alocados_ids = set()
    for aloc in resultados['alocacoes']:
        alocados_ids.update(aloc['cargas'])

    ums_nao_alocadas = [um for um in instancia['ums']
                        if um['id'] not in alocados_ids]

    if not ums_nao_alocadas:
        return

    df = pd.DataFrame(ums_nao_alocadas)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df, y='peso', ax=axes[0])
    axes[0].set_title('Distribuição de Peso das UMs Não Alocadas')

    sns.boxplot(data=df, y='volume', ax=axes[1])
    axes[1].set_title('Distribuição de Volume das UMs Não Alocadas')

    plt.tight_layout()
    plt.savefig(os.path.join(
        pasta_saida, f"{nome_base}_distribuicao_ums_nao_alocadas.png"), dpi=300)
    plt.close()


def executar_instancia_com_timeout(tipo_instancia, instancia):

    try:
        print(f"\n{'='*80}")
        print(f"INICIANDO INSTÂNCIA: {tipo_instancia.upper()}")
        print(f"{'='*80}")

        modelo, x, y, alpha = criar_modelo(instancia)

        modelo.Params.TimeLimit = TIMEOUT

        modelo.Params.LogFile = os.path.join(os.path.dirname(
            __file__), 'OtimizacaoQualif', 'Resultados', f"gurobi_log_{tipo_instancia}.log")

        modelo.Params.OutputFlag = 1

        modelo.optimize()

        resultados = {
            'tipo_instancia': tipo_instancia,
            'status': modelo.status,
            'tempo_execucao': modelo.Runtime,
            'custo_total': None,
            'veiculos_ativos': 0,
            'veiculos_inativos': len(instancia["veiculos"]),
            'ums_alocadas': 0,
            'ums_nao_alocadas': len(instancia["ums"]),
            'peso_nao_alocado': 0,
            'volume_nao_alocado': 0,
            'frete_morto_total': 0,
            'custo_transporte': 0,
            'custo_nao_alocacao': 0,
            'alocacoes': [],

            'tempo_para_otimo': modelo.RunTime if modelo.status == GRB.OPTIMAL else None,
            'melhor_solucao': modelo.ObjVal if modelo.SolCount > 0 else None,

            'solucao_relaxada': modelo.ObjBound if modelo.SolCount > 0 else None,

            'gap_otimizacao': modelo.MIPGap*100 if hasattr(modelo, 'MIPGap') else None,
        }

        if modelo.SolCount > 0:

            x_val = {(i["id"], v["id"], c["id"]): x[(i["id"], v["id"], c["id"])].x



                     for i in instancia["ums"]
                     for v in instancia["veiculos"]
                     for c in instancia["clientes"]}

            y_val = {(v["id"], c["id"]): y[(v["id"], c["id"])].x
                     for v in instancia["veiculos"]
                     for c in instancia["clientes"]}

            beta_v = 1
            frete_morto = 0.0
            for v in instancia["veiculos"]:
                v_id = v["id"]
                capacidade = v["capacidade_peso"]
                carga_real = sum(
                    i["peso"] * x[(i["id"], v_id, c["id"])].X
                    for i in instancia["ums"]
                    for c in instancia["clientes"]
                )
                ativo = alpha[v_id].X
                frete_morto_kg = max(0, capacidade * ativo - carga_real)
                frete_morto += beta_v * frete_morto_kg

            resultados['frete_morto_total'] = frete_morto

            if hasattr(modelo, 'ObjVal'):

                resultados['custo_total'] = modelo.ObjVal

            resultados['custo_total'] = modelo.objVal

            resultados['veiculos_ativos'] = sum(

                1 for v in instancia["veiculos"]
                if any(x_val.get((i["id"], v["id"], c["id"]), 0) > 0.9
                       for i in instancia["ums"]
                       for c in instancia["clientes"])

                or alpha[v["id"]].x > 0.9
            )

            resultados['veiculos_inativos'] = len(
                instancia["veiculos"]) - resultados['veiculos_ativos']

            nao_alocadas = [
                i["id"] for i in instancia["ums"]
                if all(x_val.get((i["id"], v["id"], c["id"]), 0) < 0.1

                       for v in instancia["veiculos"]

                       for c in instancia["clientes"])
            ]

            resultados['ums_nao_alocadas'] = len(nao_alocadas)
            resultados['ums_alocadas'] = len(
                instancia["ums"]) - len(nao_alocadas)

            resultados['peso_nao_alocado'] = sum(
                i["peso"] for i in instancia["ums"] if i["id"] in nao_alocadas)
            resultados['volume_nao_alocado'] = sum(
                i["volume"] for i in instancia["ums"] if i["id"] in nao_alocadas)

            resultados['custo_transporte'] = sum(
                v["custo"] * alpha[v["id"]].X
                for v in instancia["veiculos"]
            )

            resultados['custo_nao_alocacao'] = sum(
                i["peso"] * i["penalidade"] * (1 - sum(



                    x_val.get((i["id"], v["id"], c["id"]), 0)
                    for v in instancia["veiculos"]
                    for c in instancia["clientes"]))
                for i in instancia["ums"]
            )

            for v in instancia["veiculos"]:
                cargas = [
                    i["id"] for i in instancia["ums"]
                    if any(x_val.get((i["id"], v["id"], c["id"]), 0) > 0.9
                           for c in instancia["clientes"])
                ]

                if cargas:
                    tipo_carga = [next((um["tipo"]
                                        for um in instancia["ums"] if um["id"] == um_id), "Desconhecido")
                                  for um_id in cargas]

                    peso_total = sum(i["peso"]
                                     for i in instancia["ums"] if i["id"] in cargas)
                    volume_total = sum(i["volume"]
                                       for i in instancia["ums"] if i["id"] in cargas)

                    resultados['alocacoes'].append({
                        'veiculo_id': v["id"],
                        'veiculo_tipo': v["tipo"],
                        'destino': v["destino"],
                        'cargas': cargas,
                        'tipos_um': tipo_carga,
                        'peso_total': peso_total,
                        'peso_minimo': v["carga_minima"],
                        'capacidade_peso': v["capacidade_peso"],
                        'volume_total': volume_total,
                        'capacidade_volume': v["capacidade_volume"],
                        'custo_veiculo': v["custo"],
                        'frete_morto': frete_morto,
                        'taxa_utilizacao_peso': (peso_total / v["capacidade_peso"]) * 100,
                        'taxa_utilizacao_volume': (volume_total / v["capacidade_volume"]) * 100
                    })

        if resultados and modelo.SolCount > 0:
            pasta_visualizacoes = os.path.join(
                os.path.dirname(__file__), 'OtimizacaoQualif', 'Visualizacoes')
            gerar_visualizacoes(resultados, instancia, pasta_visualizacoes)

        return resultados

    except Exception as e:
        print(f"❌ Erro ao processar instância {tipo_instancia}: {str(e)}")
        return None


def imprimir_resultados_detalhados(resultados):
    print(f"\n{'='*80}")
    print(
        f" 🟢 RESULTADOS PARA INSTÂNCIA: {resultados['tipo_instancia'].upper()}")
    print(f"{'='*80}")

    status_map = {
        GRB.OPTIMAL: "Ótimo encontrado",
        GRB.TIME_LIMIT: "Tempo limite atingido",
        GRB.INFEASIBLE: "Problema inviável",
        GRB.INF_OR_UNBD: "Infinito ou ilimitado",
        GRB.UNBOUNDED: "Ilimitado"
    }

    print(
        f"\n🔷 Status: {status_map.get(resultados['status'], 'Desconhecido')}")
    print(f"⏳ Tempo de execução: {resultados['tempo_execucao']:.2f} segundos")

    if resultados['status'] == GRB.OPTIMAL:
        print(
            f"⏱️ Tempo para encontrar o ótimo: {resultados['tempo_para_otimo']:.2f} segundos")

    print(
        f"💰 Melhor solução encontrada: {resultados['melhor_solucao'] if resultados['melhor_solucao'] is not None else 'N/A'}")
    print(
        f"🔮 Solução relaxada: {resultados['solucao_relaxada'] if resultados['solucao_relaxada'] is not None else 'N/A'}")
    print(
        f"📊 GAP de otimização: {resultados['gap_otimizacao']:.2f}%" if resultados['gap_otimizacao'] is not None else "N/A")

    if resultados['status'] == GRB.OPTIMAL or resultados['status'] == GRB.TIME_LIMIT:

        def safe_format(value, fmt=".2f", prefix=""):
            return f"{prefix}{value:{fmt}}" if value is not None else "N/A"

        print(f"\n💵 CUSTOS:")
        print(
            f"  Total: {safe_format(resultados.get('custo_total'), '.2f', 'R$')}")
        print(
            f"  - Transporte: {safe_format(resultados.get('custo_transporte'), '.2f', 'R$')}")
        print(
            f"  - Frete morto: {safe_format(resultados.get('frete_morto_total'), '.2f', 'R$')}")
        print(
            f"  - Não alocação: {safe_format(resultados.get('custo_nao_alocacao'), '.2f', 'R$')}")

        print(f"\n🚚 VEÍCULOS:")
        print(f"  Ativos: {resultados.get('veiculos_ativos', 'N/A')}")
        print(f"  Inativos: {resultados.get('veiculos_inativos', 'N/A')}")

        for aloc in resultados.get('alocacoes', []):
            print(
                f"\n  Veículo {aloc.get('veiculo_id', 'N/A')} ({aloc.get('veiculo_tipo', 'N/A')} para {aloc.get('destino', 'N/A')}):")
            print(f"    Cargas: {aloc.get('cargas', 'N/A')}")
            print(f"    Peso: {safe_format(aloc.get('peso_total'), '.2f', '')}kg (min: {safe_format(aloc.get('peso_minimo'), '.2f', '')}kg, cap: {safe_format(aloc.get('capacidade_peso'), '.2f', '')}kg)")
            print(
                f"    Volume: {safe_format(aloc.get('volume_total'), '.2f', '')}m³ (cap: {safe_format(aloc.get('capacidade_volume'), '.2f', '')}m³)")
            print(
                f"    Utilização: {safe_format(aloc.get('taxa_utilizacao_peso'), '.1f', '')}% (peso), {safe_format(aloc.get('taxa_utilizacao_volume'), '.1f', '')}% (volume)")
            print(
                f"    Custo: {safe_format(aloc.get('custo_veiculo'), '.2f', 'R$')}")
            if aloc.get('frete_morto', 0) > 0:
                print(
                    f"    ℹ️ Frete morto: {safe_format(aloc.get('frete_morto'), '.2f', 'R$')}")

        print(f"\n📦 CARGAS NÃO ALOCADAS:")
        print(
            f"  Quantidade: {resultados.get('ums_nao_alocadas', 'N/A')} de {resultados.get('ums_alocadas', 0) + resultados.get('ums_nao_alocadas', 0)}")
        print(
            f"  Peso total: {safe_format(resultados.get('peso_nao_alocado'), '.2f', '')}kg")
        print(
            f"  Volume total: {safe_format(resultados.get('volume_nao_alocado'), '.2f', '')}m³")

        print(f"\n🔍 ANÁLISE DE DECISÕES:")
        if resultados.get('frete_morto_total', 0) > 0:
            print("  ℹ️ Há fretes mortos - veículos operando abaixo da capacidade mínima")
        else:
            print("  ✅ Nenhum frete morto - todos veículos atendem carga mínima")

        if resultados.get('ums_nao_alocadas', 0) > 0:
            print(
                f"  ℹ️ {resultados.get('ums_nao_alocadas', 0)} UMs não alocadas - verifique se é por restrições ou decisão ótima")
        else:
            print("  ✅ Todas UMs alocadas")

        if resultados.get('veiculos_inativos', 0) > 0:
            print(
                f"  ℹ️ {resultados.get('veiculos_inativos', 0)} veículos inativos - verifique se é esperado")

    print(f"\n{'='*80}")


def exportar_resultados_csv(resultados_lista, instancias_originais):

    caminho_saida = os.path.join(os.path.dirname(
        __file__), 'OtimizacaoQualif', 'Resultados')

    if not resultados_lista or not instancias_originais or len(resultados_lista) != len(instancias_originais):
        raise ValueError(
            "Listas de resultados e instâncias originais não correspondem")

    os.makedirs(caminho_saida, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nome_arquivo = f"resultados_completos_{timestamp}.csv"
    caminho_completo = os.path.join(caminho_saida, nome_arquivo)

    with open(caminho_completo, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file, delimiter=';')

        writer.writerow(["RELATÓRIO DE OTIMIZAÇÃO"])
        writer.writerow(
            ["Gerado em:", datetime.now().strftime('%d/%m/%Y %H:%M:%S')])
        writer.writerow([])

        for resultados, instancia in zip(resultados_lista, instancias_originais):
            if not resultados or not instancia:
                continue

            if 'ums' not in instancia or 'clientes' not in instancia:
                print(
                    f"⚠️ Estrutura inválida na instância {resultados.get('tipo_instancia', 'desconhecida')}")
                continue

            writer.writerow(
                [f"INSTÂNCIA: {resultados.get('tipo_instancia', 'N/A')}"])
            writer.writerow([])

            writer.writerow([
                "Status", "Tempo Total (s)", "Tempo para Ótimo (s)",
                "Melhor Solução", "Solução Relaxada", "GAP (%)", "Custo Total",
                "Custo Transporte", "Frete Morto", "Custo Não Alocação",
                "Veículos Ativos", "Veículos Inativos", "UMs Alocadas", "UMs Não Alocadas",
                "Peso Não Alocado", "Volume Não Alocado"
            ])

            writer.writerow([
                "Ótimo" if resultados.get(
                    'status') == GRB.OPTIMAL else "Timeout",
                f"{resultados.get('tempo_execucao', 0):.2f}",
                f"{resultados.get('tempo_para_otimo', 0):.2f}" if resultados.get(
                    'tempo_para_otimo') is not None else "N/A",
                f"{resultados.get('melhor_solucao', 0):.2f}" if resultados.get(
                    'melhor_solucao') is not None else "N/A",
                f"{resultados.get('solucao_relaxada', 0):.2f}" if resultados.get(
                    'solucao_relaxada') is not None else "N/A",
                f"{resultados.get('gap_otimizacao', 0):.2f}" if resultados.get(
                    'gap_otimizacao') is not None else "N/A",
                f"{resultados.get('custo_total', 0):.2f}" if resultados.get(
                    'custo_total') is not None else "N/A",
                f"{resultados.get('custo_transporte', 0):.2f}",
                f"{resultados.get('frete_morto_total', 0):.2f}",
                f"{resultados.get('custo_nao_alocacao', 0):.2f}",
                resultados.get('veiculos_ativos', 0),
                resultados.get('veiculos_inativos', 0),
                resultados.get('ums_alocadas', 0),
                resultados.get('ums_nao_alocadas', 0),
                f"{resultados.get('peso_nao_alocado', 0):.2f}",
                f"{resultados.get('volume_nao_alocado', 0):.2f}"
            ])
            writer.writerow([])

            writer.writerow(["VEÍCULOS ATIVOS"])
            writer.writerow([
                "ID", "Tipo", "Destino", "Cargas", "Peso Total (kg)",
                "Capacidade (kg)", "Utilização Peso(%)", "Capacidade (m3)", "Utilização Vol(%)"
            ])

            for aloc in resultados.get('alocacoes', []):
                writer.writerow([
                    aloc.get('veiculo_id', ''),
                    aloc.get('veiculo_tipo', ''),
                    aloc.get('destino', ''),
                    ";".join(map(str, aloc.get('cargas', []))),
                    aloc.get('peso_total', ''),
                    aloc.get('capacidade_peso', ''),
                    f"{aloc.get('taxa_utilizacao_peso', 0):.1f}",
                    aloc.get('volume_total', ''),
                    aloc.get('capacidade_volume', ''),
                    f"{aloc.get('taxa_utilizacao_volume', 0):.1f}"
                ])
            writer.writerow([])

            writer.writerow(["UNIDADES METÁLICAS NÃO ALOCADAS"])
            writer.writerow([
                "ID", "Tipo", "Peso (kg)", "Volume (m³)", "Cliente",
                "Destino", "Compatibilidade", "Motivo"
            ])

            alocados_ids = set()
            for aloc in resultados.get('alocacoes', []):
                alocados_ids.update(aloc.get('cargas', []))

            for um in instancia.get('ums', []):
                if um.get('id') not in alocados_ids:

                    cliente = next(
                        (c for c in instancia.get('clientes', [])
                         if c.get('id') == um.get('cliente')),
                        {}
                    )

                    motivo = "Decisão ótima"
                    if not any(
                        v.get('tipo', '') in um.get(
                            'compatibilidade', '').split(',')
                        for v in instancia.get('veiculos', [])
                    ):
                        motivo = "Incompatibilidade"

                    writer.writerow([
                        um.get('id', ''),
                        um.get('tipo', ''),
                        um.get('peso', ''),
                        um.get('volume', ''),
                        cliente.get('nome', ''),
                        um.get('destino', ''),
                        um.get('compatibilidade', ''),
                        motivo
                    ])

            writer.writerow([])
            writer.writerow(["-"*50])
            writer.writerow([])

    print(f"\n✅ Relatório salvo em: {caminho_completo}")


def executar_todas_instancias_geradas():

    PASTA_INSTANCIAS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'OtimizacaoQualif')
    PASTA_RESULTADOS = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'OtimizacaoQualif', 'Resultados')
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    arquivos_instancias = [f for f in os.listdir(PASTA_INSTANCIAS)
                           if f.endswith('.csv') and not f.startswith('00_')]

    if not arquivos_instancias:
        print("❌ Nenhuma instância encontrada na pasta!")
        return

    print(f"🔍 Encontradas {len(arquivos_instancias)} instâncias para executar")

    resultados_totais = []
    instancias_originais = []

    for arquivo in arquivos_instancias:
        try:
            nome_instancia = arquivo.replace('.csv', '')
            print(f"\n{'='*80}")
            print(f"🚀 PROCESSANDO INSTÂNCIA: {nome_instancia}")
            print(f"{'='*80}")

            caminho_completo = os.path.join(PASTA_INSTANCIAS, arquivo)
            dados = carregar_dados(caminho_completo)
            instancia = {
                "veiculos": dados['veiculos'],
                "ums": dados['ums'],
                "clientes": dados['clientes'],
                "penalidade": dados['parametros']['Penalidade por não alocação']
            }
            instancias_originais.append(instancia)

            resultados = executar_instancia_com_timeout(
                nome_instancia, instancia)

            if resultados:
                resultados_totais.append(resultados)
                imprimir_resultados_detalhados(resultados)
            else:
                print(f"❌ Falha ao executar instância {nome_instancia}")

        except Exception as e:
            print(f"❌ Erro crítico ao processar {arquivo}: {str(e)}")
            continue

    if resultados_totais:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_arquivo = f"resultados_consolidados_{timestamp}.csv"
        exportar_resultados_csv(resultados_totais, instancias_originais)
        print(
            f"\n✅ Todas instâncias processadas! Resultados em: {nome_arquivo}")
    else:
        print("\n⚠️ Nenhuma instância foi executada com sucesso!")


if __name__ == "__main__":

    executar_todas_instancias_geradas()
