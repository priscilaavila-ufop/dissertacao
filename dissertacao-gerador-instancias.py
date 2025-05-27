# coluna "penalidade" das ums de acordo com
# _ peso e volume alto, então ocupa mais espaço e deixá-la para depois pode acabar exigindo um veículo extra
# _ o cliente é muito importante, e não entregar pode gerar multas ou perdas de contrato
# _ penalidade alta: UM com penalidade específica, se não for transportada, o custo auemnta muito
# _ se a carga é restrita e só cabe em alguns veículos em específico, aí se ela não for alocada, pode sobrar sem opção de transporte
# _ se a carga for urgente

# - as ums comuns podem variar o valor da penalidade de 0.3 a 0.5
# - as um com prioridade normal (que tenha entre 0.5kg e 1kg ou clientes regulares) valores de 0.8 a 1.5
# - as ums grandes/importantes (que pesem mais de 1kg ou tenham mais que 8m3 de volume) ou que seja para clientes importantes, penalidade 2.0 a 5.0
# - as ums estratégicas, que tenham impacto operacional grave, sejam peças únicas, projetos com multa por atraso, a penalidade deve variar de 5 a 10.

"""
GERADOR DE INSTÂNCIAS:
- 10 instâncias com 400 UMs e 30 veículos
- 10 instâncias com 300 UMs e 20 veículos
- 1 instância mini com 2 veículos, 2 clientes e 5 UMs
"""

import pandas as pd
import random
import os
import warnings
# oculta alerta do Pandas sobre uma mudança futura no comportamento da função pd.concat()
warnings.simplefilter(action='ignore', category=FutureWarning)

# ====================== ⚙️ CONFIGURAÇÕES ======================
TAMANHO_GRID = 100
NUM_REGIOES = 4
PASTA_SAIDA = os.path.join(os.path.dirname(__file__), 'Instancias_Penalidade')

# Configurações de tamanho das instâncias
CONFIGURACOES = [
    {'num_veiculos': 30, 'max_ums': 500, 'num_clientes': 30,
        'min_cargas_cliente': 6, 'max_cargas_cliente': 20},  # 10 instâncias
    {'num_veiculos': 30, 'max_ums': 400, 'num_clientes': 30,
        'min_cargas_cliente': 6, 'max_cargas_cliente': 20},  # 10 instâncias
    {'num_veiculos': 20, 'max_ums': 300, 'num_clientes': 20,
        'min_cargas_cliente': 6, 'max_cargas_cliente': 20},  # 10 instâncias
    {'num_veiculos': 2, 'max_ums': 5, 'num_clientes': 2,
        'min_cargas_cliente': 2, 'max_cargas_cliente': 3}        # 1 instância mini
]

# Quantidade de instâncias por configuração
NUM_INSTANCIAS = {
    30: 5,  # 10 instâncias para 30 veículos/500 UMs
    30: 5,  # 10 instâncias para 30 veículos/400 UMs
    20: 5,  # 5 instâncias para 20 veículos/300 UMs
    2: 1     # 1 instância mini
}

# Dados dos veículos
VEICULOS_BASE = [
    {'tipo': 'Bi-trem Carga Seca', 'capacidade_peso': 36000,
        'capacidade_vol': 70, 'custo': 1500},
    {'tipo': 'Bi-trem Especializado', 'capacidade_peso': 36000,
        'capacidade_vol': 60, 'custo': 1800},
    {'tipo': 'Bi-truck', 'capacidade_peso': 18000,
        'capacidade_vol': 35, 'custo': 1200},
    {'tipo': 'Carreta L', 'capacidade_peso': 25000,
        'capacidade_vol': 50, 'custo': 1350},
    {'tipo': 'Carreta trucada (LS)', 'capacidade_peso': 30000,
     'capacidade_vol': 60, 'custo': 1600},
    {'tipo': 'Rodotrem Carga seca', 'capacidade_peso': 48000,
        'capacidade_vol': 90, 'custo': 2000},
    {'tipo': 'Rodotrem Especializado', 'capacidade_peso': 48000,
        'capacidade_vol': 80, 'custo': 2200},
    {'tipo': 'Truck', 'capacidade_peso': 13000,
        'capacidade_vol': 25, 'custo': 1000},
    {'tipo': 'Vanderléia', 'capacidade_peso': 34000,
        'capacidade_vol': 65, 'custo': 1700},
    {'tipo': 'Sem recursos', 'capacidade_peso': 0, 'capacidade_vol': 0, 'custo': 0}
]

# ====================== 🔧 FUNÇÕES AUXILIARES ======================


def criar_pasta(caminho):
    os.makedirs(caminho, exist_ok=True)


def definir_regioes():
    metade = TAMANHO_GRID // 2
    return [
        {'id': 1, 'x_min': 0, 'x_max': metade, 'y_min': 0, 'y_max': metade},
        {'id': 2, 'x_min': metade, 'x_max': TAMANHO_GRID, 'y_min': 0, 'y_max': metade},
        {'id': 3, 'x_min': 0, 'x_max': metade,
            'y_min': metade, 'y_max': TAMANHO_GRID},
        {'id': 4, 'x_min': metade, 'x_max': TAMANHO_GRID,
            'y_min': metade, 'y_max': TAMANHO_GRID}
    ]


def gerar_nome_arquivo(num_veiculos, num_clientes, num_ums, variacao, pos_raiz):
    pos = 'c' if pos_raiz == 'centro' else 'e'
    return f"{num_veiculos}v{num_clientes}c{num_ums}p_{pos}{variacao}"


def calcular_penalidade_global(veiculos):
    veiculos_validos = [v for v in veiculos if v['capacidade_peso'] > 0]
    if not veiculos_validos:
        return 0.5
    custos_por_kg = [v['custo'] / v['capacidade_peso']
                     for v in veiculos_validos]
    penalidade_ideal = (sum(custos_por_kg) / len(custos_por_kg)) * 1.2
    return max(0.3, min(1.5, penalidade_ideal))


def gerar_frota(num_veiculos):
    frota = []

    # Sempre inclui veículo sem recursos por formalidade
    frota.append({
        'tipo': 'Sem recursos',
        'capacidade_peso': 0,
        'custo': 0,
        'capacidade_vol': 0,
        'destino': 'R1',
        'id': num_veiculos + 1,
        'descricao': 'Sem recursos'
    })

    # Adiciona veículos aleatórios, garante pelo menos um veículo por região
    regioes = [f"R{i}" for i in range(1, NUM_REGIOES+1)]

    # garante um veículo por região
    for i, regiao in enumerate(regioes[:min(num_veiculos, NUM_REGIOES)]):
        veiculo = random.choice(
            [v for v in VEICULOS_BASE if v['tipo'] != 'Sem recursos'])
        frota.append({
            'tipo': veiculo['tipo'],
            'capacidade_peso': veiculo['capacidade_peso'],
            'capacidade_vol': veiculo['capacidade_vol'],
            'custo': veiculo['custo'],
            'destino': regiao,
            'id': i+1,
            'descricao': veiculo['tipo'],
            'carga_minima': max(1, veiculo['capacidade_peso'] // 2)
        })

    # preenche o restante aleatoriamente
    for i in range(len(regioes)+1, num_veiculos + 1):
        veiculo = random.choice(
            [v for v in VEICULOS_BASE if v['tipo'] != 'Sem recursos'])
        frota.append({
            'tipo': veiculo['tipo'],
            'capacidade_peso': veiculo['capacidade_peso'],
            'capacidade_vol': veiculo['capacidade_vol'],
            'custo': veiculo['custo'],
            'destino': f"R{random.randint(1, NUM_REGIOES)}",
            'id': i,
            'descricao': veiculo['tipo'],
            # 50% da capacidade total
            'carga_minima': max(1, veiculo['capacidade_peso'] // 2)
        })


# 100nhaLasos

    return frota


def distribuir_cargas_por_cliente(num_clientes, min_cargas, max_cargas, total_ums):
    cargas_por_cliente = []
    ums_distribuidas = 0

    # Distribuição inicial garantindo o mínimo
    for _ in range(num_clientes):
        cargas = random.randint(min_cargas, max_cargas)
        cargas_por_cliente.append(cargas)
        ums_distribuidas += cargas

    # Ajuste para não ultrapassar o total
    while ums_distribuidas > total_ums:  # Caso exceda
        for i in range(num_clientes):
            if cargas_por_cliente[i] > min_cargas:
                cargas_por_cliente[i] -= 1
                ums_distribuidas -= 1
                if ums_distribuidas == total_ums:
                    break

    # Distribuição das UMs restantes (se houver)
    while ums_distribuidas < total_ums:
        cliente = random.randint(0, num_clientes - 1)
        if cargas_por_cliente[cliente] < max_cargas:
            cargas_por_cliente[cliente] += 1
            ums_distribuidas += 1

    return cargas_por_cliente


def determinar_penalidade_e_criterio(peso, volume, restricao, cliente_id):

    # 5% de chance de ser uma UM estratégica (independente de outros fatores)
    if random.random() < 0.05:
        penalidade = round(random.uniform(5.0, 10.0), 2)
        criterio = "Estratégica - impacto operacional grave, peça única ou projeto com multa por atraso"

    # 15% de chance de ser um cliente importante (independente de peso/volume)
    elif random.random() < 0.15 or cliente_id % 5 == 0:  # também marca cada 5º cliente como importante
        penalidade = round(random.uniform(2.0, 5.0), 2)
        criterio = "Cliente importante - risco de multas ou perda de contrato"

    # UMs grandes (peso > 1000kg ou volume > 8m³)
    elif peso > 1000 or volume > 8:
        penalidade = round(random.uniform(2.0, 5.0), 2)
        criterio = "Carga grande - ocupa muito espaço e pode exigir veículo extra"

    # UMs com restrições especiais
    elif restricao in ['Não empilhar', 'Frágil', 'Pesado']:
        penalidade = round(random.uniform(1.5, 3.0), 2)
        criterio = f"Carga com restrição ({restricao}) - limita opções de transporte"

    # UMs com peso entre 500-1000kg ou volume médio
    elif peso >= 500:
        penalidade = round(random.uniform(0.8, 1.5), 2)
        criterio = "Prioridade normal - carga média ou cliente regular"

    # Todas as outras UMs (comuns)
    else:
        penalidade = round(random.uniform(0.3, 0.5), 2)
        criterio = "Carga comum - baixa prioridade"

    return penalidade, criterio

# ====================== 🏭 GERADOR DE INSTÂNCIAS ======================
# ter veículos para todas as regiões!!!!


def gerar_instancia(config, pos_raiz, variacao):

    colunas = [
        'tipo', 'id', 'descricao', 'valor', 'peso', 'volume', 'destino',
        'x', 'y', 'cliente', 'compatibilidade', 'restricao', 'capacidade_peso',
        'capacidade_vol', 'custo', 'carga_minima', 'penalidade', 'Criterio Penalidade'
    ]

    df = pd.DataFrame(columns=colunas)
    regioes = definir_regioes()
    veiculos = gerar_frota(config['num_veiculos'])
    penalidade_global = calcular_penalidade_global(veiculos)

    num_clientes = config['num_clientes']

    # Distribui cargas por cliente
    cargas_por_cliente = distribuir_cargas_por_cliente(
        num_clientes,
        config['min_cargas_cliente'],
        config['max_cargas_cliente'],
        config['max_ums']
    )
    total_ums = sum(cargas_por_cliente)

    nome_arquivo = gerar_nome_arquivo(
        config['num_veiculos'],
        num_clientes,
        total_ums,
        variacao,
        pos_raiz
    )

    # Penalidade global
    df = pd.concat([df, pd.DataFrame([{
        'tipo': 'parametro',
        'id': 1,
        'descricao': 'Penalidade por não alocação',
        'valor': round(penalidade_global, 4)
    }])], ignore_index=True, sort=False)

    # Nó raiz
    df = pd.concat([df, pd.DataFrame([{
        'tipo': 'no',
        'id': 0,
        'descricao': 'No_Raiz',
        'destino': 'CENTRO' if pos_raiz == 'centro' else 'CANTO'
    }])], ignore_index=True, sort=False)

    # garante distribuição por regiões
    contadores_regiao = {1: 1, 2: 1, 3: 1, 4: 1}
    clientes_por_regiao = {1: 0, 2: 0, 3: 0, 4: 0}

    # Distribui clientes pelas regiões
    for regiao_id in range(1, NUM_REGIOES+1):
        clientes_por_regiao[regiao_id] = num_clientes // NUM_REGIOES

    # Distribui clientes restantes
    clientes_restantes = num_clientes % NUM_REGIOES
    for regiao_id in range(1, clientes_restantes + 1):
        clientes_por_regiao[regiao_id] += 1

    cliente_id = 1
    for regiao_id, num_clientes_regiao in clientes_por_regiao.items():
        regiao = next(r for r in regioes if r['id'] == regiao_id)

        for _ in range(num_clientes_regiao):
            x = random.uniform(regiao['x_min'], regiao['x_max'])
            y = random.uniform(regiao['y_min'], regiao['y_max'])

            df = pd.concat([df, pd.DataFrame([{
                'tipo': 'cliente',
                'id': cliente_id,
                'descricao': f'Cliente_R{regiao_id}_{contadores_regiao[regiao_id]}',
                'destino': f"R{regiao_id}",
                'x': x,
                'y': y
            }])], ignore_index=True, sort=False)

            cliente_id += 1
            contadores_regiao[regiao_id] += 1

    # Veículos
    for veiculo in veiculos:
        df = pd.concat([df, pd.DataFrame([{
            'tipo': 'veiculo',
            'id': veiculo['id'],
            'descricao': f"Veiculo_{veiculo['tipo']}",
            'destino': veiculo['destino'],
            'capacidade_peso': veiculo['capacidade_peso'],
            'capacidade_vol': veiculo['capacidade_vol'],
            'custo': veiculo['custo'],
            # ajustar para quanto???
            'carga_minima': veiculo.get('carga_minima', max(1, veiculo['capacidade_peso'] // 2))
        }])], ignore_index=True, sort=False)

    tipos_carga = ['chapa', 'tira', 'perfil', 'tubo']
    um_id = 1

    for cliente_id in range(1, num_clientes + 1):
        num_cargas = cargas_por_cliente[cliente_id - 1]

        for _ in range(num_cargas):
            veiculos_compatíveis = [v['tipo']
                                    for v in veiculos if v['tipo'] != 'Sem recursos']
            peso = random.randint(500, 3000)
            volume = round(random.uniform(0.5, 10.0), 1)
            restricao = random.choice(['Não empilhar', 'Frágil', 'Pesado', ''])

            # Determina penalidade e critério
            penalidade, criterio_penalidade = determinar_penalidade_e_criterio(
                peso, volume, restricao, cliente_id)

            df = pd.concat([df, pd.DataFrame([{
                'tipo': 'um',
                'id': um_id,
                'descricao': random.choice(tipos_carga),
                'peso': peso,
                'volume': volume,
                'cliente': cliente_id,
                'compatibilidade': ','.join(veiculos_compatíveis),
                'restricao': restricao,
                'penalidade': penalidade,
                'Criterio Penalidade': criterio_penalidade
            }])], ignore_index=True, sort=False)

            um_id += 1

    # Salvar
    criar_pasta(PASTA_SAIDA)
    caminho_arquivo = os.path.join(PASTA_SAIDA, f"{nome_arquivo}.csv")
    df.to_csv(caminho_arquivo, sep=';', decimal='.', index=False)

    print(f'Arquivo gerado: {nome_arquivo}')

    return {
        'Veículos': config['num_veiculos'],
        'Clientes': num_clientes,
        'UMs': total_ums,
        'Min Cargas/Cliente': config['min_cargas_cliente'],
        'Max Cargas/Cliente': config['max_cargas_cliente'],
        'Posição': pos_raiz,
        'Variação': variacao,
        'Arquivo': nome_arquivo
    }

# ====================== 🚀 EXECUÇÃO PRINCIPAL ======================


def gerar_todas_instancias():
    criar_pasta(PASTA_SAIDA)
    dados_instancias = []

    for config in CONFIGURACOES:
        num_variacoes = NUM_INSTANCIAS[config['num_veiculos']]

        for variacao in range(1, num_variacoes + 1):
            # Versão centro
            dados = gerar_instancia(config, 'centro', variacao)
            dados_instancias.append(dados)

            # Versão canto
            dados = gerar_instancia(config, 'canto', variacao)
            dados_instancias.append(dados)

    # Gerar relatório
    resumo = pd.DataFrame(dados_instancias)
    resumo.to_csv(os.path.join(
        PASTA_SAIDA, '00_RESUMO_COMPLETO.csv'), index=False)

    # Tabela resumo
    resumo_consolidado = resumo.groupby(
        ['Veículos', 'Clientes', 'UMs']).size().reset_index()
    resumo_consolidado.columns = ['Veículos',
                                  'Clientes', 'UMs', 'Qtd Instâncias']
    resumo_consolidado.to_csv(os.path.join(
        PASTA_SAIDA, '00_RESUMO.csv'), index=False)

    print("\n📊 RESUMO DAS INSTÂNCIAS GERADAS:")
    print(resumo_consolidado.to_string(index=False))
    print(
        f"\n📄 Relatório completo salvo em: {os.path.join(PASTA_SAIDA, '00_RESUMO_COMPLETO.csv')}")


if __name__ == '__main__':
    gerar_todas_instancias()
