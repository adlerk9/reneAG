import heapq
import random
import time


MAPA = [
    ['S', '.', '.', '#', '.', '.', '.', '.', '.', '.'],
    ['.', '#', '.', '#', '.', '#', '#', '#', '~', '.'],
    ['.', '#', '.', '.', '.', '~', '~', '#', '.', '.'],
    ['.', '#', '#', '#', '.', '#', '.', '#', '.', '#'],
    ['.', '.', '~', '#', '.', '#', '.', '.', '.', '.'],
    ['#', '.', '#', '#', '.', '#', '#', '#', '#', '.'],
    ['.', '.', '.', '.', '.', '~', '.', '.', '#', '.'],
    ['.', '#', '#', '#', '#', '#', '#', '.', '#', '.'],
    ['.', '.', '.', '.', '.', '.', '#', '.', '~', '.'],
    ['#', '#', '#', '.', '#', '.', '.', '.', '#', 'G']
]

LINHAS = len(MAPA)
COLUNAS = len(MAPA[0])
INICIO = (0, 0)
DESTINO = (9, 9)

CUSTOS = {
    'S': 1,
    'G': 1,
    '.': 1,
    '~': 3
}

MOVIMENTOS = [
    (-1, 0),  # 0: Cima
    (1, 0),   # 1: Baixo
    (0, -1),  # 2: Esquerda
    (0, 1)    # 3: Direita
]


def custo_celula(pos):
    r, c = pos
    return CUSTOS.get(MAPA[r][c], float('inf'))


def eh_valido(pos):
    r, c = pos
    return 0 <= r < LINHAS and 0 <= c < COLUNAS and MAPA[r][c] != '#'


def distancia_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def busca_a_estrela(inicio, destino):
    """
    Busca informada A*: f(n) = g(n) + h(n)
    - g(n): custo real acumulado desde o início até n
    - h(n): heurística admissível (distância de Manhattan até o destino)
    """
    fila_prioridade = []
    heapq.heappush(fila_prioridade, (distancia_manhattan(inicio, destino), 0, inicio, [inicio]))
    melhor_g = {inicio: 0}
    nos_expandidos = 0

    while fila_prioridade:
        f, g, atual, caminho = heapq.heappop(fila_prioridade)
        nos_expandidos += 1

        if atual == destino:
            return caminho, g, nos_expandidos

        if g > melhor_g.get(atual, float('inf')):
            continue

        for dr, dc in MOVIMENTOS:
            vizinho = (atual[0] + dr, atual[1] + dc)
            if eh_valido(vizinho):
                novo_g = g + custo_celula(vizinho)
                if novo_g < melhor_g.get(vizinho, float('inf')):
                    melhor_g[vizinho] = novo_g
                    h = distancia_manhattan(vizinho, destino)
                    novo_f = novo_g + h
                    heapq.heappush(fila_prioridade, (novo_f, novo_g, vizinho, caminho + [vizinho]))

    return None, float('inf'), nos_expandidos


def simplificar_caminho(caminho):
    """Remove ciclos (loops) do caminho para obter a trajetória limpa."""
    caminho_limpo = []
    posicoes = {}
    for pos in caminho:
        if pos in posicoes:
            idx = posicoes[pos]
            for p in caminho_limpo[idx + 1:]:
                del posicoes[p]
            caminho_limpo = caminho_limpo[:idx + 1]
        else:
            posicoes[pos] = len(caminho_limpo)
            caminho_limpo.append(pos)
    return caminho_limpo


def decodificar_cromossomo(cromossomo, inicio, destino):
    """
    Converte os genes (0..3) em uma rota válida no espaço de busca.
    Se o movimento direto bater em obstáculo, usa o gene para escolher
    entre os vizinhos livres (evitando retorno imediato quando possível).
    """
    atual = inicio
    anterior = None
    caminho_bruto = [atual]
    penalidades = 0

    for gene in cromossomo:
        if atual == destino:
            break

        dr, dc = MOVIMENTOS[gene]
        tentativa = (atual[0] + dr, atual[1] + dc)

        if eh_valido(tentativa) and tentativa != anterior:
            proximo = tentativa
        else:
            penalidades += 1
            vizinhos = [
                (atual[0] + m[0], atual[1] + m[1])
                for m in MOVIMENTOS
                if eh_valido((atual[0] + m[0], atual[1] + m[1]))
            ]
            sem_retorno = [v for v in vizinhos if v != anterior]
            opcoes = sem_retorno if sem_retorno else vizinhos
            proximo = opcoes[gene % len(opcoes)]

        anterior = atual
        atual = proximo
        caminho_bruto.append(atual)

    caminho = simplificar_caminho(caminho_bruto)
    custo = sum(custo_celula(p) for p in caminho[1:])
    chegou = (caminho[-1] == destino)
    return caminho, custo, penalidades, chegou


def avaliar_fitness(cromossomo, inicio, destino):
    caminho, custo, penalidades, chegou = decodificar_cromossomo(cromossomo, inicio, destino)
    dist_final = distancia_manhattan(caminho[-1], destino)

    if chegou:
        # Prioriza chegar ao destino com o menor custo total e menor número de passos
        return 10000.0 - (custo * 100.0) - (len(caminho) * 5.0) - penalidades
    else:
        # Guia a evolução para reduzir a distância até o destino
        return 1000.0 - (dist_final * 50.0) - (custo * 2.0) - penalidades


def algoritmo_genetico(
    inicio,
    destino,
    tam_populacao=120,
    tam_cromossomo=45,
    geracoes=200,
    taxa_mutacao=0.08,
    semente=42
):
    random.seed(semente)

    populacao = [
        [random.randint(0, 3) for _ in range(tam_cromossomo)]
        for _ in range(tam_populacao)
    ]

    melhor_global = None
    melhor_fitness = float('-inf')

    for _ in range(geracoes):
        avaliados = []
        for ind in populacao:
            fit = avaliar_fitness(ind, inicio, destino)
            avaliados.append((fit, ind))
            if fit > melhor_fitness:
                melhor_fitness = fit
                melhor_global = ind[:]

        avaliados.sort(key=lambda x: x[0], reverse=True)

        # Elitismo (10% melhores passam direto)
        elite_qtd = max(2, tam_populacao // 10)
        nova_populacao = [ind[:] for _, ind in avaliados[:elite_qtd]]

        # Seleção por Torneio + Crossover de 2 Pontos + Mutação
        while len(nova_populacao) < tam_populacao:
            p1 = max(random.sample(avaliados, 4), key=lambda x: x[0])[1]
            p2 = max(random.sample(avaliados, 4), key=lambda x: x[0])[1]

            corte1, corte2 = sorted(random.sample(range(1, tam_cromossomo), 2))
            filho = p1[:corte1] + p2[corte1:corte2] + p1[corte2:]

            for i in range(tam_cromossomo):
                if random.random() < taxa_mutacao:
                    filho[i] = random.randint(0, 3)

            nova_populacao.append(filho)

        populacao = nova_populacao

    caminho, custo, _, chegou = decodificar_cromossomo(melhor_global, inicio, destino)
    return caminho, custo, chegou, melhor_fitness


def imprimir_mapa_com_rota(caminho, titulo):
    print(f"\n--- {titulo} ---")
    conjunto_caminho = set(caminho)
    for r in range(LINHAS):
        linha_str = []
        for c in range(COLUNAS):
            if (r, c) in (INICIO, DESTINO):
                linha_str.append(MAPA[r][c])
            elif (r, c) in conjunto_caminho:
                linha_str.append('*')
            else:
                linha_str.append(MAPA[r][c])
        print(" ".join(linha_str))


if __name__ == "__main__":
    print("PROBLEMA: Roteamento de Robô de Entrega em Labirinto com Custos (10x10)")

    # 1. Execução do A*
    t0 = time.perf_counter()
    caminho_astar, custo_astar, expandidos = busca_a_estrela(INICIO, DESTINO)
    tempo_astar = (time.perf_counter() - t0) * 1000

    imprimir_mapa_com_rota(caminho_astar, "Solução A* (A-Estrela)")
    print(f"Custo Total: {custo_astar} | Passos: {len(caminho_astar) - 1} | Nós expandidos: {expandidos} | Tempo: {tempo_astar:.2f} ms")
    print(f"Rota A*: {caminho_astar}")

    # 2. Execução do Algoritmo Genético
    t0 = time.perf_counter()
    caminho_ag, custo_ag, chegou_ag, fit_ag = algoritmo_genetico(INICIO, DESTINO)
    tempo_ag = (time.perf_counter() - t0) * 1000

    imprimir_mapa_com_rota(caminho_ag, "Solução AG (Algoritmo Genético)")
    print(f"Chegou ao destino: {chegou_ag} | Custo Total: {custo_ag} | Passos: {len(caminho_ag) - 1} | Fitness: {fit_ag:.1f} | Tempo: {tempo_ag:.2f} ms")
    print(f"Rota AG: {caminho_ag}")
