import math
import random
import itertools
import time
import scipy.sparse as sp
import numpy as np
import copy


def get_incidenceMatrix(fileName):
    """
    对超图数据集的处理，返回处理后最终的超图关联矩阵!
    """
    with open(fileName + '.txt', 'r') as f:
        lines = f.readlines()
        hedges = []
        nodes_origin = []
        for line in lines:
            line = line.replace('\n', '')
            line1 = tuple(map(int, line.split()))

            hedges.append(line1)
            nodes_origin.extend(list(line1))

    nodes_sorted = sorted(np.unique(nodes_origin))

    m = len(hedges)
    n = len(nodes_sorted)

    nodesIndex_dict = dict(zip(nodes_sorted, (range(n))))
    hedges_dict = dict(zip(range(m), hedges))

    csr_incidence_matrix = sp.lil_array((len(nodes_sorted), m))
    for hedgeIndex, hedges in hedges_dict.items():
        for node in hedges:
            csr_incidence_matrix[nodesIndex_dict[node], hedgeIndex] = 1
    # print(csr_incidence_matrix.toarray())
    return m, n, csr_incidence_matrix.tocsr() #csr_incidence_matrix.toarray()  #


def getHpe(inode, matrix):

    row = matrix.getrow(inode)
    inode_hyedges_list = list(row.nonzero()[1])

    return inode_hyedges_list

def chooseHpe(hpe_set):

    if len(hpe_set) > 0:
        return random.sample(list(hpe_set), 1)[0]
    else:
        return []

def getNodesofHpe(hpe, matrix):

    row = matrix.getcol(hpe)
    hedge_to_nodes = list(row.nonzero()[0])

    return hedge_to_nodes


def findAdjNode_CP(inode, incidenceMatrix):

    edges_set = getHpe(inode, incidenceMatrix)  # inode所在超边集
    edge = chooseHpe(edges_set)  # 随机选一条inode所在的超边
    adj_nodes = np.array(getNodesofHpe(edge, incidenceMatrix))  # 所选超边中的节点集

    return adj_nodes


def spreadAdj(adj_nodes, beta):

    random_list = np.random.random(size=len(adj_nodes))
    infected_list = adj_nodes[np.where(random_list < beta)[0]]
    return infected_list

def hyperSI(incidenceMatrix, seednodP, iters, betaP, C, Adj):
    I_listP = list(seednodP)
    I_total_listN = [1]

    betaPjuzhen = Adj * betaP

    for t in range(0, iters):
        infected_P = []  # 第t个时间步时，被I-list中的节点集感染成I-state的节点集
        for inodeP in I_listP:  # I_list在随时间步的变化而变大
            adj_nodes = findAdjNode_CP(inodeP, incidenceMatrix)
            betaPlist = []
            for i in range(len(adj_nodes)):
                node = adj_nodes[i]
                betaP1 = betaPjuzhen[inodeP, node]
                betaPlist.append(betaP1)
            betaPlist = np.array(betaPlist)
            infected_list_unique = spreadAdj(adj_nodes, betaPlist)
            infected_P.extend(infected_list_unique)
            betaPjuzhen[inodeP, :] *= math.exp(-C)
        I_listP.extend(infected_P)
        I_listP = list(set(I_listP))
        I_total_listN.append(len(I_listP))

    return I_total_listN[-1:][0]


def all_path(Adj, nodej, L, seeds_P):  # 找到L阶以内的所有邻居
    seeds_P = list(seeds_P)
    seed = seeds_P
    pathsetP = []

    for j in seeds_P:
        #pathsP = find_paths(Adj, j, nodej, L, seed)
        if L == 1:
            pathsP = find_paths_length_1(Adj, j, nodej, seed)
        elif L == 2:
            pathsP = find_paths_length_2(Adj, j, nodej, seed)
        pathsetP.extend(pathsP)
    return pathsetP


def find_paths_length_1(adj_matrix, start, end, seed):
    paths_length_1 = []
    if adj_matrix[start][end] != 0 and end not in seed:
        paths_length_1.append([start, end])
    return paths_length_1


def find_paths_length_2(adj_matrix, start, end, seed):
    paths_length_2 = []
    if adj_matrix[start][end] != 0:
        paths_length_2.append([start, end])
    for i in range(len(adj_matrix)):
        if adj_matrix[start][i] != 0 and adj_matrix[i][end] != 0 and i not in seed:
            paths_length_2.append([start, i, end])
    return paths_length_2


def get_neighbors_within_L_steps(adj_matrix, node, L):
    neighborslist = []
    for i in range(1, L + 1):
        # 计算邻接矩阵的 L 次幂
        adj_matrix_L_power = np.linalg.matrix_power(adj_matrix, i)
        # 提取节点的邻居
        neighbors = np.nonzero(adj_matrix_L_power[node])[0]
        neighbors = list(set(neighbors) - {node})  # 去除自身节点
        neighborslist.extend(neighbors)
    neighborsset = set(neighborslist)
    return neighborsset


def get_all_neighbors(Adj, L, seeds_N):
    seeds_N1 = list(seeds_N)
    allneighbors = []
    for i in seeds_N1:
        neighbors = get_neighbors_within_L_steps(Adj, i, L)
        allneighbors.extend(list(neighbors))
    allneighborsset = set(allneighbors)
    allneighborsset.difference_update(seeds_N)
    allneighborslist = list(allneighborsset)
    return allneighborslist


def fhanshu(t, w, C):
    if t == -1:
        result = 1
    else:
        result = 1
        for a in range(t + 1):
            result *= (1 - w*(math.exp(-C*a)))
    return result


def ptvalue(path, W, T, C):
    a = len(path)
    value = 0
    if a == 2:
        beta1 = W[path[0], path[1]]
        for t in range(1, (T + 1)):
            fvalue = fhanshu((t-2), beta1, C)
            value += beta1 * math.exp(-C*(t-1)) * fvalue
        #value = 1 - (1 - beta1) ** T
    elif a == 3:
        beta1 = W[path[0], path[1]]
        beta2 = W[path[1], path[2]]
        for t in range(2, (T + 1)):
            for j in range(1, t):
                fvalue1 = fhanshu((j - 2), beta1, C)
                fvalue2 = fhanshu((t - 2 - j), beta2, C)
                value += beta1 * math.exp(-C*(j-1)) * fvalue1 * beta2 * math.exp(-C*(t - j - 1)) * fvalue2
    elif a == 4:
        beta1 = W[path[0], path[1]]
        beta2 = W[path[1], path[2]]
        beta3 = W[path[2], path[3]]
        for t in range(3, (T + 1)):
            for j in range(2, t):
                for k in range(1, j):
                    fvalue1 = fhanshu((k - 2), beta1, C)
                    fvalue2 = fhanshu((j - 2 - k), beta2, C)
                    fvalue3 = fhanshu((t - 2 - j), beta3, C)
                    value += beta1 * math.exp(-C*(k-1)) * fvalue1 * beta2 * math.exp(-C*(j - k - 1)) * fvalue2 * beta3 * math.exp(-C*(t - j - 1)) * fvalue3
    return value


def pt_S_value(pathlist, W, T, C):
    allvalue = 1
    for path in pathlist:
        value = ptvalue(path, W, T, C)
        allvalue *= (1 - value)
    finalvalue = 1 - allvalue
    return finalvalue


def fitnessvalue(Adj, L, seeds_P, WP, T, C):
    allneighborslist = get_all_neighbors(Adj, L, seeds_P)  # 负节点种子集的所有L阶邻居
    finalvalue = len(seeds_P)  # 初始影响力
    for i in allneighborslist:  # 选择一个邻居节点
        pathsetP = all_path(Adj, i, L, seeds_P)  # 种子集到邻居节点的所有路径
        PSP = pt_S_value(pathsetP, WP, T, C)
        finalvalue += PSP
    return finalvalue


def fitness_function(X_decode, Adj, L, WP, T, C):
    """
    计算每个个体的适应度！
    """
    chromosomes_scale_list = []  # 计算每个个体 的 适应度
    for i in range(X_decode.shape[0]):
        chromosome_i = np.array(X_decode[i])  # X_decode的第i行
        chromosome_i_scale = fitnessvalue(Adj, L, chromosome_i, WP, T, C)
        chromosomes_scale_list.append(chromosome_i_scale)
    chromosomes_scale_arr = np.array(chromosomes_scale_list)
    return chromosomes_scale_arr


# 初始化种群
def initialize_population(population_size, length, degrees, hdegree, valuelist):
    pop = []
    k1 = int(population_size / 2)
    individuals_idx = np.array(list(range(len(degrees))))  # 种群中的个体数
    allshu = int(len(degrees) / 5)
    #degrees = degrees / degrees.sum()  # 获得每个个体的概率
    #hdegree = hdegree / hdegree.sum()  # 获得每个个体的概率
    degrees = list(degrees)
    sorted_id1 = sorted(range(len(degrees)), key=lambda k: degrees[k], reverse=True)
    hdegree = list(hdegree)
    sorted_id2 = sorted(range(len(hdegree)), key=lambda k: hdegree[k], reverse=True)
    X1_idx1 = sorted_id1[:allshu]
    X2_idx1 = sorted_id2[:allshu]
    sorted_id3 = sorted(range(len(valuelist)), key=lambda k: valuelist[k], reverse=True)
    X3_idx1 = sorted_id3[:allshu]
    for _ in range(k1):
        #selection1 = random.sample(X1_idx1, length)
        #pop.append(selection1)
        #selection2 = random.sample(X2_idx1, length)
        #pop.append(selection2)
        selection3 = random.sample(X3_idx1, length)
        pop.append(selection3)
        selection4 = random.sample(list(individuals_idx), length)
        pop.append(selection4)
    pop = np.array(pop)
    return pop


def select(X, fitness, lp):
    X1_idx = (np.argsort(list(fitness))[::-1])[:lp]
    X1 = X[X1_idx, :]
    return X1


# 交叉操作 - 部分映射交叉 (PMX)
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    a1_1 = copy.deepcopy(parent1)
    a2_1 = copy.deepcopy(parent2)

    # 随机选择两个交叉点
    point1, point2 = sorted(random.sample(range(size), 2))
    fragment1 = parent1[point1:point2]
    fragment2 = parent2[point1:point2]
    # 交叉
    a1_1[point1:point2], a2_1[point1:point2] = a2_1[point1:point2], a1_1[point1:point2]
    # 定义容器
    a1_2 = []  # 储存修正后的head
    a2_2 = []
    a1_3 = []  # 修正后的tail
    a2_3 = []
    # 子代1头部修正
    for i in a1_1[:point1]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        a1_2.append(i)
    # 子代2尾部修正
    for i in a1_1[point2:]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        a1_3.append(i)
    # 子代2头部修订
    for i in a2_1[:point1]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        a2_2.append(i)
    # 子代2尾部修订
    for i in a2_1[point2:]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        a2_3.append(i)

    child1 = a1_2 + fragment2 + a1_3
    child2 = a2_2 + fragment1 + a2_3
    return child1, child2


def crossover(X1_selection, pc):

    random_z = np.random.random(len(X1_selection))
    mask = random_z < pc
    true_indices = np.where(mask)[0]
    # false_indices = np.where(~mask)[0]

    parents_population = X1_selection[true_indices]

    parents_population_list = parents_population.tolist()
    combinations = list(itertools.combinations(parents_population_list, 2))

    for combination in combinations:
        xa = combination[0]
        xb = combination[1]
        child1, child2 = pmx_crossover(xa, xb)
        X1_selection = np.append(X1_selection, [child1], axis=0)
        X1_selection = np.append(X1_selection, [child2], axis=0)
    offspring_population_unique = np.unique(X1_selection, axis=0)
    return offspring_population_unique


def mutation(n_range, X_crossover, pm):
    X_crossover1 = copy.deepcopy(X_crossover)
    for i in range(X_crossover1.shape[0]):
        i_other_nodes = set(n_range) - set(X_crossover1[i])

        for j in range(X_crossover1.shape[1]):
            if np.random.rand() < pm:
                random_node = random.choice(tuple(i_other_nodes))
                X_crossover1[i, j] = random_node
                i_other_nodes.remove(random_node)
    X1_selection = np.append(X_crossover, X_crossover1, axis=0)
    X_crossover = np.unique(X1_selection, axis=0)
    return X_crossover


def getSeeds_NRHGA(n, k_seeds, num_generation, population_size, pc, pm, Adj, L, WP, T, C, degrees, hdegree, valuelist):
    chromosome_length = k_seeds  # 10
    nodesIndex = np.arange(n)
    '''
    X = np.zeros((population_size, chromosome_length), dtype=int)
    # indicess = []
    i = 0
    while i < population_size:
        indices = np.random.choice(nodesIndex, chromosome_length, replace=False)
        X[i, :] = indices
        i = i + 1
    '''
    X = initialize_population(population_size, chromosome_length, degrees, hdegree, valuelist)

    X_fitness = fitness_function(X, Adj, L, WP, T, C)
    best_fitness = [0]  # 记录每次迭代最优适应度值
    empty_array = np.empty((0,), dtype=int)
    best_newNodesIndex = [empty_array]
    generation = 0
    while generation < num_generation:
        X1_selection = select(X, X_fitness, population_size)

        # 交叉 变异
        X1_crossover = crossover(X1_selection, pc)
        X1_mutation = mutation(nodesIndex, X1_crossover, pm)  # 以pm的概率变异

        newX_fitness = fitness_function(X1_mutation, Adj, L, WP, T, C)

        indiviIndex_best = np.argmax(newX_fitness)  # 获取 最大元素对应的位置索引
        indivi_newX_best = X1_mutation[indiviIndex_best]
        fitness_newX_best = newX_fitness[indiviIndex_best]
        best_fitness.append(fitness_newX_best)
        best_newNodesIndex.append(list(indivi_newX_best))
        print("max_guji:", fitness_newX_best)
        f.write('max_guji:：' + str(fitness_newX_best) + "\n")
        X = X1_mutation
        X_fitness = newX_fitness
        generation += 1
    print("gen_num:", generation)
    f.write('gen_num:：' + str(generation) + "\n")
    seeds_individual = best_newNodesIndex[-1]
    print('GAfianl方法每次运行的个体的最优适应度值：', best_fitness)
    f.write('GAfianl方法每次运行的个体的最优适应度值:：' + str(best_fitness) + "\n")
    return seeds_individual


name = ['datasets/senate-committees', 'datasets/Algebra', 'datasets/Restaurants-Rev', 'datasets/Geometry', 'datasets/NDC-classes-unique', 'datasets/house-committees', 'datasets/citeseer', 'datasets/iAF1260b']
for na in range(len(name)):
    f = open("output/HIDE-EA.txt", 'a')
    fileName = name[na]
    print(fileName)
    f.write(fileName + "\n")
    f.write("HIDE-EA, 初始化，前20%估计值较大的节点中选取个体，50%随机选择个体" + "\n")
    T = 10
    betaP = 0.05
    num_generation = 50
    population_size = 30
    pc = 0.9
    pm = 0.02
    f.write("T:" + str(T) + "\n")
    f.write("betaP:" + str(betaP) + "\n")
    f.write("num_generation:" + str(num_generation) + "\n")
    f.write("pc:" + str(pc) + "\n")
    f.write("pm:" + str(pm) + "\n")
    m, N, H = get_incidenceMatrix(fileName)
    adj_matrix = H.dot(H.T)
    hdegree = np.array(H.sum(axis=1))
    adj_matrix[np.eye(N, dtype=np.bool_)] = 0
    adj_matrix = adj_matrix.toarray()
    degrees = np.count_nonzero(adj_matrix, axis=1)
    weight_matrix = adj_matrix / hdegree[:, np.newaxis]
    WP = weight_matrix * betaP
    L = 1
    k_seeds_list = [3,6,9,12,15,18,21,24,27,30]
    C = 0.5
    print("T:", T)
    print("betaP:", betaP)
    print("L:", L)
    adj_matrix[adj_matrix != 0] = 1

    valuelist = []
    for i in range(len(adj_matrix)):
        seeds_P = np.array([i])
        value = fitnessvalue(adj_matrix, L, seeds_P, WP, T, C)
        valuelist.append(value)

    for i, k_seed in enumerate(k_seeds_list):
        f.write("种子集大小：" + str(k_seed) + "\n")
        seeds_NRHGA = getSeeds_NRHGA(N, k_seed, num_generation, population_size, pc, pm, adj_matrix, L, WP, T, C, degrees, hdegree, valuelist)
        t2 = time.time()
        print('最优种子集：', seeds_NRHGA)
        f.write('最优种子集:：' + str(seeds_NRHGA) + "\n")
        seeds_P = np.array(seeds_NRHGA)
        num = 0
        for i in range(500):
            num += hyperSI(H, seeds_P, T, betaP, C, adj_matrix)
        average = num / 500
        print("SI:", average)
        f.write(str(average) + "\n")
    f.close()


