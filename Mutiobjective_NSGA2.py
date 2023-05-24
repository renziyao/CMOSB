import numpy as np
import copy as cy
#import matplotlib.pyplot as plt
import time

def FON(Rt):
    a = 0
    b = 0
    for i in range(Rt.shape[1]):
        a += (Rt[:, i]-1/np.sqrt(3))**2
    f1 = 1-np.exp(-a)
    for i in range(Rt.shape[1]):
        b += (Rt[:, i]+1/np.sqrt(3))**2
    f2 = 1-np.exp(-b)
    # fm = np.vstack(f1,f2)
    return f1, f2

def SCH(Rt):
    f1 = Rt[:,0]**2
    f2 = (Rt[:,0]-2)**2
    return f1,f2


#######################################################################################################################
#                                                  Cross over                                                         #
#######################################################################################################################
def crossover_SBX(Pt, nc,cross_rate):  ## ui[0,1] && nc=2/5
    child = cy.deepcopy(Pt)
    for i in range(0, len(Pt) - 1, 2):  ## Pt = 2N*M #N:pop M:dec variables dim
        if np.random.rand() < cross_rate:
            ui = np.random.rand()
            if ui <= 0.5:
                beta_qi = (2 * ui) ** (1 / (nc + 1))
            else:
                beta_qi = (1 / (2 * (1 - ui))) ** (1 / (nc + 1))
            child[i] = 0.5 * ((1 + beta_qi) * Pt[i] + (1 - beta_qi) * Pt[i + 1])
            child[i+1] = 0.5 * ((1 - beta_qi) * Pt[i] + (1 + beta_qi) * Pt[i + 1])
    return child

def cxOnePoint(Pt, rate):
    Qt = cy.deepcopy(Pt)  ### ju keng !!!###
    # Pt = np.random.permutation(Pt)
    for i in range(0, len(Qt) - 1, 2):
        if np.random.rand() < rate:
            dot_point = np.random.randint(1, min(len(Qt[i]), len(Qt[i + 1])))  ## list length can be different
            Qt[i][dot_point:], Qt[i + 1][dot_point:] = Qt[i + 1][dot_point:], Qt[i][dot_point:]  ## This doesn't in array
    return Qt


#######################################################################################################################
#                                                   Mutation                                                          #
#######################################################################################################################
def poly_mut(child, n_m, mut_rate, max_, min_):  ## ri[0,1] && n_m = 20/[20,100]
    for i in range(len(child)):
        if np.random.rand() < mut_rate:
            ri = np.random.rand()
            if ri < 0.5:
                sigmai = (2*ri)**(1/(n_m+1))-1
            else:
                sigmai = 1-(2*(1-ri))**(1/(n_m+1))
            child[i] = child[i]+(max_-min_)*sigmai  ## max-min
        print("child", child)
        # Each individual has at least two decision variables
        if type(child[i]) == "list":       
          if len(child[i]) > 1:
            for j, c in enumerate(child[i]):
                if c > max_:
                    child[i][j] = max_
                elif c < min_:
                    child[i][j] = min_ + 0.001
        else:
            if child[i] > max_:
                child[i] = max_
            elif child[i] < min_:
                child[i] = min_ + 0.001
    return child


def bi_mut(Pt, m_rate):
    for i in range(len(Pt)):
        for j in range(len(Pt[i])):
            if np.random.rand() < m_rate:
                Pt[i][j] = 1 if Pt[i][j] == 0 else 0
    return Pt


#######################################################################################################################
#                                       Fast non dominated sorting                                                    #
#######################################################################################################################
def fast_nd_sort(fm): ## input fm should be an array fm = M*N
    Sp = [[] for _ in range(fm.shape[1])] ## objective values in each class
    np = [0 for _ in range(fm.shape[1])]
    F = [[]]
    for p in range(fm.shape[1]):  ## No. of values in each objective
        Sp[p] = []
        np[p] = 0
        for q in range(fm.shape[1]):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for m in range(len(fm)): ## for each objective fm = m*N(population size)
                if fm[m,p] < fm[m,q]: ## p dominates q
                    dom_less += 1
                elif fm[m,p] == fm[m,q]:
                    dom_equal += 1
                else:
                    dom_more += 1  ## q dominates p
            if (dom_less == 0) and (dom_equal != len(fm)): ## q dominates p
                np[p] += 1
            elif (dom_more == 0) and (dom_equal != len(fm)): ## p dominates q
                Sp[p].append(q)
        if np[p] == 0:
            F[0].append(p)
    ## sorting
    i = 0
    while (F[i] != []):
        Q = []
        for p in F[i]:  ## Rt pointer/position in F
            for q in Sp[p]:  ## corresponding Rt pointer/position in Sp
                np[q] -= 1
                if np[q] == 0:
                    Q.append(q)
        i += 1
        F.append(Q)
    del F[len(F) - 1]
    return F  ## list index## [......]


#######################################################################################################################
#                                           Crowding Distance                                                         #
#######################################################################################################################
def sort_index(f, f_value, front):
    f1 = f.copy()
    sorted_index = []
    for i in f_value:
        for j in range(len(f1)): ## 2: ensure index in current front 3: avoid repeated index in current front
            if (i == f1[j]) and (j in front) and (j not in sorted_index):
                sorted_index.append(j)
    return sorted_index


def distance_sort(distance, front):
    distance_index = []
    for _ in range(len(front)):
        a = distance.index(max(distance))
        if a in front:  ## necessary
            distance_index.append(a)
            distance[a] = 0
    for i in range(len(distance)):  ## in case (when there are at least 3 distance values are the same)
        if (i in front) and (i not in distance_index): ## in case distance value = 0
            distance_index.append(i)
    return distance_index


def crowding_distance(front, fm):  ## can be used for multi-objectives
    distance = [0 for i in range(fm.shape[1])]
    # distance_index = []
    for f_m in fm:
        f_m_value = []
        for i in front:
            f_m_value.append(f_m[i])
        sorted_index_m = sort_index(f_m, sorted(f_m_value), front)
        distance[sorted_index_m[0]] = np.inf
        distance[sorted_index_m[-1]] = np.inf
        for j in range(1, len(front)-1):
            distance[sorted_index_m[j]] += (f_m[sorted_index_m[j+1]]-f_m[sorted_index_m[j-1]])/(max(f_m)-min(f_m))
    distance_index = distance_sort(distance, front)
    return distance_index


def assignCrowdingDist(front, fm):
    if len(front) == 0:
        return
    elif len(front) == 1:
        return front

    distances = [[0., 0.] for _ in range(len(front))]
    crowd = [(fm[:, f], i, f) for i, f in enumerate(front)]
    nobj = fm.shape[0]

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = [np.inf, crowd[0][2]]
        distances[crowd[-1][1]] = [np.inf, crowd[-1][2]]
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]][0] += (next[0][i] - prev[0][i]) / norm
            distances[cur[1]][1] = cur[2]
    distances.sort(key=lambda x: x[0], reverse=True)
    distances_idx = [x[1] for x in distances]
    return distances_idx


def test_run(Pt, pop_size):
    for t in range(300):
        Qt = poly_mut(crossover_SBX(Pt,2,0.9),20,0.001,4,-4)
        Rt = np.vstack((Pt,Qt))
        f1,f2 = FON(Rt)
        fm = np.vstack((f1,f2))
        F = fast_nd_sort(fm)
        distance_sort_index = []
        Pt_1 = []
        for front in F:
            if len(distance_sort_index) + len(front) < pop_size:
                distance_sort_index.extend(front)
                continue
            elif len(distance_sort_index) + len(front) == pop_size:
                distance_sort_index.extend(front)
                break
            distance_sort_index.extend(assignCrowdingDist(front, fm))
            break
        for i in distance_sort_index[0:pop_size]:
            Pt_1.append(Rt[i])
        Pt = np.array(Pt_1)
        function1, function2 = FON(Pt)
        plt.ion()
        plt.cla()
        plt.title('NSGA-2')
        plt.xlabel('Function 1')
        plt.ylabel('Function 2')
        # plt.cla()
        plt.scatter(function1, function2)
        plt.pause(0.05)
        plt.ioff()
        plt.show()
    print('### Final solutions:### ',Pt)
    print('\nnd_front: ', F)

if __name__ == '__main__':
    start = time.time()
    pop_size = 40
    Pt = -4+8*np.random.rand(pop_size,3)
    a = test_run(Pt,pop_size)
    print('Total duration time %.4f s: ', time.time()-start)
