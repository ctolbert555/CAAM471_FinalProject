import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import defaultdict
import time
from copy import deepcopy

from sys import exit

def main():

    fnames = ["att48.txt", "berlin52.txt", "gr21.txt", "hk48.txt", "ulysses22.txt"]
    # fnames = ["berlin52.txt", "gr21.txt", "hk48.txt", "ulysses22.txt"]
    idx = 0
    for fname in fnames:
        print("########################")
        print(f"# File {idx}: {fname}")
        print("########################")

        f = open(fname)
        numNodes, numEdges, edges, weights = read_file(f)

        start = time.time()
        TSP(numNodes, numEdges, edges, weights)
        end = time.time()

        print(f"\n\nProblem solved in {end-start} seconds\n\n")
        idx += 1

    # f = open("att48.txt")
    # f = open("berlin52.txt")
    # f = open("gr21.txt")
    # f = open("hk48.txt")
    # f = open("ulysses22.txt")

    # TSP(f)

def TSP(numNodes, numEdges, edges, weights):
    """
    Solves and prints the travelling salesman problem given the edges and weights
    """

    model = create_model(numNodes, numEdges, edges, weights)

    #Let Gurobi know that the model has changed
    model.update()

    #optimize model
    model.optimize()

    #add initial subtour eliminations (none should remain)
    while add_cut(model, numNodes, numEdges, edges):
        # print("ADDING CUT")
        model.update()
        model.optimize()

    #sorted weight dictionary
    wDict = dict()
    for i in range(numEdges):
        wDict[edges[i][0],edges[i][1]] = weights[i]
        wDict[edges[i][1],edges[i][0]] = weights[i]
    wDict = dict(sorted(wDict.items(), key=lambda item: item[1]))

    upper_bound = greedy_ub(wDict, numNodes, numEdges, edges, weights)

    format_str = "{:<12s} {:<12s} {:<12s}"
    print(format_str.format(*["Iteration","Lower Bound","Upper Bound"]))
    print(format_str.format(*["0","0",str(upper_bound)]))

    # iter = 1
    # while model.status == 2:

    bestModel, bestValue = solve_relaxation(model, numNodes, numEdges, edges, weights)

    # upper_bound = min([upper_bound, get_ub(bestModel, numNodes, numEdges, edges, weights, wDict)])

    # print(format_str.format(*[str(iter),str(bestValue),str(upper_bound)]))
    # iter += 1

    # if add_cut(model, bestModel, numNodes, numEdges, edges):
    #     break;

    #re-solve
    # model.update()
    # model.optimize()

    print("\n\n")
    if model.status == 2:
        #write out the lp in a lp-file
        bestModel.write("TSP.lp")

        x = bestModel.getVars()
        print("Selected edges:")
        print("Start\t End\t Weight")
        for i in range(numEdges):
            if (x[i].x > 0):
                print(f"{edges[i][0]}\t {edges[i][1]}\t {weights[i]}")
                # TODO: REMOVE AMT WHEN WE HAVE ONLY VALUES IN {0,1}
                # print(str(edges[i][0]) + " " + str(edges[i][1]) + " " + str(weights[i]) + " - AMT: " + str(x[i].x))

        print("The cost of the best tour is: " + str(bestValue))



def read_file(f):
    """
    Reads contents of file
    Returns numNodes, numEdges, edges, weights
    """
    rawNodes, rawEdges = f.readline().split()
    numNodes = int(rawNodes)
    numEdges = int(rawEdges)
    edges = []
    weights = []
    for i in range(numEdges):
        node1, node2, weight = f.readline().split()
        edges.append( (int(node1), int(node2)) )
        weights.append(float(weight))

    return numNodes, numEdges, edges, weights


def greedy_ub(wDict, numNodes, numEdges, edges, weights):
    """
    uses a greedy algorithm to solve TSP
    """
    #adjacency list
    adj_list = [[] for i in range(numNodes)]
    for edge in wDict.keys():
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])

    visited = set();
    visited.add(0)

    curr_node = 0
    total = 0;
    while len(visited) < numNodes:
        for next_node in adj_list[curr_node]:
            if not (next_node in visited):
                break;
            if next_node == adj_list[curr_node][-1]: #couldnt finish path
                return np.inf
        visited.add(next_node);
        total += wDict[(curr_node,next_node)]
        curr_node = next_node;

    return total + wDict[(curr_node,0)]



def create_model(numNodes, numEdges, edges, weights):
    """
    Creates a linear model without subtour constraints
    """
    #create model
    model = grb.Model()
    model.setParam('OutputFlag', 0)

    #define x variables and set objective values
    x = model.addVars(numEdges, obj=weights, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    #set objective function sense
    model.modelSense = GRB.MINIMIZE

    #add constraints to model
    for i in range(numNodes):
        nodeIncident = [x[j] for j in range(numEdges) if (edges[j][0] == i or edges[j][1] == i)]
        model.addConstr(sum(nodeIncident) == 2)

    return model


def solve_relaxation(model, numNodes, numEdges, edges, weights):
    """
    uses branch and bound to solve the relaxation
    returns bestModel, bestValue
    """
    x = model.getVars()
    bestModel = None
    bestValue = np.inf
    anyFractions = False
    for i in range(numEdges):
        if (x[i].x > 0 and x[i].x < 1):
            anyFractions = True
            bestModel, bestValue = model_branch(model, i, True, bestModel, bestValue,
                                                    numNodes, numEdges, edges, weights)
            bestModel, bestValue = model_branch(model, i, False, bestModel, bestValue,
                                                    numNodes, numEdges, edges, weights)
            break
    # If no variables are fractional, we return newModel
    if not anyFractions:
        bestModel = model
        bestValue = model.getObjective().getValue()

    return bestModel, bestValue



def get_ub(bestModel, numNodes, numEdges, edges, weights, wDict):
    """
    given a solution to the relaxation, create an UB by connecting the cycles
    wDict is assumed to be sorted
    """
    bestX = bestModel.getVars();

    #Use DFS to find connected components

    is_reached, connectedness = is_connected(V,E,0)

    cycles = [list(is_reached.keys()).copy()]
    s = 1
    while len(is_reached) < numNodes:
        if not is_reached[s]:
            new_reached, connectedness = is_connected(V,E,s)
            cycles.append(list(new_reached.keys()).copy())
            for v in new_reached.keys():
                is_reached[v] = True
        s += 1;

    #rearrange data into useful format
    max_edges = set()
    free_nodes = set()
    node2path = dict()
    for cycle in cycles:
        max_edge = (cycle[-1],cycle[0])
        max_weight = wDict[max_edge]
        for i in range(len(cycle) - 1):
            node1 = cycle[i];
            node2 = cycle[i+1];
            edge = (node1,node2)
            if wDict[edge] > max_weight:
                max_edge = edge
                max_weight = wDict[edge]

        fcycle = np.flip(cycle).tolist()
        free_nodes.add(max_edge[0])
        node2path[max_edge[0]] = np.roll(fcycle, -fcycle.index(max_edge[0])).tolist()

        free_nodes.add(max_edge[1])
        node2path[max_edge[1]] = np.roll(cycle, -cycle.index(max_edge[1])).tolist()

        max_edges.add((max_edge[0],max_edge[1]))
        max_edges.add((max_edge[1],max_edge[0]))



    #join cycles
    eQueue = list(wDict.keys()).copy()
    while len(free_nodes) > 2:
        if len(eQueue) == 0: #can't join loops
            return np.inf
        e = eQueue.pop(0)
        if (e[0] in free_nodes) and (e[1] in free_nodes) and not (e in max_edges):
            free_nodes.remove(e[0])
            free_nodes.remove(e[1])
            new_start = node2path[e[0]][-1]
            new_end = node2path[e[1]][-1]

            node2path[new_start] = node2path[new_start] + node2path[e[1]]
            node2path[new_end] = node2path[new_end] + node2path[e[0]]
            max_edges.add((new_start,new_end))
            max_edges.add((new_end,new_start))

    path = node2path[free_nodes.pop()]
    return wDict[(path[0],path[-1])] + sum(wDict[(path[i],path[i+1])] for i in range(numNodes - 1))


def model_branch(origModel, varIdx, isLeft, bestModel, bestVal, numNodes, numEdges, edges, weights):
    """
    Performs branch and bound for the inductive step
    """
    newModel = origModel.copy()
    x = newModel.getVars()
    if isLeft:
        x[varIdx].UB = 0
    else:
        x[varIdx].LB = 1
    newModel.update()
    newModel.optimize()

    #add initial subtour eliminations (none should remain)
    while add_cut(newModel, numNodes, numEdges, edges):
        # print("ADDING CUT IN BRANCH")
        newModel.update()
        newModel.write("TSP.lp")
        newModel.optimize()
    # print("DONE ADDING CUTS IN THIS BRANCH")

    if origModel.status == 2:
        # Only explore branch if it is has potential
        if newModel.getObjective().getValue() < bestVal:
            anyFractions = False
            for i in range(numEdges):
                if (x[i].x > 0 and x[i].x < 1):
                    anyFractions = True
                    bestModel, bestVal = model_branch(newModel, i, True, bestModel, bestVal,
                                                        numNodes, numEdges, edges, weights)
                    bestModel, bestVal = model_branch(newModel, i, False, bestModel, bestVal,
                                                        numNodes, numEdges, edges, weights)
                    break
            # If no variables are fractional, we return newModel, newVal
            if not anyFractions:
                bestModel = newModel
                bestVal = newModel.getObjective().getValue()
                # print("NEW BEST MODEL W/ VAL: " + str(bestVal))
    return bestModel, bestVal


def add_cut(model, numNodes, numEdges, edges):
    """
    Let C be the connected component containing 0.
    Adds a cut corresponding to the constraint: num edges exiting C >= 2
    returns True if best solution is feasible in the master problem, false otherwise.
    """
    x = model.getVars();

    # print(len(model.getConstrs()))
    #Create subgraph induced by tour
    V = range(numNodes)
    E = defaultdict(dict)
    for i in range(numEdges):
        #E formatted as {node: {connected node1: weight, connected node2: weight}, next_node: {etc}}
        E[edges[i][0]][edges[i][1]] = x[i].x
        E[edges[i][1]][edges[i][0]] = x[i].x

    #Stoer-wagner time!
    # mincut_set, mincut_weight = min_cut(list(V),E,0)
    cuts = min_cut(list(V),E,0)

    # if mincut_weight >= 2:
    if len(cuts) == 0:
        return False
    for mincut_set in cuts:
        gammaEdges = [x[i] for i in range(numEdges) if ((edges[i][0] in mincut_set) and (edges[i][1] in mincut_set))]
        model.addConstr(sum(gammaEdges) <= len(mincut_set) - 1)
        # print(mincut_set)
        # cycle = mincut_set
        # crossEdges = [x[i] for i in range(numEdges) if (((edges[i][0] in cycle) and not (edges[i][1] in cycle))
        #                                                 or ((edges[i][1] in cycle) and not (edges[i][0] in cycle)))]
        # model.addConstr(sum(crossEdges) >= 2)

    return True


def is_connected(V,E,s):
    """
    uses DFS to check if graph is connected.
    Returns connected component of s and whether G is connected
    V is a list of vertices
    E is a dictionary v_i --> {v_j | (v_i,v_j) in edges}
    s is the start node for DFS
    """
    is_reached = defaultdict(bool);
    DFS(V,E,s,is_reached);
    if len(is_reached) == len(V):
        return is_reached, True
    return is_reached, False


def DFS(V,E,s,is_reached):
    """
    Performs depth first search
    """
    is_reached[s] = True;
    for v in E[s]:
        if not is_reached[v]:
            DFS(V,E,v,is_reached)

def min_cut_phase(V,E,a,V2Cut):
    """
    An optimized version of MinCutPhase of the Stoer-Wagner algorithm
    We assume that we are only interested in 0 cuts
    Returns the cut of the phase and V2Cut[t]
    Modifies V2Cut
    """
    A = {a}
    s = a
    t = a
    neighbors = set(E[a])
    VmAmN = (V - A) - neighbors
    while len(A) != len(V):
        if len(neighbors) == 0 and len(VmAmN) == 0:
            exit()
        while len(neighbors) > 0: #keep adding connected neighbors
            z = neighbors.pop()
            A.add(z)
            neighbors |= {v for v in E[z] if not (v in A)}
            for v in neighbors:
                VmAmN.discard(v)

            s = t
            t = z

        if len(VmAmN) > 0:  #add unconnected neighbor
            z = VmAmN.pop()
            A.add(z)
            neighbors |= {v for v in E[z] if not (v in A)}
            for v in neighbors:
                VmAmN.discard(v)

            s = t
            t = z

    #get cut of phase
    cotp = len(E[t])
    cut = V2Cut[t]

    #modify G
    V.remove(t)
    for v in E[t]:
        E[v].remove(t)
        if (s != v) and (s not in E[v]):
            E[v].add(s)
            E[s].add(v)

    V2Cut[s] |= V2Cut[t]

    return cotp, cut


def min_cut_phase_new(V,E,a,V2Cut):
    A = {a}
    s = a
    t = a
    VmAmN = (V - A) - {y for y,w in E[a].items() if w > 0}
    while len(A) != len(V):
    #     print("A:" + str(len(A)))
    #     print("V:" + str(len(V)))
    #     print(VmAmN)
        connectedness = defaultdict(float)
        for v in A:
            for e,w in E[v].items():
                if w>0:
                    VmAmN.discard(e)
                if e not in A:
                    connectedness[e] += w
        z = None
        max_w = 0
        # print("CONNECTEDNESS:")
        for e,w in connectedness.items():
            # print(str(e) + ":" + str(w))
            if w >= max_w:
                z = e
                max_w = w
        #if rest of graph not connected
        if max_w == 0:
            z = VmAmN.pop()
        A.add(z)
        s = t
        t = z

    #get cut of the phase
    # for v in E[t].keys():
    #     if v not in V and E[t][v] != 0:
    #         print("AAAAAAAAA")
    #         exit()
    # cotp = sum(E[t][v] for v in V if v in E[t].keys())
    cotp = max_w
    cut = V2Cut[t]

    #modify G
    V.remove(t)
    for v, w in E[t].items():
        if v == s:
            E[s].pop(t)
        elif v in E[s].keys():
            E[s][v] += w
            E[v][s] += w
        else:
            E[s][v] = w
            E[v][s] = w
        if v != t and t in E[v].keys():
            E[v].pop(t)
    E.pop(t)

    V2Cut[s] |= V2Cut[t]

    # print("COTP: " + str(cotp))
    # print("CUT: " + str(cut))

    return cotp, cut


def min_cut(V,E,a):
    """
    An optimized version of the Stoer-Wagner algorithm
    Returns a list of 0 cuts that have been found.
    Returns an empty list if none exist
    """
    V = set(V)
    E = deepcopy(E)
    #dictionary that keeps track of the modifications of the graph
    V2Cut = {v:{v} for v in V}  # v --> {v_1, v_2, v_3}
    cuts = []
    cut = {}
    mincut_val = np.inf
    # print("here")
    while len(V) > 1:
        # print(len(V))
        cotp, cut = min_cut_phase_new(V,E,a,V2Cut)
        if cotp < 2 - 0.01: #floating point error
            cuts.append(cut)
        # if(cotp == 0):
        #      return cut, cotp
        # elif(cotp < mincut_val):
        #     mincut = cut
        #     mincut_val = cotp

    # return mincut, mincut_val
    return cuts

if __name__ == '__main__':
    main()
