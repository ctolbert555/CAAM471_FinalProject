import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import defaultdict

def main():
    # Read in Raw, Finals, and Demand

    f = open("att48.txt")
    # f = open("berlin52.txt")
    # f = open("gr21.txt")
    # f = open("hk48.txt")
    # f = open("ulysses22.txt")

    TSP(f)

def TSP(f):
    """
    Solves and prints the travelling salesman problem for an input file f
    """
    numNodes, numEdges, edges, weights = read_file(f)

    model = create_model(numNodes, numEdges, edges, weights)

    #Let Gurobi know that the model has changed
    model.update()

    #optimize model
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

    iter = 1
    while True:
        #if status comes back as optimal (value=2) then print out ony nonzero solution values
        if model.status != 2:
            break; #infeasible

        bestModel, bestValue = solve_relaxation(model, numNodes, numEdges, edges, weights)
        upper_bound = min([upper_bound, get_ub(bestModel, numNodes, numEdges, edges, weights, wDict)])

        print(format_str.format(*[str(iter),str(bestValue),str(upper_bound)]))
        iter += 1

        if add_cut(model, bestModel, numNodes, numEdges, edges):
            break;

        #re-solve
        model.update()
        model.optimize()

    if model.status == 2:
        #write out the lp in a lp-file
        bestModel.write("TSP.lp")

        x = bestModel.getVars()
        for i in range(numEdges):
            if (x[i].x > 0):
                # TODO: REMOVE AMT WHEN WE HAVE ONLY VALUES IN {0,1}
                print(str(edges[i][0]) + " " + str(edges[i][1]) + " " + str(weights[i]) + " - AMT: " + str(x[i].x))
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

    return total



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
    V = range(numNodes)
    E = defaultdict(list);
    for i in range(numEdges):
        if (bestX[i].x == 1):
            E[edges[i][0]].append(edges[i][1])
            E[edges[i][1]].append(edges[i][0])
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
        newModel.addConstr(x[varIdx] == 0)
    else:
        newModel.addConstr(x[varIdx] == 1)
    newModel.update()
    newModel.optimize()

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
    return bestModel, bestVal


def add_cut(model, bestModel, numNodes, numEdges, edges):
    """
    Let C be the connected component containing 0.
    Adds a cut corresponding to the constraint: num edges exiting C >= 2
    returns True if best solution is feasible in the master problem, false otherwise.
    """
    bestX = bestModel.getVars();
    x = model.getVars();

    #Create subgraph induced by tour
    V = range(numNodes)
    E = defaultdict(list);
    for i in range(numEdges):
        if (bestX[i].x == 1):
            E[edges[i][0]].append(edges[i][1])
            E[edges[i][1]].append(edges[i][0])

    #check connectedness
    is_reached, connectedness = is_connected(V,E,0)
    if connectedness:
        return True

    cycles = [set(is_reached.keys()).copy()]
    s = 1
    while len(is_reached) < numNodes:
        if not is_reached[s]:
            new_reached, connectedness = is_connected(V,E,s)
            cycles.append(set(new_reached.keys()).copy())
            for v in new_reached.keys():
                is_reached[v] = True
        s += 1;

    for cycle in cycles:
        crossEdges = [x[i] for i in range(numEdges) if (((edges[i][0] in cycle) and not (edges[i][1] in cycle))
                                                        or ((edges[i][1] in cycle) and not (edges[i][0] in cycle)))]
        model.addConstr(sum(crossEdges) >= 2)

    return False


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



if __name__ == '__main__':
    main()
