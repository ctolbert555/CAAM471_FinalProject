import gurobipy as grb
from gurobipy import GRB
import numpy as np
from collections import defaultdict
from sys import exit
import random

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



    while True:
        #if status comes back as optimal (value=2) then print out ony nonzero solution values
        if model.status != 2:
            break; #infeasible

        bestModel, bestValue = solve_relaxation(model, numNodes, numEdges, edges, weights)
        print("Relaxation Solved! Current lower bound:" + str(bestValue))

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
    is_reached, connectedness = is_connected(V,E,random.randint(0,numNodes-1))
    if connectedness:
        return True

    #add cut
    crossEdges = [x[i] for i in range(numEdges) if (is_reached[edges[i][0]] and not is_reached[edges[i][1]])
                                                    or (is_reached[edges[i][1]] and not is_reached[edges[i][0]])]
    model.addConstr(sum(crossEdges) >= 2)

    #print("\t Size of loop conatining 0: " + str(sum(1 for i in is_reached.keys() if is_reached[i])))

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
