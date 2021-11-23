import gurobipy as grb
from gurobipy import GRB
import numpy as np

# Read in Raw, Finals, and Demand
f = open("att48.txt")
rawNodes, rawEdges = f.readline().split()
numNodes = int(rawNodes)
numEdges = int(rawEdges)
edges = []
weights = []
for i in range(numEdges):
    node1, node2, weight = f.readline().split()
    edges.append( (int(node1), int(node2)) )
    weights.append(float(weight))

#create model
model = grb.Model()

#define x variables and set objective values
x = model.addVars(numEdges, obj=weights, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

#set objective function sense
model.modelSense = GRB.MINIMIZE

#add constraints to model
for i in range(numNodes):
    nodeIncident = [x[j] for j in range(numEdges) if (edges[j][0] == i or edges[j][1] == i)]
    model.addConstr(sum(nodeIncident) == 2)

#Let Gurobi know that the model has changed
model.update()

#write out the lp in a lp-file
model.write("TSP.lp")

#optimize model
model.optimize()

#if status comes back as optimal (value=2) then print out ony nonzero solution values
if model.status == 2:
    cost = 0
    for i in range(numEdges):
        if (x[i].x > 0):
            cost += weights[i] * x[i].x
            # TODO: REMOVE AMT WHEN WE HAVE ONLY VALUES IN {0,1}
            print(str(edges[i][0]) + " " + str(edges[i][1]) + " " + str(weights[i]) + " - AMT: " + str(x[i].x))
    print("The cost of the best tour is: " + str(cost))