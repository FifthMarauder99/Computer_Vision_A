import numpy as np
import sys

def DataCost(I_node_coord, i_Label, TotalBribes):
    label = {"R":0,"D":1}
    return TotalBribes[label[i_Label]][I_node_coord[0], I_node_coord[1]]

def FenceCost(I_label, J_label):
    return 0 if I_label == J_label else 1000

def InitWorkingMessageMatrix(n):
    Message_working = np.zeros((n, n, 4, 2), dtype=np.float32)
    return Message_working

def _NeighbourNodeInputMessage(i_node,j_node,i_eval_label,NGraph,Message_store):
    direction_reversal  ={"RIGHT":1,"LEFT":0,"DOWN":3,"UP":2}
    label = {"R":0,"D":1}
    MessageSum = 0

    CurrentN = [neighbour_node for neighbour_node in NGraph[i_node] if j_node not in neighbour_node]
    #print(CurrentN,i_node,j_node)
    for node in CurrentN:
        #print(f"Message_store[{i_node}][{direction[node[1]]}][label[{i_eval_label}]]")
        MessageSum += Message_store[i_node][direction_reversal[node[1]]][label[i_eval_label]]

    return MessageSum

def _CalculateMessage(i_node, j_node, TotalBribes,NGraph,Message_store):
    # Check if nodes are adjacent
    if not ((abs(i_node[0] - j_node[0]) == 1 and i_node[1] == j_node[1]) or (i_node[0] == j_node[0] and abs(i_node[1] - j_node[1]) == 1)):
        raise Exception(f"The message nodes i = {i_node} to j = {j_node} broadcasted are wrong")
    
    message_i_j_data_fence_RR = DataCost(i_node, "R", TotalBribes) + FenceCost("R","R") + _NeighbourNodeInputMessage(i_node, j_node, "R", NGraph,Message_store) 
    message_i_j_data_fence_DR = DataCost(i_node, "D", TotalBribes) + FenceCost("D","R") + _NeighbourNodeInputMessage(i_node, j_node, "D", NGraph,Message_store) 

    message_i_j_data_fence_RD = DataCost(i_node, "R", TotalBribes) + FenceCost("R","D") + _NeighbourNodeInputMessage(i_node, j_node, "R", NGraph,Message_store) 
    message_i_j_data_fence_DD = DataCost(i_node, "D", TotalBribes) + FenceCost("D","D") + _NeighbourNodeInputMessage(i_node, j_node, "D", NGraph,Message_store) 

    #print([message_i_j_data_fence_RR,message_i_j_data_fence_DR],[message_i_j_data_fence_RD,message_i_j_data_fence_DD])
        
    # Return the minimum values of message_i_jR and message_i_jD
    return np.min([message_i_j_data_fence_RR,message_i_j_data_fence_DR]), np.min([message_i_j_data_fence_RD,message_i_j_data_fence_DD])

def _NeighbourNodeBelief(i_node,i_eval_label,Message_store_final):
    label = {"R":0,"D":1}
    MessageSum = np.sum(Message_store_final[i_node],axis=0)[label[i_eval_label]]
    return MessageSum

def MessagePropItr(n,TotalBribes,NGraph,Message_store):

    Message_working = InitWorkingMessageMatrix(n)
    for x_coord in range(n):
        for y_coord in range(n):
            i_node = (x_coord,y_coord)
            for j_direction in ["RIGHT","LEFT","DOWN","UP"]:

                if j_direction == "RIGHT" and ((i_node[1]+1)<=n-1):
                    j_node = (i_node[0],i_node[1]+1)
                    Message_working[j_node][1] = _CalculateMessage(i_node,j_node,TotalBribes,NGraph,Message_store)
                    #print(f"{i_node} --> {j_node} to Message_working[{j_node}][direction[LEFT 1]]")
                    
                elif j_direction == "LEFT" and ((i_node[1]-1)>=0):
                    j_node = (i_node[0],i_node[1]-1)
                    Message_working[j_node][0] = _CalculateMessage(i_node,j_node,TotalBribes,NGraph,Message_store)
                    #print(f"{i_node} --> {j_node} to Message_working[{j_node}][direction[RIGHT 0]]")

                elif j_direction == "DOWN" and ((i_node[0]+1)<=n-1):
                    j_node = (i_node[0]+1,i_node[1])
                    Message_working[j_node][3] = _CalculateMessage(i_node,j_node,TotalBribes,NGraph,Message_store)
                    #print(f"{i_node} --> {j_node} to Message_working[{j_node}][direction[UP 3]]")

                elif j_direction == "UP" and ((i_node[0]-1)>=0):
                    j_node = (i_node[0]-1,i_node[1])
                    Message_working[j_node][2] = _CalculateMessage(i_node,j_node,TotalBribes,NGraph,Message_store)
                    #print(f"{i_node} --> {j_node} to Message_working[{j_node}][direction[DOWN 2]]")
    # if np.sum(Message_working) > 100000:
    #     Message_working /= np.max(Message_working)

    return Message_working


def _CalculateBelief(n,Message_store_final,TotalBribes):
    Label_mat = np.array([[""]*n for _ in range(n)]).astype(np.str_)
    for x_coord in range(n):
        for y_coord in range(n):
            i_node = (x_coord,y_coord)
            NodeRBelief = DataCost(i_node,"R",TotalBribes) + _NeighbourNodeBelief(i_node,"R",Message_store_final)
            NodeDBelief = DataCost(i_node,"D",TotalBribes) + _NeighbourNodeBelief(i_node,"D",Message_store_final)
            if NodeRBelief < NodeDBelief:
                Label_mat[i_node] = "R"
            else:
                Label_mat[i_node] = "D"
    return Label_mat


def _CurLabelMapCost(CurLabel,TotalBribes,n):
    cost = 0
    for i in range(n):
        for j in range(n):
            if i < n-1 and CurLabel[i][j] != CurLabel[i+1][j]:
                cost += 1000
            if j < n-1 and CurLabel[i][j] != CurLabel[i][j+1]:
                cost += 1000
            if CurLabel[i][j] == 'R':
                cost += TotalBribes[0][i][j]
            else:
                cost += TotalBribes[1][i][j]
    return cost

if __name__ == "__main__":

    Max_Iterations = 90
    convergence_counter = 0
    prev_cost = 0
    Cost_trail = []
    min_label = [10000000,0]
    
    
    n = int(sys.argv[1])  # Size of the city grid
    r_file = sys.argv[2]  # File containing R bribes "sample_r_bribes.txt"
    d_file = sys.argv[3]  # File containing D bribes "sample_d_bribes.txt"
    
    r_bribes = np.loadtxt(r_file, dtype=np.float32)
    d_bribes = np.loadtxt(d_file, dtype=np.float32)
    
    TotalBribes = np.array([r_bribes, d_bribes])
    
    NGraph = {}
    for i in range(n):
        for j in range(n):
            NGraph[(i, j)] = set([(i, j + 1, 'RIGHT'),(i, j - 1, 'LEFT'),(i + 1, j, 'DOWN'),(i - 1, j, 'UP')])
            NGraph[(i, j)] = set(((x, y), direction) for x, y, direction in NGraph[(i, j)] if 0 <= x < n and 0 <= y < n)
    
    Message_store = np.zeros((n, n, 4, 2), dtype=np.float32)
    print("Computing optimal labeling:")

    for iter_ in range(Max_Iterations):
        Message_store= MessagePropItr(n,TotalBribes,NGraph,Message_store)
        CurLabel = _CalculateBelief(n,Message_store,TotalBribes)
        CurCost = _CurLabelMapCost(CurLabel,TotalBribes,n)
        Cost_trail.append(CurCost)
        CostDiff = CurCost - prev_cost
        prev_cost = CurCost
        if CurCost < min_label[0]:
            min_label[0] = CurCost
            min_label[1] = Message_store
    Message_store_final = min_label[1].copy()

    FinalLabel =_CalculateBelief(n,Message_store_final,TotalBribes)
    FinalCost = _CurLabelMapCost(FinalLabel,TotalBribes,n)

    print("\n".join(" ".join(row) for row in FinalLabel))
    print("Total cost =", FinalCost)
