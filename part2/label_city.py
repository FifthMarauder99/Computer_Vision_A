import numpy as np
import sys

def DataCost(I_node_coord, i_Label, TotalBribes):
    label = {"R":0,"D":1}
    return TotalBribes[label[i_Label]][I_node_coord[0], I_node_coord[1]]

def FenceCost(I_label, J_label):
    return 0 if I_label == J_label else 1000

def InitWorkingMessageMatrix():
    Message_working = np.zeros((n, n, 4, 2), dtype=np.float32)
    return Message_working

def _NeighbourNodeInputMessage(i_node,j_node,i_eval_label,NGraph):
    direction ={"RIGHT":0,"LEFT":1,"DOWN":2,"UP":3}
    label = {"R":0,"D":1}
    MessageSum = 0
    for neighbour_node in NGraph[i_node]:
        if j_node not in neighbour_node:
            # print(neighbour_node)
            # print(i_node[0],i_node[1],direction[neighbour_node[1]],label[i_eval_label])
            # print(i_node,"<--",neighbour_node,neighbour_node[1],"index in Message store = ",direction[neighbour_node[1]],"for the label = ",i_eval_label, "with index = ",label[i_eval_label])
            MessageSum += Message_store[i_node[0]][i_node[1]][direction[neighbour_node[1]]][label[i_eval_label]]
        else:pass
    return MessageSum

def _CalculateMessage(i_node, j_node, TotalBribes):
    message_i_jR, message_i_jD = [], []
    
    # Check if nodes are adjacent
    if not ((abs(i_node[0] - j_node[0]) == 1 and i_node[1] == j_node[1]) or (i_node[0] == j_node[0] and abs(i_node[1] - j_node[1]) == 1)):
        raise Exception(f"The message nodes i = {i_node} to j = {j_node} broadcasted are wrong")
    
    for i_node_label in ["R", "D"]:
        j_node_label = "R"   # Initialize to "R" for first message calculation
        message_i_j_data_R = DataCost(i_node, i_node_label, TotalBribes)  # Compute data cost of node i with label i_node_label
        message_i_j_fence_R = message_i_j_data_R + FenceCost(i_node_label, j_node_label)  # Compute fence cost between nodes i and j with respective labels
        # Compute incoming message sum from all other neighbours of node i, except for j_node, with label i_node_label
        Final_Rcost = message_i_j_fence_R + _NeighbourNodeInputMessage(i_node, j_node, i_node_label, NGraph)  
        message_i_jR.append(Final_Rcost)
        
    for i_node_label in ["R", "D"]:
        j_node_label = "D"   # Initialize to "D" for second message calculation
        message_i_j_data_D = DataCost(i_node, i_node_label, TotalBribes)  # Compute data cost of node i with label i_node_label
        message_i_j_fence_D = message_i_j_data_D + FenceCost(i_node_label, j_node_label)  # Compute fence cost between nodes i and j with respective labels
        # Compute incoming message sum from all other neighbours of node i, except for j_node, with label i_node_label
        Final_Dcost = message_i_j_fence_D + _NeighbourNodeInputMessage(i_node, j_node, i_node_label, NGraph)
        message_i_jD.append(Final_Dcost)
        
    # Return the minimum values of message_i_jR and message_i_jD
    return np.min(message_i_jR), np.min(message_i_jD)

def _NeighbourNodeBelief(i_node,i_eval_label,Message_store_final):
    label = {"R":0,"D":1}
    MessageSum = np.sum(Message_store_final[i_node],axis=0)[label[i_eval_label]]
    return MessageSum

def MessagePropItr(n,TotalBribes):
    direction ={"RIGHT":0,"LEFT":1,"DOWN":2,"UP":3}
    Message_working = InitWorkingMessageMatrix()
    for x_coord in range(n):
        for y_coord in range(n):
            i_node = (x_coord,y_coord)
            for j_direction in ["RIGHT","LEFT","DOWN","UP"]:
                if j_direction == "RIGHT" and ((i_node[1]+1)<=n-1):
                    j_node = (i_node[0],i_node[1]+1)
                    #print(i_node,j_node,"RIGHT")
                    #R.append([(i_node,"-->",j_node),_CalculateMessage(i_node,j_node,TotalBribes)])
                    Raw_L = _CalculateMessage(i_node,j_node,TotalBribes)
                    if np.sum(Raw_L):
                        Norm_L = Raw_L - np.log(np.sum(Raw_L))
                        Message_working[j_node][direction["LEFT"]] = Norm_L
                    else:
                        Message_working[j_node][direction["DOWN"]] = Raw_L


                elif j_direction == "LEFT" and ((i_node[1]-1)>=0):
                    j_node = (i_node[0],i_node[1]-1)
                    #print(i_node,j_node,"LEFT")
                    #L.append([(i_node,"-->",j_node),_CalculateMessage(i_node,j_node,TotalBribes)])
                    Raw_R = _CalculateMessage(i_node,j_node,TotalBribes)
                    if np.sum(Raw_R):
                        Norm_R = Raw_R - np.log(np.sum(Raw_R))
                        Message_working[j_node][direction["RIGHT"]] = Norm_R
                    else:
                        Message_working[j_node][direction["DOWN"]] = Raw_R

                elif j_direction == "DOWN" and ((i_node[0]+1)<=n-1):
                    j_node = (i_node[0]+1,i_node[1])
                    #print(i_node,j_node,"DOWN")
                    #D.append([(i_node,"-->",j_node),_CalculateMessage(i_node,j_node,TotalBribes)])
                    Raw_U= _CalculateMessage(i_node,j_node,TotalBribes)
                    if np.sum(Raw_U):
                        Norm_U = Raw_U - np.log(np.sum(Raw_U))
                        Message_working[j_node][direction["UP"]] = Norm_U
                    else:
                        Message_working[j_node][direction["DOWN"]] = Raw_U

                elif j_direction == "UP" and ((i_node[0]-1)>=0):
                    j_node = (i_node[0]-1,i_node[1])
                    #print(i_node,j_node,"UP")
                    #U.append([(i_node,"-->",j_node),_CalculateMessage(i_node,j_node,TotalBribes)])
                    Raw_D = _CalculateMessage(i_node,j_node,TotalBribes)
                    if np.sum(Raw_D):
                        Norm_D = Raw_D - np.log(np.sum(Raw_D))
                        Message_working[j_node][direction["DOWN"]] = Norm_D
                    else:
                        Message_working[j_node][direction["DOWN"]] = Raw_D

    Message_working /= np.sum(Message_working)

    return Message_working


def _CalculateBelief(n,Message_store_final):
    Label_mat = np.array([[""]*n for _ in range(n)]).astype(np.str_)
    for x_coord in range(n):
        for y_coord in range(n):
            i_node = (x_coord,y_coord)
            NodeRBelief = DataCost(i_node,"R",TotalBribes) + _NeighbourNodeBelief(i_node,"R",Message_store_final)
            NodeDBelief = DataCost(i_node,"D",TotalBribes) + _NeighbourNodeBelief(i_node,"D",Message_store_final)
            if NodeRBelief <=NodeDBelief:
                Label_mat[i_node] = "R"
            else:
                Label_mat[i_node] = "D"
    return Label_mat


def _CurLabelMapCost(CurLabel):
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

    Max_Iterations = 150
    convergence_counter = 0
    iter_counter = 0
    prev_cost = 0
    Cost_trail = []
    min_label = [10000000,0]
    
    
    n = int(sys.argv[1])  # Size of the city grid
    r_file = sys.argv[2]  # File containing R bribes
    d_file = sys.argv[3]  # File containing D bribes
    
    r_bribes = np.loadtxt(r_file, dtype=np.float32)
    d_bribes = np.loadtxt(d_file, dtype=np.float32)
    
    TotalBribes = np.array([r_bribes, d_bribes])
    
    
    global NGraph
    NGraph = {}
    for i in range(n):
        for j in range(n):
            NGraph[(i, j)] = set([(i, j + 1, 'RIGHT'),(i, j - 1, 'LEFT'),(i + 1, j, 'DOWN'),(i - 1, j, 'UP')])
            NGraph[(i, j)] = set(((x, y), direction) for x, y, direction in NGraph[(i, j)] if 0 <= x < n and 0 <= y < n)
    
    Message_store = np.zeros((n, n, 4, 2), dtype=np.float32)
    print("Computing optimal labeling:")
    for iter_ in range(Max_Iterations):
        Message_store= MessagePropItr(n,TotalBribes)
        CurLabel = _CalculateBelief(n,Message_store)
        CurCost = _CurLabelMapCost(CurLabel)
        Cost_trail.append(CurCost)
        CostDiff = CurCost - prev_cost
        prev_cost = CurCost
        if CurCost < min_label[0]:
            min_label[0] = CurCost
            min_label[1] = Message_store
        #if iter_counter%10 == 0 :
        #    print(CurLabel)
        #    print(f"The current cost = {CurCost},{CostDiff}")
        iter_counter += 1
        
    
    Message_store_final = min_label[1].copy()
    FinalLabel =_CalculateBelief(n,Message_store_final)
    FinalCost = _CurLabelMapCost(FinalLabel)
    print("\n".join(" ".join(row) for row in FinalLabel))
    print("Total cost =", FinalCost)