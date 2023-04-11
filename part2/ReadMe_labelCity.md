-   by Balajee Devesha S (basrini)

# Objective

-   In this part we are expected to implement a min-sum loopy belief propagation for the problem of town planning by assigning the houses an affliliation of "R" or "D"
-   Each house may have some innate predilection towards a particular affiliation described by the bribe matrices.
-   If any two adjacent houses have a difference in affiliation a fence needs to be constructed resulting in a cost of 1000 per side of fence.
-   The city council's aim is to come up with a configuration with minimum cost overall for implementation

# Methodology

Main Reference: <http://nghiaho.com/?page_id=1366>

-   I have tried to implement the min sum loopy belief for the problem by using a  modular approach to the question the closely resembles the orignal mathematical expression

## DataCost (Unary function)

-    The data cost is the cost of a node to assume a given affiliation given ny either of the bribe matrices.

```sh
def DataCost(I_node_coord, i_Label, TotalBribes):
    label = {"R":0,"D":1}
    return TotalBribes[label[i_Label]][I_node_coord[0], I_node_coord[1]]
```

## FenceCost (Pairwise function)

-   A straight forward function that checks if adjacent nodes are same affiliation or not and assigns a corresponding cost.

```sh

def FenceCost(I_label, J_label):
    return 0 if I_label == J_label else 1000

```
## Label map evaluation

-    The following function calculates the arrangement cost of a given label matrix

```sh

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

```

## Message Propagation

-   The core loop has various sub parts that need to be addressed in sequence.

### main loop

```sh
def MessagePropItr(n,TotalBribes):
    direction ={"RIGHT":0,"LEFT":1,"DOWN":2,"UP":3}
    Message_working = InitWorkingMessageMatrix()
    for x_coord in range(n):
        for y_coord in range(n):
            i_node = (x_coord,y_coord)
            for j_direction in ["RIGHT","LEFT","DOWN","UP"]:
                if j_direction == "RIGHT" and ((i_node[1]+1)<=n-1):
                    j_node = (i_node[0],i_node[1]+1)
                    Raw_L = _CalculateMessage(i_node,j_node,TotalBribes)
                elif j_direction == "LEFT" and ((i_node[1]-1)>=0):
                    j_node = (i_node[0],i_node[1]-1)
                    Raw_R = _CalculateMessage(i_node,j_node,TotalBribes)
                elif j_direction == "DOWN" and ((i_node[0]+1)<=n-1):
                    j_node = (i_node[0]+1,i_node[1])
                    Raw_U = _CalculateMessage(i_node,j_node,TotalBribes)
                elif j_direction == "UP" and ((i_node[0]-1)>=0):
                    j_node = (i_node[0]-1,i_node[1])
                    Raw_D = _CalculateMessage(i_node,j_node,TotalBribes)

```
-   As can be seen above we have a nested triple for loop for i_x,i_y coordinate and the direction of message being sent which decides the j nod eand uses the function _CalculateMessage to calculate the value of the message that needs to be sent to the jth node and stores the resultant message value in a message matrix of the following format in the jth node.

```sh
[[
[["R_val","D_val"][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],
[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],
[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],
[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],
[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],[[,][,][,][,]],
]]
```

-   The message storing is done in a n x n matrix that has 4 cells each denoting a neighboring direction and each 4 direction stores 2 values w.r.t message as label = "R" or "D". once the message is calculated it also does normalization of the mseeage matrix using log normalization.
-   

### _CalculateBelief

-    This function implements the