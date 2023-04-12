# A2

# Part  1:  3d-to-2d

## Statement : 
In this part, we try to control the first person view of the plane. We first take off the plane, then turn right, then travel a certain distance, then turn right again(U-turn in total), then travel the entire air strip , then turn right twice again to be at the top of the air strip and land on the air strip.

## Approach Explanation :
We have an animate_above function in which we create the animation, we control all the movements of the plane that we want. We use the numpy library to do all the calculations (matrix multiplications) and then use the matplotlib library to actually display the animation by saving it as an mp4 file.
- First, we read the airports.pts file and set the initial pose of the plane at height 5m, as the cockpit is about 5m above the ground.
- The matrix transformations are defined by creating matrices for each transformation (focal matrix, compass matrix, tilt matrix, twist matrix, and translation T matrix) using Numpy. 
- These matrices are then multiplied together in the correct order to obtain the final projection matrix. 
- The 3D point cloud of the airport is then multiplied by the projection matrix to obtain the 2D coordinates of the projected points.
- This above matrix multiplication and the projection matrix creates a simulation in which the plane travels on the ground(air strip).
- For all the next traversals, we need to vary the Translations, tx, ty and tz. One very easy way to do this is to vary the tx,ty and tz accordint to the current frame_number. This allows us to control the point to point position of the plane.
- We use the compass variable to turn the plane right or left. The turn does not work if are turning 90 degrees in one frame, thus for such a sharp turn, we turn the plane 6 degrees every frame, and continue this for 15 frames(for every turn), while still increasing/decreasing the ty, making the plane go forward or backward.
- For landing, we use the tz to decrease the height till it is 5m(so that it does not go below the initial(starting) height)


## References:
- https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
- https://www.geeksforgeeks.org/computer-graphics-3d-translation-transformation/


# Part 2

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

### _CalculateMessage

-    This function implements the message value to be sent from i to jth node. A direct implementation of the message propagation function 

```sh

for i_node_label in ["R","D"]:
        j_node_label = "R"   
        message_i_j_data_R = DataCost(i_node,i_node_label,TotalBribes) # Data cost implementation 
        message_i_j_fence_R = message_i_j_data_R + FenceCost(i_node_label,j_node_label) # Fence cost implementation
        Final_Rcost = message_i_j_fence_R + _NeighbourNodeInputMessage(i_node,j_node,i_node_label,NGraph)   # Neighbourhood sum implementation
        message_i_jR.append(Final_Rcost)

for i_node_label in ["R","D"]:
        j_node_label = "D"   
        message_i_j_data_D = DataCost(i_node,i_node_label,TotalBribes)
        message_i_j_fence_D = message_i_j_data_D + FenceCost(i_node_label,j_node_label)
        Final_Dcost = message_i_j_fence_D + _NeighbourNodeInputMessage(i_node,j_node,i_node_label,NGraph)
        message_i_jD.append(Final_Dcost)

return np.min(message_i_jR),np.min(message_i_jD)

```

### _NeighbourNodeInputMessage

-   This function calculates the message inputs from the surrounding neighbours excluding the oneto which the message is being sent using a adjacency graph Ngraph.

```sh
def _NeighbourNodeInputMessage(i_node,j_node,i_eval_label,NGraph):
    direction ={"RIGHT":0,"LEFT":1,"DOWN":2,"UP":3}
    label = {"R":0,"D":1}
    MessageSum = 0
    for neighbour_node in NGraph[i_node]:
        if j_node not in neighbour_node:
            MessageSum += Message_store[i_node[0]][i_node[1]][direction[neighbour_node[1]]][label[i_eval_label]]
        else:pass
    return MessageSum

```

## Belief calculation

-   All of the above functions comprise of a single iteration of the message propagation which is executed util convergence is observed or the max_iteration count is reached set to 150.

-   Now we calculate the beliefs of each node based on the messages and the data cost of each node and assign the one with lower cost out of the 2 possible labels.

```sh
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

```

-   Once either convergence or max iteration count is reached we calculate the planning cost using the present labelling and display the results as such.

# Approaches considered

-   Initially I tried uing the sum-product apporach but the execution time was very high tdue to tha large amount of multiplications and summations to be performed.

-   Additionally, the Over and Underflow of the messages was very prominant and thus I chose to go with the min-sum approach. Th global minimum cost expected for the sample problem is 4500 with the following labelling,
D D D D D
D D D D D
D D R D D
D D D D D
D D D D D

however, currently the result bwing produced is suboptimal due to dicrepancy in normalization, message storing and access, we are trying to work on it and hoepfully by submission we are able to fix it.

# Difficulties

-   The main difficulty faced for this part is the apporach for Message handling, It is exceptionally difficult to trace back the message origin to a node as it already has 4 matrices for 4 directions with 2 sub values for each label.

-   While writing the message we need to write onto the destination node while simultaneously considering the the neighbouring inputs for the i node excluding j node, for which we need to access i node's stored message which is counter intuitive to the approach by looking at the neighbour node.

-   Similar issue is observed during the belief propagation as well leading to optimization issues due to such discrepancies.

# Results

The latest result from the code is:

![Result](CurRes.png)

-   The accuracy is still far from the global minima however the approach and modularization allows us to debug the code relatively more straightforward manner and we are striving to update these until the last minute to be able to achieve the global minimum.


# Part 3: Inferring Depth from Stereo

### Objectives:
- Given two images, we can find the depth of the image using Naive Stereo and Markov Random Field algorithm. 
- Produces 3D Stereoscopic Image

### Disparity Costs:
Since we are assuming that the images have been already rectified, we can match pixels from the left image to the right image's pixels which is located within the scanline in the same row as left pixels that we are looking. Ideally, we check the whole pixels to attain which pixel that has the best match. However, it is computationally expensive esepecially if we have high resolution images. Therefore, the span of area that the model will search is limitized by MAX_DISPARITY. Let's say if we set MAX_DISPARITY is equal to 10, it means that we search the area start from the same coordiante as the left pixel, scan 9 pixels to the left, and other 9 pixels to the right. Every slide, the disparity costs is calculated and stored in a matrix with a shape of (H,W,D) where H is the height of image, W is the width, and D is the number of location within the scanline in the right image that we checked. In this case, D is equal to MAX_DISPARITY * 2 - 1.

Before the learning process, we normalized the image by dividing each pixels with 255 such that every pixel value has range [0,1]. To reduce the noise, we use a window that will be convolved along the image. Larger window will give smoother result but is unable to capture the local information well. After some trials, we found that window of size 5x5 gives better result. The disparity costs is calculated based on the sum of squared difference between window in the left image and window in the right image. Based on the experiment, truncated quadratic function helps to create better output. We use the alpha value of 0.53 to control the maximum value of qudratic function.

### Naive Stereo:
This algorithm only utilize the disparity costs to find the index of 'D' where the minimum cost is located for every pixels. This index later will be converted into [0, 255] value as an image. To get the index, we have to make a slight adjustment. The principle of depth inference is to figure out the depth based on the shift distance of an object. If the object is close to the camera, it will have greater shift compared to the object that is located far from the camera. We utilize this concept for indexing when the disparity costs is stored in the matrix. The furthest shift should be located at the beginning (index 0) for left direction and at the end of matrix (index MAX_DISPARITY * 2 - 1) for the right direction. The origin (0 offset) is located at the index of MAX_DISPARITY - 1. After finding the index of minimum costs, we substract that index with the origin index (MAX_DISPARITY - 1) and take the absolute value.

### Problem & Decision:
From the given data, the training images do not require rectification as those left and right images are already taken in the same plane. This means that the camera is pointing the same direction and it only slides from the left to the right. Because of this, the object in the right image will be shifted to the left compared to the left image. Therefore, we can absoultely certain to tell the program that it only needs to check on the left direction starting from the same coordinate,from the left image that we want to check, with the number of slide of MAX_DISPARITY - 1. We have tried this approach and it gives better and faster result compared to searching in both direction (left and right). It is better because we force the model to only check to the left and ignore every pixels on the right and it is faster because it requires less iteration. However, it gives problem as this approach is not robust. If the 2 images are taken in different orientation, we need to rectify the image. This will cause local information can be shifted either to the left or to the right. As a final decision, we choose to check both direction for more robust model but with same trade-offs (slower and poorer result).
