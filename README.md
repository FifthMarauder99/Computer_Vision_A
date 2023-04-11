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
