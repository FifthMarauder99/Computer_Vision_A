import random
import sys
import math
from PIL import Image
import numpy as np
MAX_DISPARITY = 10 # Set this to the maximum disparity in the image pairs you'll use

def mrf_stereo(img1, img2, disp_costs):
    # this placeholder just returns a random disparity map
    result = np.zeros((img1.shape[0], img1.shape[1]))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            result[i,j] = random.randint(0, MAX_DISPARITY)
    return result

# This function should compute the function D() in the assignment
def disparity_costs(img1, img2):
    # this placeholder just returns a random cost map
    result = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for d in range(MAX_DISPARITY):
                result[i,j,d] = random.randint(0, 255)
    return result

# This function finds the minimum cost at each pixel
def naive_stereo(img1, img2, disp_costs):
    return np.argmin(disp_costs, axis=2)
                      
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception("usage: " + sys.argv[0] + " image_file1 image_file2 [gt_file]")
    input_filename1, input_filename2 = sys.argv[1], sys.argv[2]

    # read in images and gt
    image1 = np.array(Image.open(input_filename1))
    image2 = np.array(Image.open(input_filename2))
    
    gt = None
    if len(sys.argv) == 4:
        gt = np.array(Image.open(sys.argv[3]))[:,:,0]

        # gt maps are scaled by a factor of 3, undo this...
        gt = gt / 3.0

    # compute the disparity costs (function D_2())
    disp_costs = disparity_costs(image1, image2)
    
    # do stereo using naive technique
    disp1 = naive_stereo(image1, image2, disp_costs)
    Image.fromarray(disp1.astype(np.uint8)).save("output-naive.png")
        
    # do stereo using mrf
    disp3 = mrf_stereo(image1, image2, disp_costs)
    Image.fromarray(disp3.astype(np.uint8)).save("output-mrf.png")

    # Measure error with respect to ground truth, if we have it...
    if gt is not None:
        err = np.sum((disp1- gt)**2)/gt.shape[0]/gt.shape[1]
        print("Naive stereo technique mean error = " + str(err))

        err = np.sum((disp3- gt)**2)/gt.shape[0]/gt.shape[1]
        print("MRF stereo technique mean error = " + str(err))
        





                                                                                                                                                                                                           
