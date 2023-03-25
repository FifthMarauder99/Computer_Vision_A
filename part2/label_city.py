# Part 2: Understanding Markov Random Fields Solution script

import sys
import numpy as np

# Arranging the main input read

n = int(sys.argv[1])
Bribes_R_name = sys.argv[2]
Bribes_D_name = sys.argv[3]


R = np.loadtxt(Bribes_R_name)
D = np.loadtxt(Bribes_D_name)

