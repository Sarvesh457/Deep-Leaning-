# for numerical operations
import numpy as np

#------------------------------------------------
#   Step 1 : Define input Fertures
#------------------------------------------------
# These are the input coming to the neuron (x1,x2,x3)
# Ex : clould be marks ,pixel values,or any fetures

inputs = np.array([2.0,3.0,4.0])

#------------------------------------------------
#   Step 3 : Define Weigths
#------------------------------------------------
# Each input has a corresponding weight (w1,w2,w3)
# Weight represent importance of each input

weights = np.array([0.5,0.3,0.2])

#------------------------------------------------
#   Step 3 : Define Bais
#------------------------------------------------
#  Bias is a additional parameter that helps shift the output
# It allowes the model to fit better

bais = 1.0

#------------------------------------------------
# Step 4 : Calculate Weighted Sum (Z)
#------------------------------------------------
# Formula : 
# Z = (x1*w1 + x2*w2 + x3*w3 ) + bias

weighted_sum = np.dot(inputs,weights)+bais

# Manual Calculation :
# Z = ((2.0*0.5)+(3.0*0.3)+(4.0*0.2))+1 
# Z = 1.0 + 0.9 + 0.8 + 1.0 = 3.7

#------------------------------------------------
# Step 5 : Activation Function (ReLU)
#------------------------------------------------
# ReLU (Rectified Liner Unit)
# If value > 0 -> return value
# if value <= o -> return 0

def relu(x):
    return max(0,x)

#------------------------------------------------
# Step 5 : Final Output
#------------------------------------------------
# Pass the weighted sum though activation function

output = relu(weighted_sum)

#------------------------------------------------
# Step 5 : Diaplay Results
#------------------------------------------------

print("Input           : ",inputs)
print("Weigths         : ",weights)
print("Bias            : ",bais)
print("Weigthed Sum (Z): ",weighted_sum)
print("Final Output    : ",output)

# Input           :  [2. 3. 4.]
# Weigths         :  [0.5 0.3 0.2]
# Bias            :  1.0
# Weigthed Sum (Z):  3.7
# Final Output    :  3.7
