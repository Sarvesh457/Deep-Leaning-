#-------------------------------------------------
# Program : Artifical Neuron with Relu Actvation
# Authot : Sarvesh Atul Mahajan
#-------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------
#   Step 1 : Activation Function (ReLU)
#-----------------------------------------
# ReLU = max(0,z)
# If z is positive -> output z
# If z is negtive -> output 0

def relu(z):
    return max(0,z)

#-----------------------------------------
# Step 2 : Neuron Forward pass function
#-----------------------------------------
# This function simulates a single artificial neuron
# It performs:
# 1. Input * Weigth mulitplication
# 2. Summation + Bias
# 3. Activation (ReLU)

def Neuron_forward(inputs,weights,bias):

    print("\n---- NEURON CALCULATION START ----\n")

    # Dispaly input and weigths
    print("Inputs (x)  : ",inputs)
    print("Weigths (w) : ",weights)
    print("Bias (b)    : ",bias)

    #-----------------------------------------
    # Step 2 : Neuron Forward pass function
    # Formula : z = (x1*w1 + x2*w2 + .. + xn*wn) + bias
    #-----------------------------------------

    z = sum(w*x for w,x in zip(weights,inputs))+bias

    print("\nStep 1 : Weights Sum Calculation")
    print("z = w.x+b =",z)

    #-----------------------------------------
    # Step 2.2 : Activation Faunction
    #-----------------------------------------
    
    y_hat = relu(z)

    print("\nStep : Activation Function Applied")
    print("Activation Function : ReLU")
    print("Output (y) = y_hat")

    print("\n----- NEURON CALCULATION END -----\n")

    return z,y_hat

#-----------------------------------------
# Step 5 : Plot ReLU Function
#-----------------------------------------

# This help to VISUALIZE how ReLU behaves

def plot_relu():

    #Generate range of values for z
    z_values = np.linspace(-10, 10, 200)

    #Apply ReLU on all values
    relu_values = np.maximum(0, z_values)

    # Plot grap
    plt.figure(figsize=(8, 5))
    plt.plot(z_values, relu_values, label="ReLU Function", linewidth=2, color="green")

    # Axes lines
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.axvline(x=0, color="gray", linestyle="--")

    #Labels and title
    plt.title("ReLU Activation Function", fontsize=16)
    plt.xlabel("Input (z)", fontsize=14)
    plt.ylabel("Output", fontsize=14)

    #Grid and legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Show graph
    plt.show()


#-----------------------------------------
# Step 4 : Main function
#-----------------------------------------

def main():
    print("\n====== NEURON DEMO ======\n")

    # Ex input (fetures)
    inputs = [0.6,2.0,3.0]

    # Corresponding Weights
    weights = [0.6,0.4,-0.2]

    # Bias value
    bias = 0.5

    # Perform forward propagtion
    z,y_hat = Neuron_forward(inputs,weights,bias)

    # lot ReLU graph
    plot_relu()

#-----------------------------------------
# Starter 
#-----------------------------------------

if __name__ == "__main__":
    main()
