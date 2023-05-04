# -*- coding: utf-8 -*-

import bitstring
import random
import cupy as np
import pickle

# NEURAL NETWORK CONFIGURATION

# Number of nodes in each of the columns
COL_NODES = [808,32,16,1]

# Number of nodes in output column
OUT_NODES = 1

# END CONFIGURATION


class NeuralNet:
    """
    The class which represents operations for the neural network

    ...

    Attributes
    ----------


    Methods
    -------

    """
    
    node_list = []
    biases = []
    
    #def __init__(self):
        
       
        
        
    def loadinputbits(self, inbits: bitstring.BitArray):
        
        self.node_values = []
        self.z_values = []
        
        if(len(inbits)!=COL_NODES[0]):
            raise ValueError("Expected %d input bits, received %d" % (COL_NODES[0], len(inbits)))
            
        self.node_values.append(np.array(inbits))
        
        
    def loadeval(self, position_eval:float = 0):
        
        self.position_eval = position_eval
        if(self.position_eval > 10):
            self.position_eval = 10
        if(self.position_eval < -10):
            self.position_eval = 10
            
    def loadweights(self, filename:str):
        with open(filename, 'rb') as fp:
            self.node_list = pickle.load(fp)
                
    def loadbiases(self, filename:str):
        with open(filename, 'rb') as fp:
            self.biases = pickle.load(fp)
        
    def saveweights(self, filename:str):
        with open(filename, 'wb') as fp:
            pickle.dump(self.node_list, fp)
            print("saved to %s" % (filename))
        
    def savebiases(self, filename:str):
        with open(filename, 'wb') as fp:
            pickle.dump(self.biases, fp)
            print("saved to %s" % (filename))
        
    def processrow(self, row: list):
                
        self.loadinputbits(bitstring.BitArray(row[2]))
        
        self.loadeval(row[3])        
        
            
    def getintputbits(self):
        return self.node_values[0].bin
    
    def geteval(self):
        return self.position_eval
        
    def initializeweights(self, zeroinit:bool = False):
        
        
        randmin = -1
        randmax = 1
        
        # A list of numpy arrays representing the biases of each node
        blist = []
                
        # A temporary variable that will be used to store a weights matrix, in numpy array form
        mlist = []
        
        # A list of numpy arrays representing the node values for every column
        nlist = []
             
        
        
        # Create a new numpy object that will store the weights matrix and biases array for the interactions
        # between the input layer and first hidden layer

        # Add these numpy objects to lists for storage
        
        ### FOR MEMORY SAVINGS ###
        # Load only one weights and biases matrix at a time from disk
        
        ### INPUT LAYER <-----> FIRST HIDDEN LAYER ###
        
        # mlist.append(np.empty((COL_NODES[0], IN_NODES)))
        # blist.append(np.empty(COL_NODES[0]))
        
        # print(np.shape(mlist[0]))
        # print(np.shape(blist[0]))

        
        # for i in range(COL_NODES[0]):
        #     blist[0][i] = self.getrandom()
        #     for j in range(IN_NODES):
        #         mlist[0][i][j] = self.getrandom()
                
                
        
        # ### HIDDEN LAYER <-----> HIDDEN LAYER ###
        
        # for i in range(1, NUMCOLS):
            
        #     mlist.append(np.empty((COL_NODES[i], COL_NODES[i-1])))
        #     blist.append(np.empty(COL_NODES[i]))
            
        #     for j in range(COL_NODES[i]):
        #         blist[i][j] = self.getrandom()
        #         for k in range(COL_NODES[i-1]):
        #             mlist[i][j][k] = self.getrandom()
                    
        # ### LAST HIDDEN LAYER <-----> OUTPUT LAYER ###
        
        # mlist.append(np.empty((OUT_NODES, COL_NODES[NUMCOLS-1])))
        # blist.append(np.empty(OUT_NODES))
        
        # for i in range(OUT_NODES):
        #     blist[NUMCOLS][i] = self.getrandom()
        #     for j in range(COL_NODES[NUMCOLS-1]):
        #         mlist[NUMCOLS][i][j] = self.getrandom()
        
        # Initialize all weights and bises with random values
        
        for i in range(len(COL_NODES)-1):
            mlist.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
            blist.append(np.empty(COL_NODES[i+1]))
            
            for j in range(COL_NODES[i+1]):
                blist[i][j] = self.getrandom()
                for k in range(COL_NODES[i]):
                    mlist[i][j][k] = self.getrandom()
        
        
        self.node_list = mlist       
        self.biases = blist
                
        
                
    def getrandom(self, randmin:int = -1, randmax:int = 1, zeroinit:bool = False):
        if(zeroinit):
            return 0
        else:
            return random.random() * (randmax - randmin) + randmin
    
    def printweights(self):
        
        # print("Hidden column 1:")
        # for i in range(COL_NODES[0]):
        #     for j in range(IN_NODES):
        #         print("HC%d   %f   IN%d" % (i+1, self.node_list[0][i][j], j+1))
        #     print("HC%d bias: %f" % (i+1, self.biases[0][i]))
        
        # for i in range(1, NUMCOLS):
        #     print("Hidden column %d" % (i+1))
        #     for j in range(COL_NODES[i]):
        #         for k in range(COL_NODES[i-1]):
        #             print("HC%d   %f   HC%d" % (j+1, self.node_list[i][j][k], k+1))
        #         print("HD%d bias: %f" % (j+1, self.biases[0][j]))
        
        # print("Output Column:")            
        # for i in range(OUT_NODES):
        #     for j in range(COL_NODES[NUMCOLS-1]):
        #         print("OC%d   %f   HC%d" % (i+1, self.node_list[NUMCOLS][i][j], j+1))
        #     print("OC%d bias: %f" % (i+1, self.biases[NUMCOLS][i]))
        
        for i in range(0, len(COL_NODES)-1):
            print("Column %d" % (i))
            for j in range(COL_NODES[i+1]):
                for k in range(COL_NODES[i]):
                    print("Node%d   %f   Node%d" % (j+1, self.node_list[i][j][k], k+1))
    
    # Processes the calculations of the columns, weights, biases of the network
    # Returns the output vector of the network    
    
    def feedforward(self, input_vector:np.ndarray, weights:np.ndarray, biases:np.ndarray):
        
        for w, b in zip(weights, biases):
            z = self.getz(w, input_vector, b)
            input_vector = self.sigmafy(z)
            self.z_values.append(z)
            self.node_values.append(input_vector)
        
        #print(self.node_values)
        return input_vector
    
    def expandoutput(self, outputnode:np.array):
        
        mineval = -10
        maxeval = 10
        
        
        return outputnode[0] * (maxeval - mineval) + mineval
    
    def compressoutput(self, outputnode:np.array):
        mineval = -10
        maxeval = 10
        
        return (outputnode - mineval) / (maxeval - mineval)
        
    def calcoutput(self):        
              
        self.output_value = self.expandoutput(self.feedforward(self.node_values[0], self.node_list, self.biases))
        
                    
    def sigmafy(self, x:np.float64):
        
        return 1 / (1 + (1/np.exp(x)))
        
    def dsigmafy(self, x:np.float64):
        tmp = self.sigmafy(x)        
        return (tmp * (1 - tmp))
    
    def dCda(self, a:float, y:float):
        
        return 2 * (a - self.compressoutput(y))
    
    def dadz(self, z:float):
        
        return self.dsigmafy(z)
    
    def dzdw(self, w:float):
        
        return w
    
    def dCdw(self, next_node:float, last_node:float, y:float, w:float, b:float):
        return self.dzdw(last_node) * self.dsigmafy(next_node*w+b)*self.dCda(next_node, y)
    
    def dCdb(self, next_node:float, last_node:float, y:float, w:float, b:float):
        return self.dsigmafy(next_node*w+b)*self.dCda(next_node, y) 
    
    def getz(self, weight:np.ndarray, node:np.ndarray, bias:np.ndarray):
        return np.dot(weight, node) + bias
    
    def backpropagate2(self):
        
        gradients = []
        bias_gradients = []
        delta = 0
        
        dCda_values = []
        
        
        for i in range(len(COL_NODES)-1):
            gradients.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
            dCda_values.append(np.empty((COL_NODES[i+1])))
            bias_gradients.append(np.empty(COL_NODES[i+1]))
        for i in range(len(self.node_list)-1, -1, -1):
            for j in range(len(self.node_list[i])):
                if(i == len(self.node_list)-1):
                    dCda_values[i][j] = self.dCda(self.node_values[i+1][j], self.position_eval)
                else:
                    delta = 0
                    for jj in range(len(gradients[i+1])):     
                        delta += dCda_values[i+1][jj] * self.dadz(self.z_values[i+1][jj]) * self.node_list[i+1][jj][j]
                    dCda_values[i][j] = delta
                bias_gradients[i][j] =  dCda_values[i][j] * self.dadz(self.z_values[i][j])  
                for k in range(len(self.node_list[i][j])):
                    gradients[i][j][k] = dCda_values[i][j] * self.dadz(self.z_values[i][j]) * self.node_values[i][k]    
            self.node_list[i] -= gradients[i]
            self.biases[i] -= bias_gradients[i]
            
    def backpropagate(self):
        
        gradients = []
        bias_gradients = []
        delta = 0
        
        dCda_values = []
        
        
        for i in range(len(COL_NODES)-1):
            gradients.append(np.empty((COL_NODES[i+1], COL_NODES[i])))
            dCda_values.append(np.empty((COL_NODES[i+1])))
            bias_gradients.append(np.empty(COL_NODES[i+1]))
        for i in range(len(self.node_list)-1, -1, -1):
            if(i == len(self.node_list)-1):
                dCda_values[i] = self.dCda(self.node_values[i+1], np.array(self.position_eval))
            else:
                dCda_values[i] = np.dot(np.transpose(self.node_list[i+1]), np.multiply(dCda_values[i+1], self.dadz(self.z_values[i+1])))
            bias_gradients[i] =  np.multiply(dCda_values[i] , self.dadz(self.z_values[i])) 
  
            gradients[i] = np.dot(np.multiply(dCda_values[i] , self.dadz(self.z_values[i])).reshape(len(self.z_values[i]),1), self.node_values[i].reshape(1,len(self.node_values[i])))
            self.node_list[i] = np.subtract(self.node_list[i] , gradients[i])
            self.biases[i] = np.subtract(self.biases[i] , bias_gradients[i])
        
        
        
        
    def getoutput(self):
        return self.output_value