
# PREAMBLE:

import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

def generalized_correlation_coefficient(trajectory_data,bins):
        """ Calculates the Pearson Correlation Matrix for node pairs
        Usage: node_gen_corr_coefficient, node_mutual_information, node_average = generalized_correlation_coefficient(trajectory_data,bins)
        Arguments:
        trajectory_data: multidimensional numpy array; first index (rows) correspond to timestep, second index correspond to positions of each node; 
        bins: integer; number of bins to be used for all histograms;

        Returns:
        node_gen_corr_coefficient: a nNodes x nNodes square matrix (numpy array) filled with the Generalized Correlation Coefficient for all node pairs
        node_mutual_information: a nNodes x nNodes square matrix (numpy array) filled with the Mutual Information for all node pairs
        node_average: one dimensional numpy array containing the averages of the data

        """

        two_thirds = 2./3.
        # ----------------------------------------
        # CALCULATING THE AVERAGE OF TRAJECTORY DATA
        # ----------------------------------------
        nSteps = len(trajectory_data)
        nSteps_range = range(nSteps)
        nNodes = len(trajectory_data[0])
        nNodes_range = range(nNodes)
        for ts in nSteps_range:
                # removing center of geometry translational motion
                center_of_geometry = np.mean(trajectory_data[ts])
                trajectory_data[ts] -= center_of_geometry
                # no rotations to worry about...

        node_average = np.sum(trajectory_data,axis=0)/nSteps 
       
        # ----------------------------------------
        # PREPARE NUMPY ARRAYS
        # ----------------------------------------
        node_displacement = np.zeros((nSteps,nNodes),dtype=np.float64)
        node_mutual_information = np.zeros((nNodes,nNodes),dtype=np.float64)
        node_gen_corr_coefficient = np.zeros((nNodes,nNodes),dtype=np.float64)

        # ----------------------------------------
        # CALCULATING THE NODE DISPLACEMENTS FROM AVERAGE POSITIONS 
        # ----------------------------------------
        for ts in nSteps_range:
                for i in nNodes_range:
                        node_displacement[ts,i] = trajectory_data[ts,i] - node_average[i]   # distance for one dimensional cases

        # ----------------------------------------
        # CALCULATING THE MUTUAL INFORMATION OF NODE PAIRS
        # ----------------------------------------
        # code taken from https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
        for i in nNodes_range:
                for j in nNodes_range[i+1:]:
                        c_xy = np.histogram2d(node_displacement[:,i],node_displacement[:,j],bins)[0]
                        node_mutual_information[i,j] = mutual_info_score(None,None,contingency=c_xy)
                        node_gen_corr_coefficient[i,j] = np.sqrt(1- np.exp(-two_thirds*node_mutual_information[i,j]))**-1
                        
        # ----------------------------------------
        # OUTPUT OF AVERAGE, NODE DISPLACEMENTS, AND MUTUAL INFORMATION ARRAYS
        # ----------------------------------------
        np.savetxt('node_positional_average.dat',node_average)
        np.savetxt('node_positional_displacements.dat',node_displacement)
        np.savetxt('node_positional_mutual_infromation.dat',node_mutual_information)
        np.savetxt('node_positional_generalized_correlation_coefficient.dat',node_gen_corr_coefficient)

        # ----------------------------------------
        # PLOTTING NODE DISPLACEMENT
        # ----------------------------------------
        for i in nNodes_range:
                plt.plot(nSteps_range,node_displacement[:,i],label='Particle %d'%(i))
        plt.legend()
        plt.xlabel('Timestep',size=14)
        plt.ylabel(r'Displacement Away From Average Positions ($\AA)',size=14)
        plt.tight_layout()
        plt.savefig('node_positional_displacements.png',dpi=600,transparent=True)
        plt.close()

        # ----------------------------------------
        # PLOTTING MUTUAL INFORMATION
        # ----------------------------------------
        fig, ax = plt.subplots()
        temp = plt.pcolormesh(nNodes_range,nNodes_range,node_mutual_information,cmap='Blues')
        cb1 = plt.colorbar()
        cb1.set_label('Node-Displacement Pairwise Mutual Information')
        
        xlabels = [str(int(x)) for x in temp.axes.get_xticks()[:]]
        ylabels = [str(int(y)) for y in temp.axes.get_yticks()[:]]
        temp.axes.set_xticks(temp.axes.get_xticks(minor=True)[:]+0.5,minor=True)
        temp.axes.set_xticks(temp.axes.get_xticks()[:]+0.5)
        temp.axes.set_yticks(temp.axes.get_yticks(minor=True)[:]+0.5,minor=True)
        temp.axes.set_yticks(temp.axes.get_yticks()[:]+0.5)
        temp.axes.set_xticklabels(xlabels)
        temp.axes.set_yticklabels(ylabels)

        plt.xlim((-0.5,nNodes+0.5))
        plt.ylim((-0.5,nNodes+0.5))
        plt.xlabel('Node Index',size=14)
        plt.ylabel('Node Index',size=14)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('node_positional_mutual_information.png',dpi=600,transparent=True)
        plt.close()
        
        # ----------------------------------------
        # PLOTTING GENERALIZED CORRELATION COEFFICIENT
        # ----------------------------------------
        fig, ax = plt.subplots()
        temp = plt.pcolormesh(nNodes_range,nNodes_range,node_gen_corr_coefficient,cmap='Blues')
        cb1 = plt.colorbar()
        cb1.set_label('Generalized Correlation Coefficient')
        
        xlabels = [str(int(x)) for x in temp.axes.get_xticks()[:]]
        ylabels = [str(int(y)) for y in temp.axes.get_yticks()[:]]
        temp.axes.set_xticks(temp.axes.get_xticks(minor=True)[:]+0.5,minor=True)
        temp.axes.set_xticks(temp.axes.get_xticks()[:]+0.5)
        temp.axes.set_yticks(temp.axes.get_yticks(minor=True)[:]+0.5,minor=True)
        temp.axes.set_yticks(temp.axes.get_yticks()[:]+0.5)
        temp.axes.set_xticklabels(xlabels)
        temp.axes.set_yticklabels(ylabels)

        plt.xlim((-0.5,nNodes+0.5))
        plt.ylim((-0.5,nNodes+0.5))
        plt.xlabel('Node Index',size=14)
        plt.ylabel('Node Index',size=14)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('node_positional_generalized_correlation_coefficient.png',dpi=600,transparent=True)
        plt.close()
        
        return node_gen_corr_coefficient, node_mutual_information, node_average

