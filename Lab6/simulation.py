import os, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
from datetime import datetime
from tqdm import tqdm
from numpy.linalg import eig


class SimulationDensityOfStates():
    def __init__(self,
                 N=6,
                 nsamples=20000,
                 symmetry='goe'):
        self.N = N
        self.nsamples = nsamples
        self.symmetry = symmetry
        
        self.samples = np.zeros((self.nsamples, self.N, self.N))        
        self.date = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.init_symmetry()
    
    def gaussian_orthogonal_ensemble(self):
        self.beta = 1
        self.R = np.sqrt(2*self.N)
        self.u = 0  # mean
        self.sigma = 1  # standard deviation
        self.title = 'Gaussian Orthogonal Ensemble (GOE) - density of states'
        
        def sample_hamiltionian():
            """
            randn generates an array of shape (d0, d1, ..., dn), 
            filled with random floats sampled from a univariate 
            “normal” (Gaussian) distribution of mean 0 and variance 1.
            """
            h = np.random.randn(self.N, self.N) 
            return (h+h.T)/2
        
        def P(s):
            return np.pi/2 * s * np.exp((-np.pi/4) * s**2)
        
        self.p_func_goe = np.vectorize(P)
        
        for i in range(self.nsamples):
            self.samples[i] = sample_hamiltionian()

    
    def gaussian_unitary_ensemble(self):
        self.beta = 2
        self.R = np.sqrt(2*self.N)
        self.u = 0  # mean
        self.sigma = 1  # standard deviation
        self.title = 'Gaussian Orthogonal Ensemble (GOE) - density of states'
        
        def sample_hamiltionian():
            """
            randn generates an array of shape (d0, d1, ..., dn), 
            filled with random floats sampled from a univariate 
            “normal” (Gaussian) distribution of mean 0 and variance 1.
            """
            hx = np.random.randn(self.N, self.N) 
            hy = np.random.randn(self.N, self.N) 
            h = hx + 1j * hy
            h = h / np.sqrt(2)
            return (h+np.conjugate(h.T))* 100
        
        def P(s):
            return 32/np.power(np.pi, 2) * np.power(s, 2) * np.exp((-4/np.pi) * s**2)
        
        self.p_func_gue = np.vectorize(P)
        
        for i in range(self.nsamples):
            self.samples[i] = sample_hamiltionian()
        
    def init_symmetry(self):
        if self.symmetry == 'goe':
            self.gaussian_orthogonal_ensemble()
        elif self.symmetry == 'gue':
            self.gaussian_unitary_ensemble()
        
    def get_analytical_wigner(self, E):
        return 2/(np.pi * np.power(self.R, 2)) * np.sqrt(self.R**2 - E**2)
    
    def get_eigenvalues(self):
        self.eigenvalues, _ = eig(self.samples)
        assert self.eigenvalues.shape == (self.nsamples, self.N)
        
    def save_density_histogram(self):
        output_file = f'{self.date}_density_histogram.png'
        file_path = os.path.join('results', output_file)
        
        n, bins, _ = plt.hist(self.eigenvalues.flatten(),
                              50, 
                              density=True, 
                              facecolor='cyan', 
                              alpha=0.75)
        #print(bins)
        plt.plot(bins, 
                 self.get_analytical_wigner(bins), 
                 'r-', 
                 linewidth=2,
                 label='Analytical Wigner')
        
        plt.xlabel('Eigenvalues')
        plt.ylabel('Density')
        plt.title(self.title)
        plt.legend()
        plt.savefig(file_path)
        
    def save_energy_spacings_histogram(self):
        output_file = f'{self.date}_energy_spacings_histogram.png'
        file_path = os.path.join('results', output_file)
        
        nbins = len(self.spacings_goe) // self.N // 8
        n, bins, _ = plt.hist(self.spacings_goe,
                              nbins,
                              density=True, 
                              facecolor='cyan', 
                              alpha=1,
                              label='GOE')
        plt.plot(bins, 
                 self.p_func_goe(bins), 
                 'c-', 
                 linewidth=2, 
                 label='Analytical Wigner GOE')
        
        n, bins, __ = plt.hist(self.spacings_gue,
                                nbins,
                                density=True, 
                                facecolor='green', 
                                alpha=0.5,
                                label='GUE')
        plt.plot(bins, 
                 self.p_func_gue(bins), 
                 'g-', 
                 linewidth=2, 
                 label='Analytical Wigner GUE')
        
        
        plt.xlabel('Energy Spacings')
        plt.ylabel('Density')
        plt.legend()
        plt.title(self.title)
        plt.savefig(file_path)
    
    def make_density_histogram(self):
        self.get_eigenvalues()
        self.save_histogram()
        
    def make_energy_spacings_histogram(self):
        l = (self.N//8) * 3
        h = (self.N//8) * 5
        self.symmetry = 'goe'
        self.init_symmetry()
        self.get_eigenvalues()
        self.eigenvalues_goe = np.sort(self.eigenvalues)[:, l:h]
        self.eigenvalues_goe = self.eigenvalues_goe#.flatten()
        self.eigenvalues_goe = self.eigenvalues_goe

        self.spacings_goe = np.diff(self.eigenvalues_goe)
        self.spacings_goe = self.spacings_goe / np.mean(self.spacings_goe)
        self.spacings_goe = self.spacings_goe.flatten()

        self.samples = np.zeros((self.nsamples, self.N, self.N), dtype=np.complex128)
        
        self.symmetry = 'gue'
        self.init_symmetry()
        self.get_eigenvalues()
        self.eigenvalues_gue = np.sort(self.eigenvalues)[:, l:h]
        self.eigenvalues_gue = self.eigenvalues_gue
        self.spacings_gue = np.diff(self.eigenvalues_gue)
        self.spacings_gue = self.spacings_gue / np.mean(self.spacings_gue)
        self.spacings_gue = self.spacings_gue.flatten()
        self.save_energy_spacings_histogram()
    
    def run(self):
        #self.make_density_histogram()
        self.make_energy_spacings_histogram()
    