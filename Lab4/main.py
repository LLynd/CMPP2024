import numpy as np

from simulation import LBMSimulation

#if __name__ == 'main':

""" sim = LBMSimulation(steps=15000, n_frame_save=5, Re=220, velocity=.04, 
                     dtype=np.float64) """
""" sim = LBMSimulation(steps=5000, n_frame_save=5, Re=220, velocity=.1, 
                     dtype=np.float64) """
""" sim = LBMSimulation(steps=15000, n_frame_save=5, Re=440, velocity=.1, 
                    dtype=np.float64) """
""" sim = LBMSimulation(steps=5000, n_frame_save=5, Re=180, velocity=.1, 
                     dtype=np.float64) """
""" sim = LBMSimulation(steps=15000, n_frame_save=5, Re=660, velocity=.1, 
                     dtype=np.float32) """
""" sim = LBMSimulation(steps=5000, n_frame_save=5, Re=1, velocity=.1, 
                     dtype=np.float64) """
sim = LBMSimulation(steps=5000, n_frame_save=5, Re=320, velocity=.1, 
                     dtype=np.float64)

sim.run()

'''
v = 0
for i in [100, 200, 300, 400, 500]:
    v += 0.05
    sim = LBMSimulation(steps=5000, n_frame_save=5, Re=i, velocity=v, 
                        dtype=np.float64)
    sim.run()
'''
