**Lattice - Boltzmann method in fluid dynamics - Von Karman vortex street**

...description...

# How to run:

# To do:
- clean up numba implementation, test it out 
- try out taichi instead of numba
- add support for longer simulations (different way of storing history)
- generate requirements.txt
- add better support for different geometries of terrain
- general code review, look for optimizations and investigate better automatic viscosity and relaxation time derivations (or rather test their correctness)
- compare with more robust libraries - features inspiration
- try out jax implementation
- describe different flows (parameter list)
- perhaps add unit tests
- add graphs that monitor average velocity of flow etc
- diagnose and eliminate overflow bugs at high v and/or Re
- periodic/opposite BC easy switching

# Bibliography:
- https://www.fuw.edu.pl/~tszawello/cmpp2024/
- http://fab.cba.mit.edu/classes/864.20/people/filippos/links/final-projects/lbm/index.html
- https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lattice_boltzmann_method_python_jax.py
