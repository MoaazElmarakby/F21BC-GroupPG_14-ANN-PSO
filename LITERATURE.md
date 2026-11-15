# Literature map

Where each default or design choice comes from (decision numbers refer to docs/design.md section 6).
Entries marked `TODO(verify)` were cited from memory or from the design doc and need their bibliographic
details checking before they go into the report.

| Used for | Source |
|---|---|
| Algorithm 39 (PSO), informants, parameter roles alpha/beta/gamma/delta/epsilon | Sean Luke, *Essentials of Metaheuristics*, 2nd ed., 2013 (Algorithm 39 and the text around it; the excerpt supplied with the project was used to check the line numbers in the code comments) |
| Origin of PSO | James Kennedy and Russell Eberhart, 1995, "Particle swarm optimization", Proc. IEEE International Conference on Neural Networks, pp. 1942-1948 (as cited in Luke's footnote 37) |
| Swarm intelligence background | Kennedy, Eberhart and Shi, 2001, *Swarm Intelligence*, Morgan Kaufmann (as cited in Luke's footnote 37) |
| D3: alpha=0.7298, beta=gamma=1.4962 (constriction-equivalent) | Clerc and Kennedy, 2002, on the stability/convergence of the particle swarm. TODO(verify) title, venue, pages |
| D4: K=3 informants (SPSO default); sweep 2-8 and global | Standard PSO descriptions (Clerc and colleagues). TODO(verify) exact source |
| D5: bounded weight search helps NN training; +-2 | Rakitianskaia and Engelbrecht, 2014; Garro and Vazquez, 2015. TODO(verify) titles and venues |
| Dataset | Banknote Authentication, UCI Machine Learning Repository |
