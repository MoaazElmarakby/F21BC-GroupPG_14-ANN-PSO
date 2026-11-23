# Literature map

Where each default or design choice comes from (decision numbers refer to docs/design.md section 6).

**Every row marked `TODO(verify)` was written from memory and has NOT been checked against the source.**
Read each paper yourselves, correct the details, and only then cite it in the report: the brief says the
references must exist, must be used in the text, and that you should motivate value ranges and compare
with published results with them.

| Used for | Source |
|---|---|
| Algorithm 39 (PSO), informants, parameter roles alpha/beta/gamma/delta/epsilon | Sean Luke, *Essentials of Metaheuristics*, 2nd ed., 2013 (Algorithm 39 and the text around it; the excerpt supplied with the project was used to check the line numbers in the code comments) |
| Origin of PSO | James Kennedy and Russell Eberhart, 1995, "Particle swarm optimization", Proc. IEEE International Conference on Neural Networks, pp. 1942-1948 (as cited in Luke's footnote 37) |
| Swarm intelligence background | Kennedy, Eberhart and Shi, 2001, *Swarm Intelligence*, Morgan Kaufmann (as cited in Luke's footnote 37) |
| D3: alpha=0.7298, beta=gamma=1.4962 (constriction-equivalent) | Clerc and Kennedy, 2002, "The particle swarm: explosion, stability, and convergence in a multidimensional complex space", IEEE Transactions on Evolutionary Computation 6(1), 58-73. TODO(verify) |
| D4: K=3 informants (SPSO default); Q4 informant study | Standard PSO descriptions (Clerc and colleagues). TODO(verify) exact source. For neighbourhood structure: Kennedy and Mendes, 2002, "Population structure and particle swarm performance", Proc. CEC. TODO(verify) |
| D5: bounded weight search; +-2 | Rakitianskaia and Engelbrecht, 2014 (training high-dimensional neural networks with PSO); Garro and Vazquez, 2015, "Designing artificial neural networks using particle swarm optimization algorithms", Computational Intelligence and Neuroscience. TODO(verify) titles and venues |
| Q5: boundary handling | Helwig, Branke and Mostaghim, 2013, "Experimental analysis of bound handling techniques in particle swarm optimization", IEEE Transactions on Evolutionary Computation 17(2). TODO(verify); also the lecture slide on boundary handling |
| Q8 / D19: linearly decreasing inertia weight, alpha 0.9 to 0.4 | Shi and Eberhart, 1999, "Empirical study of particle swarm optimization", Proc. 1999 Congress on Evolutionary Computation, Washington DC, pp. 1945-1950. (The inertia weight itself was introduced in Shi and Eberhart, 1998, "A modified particle swarm optimizer", Proc. IEEE International Conference on Evolutionary Computation, Anchorage, pp. 69-73.) Checked 23 Nov 2026 through later papers that cite both, not the originals |
| Q8 / D19: time-varying acceleration coefficients, beta 2.5 to 0.5 and gamma 0.5 to 2.5 | Ratnaweera, Halgamuge and Watson, 2004, "Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients", IEEE Transactions on Evolutionary Computation 8(3), 240-255. Checked 23 Nov 2026: citation and both value ranges confirmed through a later paper that reproduces them, not the original |
| Q1: PSO and dimension | Lecture slide 8 (about 20 particles suit up to about 100 dimensions) |
| Q3: coefficient balance | Lecture slide 14 (too much beta: egocentric; too much gamma: stagnation; too much delta: early convergence) |
| Q6: activation functions | TODO: find a source on activation choice (ReLU against saturating functions) and on evolving activations with metaheuristics |
| Published results on the banknote data | TODO: find at least two published accuracies on this dataset to compare against |
| Statistics: A12 effect size | Vargha and Delaney, 2000, "A critique and improvement of the CL common language effect size statistics of McGraw and Wong", Journal of Educational and Behavioral Statistics 25(2). TODO(verify) |
| Statistics: Holm correction | Holm, 1979, "A simple sequentially rejective multiple test procedure", Scandinavian Journal of Statistics 6(2), 65-70. TODO(verify) |
| Dataset | Banknote Authentication, UCI Machine Learning Repository |
