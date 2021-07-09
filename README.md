# Scientific Machine Learning for the (Simple) Harmonic Oscillator
A compendium of examples utilizing "Scientific Machine Learning" for the harmonic oscillator. 

Note: GitHub markdown does not support mathjax. See [this](https://latex.codecogs.com) link to generate embedded math images.

# Overview 
## Stages of the Harmonic Oscillator Problem 

| Type | Description |
| ----------- | ----------- |
| Simple Harmonic Oscillator | ![SHO](https://latex.codecogs.com/svg.image?m\ddot{x}&space;&plus;&space;kx&space;=&space;0) | 
| Damped Harmonic Oscillator | ![DSHO](https://latex.codecogs.com/svg.image?m\ddot{x}&space;&plus;&space;2\Gamma&space;\dot{x}&space;&plus;&space;kx&space;=&space;0) | 
| Forced Harmonic Oscillator | ![FSHO](https://latex.codecogs.com/svg.image?m\ddot{x}&space;&plus;&space;kx&space;=&space;F(t)) | 
| Damped-Forced Harmonic Oscillator | ![DFSHO](https://latex.codecogs.com/svg.image?m\ddot{x}&space;&plus;2\Gamma&space;\dot{x}&space;&plus;&space;kx&space;=&space;F(t)) | 
| System of Coupled Harmonic Oscillators | ![coupled](https://latex.codecogs.com/svg.image?\mathbf{M}\ddot{\vec{x}}&plus;&space;\mathbf{K}\vec{x}&space;=&space;0) | 

## Methods 
1. Standard Numerical Integration via `DifferentialEquations.jl`
   - Checkout *Differential Algebraic Equations* (DAEs)
2. Parameter Fitted Differential Equations 
3. Physics Informed Neural Networks (i.e. physics-based loss regression)
4. Universal Differential Equations a la [this](https://arxiv.org/abs/2001.04385)
5. Neural ODEs/PDEs
6. Green's Function Approach
7. Hamiltonian/Lagrangian Neural Networks 
8. Graph Neural Networks 
9. *Geometric* Machine Learning 
10. Sparse Identification (e.g. SINDy)
11. FEM style Physics-Based ML on irregular grid 
12. Manifold Methods
13. Interaction Networks 
14. Relational Networks



