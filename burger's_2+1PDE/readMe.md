# DeepONet for 2D Burgers’ Equation (2+1D)

This repository contains an end-to-end research pipeline for **learning the solution operator of the 2D viscous Burgers’ equation** using **Deep Operator Networks (DeepONets)**.

The work proceeds from:
1. **Numerical PDE solving (ground-truth generation)**  
2. **Dataset construction for operator learning**  
3. **DeepONet design with CNN-based branch networks**  
4. **Careful training with controlled mini-batch scaling**  

The goal is to approximate the operator:
\[
\mathcal{G}: u_0(x,y) \;\longmapsto\; u(x,y,t),
\]
where \(u_0\) is the initial condition and \(u(x,y,t)\) is the PDE solution over space–time.

---

## 1. PDE Problem Definition

We consider a **2D viscous Burgers-type equation**:
\[
u_t + u (u_x + u_y) = \nu \Delta u,
\]
defined on a periodic spatial domain \([0,2\pi]^2\) and time interval \([0,T]\).

Key properties:
- Nonlinear advection
- Diffusive regularization via viscosity \(\nu > 0\)
- Periodic boundary conditions
- Smooth solutions for all \(t > 0\)

This equation serves as a canonical nonlinear testbed for **operator learning**.

---

## 2. Numerical Solver: Pseudo-Spectral Method

### Why a pseudo-spectral solver?
- Periodic domain → Fourier basis is natural
- Spatial derivatives computed exactly in spectral space
- High accuracy for smooth solutions
- Standard approach for Burgers / Navier–Stokes equations

### Method overview
- Spatial discretization via **Fourier modes**
- Time integration using **RK4 with integrating factor**
- Nonlinear terms computed in physical space
- **2/3-rule dealiasing** to prevent aliasing instability

### Mathematical splitting
\[
u_t = \underbrace{-u(u_x + u_y)}_{\text{nonlinear}} + \underbrace{\nu \Delta u}_{\text{linear}}
\]

- Linear diffusion handled exactly using an integrating factor
- Nonlinearity advanced via explicit Runge–Kutta

The solver is implemented in `Burgers2DSolver` and follows classical references in spectral methods.

---

## 3. Initial Conditions: Gaussian Random Fields

Each training sample corresponds to a **different initial condition** \(u_0(x,y)\).

### Generation process
1. Sample white noise in physical space
2. Transform to Fourier space
3. Apply **spectral Gaussian filter**:
   \[
   \hat u_0(k) \sim \exp(-\tfrac{1}{2} \ell^2 |k|^2)
   \]
4. Transform back to physical space
5. Normalize to **O(1)** magnitude

### Why this choice?
- Produces smooth, random, physically meaningful initial conditions
- Controls correlation length via \(\ell\)
- Standard in operator-learning benchmarks

---

## 4. Dataset Construction

Each dataset element consists of:
- One initial condition \(u_0(x,y)\)
- The corresponding full space–time solution \(u(x,y,t)\)

### Important distinction
- **One sample = one entire function**
- Increasing grid resolution does *not* increase dataset size
- Generalization depends on number of distinct initial conditions

### Current dataset
- Number of function pairs: **1600**
- Grid resolution: **64 × 64 × 200** (space–time)
- Domain: uniform grid over \([0,2\pi]^2 \times [0,T]\)

---

## 5. DeepONet Architecture

We use the **DeepONet framework** to learn the solution operator.

### Operator approximation
\[
u(x,y,t) \approx \sum_{k=1}^p b_k(u_0)\, t_k(x,y,t)
\]

- **Branch network** computes coefficients \(b_k\)
- **Trunk network** computes basis functions \(t_k\)

---

### 5.1 Branch Network (CNN-based)

**Input:** initial condition \(u_0(x,y)\) as a rank-3 tensor  
**Architecture:** convolutional neural network (CNN)

Why CNN?
- Preserves spatial locality
- Captures gradients and multi-scale features
- Avoids information loss from flattening

This design choice significantly improved training and test performance compared to an MLP-based branch.

---

### 5.2 Trunk Network (MLP-based)

**Input:** space–time coordinates \((x,y,t)\)  
**Architecture:** multi-layer perceptron (MLP)

- Processes query locations
- Outputs basis coefficients in \(\mathbb{R}^p\)
- Depth increased to improve space–time expressivity

---

## 6. Training Strategy

### Key distinction (critical)
We explicitly distinguish between:
- **Dataset size** (number of function pairs, fixed)
- **Grid resolution** (physics accuracy, fixed)
- **Mini-batch size** (optimization hyperparameter)

---

### Mini-batch parameterization

Mini-batches are parameterized as:
\[
2^p \text{ initial conditions} \times 2^q \text{ space–time points}
\]

- \(p\): number of functions per batch
- \(q\): number of space–time query points per function

These can be tuned independently for performance and efficiency.

---

### Training objective
- Mean-squared error (or SmoothL1) over predicted vs true solutions
- Loss evaluated consistently on the same space–time grid

---

## 7. Experimental Progress and Results

Key milestones:
- Correct numerical solver with stability checks
- CNN-based branch network (major improvement)
- Clear separation of dataset size vs mini-batch size
- Systematic mini-batch scaling experiments

### Current performance
- Achieved losses as low as **~5×10⁻³**
- This places the model in the expected accuracy regime for smooth Burgers solutions
- Further gains depend on mini-batch scaling and compute limits

---

## 8. Key Lessons Learned

- Operator learning is **data-hungry**: number of functions matters more than grid density
- Preserving tensor structure in the branch is essential
- Numerical solver correctness is foundational
- Mini-batch size affects optimization speed, not generalization
- Clean baselines must precede optimization tricks

---

## 9. References

1. Lu et al., *Learning Nonlinear Operators via DeepONet*, Nature Machine Intelligence (2021)  
2. Canuto et al., *Spectral Methods in Fluid Dynamics*, Springer  
3. Trefethen, *Spectral Methods in MATLAB*, SIAM  
4. Karniadakis et al., *Physics-Informed Machine Learning*, Nature Reviews Physics  
5. Pope, *Turbulent Flows*, Cambridge University Press  

---

## 10. Notes

- The numerical solver is **standard and well-established**, not novel.
- The research contribution lies in:
  - correct operator-learning formulation,
  - architectural choices,
  - experimental protocol,
  - and systematic scaling analysis.

---

## Contact

For questions or discussions related to this project, please open an issue or contact the repository authors.
