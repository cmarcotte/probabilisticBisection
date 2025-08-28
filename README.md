# Probabilistic Bisection Algorithm (PBA)

This is an implementation of the PBA following Waeber _et al_ (2013) [^1].

The method uses the prior belief $f: x \in \Omega \to \mathbb{R}$ for $\Omega = (a, b) \subset \mathbb{R}$ for the location of the sign-change $x^*$ (hereafter, I'll refer to $x^*$ as a root, but the method works for the less specific condition).

The procedure requires an oracle, $Z: \Omega \to \mathbb{B}$, where $\mathbb{B} = \{0,1\}$ is a boolean, which returns $Z(x_i) = z_i = 1$ if input $x$ has overshot the sign-change $x^*$, with _constant_ probability $1/2 < p \leq 1$, otherwise $z_i = 0$ with probability $p$.

We define the CDF of the belief $F(x) = (b-a)^{-1}\int_{a}^{x} f(x^\prime) \mathrm{d}x^\prime$, and define

$$
	\gamma(x) = (1 - p) F(x) + p (1 - F(x)).
$$

The update to the belief with each new sample $x_i$ and oracle value $z_i = Z(x_i)$ is given by:

$$
	f(y | x_i) =
		\begin{cases}
			\begin{cases}
				p f(y)/\gamma(y), &y \geq x_i \\
				(1 - p) f(y) / \gamma(y), &y < x_i \\
			\end{cases} &z_i = 1, \\
			\begin{cases}
				(1 - p) f(y) / (1 - \gamma(y), &y \geq x_i\\
				p f(y) / (1 - \gamma(y)), &y < x_i
			\end{cases} &z_i = 0,
		\end{cases}
$$

### Example

Let $f(x) = \mathcal{U}(0,1) = 1$, with fixed $1/2 < p \leq 1$, and let $x_1 \in (0,1)$. Then $F(x) = x$, and $\gamma(x) = (1 - p) x + p (1 - x)$.
Let us assume that $z_1 = 1$; then the updated belief is:

$$
	f(x) =
		\begin{cases}
			\frac{(1-p)}{(1-p)x + p(1-x)}, &x < x_i, \\
			\frac{p}{(1-p)x + p(1-x)}, &x \geq x_i
		\end{cases}
$$

## Base Implementation

One should specify a prior belief $f(x)$ -- this is typically just an uninformative uniform distribution over $\Omega: f(x) = \mathcal{U}(a,b)$.
In principle, one can approximate any prior with a $\mathcal{C}^0$ (piecewise constant) input, if desired.
The module uses an in-built `Sparse[Cumulative]Distribution{T}` type to represent the belief $f$, which assumes a piecewise constant (i.e. $\mathcal{C}^0$) prior.

To update the prior, we require the CDF of the belief $F(x) = (b-a)^{-1}\int_{a}^{x} \mathrm{d}x' f(x')$, which is represented as a `SparseCumulativeDistribution{T}` which is not required to be normalized (like the `SparseDistribution{T}`) and is $\mathcal{C}^1$, i.e. represented by linear segments.

The bisection procedure will then sample the initial domain
according to the median rule (i.e. $x_1 = x: F(x) = \frac{1}{2}$), and update the belief according to the oracle values returned from these sample points $z_i = Z(x_i)$: $f(x | \{z_i\}_{i=1}^{n})$.

The method uses `push!(f.x, x′)` to append the new sample point `x′` to the existing support points which is $O(\log_2(n))$ reallocations, followed by a `sortperm(f.x)` which is $O(n \log_2(n))$.
In practice, this is quite cheap; the number of points used depends on $p$ the confidence in the oracle $Z$ outputs and the tolerance required for convergence, and in the typical case one begins with two support points $(a,b)$ and the subsequent refinement is often $\approx 30$ additional samples.
Even for initial $n \approx 2^{10}$, the time taken by the method is negligible.

[^1]: [Waeber, Rolf, Peter I. Frazier, and Shane G. Henderson. "Bisection search with noisy responses." SIAM Journal on Control and Optimization 51, no. 3 (2013): 2261-2279.](https://doi.org/10.1137/120861898)

## Extensions

The package includes several extensions which rely on external packages to extend the functionality.

### `ApproxFun.jl`

The most significant extension relies on the excellent spectral approximation package [`ApproxFun.jl`](https://juliaapproximation.github.io/ApproxFun.jl/latest/).
The extension identifies a subset of the `ApproxFun.Fun` type:

```jl
const ℱ = Fun{PiecewiseSpace{NTuple{N, Chebyshev{Segment{T}, T}}, DomainSets.UnionDomain{T, NTuple{N, Segment{T}}}, T}, T, Vector{T}} where {N, T<:Real}
```

with relevance to our purposes.
This type sub-divides the domain $\Omega = (a,b) \subset \mathbb{R}^1$ at all sample locations $x \in \Omega$.
Thus this treats $\Omega = (a,b) = \cup_{i=1}^{n}\Omega_i$ where $\Omega_i = (a_i, b_i)$ and $a_1 = a$ and $b_n = b$.
This type has the benefit of retaining low-order approximations on each `Segment{T}(a,b)`, rather than combining them into a single `Interval(a,b)` which requires a substantially higher order modal expansion.

Since this extension and type allows arbitrarily high-order polynomial expansions $p$ [^2], and the CDF required for the updating of the belief is $p_i+1$ on each $\Omega_i$, and the update to the belief involves the multiplicative inverse of the integral of the belief, we will find that the degree of $f$ on each `Segment{T}` increases with each iteration and the number of `Segment{T}`s increases by one on each iteration.

[^2]: That is, `ApproxFun.Fun` tops out at `length(coefficients(f)) == 1 + 2^20`, which might as well be arbitrarily high.

### `Distributions.jl`

This extension is only included for convenient definition of _informative_ priors (i.e. not the typical $\mathcal{U}(a,b)$).
One simply passes a `Distribution{T}` type to the `PBA` as the prior with a finite domain.
Internally, the prior is immediately approximated with
