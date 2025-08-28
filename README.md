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

Let $f(x) = \mathcal{U}(0,1) = 1$, with fixed $1/2 < p \leq 1$, and let $x_1 \in (0,1)$. Then $F(x_1) = x_1$, and $\gamma(x_1) = (1 - p) x_1 + p (1 - x_1)$.
Let us assume that $z_1 = 1$; then the updated belief is:

$$
	f(x) =
		\begin{cases}
			\frac{(1-p)}{(1-p)x_1 + p(1-x_1)}, &x < x_1, \\
			\frac{p}{(1-p)x_1 + p(1-x_1)}, &x \geq x_1
		\end{cases},
$$
i.e., it shifts the previously uniformly distributed probability density to the left and right of $x_1$ according to the oracle.

## Implementation

One should specify a prior belief $f(x)$ -- this is typically just an uninformative uniform distribution over $\Omega: f(x) = \mathcal{U}(a,b)$.
In principle, one can approximate any prior with a $\mathcal{C}^0$ (piecewise constant) input, if desired.
The module uses an in-built `Sparse[Cumulative]Distribution{T}` type to represent the belief $f$, which assumes a piecewise constant (i.e. $\mathcal{C}^0$) prior.
The restriction to $\mathcal{C}^0$ beliefs is well-behaved because the update to the belief is multiplicative on each subinterval, so subdivision of each subdomain is sufficient for the posterior belief to have the same polynomial order as the prior belief (i.e. piecewise constant).

To update the prior, we require an evaluation of the CDF of the belief $F(x) = (b-a)^{-1}\int_{a}^{x} \mathrm{d}x' f(x')$ at the sample point.
The CDF $F(x)$ is represented as a `SparseCumulativeDistribution{T}` which is not required to be normalized (like the `SparseDistribution{T}`) and is $\mathcal{C}^1$, i.e. evaluation of $F(x)$ for $a_i < x < b_i$ is performed by linear interpolation.

The bisection procedure will then sample the initial domain
according to the median rule (i.e. $x_1 = x: F(x) = \frac{1}{2}$), and update the belief according to the oracle values returned from the sample point $z_i = Z(x_i)$: $f(x | \{z_i\}_{i=1}^{n})$.

The method uses `insert!(a::Vector, index::Integer, item)` to insert the new sample point `x′` to the existing support points of a `SparseDistribution{T}`.
For few initial support points (the typical case), this insertion requires $O(1)$ memory and $O(1)$ time.
In practice, this is quite cheap; the number of points used depends on $p$ the confidence in the oracle $Z$ outputs and the tolerance required for convergence, and in the typical case one begins with two support points $(a,b)$ and the subsequent refinement is often $\approx 30$ additional samples.
In testing, for $1 + 2^{10}$ initial support points, the time taken to update the `SparseDistribution` is negligible.

[^1]: [Waeber, Rolf, Peter I. Frazier, and Shane G. Henderson. "Bisection search with noisy responses." SIAM Journal on Control and Optimization 51, no. 3 (2013): 2261-2279.](https://doi.org/10.1137/120861898)