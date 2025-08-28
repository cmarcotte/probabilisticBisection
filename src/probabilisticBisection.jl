module probabilisticBisection

export SparseDistribution, SparseCumulativeDistribution, probabilisticBisection, sum, cumsum

abstract type AbstractSparseDistribution end

struct SparseDistribution{T} <: AbstractSparseDistribution
   x::Vector{T} # bounding positions
   y::Vector{T} # distribution values
   function SparseDistribution(x::Vector{T}, y::Vector{T}) where {T<:Real}
      @assert allunique(x)
      #=
      # Not entirely clear if this is needed...
      # @assert all(y .> zero(T))
      =#
      @assert length(x) == length(y) + 1
      @debug "Making tmp_f"
      tmp_f = new{T}(x, y)
      @debug "tmp_f made; integrating tmp_f"
      integral = sum(tmp_f)
      @debug "integral = $(integral); normalizing tmp_f"
      normalize!(tmp_f)
      @debug "tmp_f normalized!"
      return tmp_f
   end
end

struct SparseCumulativeDistribution{T} <: AbstractSparseDistribution
   x::Vector{T} # bounding positions
   y::Vector{T} # distribution values

   function SparseCumulativeDistribution(x::Vector{T}, y::Vector{T}) where {T<:Real}
      @assert allunique(x)
      @assert length(x) == length(y)
      @debug "Making tmp_f"
      tmp_f = new{T}(x, y)
      return tmp_f
   end
end

SparseDistribution(x::Vector{T}) where {T<:Real} = SparseDistribution(x, ones(T, length(x) - 1))

function sum(f::SparseDistribution{T}) where {T<:Real}
   integral = zero(T)
   for i in eachindex(f.x)
      if i != firstindex(f.x)
         integral += (f.x[i] - f.x[i-1]) * f.y[i-1]
      end
   end
   return integral
end

function normalize!(f::SparseDistribution{T}) where {T<:Real}
   integral::T = sum(f)
   if !(integral ≈ one(T))
      f.y .= f.y ./ integral
   end
   return nothing
end

function (f::SparseDistribution{T})(x::T) where {T<:Real}
   # this permits direct evaluation of a SparseDistribution
   if minimum(f.x) <= x <= maximum(f.x)
      if x in f.x
         idx = findfirst(y -> y == x, f.x)
         return f.y[max(1, idx - 1)]
      else
         leftIndex = findlast(X -> X < x, f.x)
         return f.y[leftIndex]
      end
   else
      return zero(T)
   end
end

function (F::SparseCumulativeDistribution{T})(x::T) where {T<:Real}
   # this permits direct evaluation of a SparseDistribution
   if minimum(F.x) <= x <= maximum(F.x)
      if x in F.x
         idx = findfirst(y -> y == x, F.x)
         return F.y[idx]
      else
         ia = findlast(X -> X < x, F.x)
         return F.y[ia] + (x - F.x[ia]) * (F.y[ia+1] - F.y[ia]) / (F.x[ia+1] - F.x[ia])
      end
   elseif x < minimum(F.x)
      return zero(T)
   elseif x > maximum(F.x)
      return one(T)
   end
end

function cumsum(f::SparseDistribution{T})::SparseCumulativeDistribution{T} where {T<:Real}
   #=		 / f_1, x[1] <= x < x[2]
   We assume f(x) = | f_2, x[2] <= x < x[3]
   		 \ ...
   Such that we can simply cumsum the contributions in the gaps between samples,
   	F(x) = f(x[i]) * (x - x[i]) + Σᵢ f(x[i-1]) * (x[i-1] - x[i-2])
   =#
   domain = extrema(f.x)
   normalize!(f)
   F = SparseCumulativeDistribution(f.x, ones(T, size(f.x)))
   for n in 1:length(F.x)
      if n == firstindex(F.x)
         F.y[n] = zero(T)
      elseif n == lastindex(F.x)
         F.y[n] = one(T)
      else
         F.y[n] = F.y[n-1] + f.y[n-1] * (F.x[n] - F.x[n-1]) / (last(domain) - first(domain))
      end
   end
   return F
end

function median(f::SparseDistribution{T}) where {T<:Real}
   #=
   	Median[f] := x: (b-a)^(-1) integral_a^x f(x') dx' == 1/2
   =#
   F = cumsum(f)
   bounds = (findlast(X -> X <= T(1 // 2), F.y), findfirst(X -> X >= T(1 // 2), F.y))
   x = F.x[first(bounds)]
   if first(bounds) != last(bounds)
      a′ = F.x[first(bounds)]
      b′ = F.x[last(bounds)]
      Fa′ = F.y[first(bounds)]
      Fb′ = F.y[last(bounds)]
      dx = ((T(1 // 2) - Fa′) * (b′ - a′)) / (Fb′ - Fa′)
      x = x + dx
   end
   return x
end

function appendSample!(f::SparseDistribution{T}, sample::NTuple{2,T}) where {T<:Real}
   if !(first(sample) in f.x)
      if length(f.x) > 2
         n = findlast(X -> X < first(sample), f.x[begin:end-1])
         insert!(f.x, n + 1, first(sample))
         insert!(f.y, n + 1, last(sample))
      else
         insert!(f.x, 2, first(sample))
         insert!(f.y, 2, last(sample))
      end
      @assert issorted(f.x)
      @assert length(f.x) == length(f.y) + 1
      normalize!(f)
   end
   return nothing
end

function bisection(Z::Function, interval::NTuple{2,T}, p::T; reltol::T=T(1e-6), abstol::T=zero(T), maxiters::Integer=1024) where {T<:Real}
   f = SparseDistribution([first(interval), last(interval)])
   bisection!(Z, f, p; reltol=reltol, abstol=abstol, maxiters=maxiters)
end

function bisection!(Z::Function, f::SparseDistribution{T}, p::T; reltol::T=T(1e-6), abstol::T=zero(T), maxiters::Integer=1024) where {T<:Real}
   #=
      let f0(x) ~ U(interval)
      let Z(x::Real)::Bool tell us with probability p if x >= the root (true) or not (false)
      let x1 = median(f0(x)) from the interval be sampled using the oracle, Z1 = Z(x1)
      f1 is then updated using Bayes' rule from f0 (see update!), and
      then f0 <- f1,
      and then you check if the root is converged
      =#
   iters = 0
   x = median(f)
   while !converged(f, x; abstol=abstol, reltol=reltol) && iters < maxiters
      z = Z(x)
      update!(f, p, x, z)
      iters += 1
   end
   return median(f), f
end

function update!(f::SparseDistribution{T}, p::T, x::T, z::Bool) where {T<:Real}
   #=
   Update step:
   	if Z1 == true
     		f1(y >= x1) = inv(γ(x1)) * p * f0(y)
     		f1(y < x1) = inv(γ(x1)) * (1 - p) * f0(y)
     	elseif Z1 == false
     		f1(y >= x1) = inv(1 - γ(x1)) * (1 - p) * f0(y)
     		f1(y < x1) = inv(1 - γ(x1)) * p * f0(y)
     	end
   where
     	γ(x) = (1 - F0(x)) * p + F0(x) * (1 - p)
   and
     	F0(x) = cumsum(f0(x))
   =#
   appendSample!(f, (x, f(x)))
   F = cumsum(f)
   γ = SparseCumulativeDistribution(F.x, (one(T) .- F.y) .* p + F.y .* (one(T) - p))
   Γ = γ(x)
   for n in eachindex(f.x[begin:end-1], f.y)
      if z
         f′ = (inv(Γ) * p, inv(one(T) - Γ) * (one(T) - p))
      else
         f′ = (inv(Γ) * (one(T) - p), inv(one(T) - Γ) * p)
      end
      x′ = (f.x[n] + f.x[n+1]) * T(1 // 2)
      f.y[n] *= (x′ >= x ? f′[1] : f′[2])
   end
   normalize!(f)
   return nothing
end

function converged(f::SparseDistribution{T}, x::T; abstol::T=zero(T), reltol::T=T(1e-6)) where {T<:Real}
   #=
   # Convergence is determined by b_i - a_i for x in (a_i, b_i).
   # (a --------- aᵢ - x - bᵢ ---- b) =>
   # Require that |b_i - a_i| < |b - a| * reltol + abstol
   # =#
   n = findlast(X -> X < x, f.x)
   aᵢ, bᵢ = f.x[n], f.x[n+1]
   a, b = f.x[begin], f.x[end]
   #=
   # There should perhaps be a considation of what proportion of the total probability (ai, bi) * f.y[n] covers compared to p?
   =#
   return abs(bᵢ - aᵢ) < abs(b - a) * reltol + abstol
end

#=
# For compatibility with older Julia versions
# =#
if !isdefined(Base, :get_extension)
   include("../ext/ApproxFunExtension.jl")
end

end #module
