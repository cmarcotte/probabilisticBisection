@testset "Base probabilisticBisection Tests" begin

   @testset "SparseDistribution Tests" begin

      @testset "SparseDistribution Evaluation Tests" begin
         for T in (Float32, Float64,)
            let a = zero(T), b = one(T), s::T = a + (b - a) * rand(T), f(x) = (a <= x <= b ? one(T) : zero(T))
               x::Vector{T} = T[a, b]
               y::Vector{T} = T[f(T(1 // 2) * (a + b))]
               𝒻 = SparseDistribution(x)
               @test 𝒻(s) ≈ f(s)
               𝒻 = SparseDistribution(x, y)
               @test 𝒻(s) ≈ f(s)
            end
            let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T), s1::T = a + (r - a) * rand(T), s0::T = r + (b - r) * rand(T), f(x) = (a <= x <= b ? x <= r ? one(T) : zero(T) : zero(T))
               x::Vector{T} = T[a, r, b]
               y::Vector{T} = T[f(T(1 // 2) * (a + r)), f(T(1 // 2) * (r + b))]
               𝒻 = SparseDistribution(x, y)
               # f(x) is unnormalized: sum(f) = 1 * (r-a) + 0 * (b-r) = (r-a), so 𝒻(x) = (x <=r ? 1/(r-a) : 0)
               @test 𝒻(s1) ≈ f(s1) / (r - a)
               @test 𝒻(s0) ≈ zero(T)
            end
            let a = one(T), b = T(100), r::T = a + (b - a) * rand(T), s1::T = a + (r - a) * rand(T), s0::T = r + (b - r) * rand(T), f(x) = (a <= x <= b ? x <= r ? one(T) : zero(T) : zero(T))
               x::Vector{T} = T[a, r, b]
               y::Vector{T} = T[f(T(1 // 2) * (a + r)), f(T(1 // 2) * (r + b))]
               𝒻 = SparseDistribution(x, y)
               # f(x) is unnormalized: sum(f) = 1 * (r-a) + 0 * (b-r) = (r-a), so 𝒻(x) = (x <=r ? 1/(r-a) : 0)
               @test 𝒻(s1) ≈ f(s1) / (r - a)
               @test 𝒻(s0) ≈ zero(T)
            end
         end
      end
      @testset "Median Tests" begin
         for T in (Float32, Float64,)
            let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T)

               𝒻 = SparseDistribution(T[a, b])
               @test probabilisticBisection.median(𝒻) ≈ T(1 // 2) * (a + b)

               𝒻 = SparseDistribution(T[a, r, b], T[one(T), zero(T)])
               @test probabilisticBisection.median(𝒻) ≈ T(1 // 2) * (a + r)

               𝒻 = SparseDistribution(T[a, r, b], T[zero(T), one(T)])
               @test probabilisticBisection.median(𝒻) ≈ T(1 // 2) * (r + b)

               let s1::T = T(10) * rand(T), s2::T = T(2) * rand(T)
                  nrm = (s2 * (b - r) + s1 * (r - a))
                  s1′ = s1 / nrm
                  s2′ = s2 / nrm
                  𝒻 = SparseDistribution(T[a, r, b], T[s1, s2])
                  @test 𝒻(T(1 // 2) * (a + r)) ≈ s1′
                  @test 𝒻(T(1 // 2) * (r + b)) ≈ s2′
                  @test probabilisticBisection.median(𝒻) ≈ (s1′ * r >= T(1 // 2) ? one(T) / (T(2) * s1′) : r + ((T(1 // 2) - s1′ * r) / (s2′)))
               end
            end
         end
      end
      @testset "SparseDistribution Update Tests" begin
         for T in (Float32, Float64,)
            let a = zero(T), b = one(T), s::T = a + (b - a) * rand(T), f(x) = (a <= x <= b ? one(T) : zero(T)), p = T(1 // 2) + T(1 // 2) * rand(T), x = a + (b - a) * rand(T), z::Bool = rand(Bool)
               𝒻 = SparseDistribution(T[a, b], T[f(T(1 // 2) * (a + b))])
               nx = length(𝒻.x)
               probabilisticBisection.update!(𝒻, p, x, z)
               @test nx + 1 == length(𝒻.x)
               nx = length(𝒻.x)
               probabilisticBisection.update!(𝒻, p, a, z)
               probabilisticBisection.update!(𝒻, p, b, z)
               @test nx == length(𝒻.x)
               @test probabilisticBisection.sum(𝒻) ≈ one(T)
            end
         end
      end
   end

   @testset "SparseCumulativeDistribution Tests" begin
      @testset "SparseCumulativeDistribution Evaluation Tests" begin
         for T in (Float32, Float64,)
            let a = zero(T), b = one(T), s::T = a + (b - a) * rand(T), f(x) = (a <= x <= b ? one(T) : zero(T))
               𝒻 = SparseDistribution(T[a, b], T[f(T(1 // 2) * (a + b))])
               F = probabilisticBisection.cumsum(𝒻)
               @test F(s) ≈ f(s) * T(s - a) / (b - a)
               @test F(b) - F(a) ≈ one(T)
            end
         end
      end
   end

   @testset "probabilisticBisection Tests" begin
      @testset "Deterministic probabilisticBisection Test" begin
         for T in (Float32, Float64,)
            let p::T = one(T)
               let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T)
                  function Z(x::T; r::T=r, p::T=p) where {T}
                     return (rand(T) <= p ? r >= x : !(r >= x))
                  end
                  𝒻 = SparseDistribution(T[a, b])
                  q = probabilisticBisection.median(𝒻)
                  @test !(q ≈ r)
                  @test !probabilisticBisection.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
                  for n in 1:6
                     x = probabilisticBisection.median(𝒻)
                     z = Z(x)
                     probabilisticBisection.update!(𝒻, p, x, z)
                  end
                  q = probabilisticBisection.median(𝒻)
                  @test isapprox(q, r; atol=zero(T), rtol=T(2.0^-3))
                  @test probabilisticBisection.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
                  𝒻 = SparseDistribution(T[a, b])
                  q, 𝒻′ = probabilisticBisection.bisection!(Z, 𝒻, p; reltol=one(T), abstol=zero(T))
                  @test isapprox(q, r; rtol=one(T), atol=zero(T))
               end
            end
         end
      end
      @testset "Stochastic probabilisticBisection Test" begin
         for T in (Float32, Float64,)
            let p::T = one(T) - T(1 // 8)
               let a = zero(T), b = one(T), r::T = a + (b - a) * rand(T)
                  function Z(x::T; r::T=r, p::T=p) where {T}
                     return (rand(T) <= p ? r >= x : !(r >= x))
                  end
                  𝒻 = SparseDistribution(T[a, b])
                  q = probabilisticBisection.median(𝒻)
                  @test !(q ≈ r)
                  @test !probabilisticBisection.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
                  for n in 1:129
                     x = probabilisticBisection.median(𝒻)
                     z = Z(x)
                     probabilisticBisection.update!(𝒻, p, x, z)
                  end
                  q = probabilisticBisection.median(𝒻)
                  @test isapprox(q, r; atol=zero(T), rtol=T(2.0^-3))
                  @test probabilisticBisection.converged(𝒻, q; reltol=T(2.0^-3), abstol=zero(T))
                  𝒻 = SparseDistribution(T[a, b])
                  q, 𝒻′ = probabilisticBisection.bisection!(Z, 𝒻, p; reltol=one(T), abstol=zero(T))
                  @test isapprox(q, r; rtol=one(T), atol=zero(T))
               end
            end
         end
      end
   end
end
