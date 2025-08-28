using probabilisticBisection, GLMakie

function Z(x::T; r::T=T(0.55), p::T=T(1.0)) where {T}
   return (rand(T) <= p ? r >= x : !(r >= x))
end

function main()

   a, b = 0.0, 1.0
   X = 0.25 + 1e-3 * rand() #rand(a .. b)
   p = 0.5 + 0.5 * rand()
   @assert 0.5 < p <= 1.0
   ZZ(x; p=p) = Z(x; r=X, p=p)

   xr = range(; start=a, stop=b, length=256)
   _xr = collect(xr)
   push!(_xr, X)
   push!(_xr, X + eps(1.0))
   push!(_xr, X - eps(1.0))
   sort!(_xr)

   fig = Figure()
   axs = [Axis(fig[1, 1], ylabel="Z(x; p)", title="p=$(p), 1-p = $(1-p)"), Axis(fig[2, 1], xlabel="x", ylabel="f(x)", yscale=log2)]
   linkxaxes!(axs[1], axs[2])

   N = 33
   clist = range(Makie.HSL(colorant"tomato"), stop=Makie.HSL(colorant"olive"), length=N + 1)
   ylims!(axs[2], 2.0^-1, 2.0^+1)

   zl = lines!(axs[1], _xr, ZZ.(_xr; p=1.0), color=:black, linewidth=2, label="E[Z(x; p)] = Z(x; p = 1)")

   f = SparseDistribution{Float64}([a, b])

   stairs!(axs[2], f.x[begin+1:end], f.y; step=:pre, linewidth=1, color=clist[1], label="f(x)")
   stairs!(axs[2], f.x[begin:end-1], f.y; step=:post, linewidth=1, color=clist[1], label="f(x)")
   n = 1
   x = PBA.median(f)
   while !PBA.converged(f, x; reltol=1e-3, abstol=0.0) && n < N
      x = PBA.median(f)
      z = ZZ(x)
      scatter!(axs[1], x, z, color=:black, label="Z(x; p)")
      PBA.update!(f, p, x, z)
      n += 1
      stairs!(axs[2], f.x[begin+1:end], f.y; step=:pre, linewidth=1 + (n / N), color=clist[n], label=((n == N) ? "f(x | x_n)" : nothing))
      stairs!(axs[2], f.x[begin:end-1], f.y; step=:post, linewidth=1 + (n / N), color=clist[n], label=((n == N) ? "f(x | x_n)" : nothing))
   end
   axislegend(axs[1]; merge=true, unique=true)
   axislegend(axs[2]; merge=true, unique=true)
   ylims!(axs[2]; low=nothing, high=nothing)
   fig
end

main()
