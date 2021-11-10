using ForwardDiff
using Plots
using ReverseDiff
using DifferentialEquations
using Flux
using Statistics
# gr()
# pyplot()
plotly()

# Define SHO Hamiltonian Function
H(q, p) = 0.5*q^2 + 0.5*p^2


function dH(q, p)
    dHdq, dHdp = ForwardDiff.gradient(x->H(x[1], x[2]), [q, p])
    return dHdq, dHdp
end

function dH_analytic(q, p)
    return q, p
end

function H_vec_field(q, p)
    dHdq, dHdp = dH(q, p)
    q̇ = dHdp
    ṗ = - dHdq

    return q̇, ṗ
end


# Visualize the H, it's level curves, and the flow as
# as tangent vectors to level curves
# i.e. governing idea is H = const
qs = -1.5:0.1:1.5
ps = -1.5:0.1:1.5

Q = []
P = []
Q̇ = []
Ṗ = []

for q ∈ qs[1:2:end], p ∈ ps[1:2:end]
    q̇, ṗ = H_vec_field(q, p)
    push!(Q, q)
    push!(P, p)
    push!(Q̇, q̇)
    push!(Ṗ, ṗ)
end

p1 = contour(qs, ps, H, xlabel="q", ylabel="p", aspect_ratio=1.0, fill=true, colorbar_title="H", size=(800,800), c=:viridis)
α=0.1
quiver!(p1, Q, P, quiver=(α .* Q̇, α .* Ṗ), aspect_ratio=1.0, color=:white)
xlims!(p1, -1.5, 1.5)
ylims!(p1, -1.5, 1.5)





Ham(q,p, param) = q^2 + p^2
p0=1.0
q0=0.0
tspan=(0.0, 10.0)
prob = HamiltonianProblem(Ham, p0, q0, tspan, dt=0.01)
sol = solve(prob, SymplecticEuler())

plot!(p1, sol[1,:], sol[2, :], color=:green, label="q₀=$(q0), p₀=$(p0)")

p0=1.5
q0=0.0
tspan=(0.0, 10.0)
prob = HamiltonianProblem(Ham, p0, q0, tspan, dt=0.01)
sol = solve(prob, SymplecticEuler())

plot!(p1, sol[1,:], sol[2, :], color=:cyan, label="q₀=$(q0), p₀=$(p0)")



# Let's set up the Neural Network to approximate the Hamiltonian
# generate training data with H(q,p) <= 1
Npoints = 500
r_train = sqrt.(rand(1, Npoints))
φ_train = 2π.*rand(1, Npoints)
data = vcat(r_train .* cos.(φ_train), r_train .* sin.(φ_train))


# set up target data: here we use analytic result for time derivatives
target = zero(data)
# i.e. q̇= dHdp, ṗ = -dHdq
target[1,:] .= data[2, :]
target[2,:] .= -data[1, :]


p2 =  scatter!(p1, data[1,:], data[2,:], c=:green, label="training points")
p3 = plot(Q, P, H.(Q, P), st = :surface, xlabel = "q", ylabel = "p", zlabel = "H")



# -------------------------------------------------------------------------------------------------------
# create model for HNN

# define the struct
struct HNN{M, R, P}
    model::M  # the internal NN
    re::R   # for recreating internal NN
    p::P  # for holding current params of internal NN


    # define the constructor
    function HNN(model)
        p, re = Flux.destructure(model)
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

# define the trainable paramaters
Flux.trainable(hnn::HNN) = (hnn.p,)

function _hamiltonian_flow(re, p, x)
    dHdX = Flux.gradient(x->sum(re(p)(x)), x)[1]  # note: re(p)(x) == Model(x)
    n = size(x,1) ÷ 2  # i.e. how many p's and q's
    return cat(dHdX[(n+1):2n, :], dHdX[1:n, :], dims=1)
end

# define how to call the HNN on data
(hnn::HNN)(X, p=hnn.p) = _hamiltonian_flow(hnn.re, p, X)


hnn = HNN(
    Chain(Dense(2, 200, relu), Dense(200, 200, relu), Dense(200, 1))
)

dataloader = Flux.Data.DataLoader((data, target), batchsize=500, shuffle=true)

p = hnn.p

opt = ADAM(10^(-3))

loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2)
loss(data, target, p)


callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")
callback()


test_gs = ReverseDiff.gradient(p -> loss(data, target, p), p)

epochs = 2000
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p)
        Flux.Optimise.update!(opt, p, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end
callback()


#p3 = plot(Q, P, H.(Q, P), st = :surface, xlabel = "q", ylabel = "p", zlabel = "H")
#p3 = plot(Q, P, HNN.model.(Q, P), st = :surface, xlabel = "q", ylabel = "p", zlabel = "H")
