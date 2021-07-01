using ForwardDiff
using Plots
gr()
using DifferentialEquations


# Define SHO Hamiltonian Function
H(q, p) = 0.5*q^2 + 0.5*p^2

# try out the function
H(1.0, 1.0)


function dH(q, p)
    dHdq, dHdp = ForwardDiff.gradient(x->H(x[1], x[2]), [q, p])
    return dHdq, dHdp
end

function dH_analytic(q, p)
    return q, p
end


# try it out
dH(1.0, 1.0) == dH_analytic(1.0, 1.0)
dH(1.0, -1.0) == dH_analytic(1.0, -1.0)
dH(-1.0, 1.0) == dH_analytic(-1.0, 1.0)
dH(-1.0, -1.0) == dH_analytic(-1.0, -1.0)


function H_vec_field(q, p)
    dHdq, dHdp = dH(q, p)
    q̇ = dHdp
    ṗ = - dHdq

    return q̇, ṗ
end

# test it out
dH(1.0, 2.0)
H_vec_field(1.0, 2.0)  # (2.0, -1.0)


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
using Distributions
X_train = rand(Uniform(-1, 1), 2, 100)

# set up target data: here we use analytic result for time derivatives
Y_train = zero(X_train)
# i.e. q̇= dHdp, ṗ = -dHdq
Y_train[1,:] .= X_train[2, :]
Y_train[2,:] .= -X_train[1, :]

train_loader = DataLoader((X_train, Y_train), batchsize=2, )



scatter!(p1, X_train[1,:], X_train[2,:], c=:green, label="training points")

using Flux

HNN = Chain(Dense(2, 50, tanh), Dense(50,50, tanh), Dense(50, 1), first)


H_true = H.(X_train[1,:], X_train[2, :])
H_approx = HNN(X_train)

# visualize pre-traning error
scatter(H_approx, H_true)
dHNN(q::AbstractFloat, p::AbstractFloat) = collect(Flux.gradient.((q,p)->HNN([q, p]), q, p))

function dHNN(x::AbstractArray)
    return [dHNN(col[1], col[2]) for col ∈ eachcol(x)]
end

# return flow from HNN
function HNN_vec_field(X)
    DH = dHNN(X)

    Y = zero(DH)
    Y[1, :] .= DH[2, :]
    Y[2, :] .= -DH[1,:]
    return Y
end

# Compute Loss
function loss(X, Y)
    Ŷ = HNN_vec_field(X)
    Lq = Flux.mse(Ŷ[1,:], Y[1,:])
    Lp = Flux.mse(Ŷ[2,:], Y[2,:])

    return Lq + Lp
end

# test it out
loss(X_train, Y_train)
