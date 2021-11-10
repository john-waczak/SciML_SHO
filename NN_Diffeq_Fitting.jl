using Flux
using Statistics
using Plots
using DifferentialEquations


# Solve the ode given by $$ u' = \cos 2\pi t $$ by approximation of $u$ with a Neural Network
NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
              Dense(1,32,tanh),
              Dense(32,1),
              first) # Take first value, i.e. return a scalar
NNODE(1.0)

# construct function $g(t)$ to enforce initial condition $$ g(t) = t NNODE(t) + u(0)
u₀ = 1.0
g(t) = t*NNODE(t) + u₀



#Construct loss function to numericaly compute derivative and check against R.H.S.
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)



# compare against true solution
t = 0:1e-3:1.0

plot(t, g.(t), label="NN")
plot!(t, u₀ .+ sin.(2π.*t)/2π, label="True Solution")



