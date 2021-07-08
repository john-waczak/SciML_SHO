using Flux
using Plots
using DifferentialEquations

ω0 = 0.5
γ = 0.1
x₀ = 10.0
v₀ = 0.0
tspan = (0.0, 20.0)
acceleration(dx, x, p, t) = -ω0^2*x-2*γ*dx
prob = SecondOrderODEProblem(acceleration, v₀, x₀, tspan)
sol = solve(prob)

plot(sol,label=["Velocity" "Position"])


# generate the training dataset
plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,t) for state in data_plot]

t = 0:3.3:10
dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

plot(plot_t,force_plot,xlabel="t",label="True Force")
scatter!(t,force_data,label="Force Measurements")


# create NN
# create NN
NNForce = Chain(x -> [x],
                Dense(1,32,tanh),
                Dense(32,1),
                first)

loss() = sum(abs2,NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
loss()


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

# train the NN
Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)



# visualize the result
learned_force_plot = NNForce.(positions_plot)
plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")




# solve SHO problem to justify PINN
acceleration2(dx,x,k,t) = -ω0^2*x
prob_simplified = SecondOrderODEProblem(acceleration2, v₀, x₀, tspan)
sol_simplified = solve(prob_simplified)
p2 = plot(sol_simplified, label=["Velocity" "Position"])
plot!(p2, sol_simplified,label=["Velocity Simplified" "Position Simplified"])




# create Physics Loss Term to force solution to satisfy SHO ode as well
random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2,NNForce(x) - (-(ω0^2)*x) for x in random_positions)
loss_ode()

λ = 0.025  # factor controlling physics regularization
composed_loss() = loss() + λ*loss_ode()


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(composed_loss())
    end
end
display(composed_loss())

# train the new model
Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)


loss()
λ * loss_ode()

# visualize the final result!
learned_force_plot = NNForce.(positions_plot)
plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")










