using Flux
using Plots
using DifferentialEquations


# solve 2nd order ode numerically
k = 1.0
force(dx,x,k,t) = -k*x + 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
plot(sol,label=["Velocity" "Position"])


# Generate the dataset
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
force2(dx,x,k,t) = -k*x
prob_simplified = SecondOrderODEProblem(force2,1.0,0.0,(0.0,10.0),k)
sol_simplified = solve(prob_simplified)
plot(sol,label=["Velocity" "Position"])
plot!(sol_simplified,label=["Velocity Simplified" "Position Simplified"])


# create Physics Loss Term to force solution to satisfy SHO ode as well
random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)
loss_ode()

λ = 0.1  # factor controlling physics regularization
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

# visualize the final result!
learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")






