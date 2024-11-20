using DifferentialEquations
using LinearAlgebra
using Random


using Plots
# Simulation Parameters
dt = 0.01
T = 100.0
t = 0.0:dt:T
beta = 8/3
sigma = 10.0
rho = 28.0
n = 6



function rossler(du, u,p,t)
    x1, y1, z1, x2, y2, z2 = u
    eps = 0  # Set to a desired value for coupling
    a, b, c = 0.2, 0.2, 5.7  # Typical Rossler parameters

    # Rossler equations for first oscillator with optional coupling
    du[1] = -y1 - z1             + eps * (x2 - x1)  # Uncomment if coupling is desired
    du[2] = x1 + a * y1          + eps * (y2 - y1)  # Uncomment if coupling is desired
    du[3] = b + z1 * (x1 - c)    + eps * (z2 - z1)  # Uncomment if coupling is desired

    # Rossler equations for second oscillator with optional coupling
    du[4] = -y2 - z2             + eps * (x1 - x2)  # Uncomment if coupling is desired
    du[5] = x2 + a * y2          + eps * (y1 - y2)  # Uncomment if coupling is desired
    du[6] = b + z2 * (x2 - c)    + eps * (z1 - z2)  # Uncomment if coupling is desired

    return du
end


# Initial Condition
Random.seed!(123)
x0 = [-8.0, 10.0, 27.0, -14.0, 12.0, 2.0]

# Solve the Lorenz System
prob = ODEProblem(rossler, x0, (0.0, T))
sol = solve(prob, RK4(), saveat=dt, reltol=1e-12, abstol=1e-12)

# Extract solution values
x = Array(sol)'
x = x[2000:end,:]
# Plot results

plot(x[:,1], x[:,2], x[:,3], label="Trajectory 1", xlabel="x", ylabel="y", zlabel="z");
plot!(x[:,4], x[:,5], x[:,6], label="Trajectory 2")



# Compute Derivative
dx = zeros(size(x))
for j in 1:length(x[2000:end,1])
    dx[j, :] = lorenz!(similar(x[j, :]), x[j, :])
end


# SINDy Function Definitions
function poolData(yin, nVars, polyorder)
    n = size(yin, 1)
    yout = ones(n, 1)

    # Poly order 1
    for i in 1:nVars
        yout = hcat(yout, yin[:, i])
    end

    # Poly order 2
    if polyorder >= 2
        for i in 1:nVars
            for j in i:nVars
                yout = hcat(yout, yin[:, i] .* yin[:, j])
            end
        end
    end

    # Poly order 3
    if polyorder >= 3
        for i in 1:nVars
            for j in i:nVars
                for k in j:nVars
                    yout = hcat(yout, yin[:, i] .* yin[:, j] .* yin[:, k])
                end
            end
        end
    end

    return yout
end

function sparsifyDynamics(Theta, dXdt, lamb, n)
    Xi = Theta \ dXdt  # Initial guess using least squares
    for _ in 1:10
        smallinds = abs.(Xi) .< lamb
        Xi[smallinds] .= 0  # Threshold small coefficients to zero

        for ind in 1:n
            biginds = .!smallinds[:, ind]
            Xi[biginds, ind] .= Theta[:, biginds] \ dXdt[:, ind]
        end
    end
    return Xi
end

# Apply SINDy
Theta = poolData(x, n, 3)  # Up to third-order polynomials
lamb = 0.025  # Sparsification parameter
Xi = sparsifyDynamics(Theta, dx, lamb, n)

Xi
println(Xi)








