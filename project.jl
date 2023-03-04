# using Pkg; Pkg.add("POMDPs"); Pkg.add("QuickPOMDPs"); Pkg.add("POMDPSimulators"); Pkg.add("QMDP")
using POMDPs, QuickPOMDPs, POMDPSimulators, QMDP
using POMDPTools: Deterministic, Uniform
using Random

satellite = QuickPOMDP(
    states = [(P, S1, E, O, T) for P in 0:100, S1 in 0:100, E in 0:1, O in 0:1, T in 0:4],
    actions = [0, 1],
    observations = [(P, S1, E, O, T) for P in 0:100, S1 in 0:100, E in 0:1, O in 0:1, T in 0:4],
    initialstate = Deterministic((100,100,1,1,rand(0:4))),
    discount = 0.95,

    transition = function (s, a)
        if s == 0
            return Deterministic(s)
        elseif s[1] != 0
            return Deterministic((s[1] - rand(1:s[1]),s[2],s[3],s[4],s[5]))
        else
            return Deterministic(s)
        end
    end,

    observation = function (s, a, sp)
        return Deterministic(s)
    end,

    reward = function (s, a)
        if s[3] == 1 && s[4] == 1 # functional satellite
            r = 10
        elseif s[3] == 1 || s[4] == 1 # partially functional
            r = 0
        else # broken
            r = -100
        end
    end,

    isterminal = s -> s[1] == 0
)

solver = QMDPSolver()
policy = solve(solver, satellite)

rsum = 0.0
for (s,a,o,r) in stepthrough(satellite, policy, "s,a,o,r", max_steps=10)
    println("in state $s")
    println("took action $a")
    println("received observation $o and reward $r")
    global rsum += r
end
println("Undiscounted reward was $rsum.")