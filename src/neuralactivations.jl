ACTV_FUNCS = ["relu", "sigmoid", "σ", "tanh"]

function σ(X)
    return 1 ./(1 .+ exp.(.-X))
end

function relu(X)
    return max.(0,X)
end

function dσ(z)
    return σ(z).*(1 .-σ(z))
end

function drelu(z)
    return z.*(z.>0)
end
