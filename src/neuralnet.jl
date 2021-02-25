"""
__author__ = "Gil Miranda"
__copyright__ = "MIT"
__credits__ = ["Gil Miranda", "Ricardo R. Rosa", "Beatriz Farah"]
__license__ = "MIT"
__version__ = "0.1"
__email__ = "gil@matematica.ufrj.br"
__status__ = "Production
"""

include("neuralactivations.jl")

"""
Estrutura para rede neural
"""
mutable struct NeuralNetwork
    layer_dimensions::Array
    θ::Dict
    ϕ::Dict
end

"""
Construtor da estrutura
"""
function NeuralNetwork(layer_dimensions, all_activations = "relu")
    parameters, activations = init_param(layer_dimensions, all_activations)
    NeuralNetwork([l[1] for l in layer_dimensions], parameters, activations)
end

"""
Inicializa os parâmetros da rede
"""
function init_param(layer_dimensions::Vector{Int64}, all_activations = "relu")
    param = Dict()
    actv = Dict()
    @assert(all_activations ∈ ACTV_FUNCS, "Please, use a valid activation function from $ACTV_FUNCS")
    for l = 2:length(layer_dimensions)
        param["W_$(l-1)"] = rand(layer_dimensions[l], layer_dimensions[l-1])*.1
        param["b_$(l-1)"] = zeros(layer_dimensions[l],1)
        actv["ϕ_$(l-1)"] = all_activations
    end
    return param, actv
end

function init_param(layer_dimensions::Vector{Any}, all_activations = "relu")
    param = Dict()
    actv = Dict()
    @assert(all_activations ∈ ACTV_FUNCS, "Please, use a valid activation function from $ACTV_FUNCS")
    for l = 2:length(layer_dimensions)
        if length(layer_dimensions[l-1]) > 1
            @assert(layer_dimensions[l-1][2] ∈ ACTV_FUNCS, "Please, use a valid activation function from $ACTV_FUNCS")

            param["W_$(l-1)"] = rand(layer_dimensions[l][1], layer_dimensions[l-1][1])*.1
            param["b_$(l-1)"] = zeros(layer_dimensions[l][1],1)
            actv["ϕ_$(l-1)"] = layer_dimensions[l-1][2]
        else
            param["W_$(l-1)"] = rand(layer_dimensions[l][1], layer_dimensions[l-1][1])*.1
            param["b_$(l-1)"] = zeros(layer_dimensions[l][1],1)
            actv["ϕ_$(l-1)"] = all_activations
        end
    end
    return param, actv
end

"""
Modelo de predição com rede neural, retorna o output da rede dada uma entrada
"""

function model(nn::NeuralNetwork, X)
    activation = X
    for l in 1:div(length(nn.θ),2)
        activation, _  = forward_activation(activation, nn.θ["W_$l"], nn.θ["b_$l"], nn.ϕ["ϕ_$l"])
    end
    return activation
end


function forward_linear(A,w,b)
    Z = w*A .+ b
    return Z
end

function forward_activation(A_pre, W, b, func = "sigmoid")
    @assert(func ∈ ACTV_FUNCS, "Please, use a valid activation function from $ACTV_FUNCS")
    if func == "sigmoid" || func == "σ"
        activation_func = σ
    elseif func == "relu"
        activation_func = relu
    end
    Z = forward_linear(A_pre,W,b)
    A = activation_func(Z)

    return A, Z
end

function loss(Ŷ, Y)
    return 1/length(Y) * sum(abs2, Y .- Ŷ)
end

function d_activation(z, actv)
    if actv == "σ" || actv == "sigmoid"
        deriv = dσ(z)
    elseif actv == "relu"
        deriv = drelu(z)
    elseif actv == "tanh"
        deriv = dtanh(z)
    else
        deriv = 0
    end
    return deriv
end

function GradDescent!(nn::NeuralNetwork, grads, η, batch_size = 1)
    for l=1:div(length(nn.θ),2)
        nn.θ["W_$l"] -= (η/batch_size) * grads["W_$l"]
        nn.θ["b_$l"] -= (η/batch_size) * grads["b_$l"]
    end
end

function backprop(nn::NeuralNetwork, X, Y)
    grad = Dict()

    activation = X
    activation_cache = Any[X]
    z_cache = Any[]
    L = div(length(nn.θ),2)

    for l in 1:L
        activation, z  = forward_activation(activation, nn.θ["W_$l"], nn.θ["b_$l"], nn.ϕ["ϕ_$l"])
        push!(z_cache, z)
        push!(activation_cache, activation)
    end
    if length(size(Y)) > 1
        delta = (2/size(Y)[2] * (activation_cache[end] - Y)) .* d_activation(z_cache[end], nn.ϕ["ϕ_$L"])
    else
        delta = (2/size(Y)[1] * (activation_cache[end] - Y)) .* d_activation(z_cache[end], nn.ϕ["ϕ_$L"])
    end
    grad["b_$L"] = sum(delta,dims=2)
    grad["W_$L"] = delta*activation_cache[end-1]'
    for l in L:-1:2
        z = z_cache[l-1]
        delta = nn.θ["W_$(l)"]'*delta .* d_activation(z, nn.ϕ["ϕ_$l"])
        grad["b_$(l-1)"] = sum(delta,dims=2)
        grad["W_$(l-1)"] = (delta*activation_cache[l-1]')
    end
    return grad
end

function train!(nn::NeuralNetwork, X, Y; η = 0.5, epochs = 100, verbose = true, batch_size = 2)
    N = size(Y)[2]

    costs = []
    epch = []
    accuracy = []
    for k in 1:epochs

        randomized_index = shuffle(1:N)
        xrandom = X[:, randomized_index]
        yrandom = Y[:, randomized_index]

        batch_index = []
        for k in 1:batch_size:div(N,batch_size)*batch_size
            push!(batch_index, (k,k+batch_size-1))
        end
        if N%batch_size != 0
            push!(batch_index, (div(N,batch_size)*batch_size, div(N,batch_size)*batch_size + N%batch_size))
        end

        for i in batch_index
            grads = []
            grads = backprop(nn, xrandom[:,i[1]:i[2]], yrandom[:,i[1]:i[2]])
            GradDescent!(nn, grads, η, batch_size)
        end

        cost = loss(model(nn,xrandom), yrandom)
        acc = check_accuracy(nn, xrandom, yrandom)

        push!(costs, cost)
        push!(epch, k)
        push!(accuracy, acc)

        if verbose == true
            println("Loss at epoch $k = $cost")
        end
    end

    return (costs=costs, epochs=epch, accuracy=accuracy)
end

##################### Gerando dados para função f(x,y) = x+y
function get_train_mod2(n)
    xtrain=rand(2,n)/2
    ytrain=sum(xtrain, dims=1)
    return xtrain, ytrain
end
