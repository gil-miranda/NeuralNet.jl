function check_accuracy(NN::NeuralNetwork, X, Y, ϵ = 0.01)
    n = size(Y)[2]
    acc = sum(abs.((Y-model(NN, X))./Y) .<= ϵ)
    return acc/n
end
