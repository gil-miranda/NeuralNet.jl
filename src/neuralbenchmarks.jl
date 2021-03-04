using Flux: onecold

function check_accuracy(NN::NeuralNetwork, X, Y, ϵ = 0.01)
    # n = size(Y)[2]
    # acc = sum(abs.((Y-predict(NN, X))./Y) .<= ϵ)
    # return acc/n
    acc = 0
    for i in 1:size(Y)[2]
        x = onecold(predict(NN,X[:,i])).-1
        y = Y[i]
        if x[1] ≈ y
            acc += 1
        end
    end
    acc /= size(Y)[2]
    return acc
end
