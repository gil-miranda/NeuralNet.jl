function loss()
    mse(ŷ, y) = loss_mse(ŷ, y)
    quadratic(ŷ,y) = loss_quadratic(ŷ, y)
    ce(ŷ, y) = loss_cross_entropy(ŷ, y)
    exp(ŷ, y) = loss_exponential(ŷ, y)
    () -> (mse, quadratic, ce, exp)
end

function ∂loss()
    mse(ŷ, y) = ∂loss_mse(ŷ, y)
    quadratic(ŷ,y) = ∂loss_quadratic(ŷ, y)
    ce(ŷ, y) = ∂loss_cross_entropy(ŷ, y)
    exp(ŷ, y) = ∂loss_exponential(ŷ, y)
    () -> (mse, quadratic, ce, exp)
end

function loss_mse(Ŷ, Y)
    return 1/length(Y) * sum(abs2, Y .- Ŷ)
end

function ∂loss_mse(Ŷ, Y)
    return (2/size(Y)[2]) *  Ŷ .- Y
end

function loss_quadratic(Ŷ, Y)
    return 1/2 * sum(abs2, Y .- Ŷ)
end

function ∂loss_quadratic(Ŷ, Y)
    return Ŷ .- Y
end

function ∂loss_cross_entropy(Ŷ, Y)
    return 0
end

function ∂loss_cross_entropy(Ŷ, Y)
    return 0
end

function ∂loss_exponential(Ŷ, Y)
    return 0
end

function ∂loss_exponential(Ŷ, Y)
    return 0
end