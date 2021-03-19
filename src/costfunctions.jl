function loss()
    mse(ŷ, y, ∂) = loss_mse(ŷ, y, ∂)
    quadratic(ŷ, y, ∂ = false) = loss_quadratic(ŷ, y, ∂)
    ce(ŷ, y, ∂ = false) = loss_cross_entropy(ŷ, y, ∂)
    exp(ŷ, y, ∂ = false) = loss_exponential(ŷ, y, ∂)
    () -> (mse, quadratic, ce, exp)
end

function loss_mse(Ŷ, Y, ∂ = false)
    if ∂ == true
        if length(size(Y)) > 1
            return (2/size(Y)[2] * (Ŷ - Y))
        else
            return (2/size(Y)[1] * (Ŷ - Y))
        end
    else
        return 1/length(Y) * sum(abs2, Y .- Ŷ)
    end
end

function loss_quadratic(Ŷ, Y, ∂ = false)
    return 1/2 * sum(abs2, Y .- Ŷ)
end

function loss_cross_entropy(Ŷ, Y, ∂ = false)
    return 0
end

function loss_exponential(Ŷ, Y, ∂ = false)
    return 0
end