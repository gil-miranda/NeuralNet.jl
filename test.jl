include("src/NeuralNet.jl")

model_1 = NeuralNet.model_1(100,30);

nn = NeuralNet.NeuralNetwork([2,1]);
train_data = NeuralNet.train!(nn, model_1.xtrain, model_1.ytrain, batch_size = 2, epochs = 30);


### Plotting

using Plots

function getPlot_accuracy(train_data)
    the_plot = plot(train_data.accuracy,
             label="Accuracy",
             xlabel="Epochs",
             ylabel="Accuracy as %",
             title="Development of accuracy at each iteration");
    return the_plot
end

function getPlot_cost(train_data)
    the_plot = plot(train_data.costs,
             label="Cost Function",
             xlabel="Epochs",
             ylabel="Cost",
             color="red",
             title="Development of cost at each iteration");
    return the_plot
end

function plot_nn(train_data)
    Plots.plot(getPlot_accuracy(train_data), getPlot_cost(train_data), layout = (2, 1), size = (800, 600))
end

plot_nn(train_data)

## Predicting

NeuralNet.model(nn, [100,500])
