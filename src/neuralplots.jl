function getPlot_accuracy(train_data)
    the_plot = plot(treino.accuracy,
             label="Accuracy",
             xlabel="Epochs",
             ylabel="Accuracy as %",
             title="Development of accuracy at each iteration");
    return the_plot
end

function getPlot_cost(train_data)
    the_plot = plot(treino.costs,
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
