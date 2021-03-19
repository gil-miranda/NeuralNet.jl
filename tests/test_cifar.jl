using MLDatasets: FashionMNIST
using Plots
using Flux: onehotbatch, onecold, onehot
using Random
include("src/NeuralNet.jl")

### Preparando dados


xtrain, ytrain = FashionMNIST.traindata();
xtest, ytest = FashionMNIST.testdata();


ytrain_OneHot = onehotbatch(ytrain,0:9);
ytest_OneHot = onehotbatch(ytest,0:9);


xtrain_vec = zeros(28*28,60000)
xtest_vec = zeros(28*28,10000)

for i in 1:size(xtrain)[3]
    xtrain_vec[:,i] = vec(xtrain[:,:,i])
end

for i in 1:size(xtest)[3]
    xtest_vec[:,i] = vec(xtest[:,:,i])
end


nn = NeuralNet.NeuralNetwork([28*28,30,10],"sigmoid");

train_data = NeuralNet.train!(nn, xtrain_vec, ytrain_OneHot, batch_size=10, epochs=30, η=2.0)
# train_data = NeuralNet.train!(nn, xtrain_vec, ytrain_OneHot, batch_size=10, epochs=30, η=3.0)

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

acc = 0
for i in 1:10000
    x = onecold(NeuralNet.predict(nn,xtest_vec[:,i])).-1
    y = ytest[i]
    if x[1] ≈ y
        acc += 1
    end
end
acc /= 10000