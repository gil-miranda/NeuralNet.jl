using MLDatasets: MNIST
using Plots

include("src/NeuralNet.jl")

### Preparando dados

xtrain, ytrain = MNIST.traindata();
xtest, ytest = MNIST.testdata();


ytrain_OneHot = Flux.onehotbatch(ytrain,0:9);
ytest_OneHot = Flux.onehotbatch(ytest,0:9);


xtrain_vec = zeros(28*28,10000)
xtest_vec = zeros(28*28,10000)

for i in 1:size(xtest)[3]
    xtrain_vec[:,i] = vec(xtrain[:,:,i])
    xtest_vec[:,i] = vec(xtest[:,:,i])
end

nn = NeuralNet.NeuralNetwork([28*28,32,32,16,10],"sigmoid");

train_data = NeuralNet.train!(nn, xtrain_vec, ytrain_OneHot, batch_size=20, epochs=100)

NeuralNet.model(nn, xtrain_vec[:,1])
NeuralNet.backprop(nn, xtrain_vec[:,1:3], ytrain_OneHot[:,1:3])
xtrain_vec[1:784, 1:3]