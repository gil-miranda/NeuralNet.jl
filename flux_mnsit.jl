using MLDatasets: MNIST
using Plots
using Flux


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


flux_model = Chain(Dense(28*28, 32, sigmoid), Dense(32,32, sigmoid), Dense(32,16, sigmoid), Dense(16,10,sigmoid))

flux_loss(x,y) = Flux.mse(flux_model(x),y)

ps = params(flux_model)
dataset = [(xtrain_vec', ytrain_OneHot')];
opt = ADAM(0.1)

cb = () -> println(flux_loss(xtrain_vec',ytrain_OneHot'))

Flux.@epochs 100 Flux.train!(flux_loss, ps, dataset, opt, cb = cb)
