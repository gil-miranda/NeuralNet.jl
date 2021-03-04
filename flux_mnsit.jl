using MLDatasets: MNIST
using Plots
using Flux


xtrain, ytrain = MNIST.traindata();
xtest, ytest = MNIST.testdata();


ytrain_OneHot = Flux.onehotbatch(ytrain,0:9);
ytest_OneHot = Flux.onehotbatch(ytest,0:9);


xtrain_vec = zeros(28*28,60000)
xtest_vec = zeros(28*28,10000)

for i in 1:size(xtrain)[3]
    xtrain_vec[:,i] = vec(xtrain[:,:,i])
end

for i in 1:size(xtest)[3]
    xtest_vec[:,i] = vec(xtest[:,:,i])
end

xtrain_vec

flux_model = Chain(Dense(28*28, 30, sigmoid), Dense(30,10, sigmoid))

flux_loss(x,y) = Flux.mse(flux_model(x),y)

ps = params(flux_model)
dataset = [(xtrain_vec, ytrain_OneHot)];
opt = Descent(3.0)

cb = () -> println(flux_loss(xtrain_vec,ytrain_OneHot))

Flux.@epochs 30 Flux.train!(flux_loss, ps, dataset, opt, cb = cb)

acc = 0
for i in 1:10000
    x = Flux.onecold(flux_model(xtest_vec[:,1])).-1
    y = ytest[i]
    if x[1] ≈ y
        acc += 1
    end
end
acc /= 10000
flux_model(xtest_vec[:,2])