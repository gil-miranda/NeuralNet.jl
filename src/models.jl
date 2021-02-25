
"""
f(x,y) = x+y
"""
function model_1(n_train = 100, n_test = 30)
    n_train = 100
    xtrain=rand(2,n_train)
    ytrain=sum(xtrain, dims=1)

    xtest=10rand(2,n_test)
    ytest=sum(xtest, dims=1)

    return (xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
end
