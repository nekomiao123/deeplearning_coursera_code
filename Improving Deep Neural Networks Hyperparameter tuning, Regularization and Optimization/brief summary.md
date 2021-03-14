# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

## Initialization

A well chosen initialization can:

- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error

### 1. Zero initialization

There are two types of parameters to initialize in a neural network:

- the weight matrices (W[1],W[2],W[3],...,W[L−1],W[L])
- the bias vectors (b[1],b[2],b[3],...,b[L−1],b[L])

Initialize all parameters to zeros

In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with n[l]=1 for every layer, and the network is no more powerful than a linear classifier such as logistic regression.

- The weights W[l] should be initialized randomly to break symmetry. 

- It is however okay to initialize the biases b[l] to zeros. Symmetry is still broken so long as W[l] is initialized randomly.

![image-20210312214228628](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210312214228628.png)

### 2. Random initialization

To break symmetry, lets intialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. But if we use a large random value, the result will be  very bad.

**Observations**:

- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when log(a[3])=log(0), the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm. 
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.

**In summary**:

- Initializing weights to very large random values does not work well. 
- Hopefully intializing with small random values does better. The important question is: how small should be these random values be? Lets find out in the next part!

![image-20210312214244669](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210312214244669.png)

### 3. He initialization

Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

**Hint**: This function is similar to the previous `initialize_parameters_random(...)`. The only difference is that instead of multiplying `np.random.randn(..,..)` by 10, you will multiply it by $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$, which is what He initialization recommends for layers with a ReLU activation. 

### 4. Conclusions

**What you should remember from this notebook**:

- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.



## Regularization



## Optimization

#### 1. Mini-batch-size

#### 2. Momentum

**Note** that:

- The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
- If β=0, then this just becomes standard gradient descent without momentum. 

**How do you choose β?**

- The larger the momentum β is, the smoother the update because the more we take the past gradients into account. But if β is too big, it could also smooth out the updates too much. 
- Common values for β range from 0.8 to 0.999. If you don't feel inclined to tune this, β=0.9 is often a reasonable default. 
- Tuning the optimal β for your model might need trying several values to see what works best in term of reducing the value of the cost function J.

#### 4. Adam

Some advantages of Adam include:

- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except α)