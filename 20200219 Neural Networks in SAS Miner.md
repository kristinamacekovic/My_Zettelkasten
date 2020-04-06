#data #SAS #neural_network

# Neural Networks Using EM

- most used type of NN is MLP = multilayer perceptron

- built out of *hidden units* (=neuron)

  - each receives a linear combination of variables
    - coefficients are called (synaptic) *weights*
    - an *activation function* transforms these combinations and outputs them to another unit that uses them as inputs
  - include a link function (in NN terms, an *activation function*)
    - default
      - interval target: hyperbolic tangent = shift and rescaling of logistic function
      - binary target: logit function

- they are arranged into *layers*

  - the 1st one is called the *input layer* with any number of inputs. It connects to a hidden layer, which connects to the final, *target layer*
  - there could be 1 or more hidden layers between the input and target layers
  - hidden layers can contain any number of hidden units

- the terminology is somewhat different than regression

  - intercept term = *bias term*
  - parameter estimates = *weight estimates*

- good for problems

  - where there is no known math formula to describe the relationship between the inputs and outputs
  - predicting is more important than explanation
  - there is a lot of training data

- weights are estimated with 

  - interval target: least-squares

  - binary target: maximum likelihood

    Maximize $\sum_{\text{primary cases}}log(\hat{p_i})+\sum_{\text{non-primary cases}}log(1-\hat{p}_i)$

  Interval target:

  ![Screenshot 2020-02-19 at 16.35.54](/Users/kristinamacekovic/Library/Application Support/typora-user-images/Screenshot 2020-02-19 at 16.35.54.png)

  Binary target:

  ![Screenshot 2020-02-19 at 16.36.27](/Users/kristinamacekovic/Library/Application Support/typora-user-images/Screenshot 2020-02-19 at 16.36.27.png)

  ![Screenshot 2020-02-19 at 16.41.43](/Users/kristinamacekovic/Desktop/Screenshot 2020-02-19 at 16.41.43.png)

  

Similarities to regression:

- missing values need to be imputed
- extreme or unusual values - mitigated by the tanh function which compresses everything to [-1,+1]

Differences from regression:

- nonnumeric inputs are less of a problem for NN than for regression
- easily accomodate nonlinear and nonadditive associations between inputs and the target (sometimes too easily, i. e. overfits)



- the method lacks a built-in method for selecting useful inputs
  - regression can be used before running NN for this
- optimizing complexity
  - each iteration of the optimization process is treated as a separate model
  - the iteration with the smallest value of the selected fit statistic is chosen as the final model (called *stopped training*)
    - balances overfitting
      - can be adjusted by changing the number of hidden units (manually with NN node, automatically by AutoNeural node)
- AutoNeural
  - automatically explores alternative network achitectures and hidden unit counts