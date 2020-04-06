#data #machinelearning

**Batch**
The number of examples that are used to calculate gradient in a single iteration in #gradient_descent. This could be the whole dataset, but for extremely large datasets this becomes inefficient. 

**SGD (Stochastic Gradient Descent)**
1 example is chosen at random to calculate the gradient. Given enough iterations, SGD works but is very noisy.

**Mini-batch SGD**
10-1,000 examples are chosen at random to calculate the gradient in one iteration. Reduces the noise compared to SGD but is still much more efficient than SGD itself.

# Links
[[20200212100944]] 20200212100944 - gradient descent
