# Supporting Materials
## Table of Contents
- [Supporting Materials](#supporting-materials)
  - [Table of Contents](#table-of-contents)
- [1. Gradient Boosting](#1-gradient-boosting)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Draft #1](#12-draft-1)
  - [1.3 Draft #2 - metrics and initial value](#13-draft-2---metrics-and-initial-value)
  - [1.4 Draft #3 - other metrics](#14-draft-3---other-metrics)
    - [Gradient Descent](#gradient-descent)
    - [Gradient Descent in Gradient Boosting](#gradient-descent-in-gradient-boosting)
  - [1.5 Draft #4](#15-draft-4)
  - [1.6 Draft #5](#16-draft-5)
  - [1.7 Gradient Boosting in SAS](#17-gradient-boosting-in-sas)
    - [Loss functions](#loss-functions)
    - [**Variable Importance**](#variable-importance)
  - [1.8 Example](#18-example)
- [2. Decision and Regression Trees Splitting Criteria](#2-decision-and-regression-trees-splitting-criteria)
  - [Decision tree criteria](#decision-tree-criteria)
  - [Regression tree criteria](#regression-tree-criteria)
- [3. Logistic Regression](#3-logistic-regression)
  - [3.1 Weight of Evidence and Information Value](#31-weight-of-evidence-and-information-value)
    - [**Weight of Evidence**](#weight-of-evidence)
    - [**Information Value**](#information-value)
  - [3.2 Interactive Grouping in SAS](#32-interactive-grouping-in-sas)
    - [Algorithm](#algorithm)
- [4. Random Forests](#4-random-forests)
- [5. Transformations](#5-transformations)
  - [5.1 Maximize normality](#51-maximize-normality)
  - [5.2 Maximize correlation](#52-maximize-correlation)
- [6. Imputation](#6-imputation)
  - [Count](#count)
  - [Mean](#mean)
  - [Median](#median)
  - [Tree Surrogate](#tree-surrogate)
- [7. Variable Selection](#7-variable-selection)
  - [Variable Selection](#variable-selection)
  - [Variable Clustering](#variable-clustering)
- [Sources and Further Reading](#sources-and-further-reading)

# 1. Gradient Boosting

## 1.1 Motivation

Problem: we would like to predict someone's AGE based on them liking

- GARDENING - logic and a quick look at the data suggests older people will like gardening more than young ones
- VIDEO GAMES - reverse; younger will enjoy it more
- HATS - this will probably introduce some noise, since both young and old persons can enjoy hats

Let's make a regression tree with 1 split (use min. 3 samples in leaves). It splits it based on the *LikesGardening* variable.

Cool, it seems to capture some interesting info, but we'd like to also use some info from the *LikesVideo* variable!

Let's make a regression tree with 2 splits (min. 2 samples in leaves). The tree splits first on *LikesGardening* (T or F)*,* and then on *LikesVideo* (T or F) in one branch, but splits on *LikesHats* on another branch! This is an indicator of overfitting because it's trying to split on random noise.

*Lesson: a single regression tree fails to include predictive power from multiple overlapping regions of the feature space.*

Let's improve on this by getting the training errors: 

$$AGE - AGE_{predicted}$$

Now, let's fit a new regression tree with the residuals of the first tree as targets. Features are the same as before (*LikesGardening, LikesVideo, LikesHats*).

This new tree has in its root all residuals. It splits on *LikesVideo* (T or F) and doesn't include *LikesHats,* since it is able to consider HATS and VIDEOS with respect to **all samples**, unlike the previous tree which only considered each feature inside a small region of the input space, which allowed HATS (noise) to be selected as a splitting feature. Now we can add those error-correcting residuals to the predictions.

## 1.2 Draft #1

So now we have a first draft of what would eventually become a gradient boosting method. Let's summarize and formalize:

1. Fit a model to the data: 

    $$F_1(x)=y$$

2. Fit a model to the residuals:

    $$h_1(x)=y-F_1(x)$$

3. Create a new model by summing the two outcomes:

    $$F_2(x) = F_1(x)+h_1(x)$$

4. Repeat

    We generalize by fitting more models that correct the results of the previous model(s):

    $$F(x)=F_1(x)\rightarrow F_2(x)=F_1(x)+h_1(x)\rightarrow \dots\rightarrow F_M(x)=F_{M-1}(x)+h_{M-1}(x)$$

The task is to find/predict *h*, the residual, at each step.

Notice that the model used can be any, but usual choice are weak learners like decision/regression trees.

## 1.3 Draft #2 - metrics and initial value

The goal is to minimize ASE (average squared error), which is the *loss function*.

We initialize the 1st value to be the mean of the training target values. In terms of the loss function, this is:

$$F_0(x)=argmin_{\gamma}\sum_{i=1}^{n}L(y_i,\gamma)=argmin_{\gamma}\sum_{i=1}^{n}(\gamma - y_i)^2=\text{derive by }\gamma={1}/{n} \sum_{i=1}^{n}y_i$$

Where L is the loss function, y-s are training samples.

Recursively defining, we get:

$$F_{m+1}(x)=F_{m}(x)+h_{m}(x) = y, \text{for }  m \ge0$$

## 1.4 Draft #3 - other metrics

Instead of minimizing ASE, we could minimize absolute error.

Why? Suppose we have this example:

[Example](https://www.notion.so/1e889d3f2c94490ea2636419683f7058)

Note: in this case, F_0 (initial guess) was calculated as the minimum of absolute error, which is the median:

$$argmin_{\gamma}\sum_{i}|\gamma-y_i|$$

Squared error in case 1: 484

Squared error in case 2: 100

Imagine now we managed to lower the residuals in both cases by 1. Then the reduction in error is:

case 1: 484-(-21)^2 = 43

case 2: 100 - (-9)^2 = 19

By using ASE, a regression tree would concentrate on reducing the larger error, i. e. case 1.

What are the reductions if we use absolute error?

case 1: |-22| - |-21| = 1

case 2: |-10| - |-9| = 1

By using **absolute error**, moving predictions 1 unit closer to the target produces an equal reduction in the cost function.

Now suppose, we instead of training h_0 on the residuals of F_0, we train it on the gradient of the loss function L(y, F_0(x)) with respect to the prediction values of F_0(x)

We are therefore, training h0 on the cost reduction for each sample if the predicted value would become 1 unit closer to the observed value.

Absolute error: h_m considers the **sign** of every F_m residual                                              Squared error:  h_m considers the **magnitude** of every F_m residual

Samples in h_m (a regression tree in this case) are grouped into leaves. On those, we calculate **average gradient descent**, which is then scaled by some factor gamma so that

$$F_m + \gamma h_m$$

minimizes the loss function for the samples in each leaf (with different factor for each leaf!).

### Gradient Descent

Minimizing "nice" functions directly by differentiation is easy, but we run into problems when encountering harder examples. That's where gradient descent comes into play. Here we describe the method in algorithm form:

**Initialization**

Number of iterations: M

Starting point: S_0 = (0,0)

Step size: gamma

**Algorithm**

    For iteration i=1 to M:
    	1. Calculate gradient of L at point S(i-1)
    	2. "Step" in the direction of the greatest gradient descent 
    			(- gradient) with step size gamma
    	3. New point is: S(i) = S(i-1) - gamma * gradient(L(S(i-1)))

If gamma is small and M sufficiently large, the point after the final interation, S(M), will be the location of the loss functions minimum value.

**Potential additions**

- instead of a fixed number of iterations, we could move until a sufficiently small improvement is reached
- instead of a fixed step size, we could use e. g. line search to adjust it along the way

### Gradient Descent in Gradient Boosting

Here we combine the 2 described methods together.

1. Initialize with a constant value

    $$F_0(x)=argmin_{\gamma}\sum_{i=1}^{n}L(y_i, \gamma)$$

2. For iterations m = 1 to M
    1. Compute pseudo residuals

        $$r_{im}=-\frac{\partial L(y_i,F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}, i=1,\dots,n$$

    2. Fit a learner to the residuals
    3. (Compute step magnitude, gamma_m)
    4. Update:

        $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

## 1.5 Draft #4

We could also add *shrinkage*, from [0,1], which slows down convergence to reduce overfitting.

## 1.6 Draft #5

It is also possible to add random sampling of rows (observations) and columns (features) used in training, also in order to reduce overfitting.

## 1.7 Gradient Boosting in SAS

The general approach is the same as previously described: each time the data is used to grow a tree and the accuracy of the tree is computed. The successive samples are adjusted to accommodate previously computed inaccuracies.

### Loss functions

**Interval targets** define the residual using the **squared error** loss function.

**Binary targets** define the residual using the negative binomial log-likelihood loss function (**logistic loss**). This is a classic function used, since it heavily penalizes wrong predictions.

$$\text{Logistic loss function}=-[ylog(p)+(1-y)log(1-p)], \newline \text{ where }y\text{ is the target} \in\{0,1\} \text{, and }p\text{ is the predicted probability}$$

### **Variable Importance**

The **Gradient Boosting node** provides two approaches to evaluating the importance of a variable: 

- **Split-Based Variable Importance**

    *Uses the reduction in the sum of squares from splitting a node, summing over all nodes.*

    The split-based approach to evaluating the importance of a variable utilizes the idea that the **reduction in the sum of squares due to the model may be expressed as a sum over all nodes of the reduction in the sum of squares due to splitting the data in the node**. The credit for the reduction in a particular node goes to the variable used to split the node. The formula for variable importance **sums the credits over all splitting rules, scales the sums so that the largest sum is one, and takes a square-root to revert to linear units**.

    Formalized, the relative importance I of an input variable v in subtree T:

    $$I(v;T) = \sqrt{\sum_{\tau \in T}a(S_v,\tau)\Delta SSE(\tau)}$$

    Here we sum over nodes tau in subtree T. S denotes a splitting rule using the input variable v. a(S, tau) is the measure of agreement for the rule using v in node tau. It is equal to 1 is the primary splitting rule, and 0 if not.

    The second factor is the reduction in sum of squared errors from the predicted values:

    $$\Delta SSE(\tau) = SSE(\tau) - \sum_{b \in B(\tau)}SSE(\tau_b)$$

    Where SSE for an interval target Y is:

    $$\sum_{\text{obs i in }\tau}(Y_i - \hat{Y}(\tau))^2$$

    And for a target with J categories:

    $$\sum_{\text{obs i in }\tau}[\sum_{\text{class j of J classes}}(\delta_{ij}-\hat{p}_{j}(\tau))^2]$$

    Where delta is 1 in case Y_i is class j, otherwise 0, and p is average delta in training data in tau.

    The relative importance is computed with the training data and again computed with the validation data. If the validation data indicates a much lower importance than the training data, then the variable is over-fitting. The over-fitting usually occurs in an individual node that uses the variable to split with. The validation statistics in the branches will differ substantially from the training data statistics in such a node.

- **Observation-Based Variable Importance**

    *Uses the increase in a fit statistic due to making the observation values of a variable uninformative.*

    The observation-based approach to evaluating the importance of a variable entails reading a data set and applying the model several times to each observation, first to compute the standard prediction of the observation, and then once for each variable or pairs of variables being evaluated to compute a prediction in which variables being evaluated are made uninformative. The difference between the standard prediction and the prediction using an uninformative rendering of a variable is a measure of the influence of the variable. The **difference in a fit statistic computed first the standard way and then computed using the uninformative rendering of the variable is a measure of the overall importance of the variable**. To make a variable uninformative the Gradient Boosting node replaces its value in a given observation with the empirical distribution of the variable, and replaces the standard prediction with the expected prediction integrated over the distribution. In simpler words, to make a variable uninformative imagine making several copies of a given observation, altering the values of the variable being evaluated, and then taking the average of the standard predictions of these copies. The choice of altered values follows the empirical distribution: each value that appears in the input data set also appears among the imaginary copies of the observation, and appears with the same frequency. Notice that the uninformative prediction of an observation depends on the other observations in the input data set because of the empirical distribution.

## 1.8 Example

![Supporting%20Materials/Untitled.png](Supporting%20Materials/Untitled.png)

![Supporting%20Materials/Untitled%201.png](Supporting%20Materials/Untitled%201.png)

![Supporting%20Materials/Untitled%202.png](Supporting%20Materials/Untitled%202.png)

![Supporting%20Materials/Untitled%203.png](Supporting%20Materials/Untitled%203.png)

# 2. Decision and Regression Trees Splitting Criteria

Variables used for splitting in decision or regression trees are selected based on the *(im)purity* achieved by splitting on that variable. For a specific node and input, the decision or regression tree seeks the split that maximizes the measure of worth associated with the splitting criterion specified. Here we introduce main criteria used.

The impurity of a parent node *T* is defined as *I(T)*, a nonnegative number that is equal to zero for a pure node—in other words, a node for which all the observations have the same value of the response variable. Parent nodes for which the observations have very different values of the response variable have a large impurity. The general formula for quantifying gain from a split is:

$$I(node)-\sum_{b \in branches}P(b)I(b)$$

Here *I(node)* denotes the impurity measure used, and *P(b)* denotes the proportion of observations in the node that are assigned to branch *b*. The 3 most widely used impurity measures are described below.

## Decision tree criteria

- **Entropy**

    $$I(node)=-\sum_{\text{all classes}}P_{class}log_2(P_{class})$$

- **Gini**

    $$1-\sum_{i}^{classes}(\frac{\text{number of class i cases}}{\text{all cases in node}})^2$$

## Regression tree criteria

- **Variance (RSS)**

    $$I(node)=\sum_{\text{obs }i}(Y_i-\bar{Y})^2$$

    , where Y-bar is the average of Y in the node

# 3. Logistic Regression

The outcome of a logistic regression model is defined in terms of log-odds.

$$logit(p)=log(\frac{p}{1-p})=\beta_0+\beta_1x_1+\dots+\beta_nx_n$$

## 3.1 Weight of Evidence and Information Value

### **Weight of Evidence**

Weight of evidence (WOE) measures the relative risk of an attribute or group level. The value depends on the value of the binary target variable, which is either "non-event" (target = 0) or "event" (target = 1). An attribute's WOE is defined as follows:

![https://documentation.sas.com/api/docsets/emref/15.1/content/images/woe.png?locale=en](https://documentation.sas.com/api/docsets/emref/15.1/content/images/woe.png?locale=en)

, where ps are proportions of (non)events in total (non)events for that attribute.

[Example](https://www.notion.so/744af09b7dbe4533b5c069354c287eaf)

For the attribute Age<24, WOE = ln{(1440/18000)/(141/600)} = -1.07756.

A monotonically increasing or decreasing WOE is desired. WOE is inversely related to the event rate. Thus, when you use the Interactive Grouping node's interactive grouping editor, grouping so that the event rate is monotonically increasing or decreasing ensures a monotonically decreasing or increasing WOE, respectively.

The **Interactive Grouping** node generates and exports the WOE variables as interval variables.

### **Information Value**

The predictive power of a characteristic (that is, its ability to separate high-risk applicants from low-risk ones) is assessed by its Information Value or Gini Index. You can choose to use either method for variable selection, or you can choose not to perform variable selection at all.

The **Information Value is a weighted sum of the WOE of the characteristic's attributes**. The weight is the difference between the conditional probability of an attribute given an event and the conditional probability of an attribute given a non-event.

![https://documentation.sas.com/api/docsets/emref/15.1/content/images/information_value_0414.png?locale=en](https://documentation.sas.com/api/docsets/emref/15.1/content/images/information_value_0414.png?locale=en)

Information values can be **any real number**. Generally, the higher the information value, the more predictive a characteristic is likely to be. However, there is no way to determine an optimal cutoff value for characteristic selection, so ad hoc rules developed from anecdotal evidence are typically used.

For example, some practitioners use the following criteria:

- less than 0.02: unpredictive
- 0.02 to 0.1: weak
- 0.1 to 0.3: medium
- 0.3 +: strong

The **Interactive Grouping** node uses a default cutoff value of 0.1, but a user can specify another value.

## 3.2 Interactive Grouping in SAS

Characteristic=variable

Attribute=value of a variable

Before any binning or grouping is performed, binary, ordinal, and nominal variables have a fixed, finite number of attributes while interval characteristics have an infinite number of attributes.

Grouping refers to the process of purposefully censoring your data.

Advantages of grouping:

- It offers an **easier way to deal with rare classes and outliers with interval variables**.
- **Nonlinear dependencies can be modeled with linear models**.

**Initial characteristic screening** refers to the process of **assessing the strength of each characteristic individually as a predictor of performance**. The strength of a characteristic is gauged using four main criteria.

- **Predictive power of each attribute**. The **weight of evidence measure** (WOE) is used for this purpose.
- **Predictive power of the characteristic**. The **Information Value** or **Gini Statistic** is used for this.

### Algorithm

1. The Interactive Grouping node first performs **binning on the interval characteristic**. You can choose between two binning methods: **quantile** and **bucket**. The quantile method generates groups. The groups are formed by ranked quantities with **approximately the same frequency** in each group. The bucket method generates groups by dividing the data into **evenly spaced intervals** that are based on the difference between the maximum and minimum values.
2. A **decision tree** model is fitted for **each characteristic**. You can choose among four grouping methods: 
    - optimal criterion
        - Uses one of two criteria:
            - reduction in entropy measure or
            - the p-value of the Pearson Chi-square statistic
    - quantile
        - generates groups with approximately the same frequency in each group
    - monotonic event rate
        - Generates groups that result in a monotonic distribution of event rates across all attributes. The event rate is equal to P(event | attribute)—the conditional probability of an event given that an applicant exhibits a particular attribute.
    - constrained optimal
        - finds an optimal set of groups and simultaneously imposes additional constraints, as specified in the node property panel settings
3. After the characteristics have been grouped, the Interactive Grouping node computes the WOE for each attribute for every characteristic.

Note: The Interactive Grouping node uses normalized values of categorical variables. It considers two categorical values the same if the normalized values are the same. Normalization removes any leading blank spaces from a value, converts lowercase characters to uppercase characters, and truncates the value to 32 characters.

# 4. Random Forests

![Supporting%20Materials/ensemble_methods.png](Supporting%20Materials/ensemble_methods.png)

via [https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)

Random forest is an ensemble learning method that aggregates results from learning multiple weak learners (usually trees). It is different from boosting methods in that it parallelizes learners, while boosting is done sequentially. It introduces 2 sources of randomness that control for overfitting:

- each learner draws a random sample through bootstrapping with replacement
- each learner uses a random subset of inputs, instead of all available inputs

To predict an observation, RF assigns the observation to a single leaf in each decision tree in the forest. That leaf is used to make a prediction based on the tree that contains the leaf. Finally, RF averages the predictions over all the trees. For an interval target, the prediction in a leaf equals the average of the target values among the bagged training observations in that leaf. For a nominal target, the posterior probability of a target category equals the proportion of that category among the bagged training observations in that leaf. The predicted nominal target category is the category with the largest posterior probability. In case of a tie, the first category that occurs in the training data is the prediction.

**Out-of-bag error**

Since not all samples are used in training a learner, the leftover samples are used to predict unseen samples, which produces "out-of-bag" error.

# 5. Transformations

Following transformation methods can be found within the properties of the **Transform variables node**. They are part of "best power" transformations, a subset of the general class of transformations that are known as Box-Cox transformations. Variables are scaled before they are transformed by the power transformations. Variables are scaled so that their values range from 0 to 1. The scaled value of variable x is equal to (x - min) / (max - min).  All of the "Best Power" transformations evaluate the transformation subset listed below, and choose the transformation that has the best results for the specified criterion. In the list below, x represents the transformed variable.

- x
- ln(x)
- sqrt(x)
- e^x
- x^1/4
- x^2
- x^4

## 5.1 Maximize normality

The **Maximize Normality** power transformation is useful when you want to normalize a variable that has a skewed distribution or an overly peaked or flat distribution. Skewness measures the deviation of the distribution from symmetry. A skewed distribution has a heavy tail or extreme values located on one side of the distribution. If the skewness statistic for a variable is clearly different from 0, then the distribution is asymmetrical. Normal distributions are perfectly symmetrical (bell shaped). The degree of flatness is measured by kurtosis, where a normal distribution (in SAS) has a value of 1.

## 5.2 Maximize correlation

The **Maximize Correlation with Target** power transformation enables you to straighten (linearize) the relationship between an interval target and an interval input variable. Although neural networks and decision tree methods are quite capable of predicting nonlinear relationships, increasing the correlation with the target tends to simplify the relationship between the target and the input variable, making it more easily understood and communicable to others. The overall model fit is also often improved by linearizing the data.

# 6. Imputation

## Count

Missing values are replaced with the modal value (used for categorical inputs).

## Mean

Replace missing interval variable values with the arithmetic average, calculated as the sum of all values divided by the number of observations. The mean is the most common measure of a variable's central tendency; it is an unbiased estimate of the population mean. The mean is the preferred statistic to use to replace missing values if the variable values are at least roughly symmetric (for example, a bell-shaped normal distribution).

## Median

Replace missing interval variable values with the 50th percentile, which is either the middle value or the arithmetic mean of the two middle values for a set of numbers arranged in ascending order. The mean and median are equal for a symmetric distribution. The median is less sensitive to extreme values than the mean or midrange. Therefore, the median is preferable when you want to impute missing values for variables that have skewed distributions. The median is also useful for ordinal data.

## Tree Surrogate

Replace missing interval variable values with replacement values that are estimated by analyzing each input as a target. The remaining input variables are used as predictors. Because the imputed value for each input variable is based on the other input variables, this imputation technique may be more accurate than simply using the variable mean or median to replace the missing values. A surrogate rule is a back-up to the main splitting rule. When the main splitting rule relies on an input whose value is missing, the next surrogate is invoked.

# 7. Variable Selection

## Variable Selection

The Variable Selection node quickly identifies input variables which are useful for predicting the target variable.

It uses a forward stepwise least squares regression that maximizes the model R-square value. The following three step process is performed when you apply the R-Square variable selection criterion to a (binary) target (the last step is not applied if the target variable is non-binary):

- **Compute R2** — The squared correlation coefficient (simple R2) for each input variable is computed using a simple linear regression (or a one-way ANOVA for class variables) and compared to the default Minimum R2 of 0.005. If the R2 for an input is less than the cut-off-criterion, then the input variable is rejected. The R2 is the proportion of target variation explained by a single input variable; the effect of the other input variables is not included in it's calculation.
- **Forward Stepwise Regression** — After R2 for each variable, the remaining significant variables are evaluated using a forward stepwise regression. The sequential forward selection process starts by selecting the input variable that explains the largest amount of variation in the target. This is the variable that has the highest R2. At each successive step, an additional input variable is chosen that provides the largest incremental increase in the model R2. The stepwise process terminates when no remaining input variables can meet the Stop R2 criterion (the default value is 0.0005).
- **Logistic Regression for Binary Targets** — If the target is a binary variable, then a final logistic regression analysis is performed using the predicted values that are output from the forward stepwise selection as the independent input.

**Option: use Group variables**

Reduces class variables to group variables. The levels of the class variables are reduced based on the relationship of the variable to the target. The group class variables method can be viewed as an analysis of the ordered cell means to determine if specific levels of a class input can be collapsed into a single group. For example, suppose that you have a four level class input that explains 20% of the total variation in the target (R-square value = 0.2). The node first orders the input levels by the mean responses (target values). The ranking determines the order that the node considers the input levels for grouping. The node always works from top to bottom when considering the levels that can be collapsed.

[Example](https://www.notion.so/fb3514f6eb4c408b8724e5c1d2c0d166)

The node combines the first two levels (C and B) into a single group if the reduction in the explained variation is ≤5% of the target variable (R2 value of at least 19). If the C and B levels can be combined into a group, then the node determines if CB and D can be collapsed. If C and B cannot be combined, then the node determines if B and D can be combined into one group. The node stops combining levels when the R2 threshold is not met for all possible ordered, adjacent levels or when it reduces the original variable to two groups.

## Variable Clustering

Variable clustering is a useful tool for data reduction, such as choosing the best variables or cluster components for analysis. Variable clustering removes collinearity, decreases variable redundancy, and helps reveal the underlying structure of the input variables in a data set. 

Variable clustering divides numeric variables into disjoint or hierarchical clusters. The resulting clusters can be described as a linear combination of the variables in the cluster. The linear combination of variables is the first principal component of the cluster. The first principal components are called the cluster components. Cluster components provide scores for each cluster. The first principal component is a weighted average of the variables that explains as much variance as possible. The algorithm for Variable Clustering seeks the maximum variance that is explained by the cluster components, summed over all the clusters.

The cluster components are oblique, and not orthogonal, even when the cluster components are first principal components. In an ordinary principal component analysis, all components are computed from the same variables. The first principal component is orthogonal to the second principal component and to each other principal component. With the **Variable Clustering** node, each cluster component is computed from a different set of variables than all the other cluster components. The first principal component of one cluster might be correlated with the first principal component of another cluster. The **Variable Clustering** node performs a type of oblique component analysis.

**The algorithm**

It begins with all variables in a single cluster, then repeats the following steps:

1. A cluster is chosen for splitting. Depending on the options specified, the selected cluster has either
    - the smallest percentage of variation explained by its cluster component or
    - the largest eigenvalue that is associated with the second principal component.
2. The chosen cluster is split into two clusters by finding the first two principal components, performing an orthoblique rotation and assigning each variable to the rotated component with which it has the higher squared correlation.
3. Variables are iteratively reassigned to clusters to try to maximize the variance accounted for by the cluster components. This iterative reassignment of variables to clusters proceeds in two steps. 
    1. Nearest component sorting phase. In each iteration, the cluster components are computed, and each variable is assigned to the component with which it has the highest squared correlation. 
    2. A search algorithm in which each variable is tested to see whether assigning it to a different cluster increases the amount of variance explained. If a variable is reassigned during the search phase, the components of the two clusters involved are recomputed before the next variable is tested. 

Using default settings, the **Variable Clustering** node stops splitting when each cluster has only one eigenvalue greater than 1.

Instead of outputting cluster components from the node as inputs, there is a possibility to output the best variables from each cluster or even manually adjust which variables are selected. The best variable in each cluster is the variable that has the lowest 1–R2 Ratio.

# Sources and Further Reading

1. [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
2. [https://explained.ai/gradient-boosting/](https://explained.ai/gradient-boosting/)
3. [https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
4. [https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
5. [https://www.saedsayad.com/decision_tree.htm](https://www.saedsayad.com/decision_tree.htm)
6. [Example](https://datascience.stackexchange.com/a/23339)
7. [Splitting Criteria](https://support.sas.com/documentation/cdl/en/stathpug/68163/HTML/default/viewer.htm#stathpug_hpsplit_details02.htm)
8. [Transformation](https://documentation.sas.com/?docsetId=emref&docsetTarget=n0f3ix7imzm4xrn1773i0xuq47mj.htm&docsetVersion=14.3&locale=en#p1ks4t7pdyk23bn1xnslz8zwc64i)
9. [Variable Selection](https://documentation.sas.com/?docsetId=emref&docsetTarget=n1m7rvh6yyb3mmn0zavezsher4ml.htm&docsetVersion=14.3&locale=en)
10. [Variable Clustering](https://documentation.sas.com/?docsetId=emref&docsetTarget=p19e837tepmjz0n1hjt2gdk3sqfg.htm&docsetVersion=14.3&locale=en)
11. [Interactive grouping, WoE, IV](https://documentation.sas.com/?docsetId=emref&docsetTarget=p1qzwz7onopjqcn11uc04i18urg7.htm&docsetVersion=15.1&locale=en)