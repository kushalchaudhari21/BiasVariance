## Description

This is an octave prototype to reflect on the general advice to avoid high bias/variance errors(the two most common errors that might affect performance of a trained model). The question is - **"When you test your hypothesis on a new set of examples, you find that it makes unacceptably large errors in its predictions. What should you try next?"** 

Following are some solutions:
* *Fix for **High Variance** problems:*                                   
 Get more training examples.    
 Try smaller set of features.     
 Try increasing lambda. 
* *Fix for **High Bias** problems:*                                   
 Try getting additional features.    
 Try adding polynomial features.     
 Try decreasing lambda.
 
To understand how these solutions fix high Bias/Variance refer to [this pdf](https://github.com/kushalchaudhari21/BiasVariance/blob/master/BiasVarianceInsight.pdf). 

To rule out half of these possibilities we make use of different machine learning diagnostic. One of the most effective diagnostic to debug the learning algorithms is to plot learning curves.
 
The prototype implements regularised linear regression model with one variable. Using diagonostic we figure it out that linear regression model is not a proper fit. We further train the model using polynomial regression which better fits the data. By running another diagnostic on this, we calculate the value of lambda to get a more generalised model. Thereby increasing the accuracy of the model on unseen examples. 

The dataset is divided into training set, test set and cross validation set. Training set(X,y) that the model will learn on. The parameters evaluated from training set are used to see how well they perform on the examples it hasn't seen(cross validation or test set). More precisely, cross validation set(Xval,yval) to determine the regularisation parameters. And test set(Xtest,ytest), to evaluate the performance of pre-trained model on unseen examples.

The complete details for the implementation and procedure can be found in [ex5](https://github.com/kushalchaudhari21/BiasVariance/blob/master/ex5.pdf).

## Sample Invocation

After running Octave cli, in the project directory input the following to execute the algorithm
```
ex5
```

## General Procedure

**1.**  It is better to visualise the training dataset before execution. 
![Visualising data](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/VisualiseData.png)

**2.** Implement algorithm for regularised linear regression cost function to get *J* and *gradients*. To gain any insights regarding the same refer to my [LinearRegression](https://github.com/kushalchaudhari21/LinearRegression) repository. Cost and gradient computed using vector theta = [1 ; 1] are:

![Cost Evaluation](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/CostFunctionOP.png)    
![Gradient Evaluation](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/GradientOP.png)  

**3.** Next, train the model with *fmincg* optimisation using file [trainLinearReg.m](https://github.com/kushalchaudhari21/BiasVariance/blob/master/Solution%20for%20ex5%20Andrew%20NG%20Machine%20learning/trainLinearReg.m). After this fit the regression line to the dataset.

![LinearFit](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/LinearFit.png) 

From the plot, model clearly has a high bias error. But unlike above example, it is not always possible to visualise the data and the model. Hence we plot the learning curves to debug the algorithm.

**4.** A learning curve plots training and cross validation error as a function of training set size. We train models for different training set sizes. For every model we get some *parameters* and *training errors*. We use each of these parameters to computer errors on the complete cross validation set. The following code might be more insightful:
```matlab
for i = 1:m
          % To obtain different training set sizes, you should use 
          %different subsets of the original training set X. Specifically, 
          %for a training set size of i, you should use the first i examples (i.e., X(1:i,:) and y(1:i)). 
          
          % NOTE- The training error here does not include the regularisation parameter
          % as we are merely interested in finding the errors and not minimising them.
          % So set lambda to zero while calculating training and cross validation errors to calculate true errors.
          
          t = trainLinearReg(X(1:i,:), y(1:i), lambda); %here lambda is set to zero in ex5.m while training
          error_train(i) = linearRegCostFunction(X(1:i,:), y(1:i), t, 0);
          error_val(i) = linearRegCostFunction(Xval, yval, t, 0);
          
endfor
```
![ErrorComparison](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/ErrorComparison.png) 
![LRLearningCurve](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/LRLearningCurve.png) 

You can observe that both the train error and cross validation error are high when the number of training examples is increased. This reflects a high bias problem in the model – the linear regression model is too simple and is unable to fit our dataset well.


**5.** To fix the poorly fitted model, we try using polynomial regression(i.e adding more features to fix high bias). Add more features using the higher powers of the existing feature X in the dataset. Specifically, when a training set X of size m × 1 is passed into the function, the function should return a m×p matrix X_poly, where column 1 holds the original values of X, column 2 holds the values of X.^2, column 3 holds the values of X.^3, and so on.
```
%Using broadcasting method for faster execution instead of using a "for loop".	   
p_v = 1:1:p;

    %Using built in broadcasting function
X_poly = bsxfun(@power, X, p_v);

    %Alternatively bitwise operator can be used which is mostly same as the bsxfun function.
%X_poly = X.^p_v;

%"for loop" as shown below can also be used but it is better to go for vectorised implementations.
%for j = 1 : p
%    X_poly(:,j) = X.^j;
%endfor
```

**6.** Next, train the model upto 8th degree polynomial and plot the regression fit along with its learning curve. Since we are using powers of X, we need to use normalisation.

Polynomial Regression fit:

![PolyRegressionFit.png](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/PolyRegressionFit.png)

Learning curve:

![PolyLearningCurve](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/Learning%20curve%20for%20poly%20fit.png)

The polynomial fit is very complex and even drops off at the extremes. From learning curve, Training error is low, but the cross validation error is high. There is a gap between the training and cross validation errors. Both these insights refer to a high variance problem in the model.

**7.** One way to combat the overfitting (high-variance) problem is to add regularization to the model. Auto select *lambda* to get the best fitted model. Try different values of λ in the range: {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} and plot errors as a function of these values.
```matlab
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    t = trainLinearReg(X, y, lambda);
    error_train(i) = linearRegCostFunction(X, y, t, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, t, 0);
endfor
```
![ValidationCurve](https://github.com/kushalchaudhari21/BiasVariance/blob/master/outputScreenshots/ValidationCurve.png)

In this figure, we can see that the best value of λ is around 3.

## Important Insights

* Always go for vectorised implementation if possible.
* To debug vectorised implementation it is insightful to print out sizes of matrices using octave function *whos*.
* The desirable split in dataset should approximately be - Training set(60%), Cross Validation Set(20%), Test Set(20%).
* Training and cross validation errors for learning curves should be found out using λ = 0 for training and cross validation set. 
* To get more insight regarding bias/variance diagnostic refer to [this video](https://www.coursera.org/lecture/machine-learning/diagnosing-bias-vs-variance-yCAup). 


