# Simple Linear Regression Project 


## Modelling the linear relationship between Sales and Advertising Dataset

&nbsp; &nbsp; &nbsp; &nbsp;


The contents of this project are divided into following topics which are listed as follows:- 




## Table of Contents


1.	Introduction
2.	License information
3.	Python libraries
4.	The problem statement
5.	Linear Regression
6.	Independent and dependent variable
7.	Simple Linear Regression (SLR)
8.	About the dataset
9.	Exploratory data analysis
10.	Mechanics of Simple Linear Regression
11.	Model slope and intercept term
12.	Making predictions
13.	Regression metrics for model performance
i.	    RMSE
ii.	    R2 Score
14.	Interpretation and conclusion
15.	Residual analysis
16.	Checking for Overfitting or Underfitting the data
17.	Simple Linear Regression - Model Assumptions
18.	References

&nbsp; &nbsp; &nbsp; &nbsp;

## 1.	Introduction

In this project, I build a Simple Linear Regression model to study the linear relationship between Sales and Advertising dataset for a dietary weight control product. I discuss the basics of linear regression and its implementation in Python programming language using Scikit-learn. Scikit-learn is the popular machine learning library of Python programming language. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 2.	License information

The work done in this Jupyter notebook is made available under the Creative Commons Attribution License 4.0. 

You are free to:

•	Share—copy and redistribute the material in any medium or format
•	Adapt—remix, transform, and build upon the material.

I have licensed this Jupyter notebook for general public. The work done in this project is for learning and demonstration purposes. 
  
&nbsp; &nbsp; &nbsp; &nbsp;


## 3.	Python libraries

I have Anaconda Python distribution installed on my system. It comes with most of the standard Python libraries I need for this project. The basic Python libraries used in this project are:-

 •	Numpy – It provides a fast numerical array structure and operating functions.
 
 •	pandas – It provides tools for data storage, manipulation and analysis tasks.
 
 •	Scikit-Learn – The required machine learning library in Python.
 
 •	Matplotlib – It is the basic plotting library in Python. It provides tools for making plots. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 4.	The problem statement

The aim of building a machine learning model is to solve a problem and to define a metric to measure model performance. So, first of all I have to define the problem to be solved in this project.
As described earlier, the problem is to model and investigate the linear relationship between Sales and Advertising dataset for a dietary weight control product. I have used two performance metrics RMSE (Root Mean Square Value) and R2 Score value to compute our model performance.


&nbsp; &nbsp; &nbsp; &nbsp;

## 5.	Linear Regression

Linear Regression is a statistical technique which is used to find the linear relationship between dependent and one or more independent variables. This technique is applicable for Supervised Learning Regression problems where we try to predict a continuous variable.
Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. In this project, I employ Simple Linear Regression technique where I have one independent and one dependent variable. It is the simplest form of Linear Regression where we fit a straight line to the data.


&nbsp; &nbsp; &nbsp; &nbsp;

## 6.	Independent and Dependent Variables

In this project, I refer Independent variable as Feature variable and Dependent variable as Target variable. These variables are also recognized by different names as follows: -


### Independent variable

Independent variable is also called Input variable and is denoted by X. In practical applications, independent variable is also called Feature variable or Predictor variable. We can denote it as: -
Independent or Input variable (X) = Feature variable = Predictor variable 


### Dependent variable

Dependent variable is also called Output variable and is denoted by y. Dependent variable is also called Target variable or Response variable. It can be denote it as follows: -

Dependent or Output variable (y) = Target variable = Response variable


&nbsp; &nbsp; &nbsp; &nbsp;

## 7.	Simple Linear Regression (SLR)

Simple Linear Regression (or SLR) is the simplest model in machine learning. It models the linear relationship between the independent and dependent variables. 

In this project, there is one independent or input variable which represents the Sales data and is denoted by X. Similarly, there is one dependent or output variable which represents the Advertising data and is denoted by y. We want to build a linear relationship between these variables. This linear relationship can be modelled by mathematical equation of the form:-

				Y = β0   + β1*X    -------------   (1)

In this equation, X and Y are called independent and dependent variables respectively, 
β1 is the coefficient for independent variable and
β0 is the constant term.
β0 and β1 are called parameters of the model.
 
For simplicity, we can compare the above equation with the basic line equation of the form:-
 
        			y = ax + b       ----------------- (2)
        

We can see that 


slope of the line is given by, a =  β1,  and

intercept of the line by b =  β0. 

In this Simple Linear Regression model, we want to fit a line which estimates the linear relationship between X and Y. So, the question of fitting reduces to estimating the parameters of the model β0 and β1. 

 
### Ordinary Least Square Method (OLS)


As I have described earlier, the Sales and Advertising data are given by X and y respectively. If we draw a scatter plot of X and y, then it resembles the diagram which looks like as follows:-

&nbsp; &nbsp; &nbsp; &nbsp;

![](Images/Scatter%20plot%20of%20X%20and%20y.png)



 
Now, our task is to find a line which best fits the above scatter plot. This line will help us to predict the value of any Target variable for any given Feature variable. This line is called regression line. 
We can define an error function for any line. Then, the regression line is the one which minimizes the error function. Such an error function is also called a Costor a Cost function. 

&nbsp; &nbsp; &nbsp; &nbsp;


### Cost Function



We want the above line to resemble the dataset as closely as possible. In other words, we want the line to be as close to actual data points as possible. It can be achieved by minimizing the vertical distance between the actual data point and fitted line. We calculate the vertical distance between each data point and the line. This distance is called the residual. So, in a regression model, we try to minimize the residuals by finding the line of best fit. 


&nbsp; &nbsp; &nbsp; &nbsp;

Diagrammatic representation of residuals is given below. In this diagram, the residuals are represented by the vertical dotted lines from actual data points to the line.


&nbsp; &nbsp; &nbsp; &nbsp;

![](Images/Diagrammatic%20Representation%20of%20Residuals.png)


We can try to minimize the sum of the residuals, but then a large positive residual would cancel out a large negative residual. For this reason, we minimize the sum of the squares of the residuals. 

Mathematically, we denote actual data points by yi and predicted data points by ŷi. So, the residual for a data point i would be given as 
				
       				
				di = yi -  ŷi
				

Sum of the squares of the residuals is given as:


				 D = Ʃ di2       for all data points

This is the Cost function. It denotes the total error present in the model which is the sum of the total errors of each individual data point. We can represent it diagrammatically as follows:-

&nbsp; &nbsp; &nbsp; &nbsp;


![](Images/Ordinary%20Least%20Squares.png)


We can estimate the parameters of the model β0 and β1 by minimize the error in the model by minimizing D. Thus, we can find the regression line given by equation (1).  This method of finding the parameters of the model and thus regression line is called Ordinary Least Square Method.

&nbsp; &nbsp; &nbsp; &nbsp;


## 8.	About the dataset

The data set has been imported from the econometrics website with the following url-


http://www.econometrics.com/intro/sales.htm


This data set contains Sales and Advertising expenditures for a dietary weight control product. It contains monthly data for 36 months. 
The variables in this data set are Sales and Advertising.

&nbsp; &nbsp; &nbsp; &nbsp;


## 9.	Exploratory data analysis

First, I import the dataset into the dataframe with the standard read_csv () function of pandas library and assign it to the df variable. Then, I conducted exploratory data analysis to get a feel for the data.
I checked the dimensions of dataframe with the shape attribute of the dataframe. I viewed the top 5 rows of the dataframe with the pandas head() method. I viewed the dataframe summary with the pandas info() method and descriptive statistics with the describe() method. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 10.	Mechanics of Simple Linear Regression

The mechanics of Simple Linear Regression model starts with splitting the dataset into two sets – the training set and the test set. We instantiate the regressor lm and fit it on the training set with the fit method. In this step, the model learned the correlations between the training data (X_train, y_train). 
Now the model is ready to make predictions on the test data (X_test). Hence, I predict on the test data using the predict method. 

&nbsp; &nbsp; &nbsp; &nbsp;


## 11.	Model slope and intercept term

The model slope is given by lm.coef_ and model intercept term is given by lm.intercept_. The estimated model slope and intercept values are 1.60509347 and  -11.16003616.

So, the equation of the fitted regression line is

y = 1.60509347 * x - 11.16003616  


&nbsp; &nbsp; &nbsp; &nbsp;

## 12.	Making predictions

I have predicted the Advertising values on first five 5 Sales datasets by writing code

		lm.predict(X) [0:5]  

If I remove [0:5], then I will get predicted Advertising values for the whole Sales dataset.

To make prediction, on an individual Sales value, I write

		lm.predict(Xi)

where Xi is the Sales data value of the ith observation.

&nbsp; &nbsp; &nbsp; &nbsp;



## 13.	Regression metrics for model performance

Now, it is the time to evaluate model performance. For regression problems, there are two ways to compute the model performance. They are RMSE (Root Mean Square Error) and R-Squared Value. These are explained below:-  

###	i.	RMSE

RMSE is the standard deviation of the residuals. So, RMSE gives us the standard deviation of the unexplained variance by the model. It can be calculated by taking square root of Mean Squared Error.
RMSE is an absolute measure of fit. It gives us how spread the residuals are, given by the standard deviation of the residuals. The more concentrated the data is around the regression line, the lower the residuals and hence lower the standard deviation of residuals. It results in lower values of RMSE. So, lower values of RMSE indicate better fit of data. 

###	ii.	R2 Score

R2 Score is another metric to evaluate performance of a regression model. It is also called coefficient of determination. It gives us an idea of goodness of fit for the linear regression models. It indicates the percentage of variance that is explained by the model. Mathematically, 


R2 Score = Explained Variation/Total Variation


In general, the higher the R2 Score value, the better the model fits the data. Usually, its value ranges from 0 to 1. So, we want its value to be as close to 1. Its value can become negative if our model is wrong.


&nbsp; &nbsp; &nbsp; &nbsp;

## 14.	Interpretation and Conclusion

The RMSE value has been found to be 11.2273. It means the standard deviation for our prediction is 11.2273. So, sometimes we expect the predictions to be off by more than 11.2273 and other times we expect less than 11.2273. So, the model is not good fit to the data. 

In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value >= 0.7, then the model is good enough to deploy on unseen data whereas if R2 score value < 0.7, then the model is not good enough to deploy. Our R2 score value has been found to be .5789. It means that this model explains 57.89 % of the variance in our dependent variable. So, the R2 score value confirms that the model is not good enough to deploy because it does not provide good fit to the data.

&nbsp; &nbsp; &nbsp; &nbsp;


## 15.	Residual analysis

A linear regression model may not represent the data appropriately. The model may be a poor fit to the data. So, we should validate our model by defining and examining residual plots.

The difference between the observed value of the dependent variable (y) and the predicted value (ŷi) is called the residual and is denoted by e. The scatter-plot of these residuals is called residual plot.

If the data points in a residual plot are randomly dispersed around horizontal axis and an approximate zero residual mean, a linear regression model may be appropriate for the data. Otherwise a non-linear model may be more appropriate.

If we take a look at the generated ‘Residual errors’ plot, we can clearly see that the train data plot pattern is non-random. Same is the case with the test data plot pattern.
So, it suggests a better-fit for a non-linear model. 

&nbsp; &nbsp; &nbsp; &nbsp;


## 16.	Checking for Overfitting and Underfitting

I calculate training set score as 0.2861. Similarly, I calculate test set score as 0.5789. 
The training set score is very poor. So, the model does not learn the relationships appropriately from the training data. Thus, the model performs poorly on the training data. It is a clear sign of Underfitting. Hence, I validated my finding that the linear regression model does not provide good fit to the data. 

Underfitting means our model performs poorly on the training data. It means the model does not capture the relationships between the training data. This problem can be improved by increasing model complexity. We should use more powerful models like Polynomial regression to increase model complexity. 

&nbsp; &nbsp; &nbsp; &nbsp;


## 17.	Simple Linear Regression - Model Assumptions


The Linear Regression Model is based on several assumptions which are listed below:-


i.	Linear relationship

ii.	Multivariate normality

iii.	No or little multicollinearity

iv.	No auto-correlation

v.	Homoscedasticity


### i.	Linear relationship

The relationship between response and feature variables should be linear. This linear relationship assumption can be tested by plotting a scatter-plot between response and feature variables.


### ii.	Multivariate normality

The linear regression model requires all variables to be multivariate normal. A multivariate normal distribution means a vector in multiple normally distributed variables, where any linear combination of the variables is also normally distributed.


### iii.No or little multicollinearity

It is assumed that there is little or no multicollinearity in the data. Multicollinearity occurs when the features (or independent variables) are highly correlated.


### iv.	No auto-correlation

Also, it is assumed that there is little or no auto-correlation in the data. Autocorrelation occurs when the residual errors are not independent from each other.


### v.	Homoscedasticity

Homoscedasticity describes a situation in which the error term (that is, the noise in the model) is the same across all values of the independent variables. It means the residuals are same across the regression line. It can be checked by looking at scatter plot.

&nbsp; &nbsp; &nbsp; &nbsp;


## 18.	 References


The concepts and ideas in this project have been taken from the following websites and books:-


 i.  Machine learning notes by Andrew Ng
 
 ii. https://en.wikipedia.org/wiki/Linear_regression
 
 iii.https://en.wikipedia.org/wiki/Simple_linear_regression
 
 iv  https://en.wikipedia.org/wiki/Ordinary_least_squares
 
 v.  https://en.wikipedia.org/wiki/Root-mean-square_deviation
 
 vi. https://en.wikipedia.org/wiki/Coefficient_of_determination
 
 vii https://www.statisticssolutions.com/assumptions-of-linear-regression/
 
 viii.Python Data Science Handbook by Jake VanderPlas
 
 ix. Hands-On Machine Learning with Scikit Learn and Tensorflow by Aurilien Geron
 
 x.  Introduction to Machine Learning with Python by Andreas C Muller and Sarah Guido


























