# housing-regularized-regression-models
The research question of interest is to accurately predict a house's sale price in Ames, IA. The resulting model can be used by buyers, sellers, and realtors to obtain an estimate of what a house is worth. 

The code can be run interactively through a [Google Colab Notebook](https://colab.research.google.com/github/jhancuch/housing-regularized-regression-models/blob/main/housing-price-predictions.ipynb).

## Research design and modeling methods
The research design is to use regularized linear regression models. These models contain all the predictors we deem having any potential predictive power for SalePriceLn (through correlation heat maps) and then constrains/shrinks the coefficient estimates towards zero. By shrinking their coefficient, this reduces the models variance. The most popular are Ridge, Lasso, and ElasticNet. I end up generating a Ridge model, a Lasso Model, an ElasticNet model, and a tuned ElasticNet model. The general concept of wy regularized linear regressions may be preferred over OLS is due to the bias-variance trade off. As a parameter (discussed later) increases, the flexability of the fit decreases as seen through the constrained/shrinking coefficients leading to decrease variance but increased bias.

For each of these models, the target feature is SalePriceLn. By taking the natural log of SalePrice, we can shift the distribution to a more normal distribution and reduce the effect of outliers. This helps remove some of the non-linearity in the target variable itself.

In the preprocessing stage, I handle outliers, miscodings, missing values, dummy variable generation, and correlation/multicollinearity. 

Ridge regression is very similar to least squares, except that the coefficients are estimated by minimizing via a tradeoff between RSS and a shrinkage penalty. The tuning parameter (alpha or lamda) is used to control the tradeoff. If the parameter is 0, then the penalty term has no effect and it is an OLS regression. If the parameter approaches infinity, the impact of the shrinkage penatly increases and the coefficients approach zero. The parameter is determined using the cross validation method.

Lasso regression is similar to the Ridge regression but has some key differences. First, unlike the ridge regression that contains all the potential predictors and constrains towards zero but doesn't reach zero, Lasso allows coeffiencts to equal zero. Thus Lasso performs variable selection and can result in sparse models if the parameter is large enough.

In general, a rule of thumb is that Lasso is expected to perform better when a small number of independent variables have large coefficients and the remaining independent variables have small coefficients or equal to zero. Ridge is expected to perform better when the dependent variable is the function of many of the independent variables. 

ElasticNet model is a compromise between Ridge and Lasso. It introduces estic-net penalty that governs the trade-off between l1 (Lasso) and l2 (Ridge). By having a penalty that favors l1, you have a model that is more similar to a pure Lasso model but having Ridge characteristics in terms of shrinkage of the coeffiencts. And vice versa.

I additionally use GridSearchCV to examine the grid of hyperparameters associated with the ElasticNet model to find the optimal equation. I check the l1_ratio which determines the scaling between l2 penalty (Ridge) and l1 penalty (Lasso), number of cross validations, the number of alphas along the regularization path, and the max number of iterations. The set of parameters that had the best score was different than my initial parameters but the RMSE was similar, but higher by roughly .001. 

# Results and Evaluation
| Metric | Ridge | Lasso | ElasticNet | Tuned ElasticNet |
|---     | ---   | ---   |---         |---               |
| Cross Validation RMSE | 0.00164 | 0.00164 | 0.00164 | 0.00166 |
| Test Set RMSE | 0.18086 | 0.18607 | 0.18543 | 0.19252 |


discussion of why certain ones performed better

# Discussion
depending on which one performed better, discuss why I think its the case. Also talk about overfitting with tuned ElasticNet
