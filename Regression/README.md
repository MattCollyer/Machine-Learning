#hello

Regression Writeup.


Hi Justin. I'm behind! 
This writeup probably won't be as robust as I would like. Sorry! Next time for sure...



###Where to begin??
Linear Regression I guess. This was fairly simple. Nothing stumped me here. 
The jump from linear regression to polynomial took time... but overall it was good.

The dynamic step size (backtracking line search) however took me a while to figure out. It was messy, ugly and confusing at first. (You helped me make it neater... ty). In the end it made sense and got cleaned up. That was one of the larger hurdles of all this I'd say.



My gradient descent:


```python
def gradient_descent(W, X, Y, tolerance, iterations, beta):
  #Where the magic happens. 
  #Returns Weights, Steps taken, MSE (RSS/N)
  #Uses backtracking line search to find step size.
  for t in range(1, iterations+1):
    step_size = 1.0
    derivative_weights = np.zeros(X.shape[0]-1)
    y_hat = X @ W
    errors = y_hat - Y
    derivative_weights = X.T @ errors
    gradient_mag_squared = np.dot(derivative_weights, derivative_weights)
    current_cost = cost(W, X, Y)
    while( cost(W - derivative_weights * step_size, X, Y) > (current_cost - ((step_size/2) * gradient_mag_squared))) :
      step_size *= beta
    W = W - derivative_weights * step_size
    if((t >= iterations) or (gradient_mag_squared < tolerance ** 2)):
      return({'Weights': W, 'Steps':t, 'MSE':sum([e**2 for e in errors])/X.shape[0]})
```
Pretty nice.


###Data time
Cleaning data is something I've done a thousand times. It ain't fun, but it aint hard. Just tedious... so tedious. 

I focused mainly on the columns "Land SF", "TotalFinishedArea" and "TotalAppraisedValue".



```python
data = pd.read_csv('Data/Housing/Real_Estate_Sales_730_Days.csv')
data = data[['LandSF','TotalFinishedArea','TotalAppraisedValue']]
data = data.dropna()
data = data[data > 0]
data = data[data['TotalAppraisedValue'] < 5000000]
data = data[data['LandSF'] < 350000]
shuffle_divide_export(data)

```
Here's the main cleaning itself. Nothing too crazy. Grab the columns I like, drop the empty rows, filter outliers, shuffle divide and export. I came up with those cutoff values just by trial and error. I would graph them and see the super outliers. In the end, there still appears to be some outliers, but not nearly as out there as the ones I cut off. (I could've done some stats stuff to take out the real outliers... but I am a lazy man who does not like statistics... next time for sure though)


Ok! Now we've got three beautiful csv's. Training, validation and test.


Now for the not fun stuff.
I mean.. fun ! YEAH.

###What the hell am I gonna choose ?
####both

First, I wanted to choose BOTH landSF and totalFinishedArea against the totalAppraisedValue

So I set my poor poor machine to the task. I let it run up to a polynomial of EIGHT (Each!).

For the training with both of them, I got this:
```
8th degree 
MSE: 0.0006038654565721106,
Steps: 14140
 TOTAL APPRAISED = 1.07878098e-03*LANDSF^8 + 8.11974341e-01 *LANDSF^7 - 5.87133491e-01 *LANDSF^6 + 6.59019988e-01*LANDSF^5 + 1.12655324 * LANDSF^4+ 7.86595013e-01 *LANDSF^3 + 4.44828851e-02 *LANDSF^2 -8.46061920e-01*LANDSF -1.75772269*TFA^8 + 3.19034542e-01*TFA^7 + 4.81801337e-01*TFA^6 -5.02154833e-01*TFA^5-7.00213023e-01*TFA^4 -4.49279353e-01*TFA^3 -6.39765155e-02*TFA^2 +3.30600966e-01*TFA + 7.02834290e-01

```
 boyo boy is dat ugly. 
Welp. Here's the graph. 
![BothTrain]("/Regression Graphs/BothTrain.png")

Ok lets see how it works with the test and validation sets:

TEST MSE: 0.0029523007884549876

VALIDATION MSE: 0.003019297713348949

Eh. not great on the test + validation. those outliers.....
I next wanted to try one variable at a time.

####LandSF only


Land SF against Total Appraised Value. Training yielded me this:
```
6th degree
24390 Steps
'MSE': 0.0014746153594685303
Total Appraised Value =  0.00585349 * LandSF^6 + 0.61867204* LandSF^5 +  2.17337952* LandSF^4 -2.83073102* LandSF^3  -2.37963248* LandSF^2 + 0.11841184* LandSF + 2.63007589
```

Test
MSE is 0.004436810105258282
Validation
MSE is 0.005201322387266734

Not great.

####Total Finished Area only
Total Finished Area against Total Appraised Value. Training yielded me this:

```
degree: 8
Steps: 11693
MSE': 0.0007063312674572078
Total Appraised Value = 0.00838731 * TFA^8 + 1.07213719 * TFA^7 - 0.9134273 * TFA^6 + 0.59034817* TFA^5 + 1.33027531* TFA^4 + 1.02969104 * TFA^3 + 0.17254856* TFA^2 -0.90578411 * TFA -2.02932659
```

Test
MSE is 0.0035142796870517015
Validation
MSE is 0.002587949461774681








##Conclusion
Alrighty. It is now the moment you've been patiently awaiting. How will Matt truly show me he knows his stuff? Sure... he can code it. But does he understand it? No more hand waving. Whats really the conclusions he can draw here? 

Well prepare for more hand waving. I'm impatient and wanna get caught up !!!
I really gave my computer a hard time here, especially with the multivariate. 
Surprisingly, the training model for BOTH yielded the lowest MSE. The graph sure looks like some outliers made it go a little funky towards the end... but overall it looked like it hit its mark.

Seems like my model for total finished area v total appraised value was better than the land sqft  v total appraised value.
Not really surprised there. That's about as far as my real estate intuition goes though.

I was suprised on how dissapointing my test and validations sets performed. Big mEH.
Lots to change if I were to do this again.

###What I didn't do.
Yeah. As I've said... I am one lazy sob. I shoud've taken the outliers out properly and I should have ACTUALLY saved all descents from my training to run them against the validation and test. I should have then decided if one had a lower MSE with the validation and test to maybe have changed my function... I mean... that's what they're there for. oh well. Next time... for sure ;P







