# Binary Classification | Logistic Regression.
## Matt Collyer
 
--- 
This project took me a while. My schedule and I have not been getting along lately.


So, I begin with an apology -- I AM SORRY THIS IS ATROCIOUSLY LATE.

### Assignment.
```
Gradient ascent

gradient_ascent(X : np.ndarray, 
                Y : np.ndarray, 
        step_size : float,
           epochs : int = 1000) -> np.ndarray
```

* Should take in X as numpy matrix. # of columns is the # of features, # of rows is the # of training examples
* Should take in Y as a numpy vector, # of elements is # of training examples, the values of Y are either 0 or 1
* Should return a numpy array of weights 1 + # of features long
* Should be stochastic. It should update the weights after each training example.
* Will test this on the sample data provided:
X = np.array([[9, 8], [5, 7], [8, 7], [4, 8], [0, 1], [9, 2], [8, 4], [4, 4], [3, 2], [3, 0]])
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

The other functions I’ll check for:

* Something to calculate f-score
* preprocessing

Try to train around 3-5 different models on the training set. For example:

* Vary the number of epochs
* Vary the step size
* Vary the number of words you include or how you count them
Or compare stochastic vs batch gradient ascent
Include in your write-up how each of your models performed on the test set and a conclusion.

### What I did.
Ok ! So to start, of course, I modified my gradient descent from the previous assignment into a gradient ASCENT. Time to rise.
The main difference was the implementation of sigmoid

Here's an example of my BATCH ascent.

```python
def batch_gradient_ascent(W, X, Y, tolerance, iterations, step_size, updates = True):
  for i in range(1, iterations+1):
      if(updates and i % 100 == 0):
        print("Step:",i)
      scores = X @ W
      probabilities = sigmoids(scores)
      partial_derivatives = X.T @ (Y - probabilities)
      W += step_size * partial_derivatives
      if(sum(np.square(partial_derivatives)) < tolerance ** 2):
        break
  return {'Weights': W, 'Steps Taken': i-1}
```
As we're working with probabilities -- 0 ->1, we're using our new friend sigmoid. 

Embarassingly enough, I got stuck for a long time on this because I forgot I needed to ADD the partials to my weights. BECAUSE WE'RE ASCENDING. duh. Big duh. Oh well.

Next up I needed to implement stochastic ascent. It's different than our batch -- instead of going through all the rows at once a hundreds/thousands of times, we go through each row individually a few (epoch) times.

It's a weird idea, and doesn't always work well -- but it can be loads faster. Even though an epoch takes longer than a single iteration, we don't do it nearly as many times.

![stochastic vs batch](https://www.researchgate.net/profile/Xudong-Huang-4/publication/328106221/figure/fig3/AS:678422925807621@1538760038610/Stochastic-gradient-descent-compared-with-gradient-descent.png)


Stochastic implementation
```python
def stochastic_gradient_ascent(W, X, Y, num_epochs, step_size):
  X, Y = shuffle(X,Y)
  for epoch in range(num_epochs):
    for i in range(X.shape[0]):
      score = W.T * X[i]
      y_hat = sigmoid(score)
      error = Y[i] - y_hat
      partial_derivatives = X[i] * error
      W += step_size * partial_derivatives
  return {'Weights': W, 'epochs': num_epochs}
```

I got confused for a while because I was not getting the same weights as before. Obviously I overlooked that this could ALSO be a pretty good boundary line. Another time consuming DUH.

I also implemented a function that calcualtes F-score.
This however I won't paste as its messy and rather simple. It just finds 
True/False positives & True/False negatives and shoves them together. 

Preprocessing the data was simple enough. I didn't implement any of my own ideas (like choosing the words I care about) however. I just counted the word occurences of all that appear more than once in the positive + neg file. 


### Where I got confused.
As I said before I got stuck on the stochastic for a while as I naiively assumed similar weights. Lesson learned. I also forgot to actually ascend. 
I got confused on what to do with the data for a little. I realized I needed to format the TEST data at the same time as the train data to make sure the test is using the same corpus as the TRAIN. 

### What I got.
I created 4 models. 2 batch, 2 stochastic.
I used two sets of params, doing one batch and one stochastic for each.

```python
  step_size = 0.05
  iterations = 1000
  epochs = 10
  tolerance = 0.1
  and 
  step_size = 0.03
  iterations = 2000
  epochs = 5
  tolerance = 0.1
```

### Outputs
```
collyer@lilith Binary Classification % python movie_reviews.py    

Parameters: step size 0.05 iteration max 1000 stochastic epochs 10 tolerance 0.1
Batch:
{'Weights': array([-277.83371648, -424.67581009,  308.2691353 , ..., -150.05000004,
       -150.05      , -150.15      ]), 'Steps Taken': 999}
fscore for training: 0.946066303809995
fscore against test set 1.0

Stochastic:
{'Weights': array([-5.65731998e-02, -1.01772757e-01,  2.71269135e+02, ...,
       -1.50050000e+02, -1.50050000e+02, -1.50150000e+02]), 'epochs': 10}
fscore for training: 0.8787241500175254
fscore against test set 0.9473684210526316


Changing Params!
Parameters: step size 0.03, iteration max 2000, stochastic epochs 5, tolerance 0.1
Batch:
{'Weights': array([-159.2644676 , -324.7011872 ,  261.2701259 , ..., -270.74999396,
       -300.05      , -300.15      ]), 'Steps Taken': 1999}

fscore for training: 0.9561484207298375
fscore against test set 1.0

Stochastic:
{'Weights': array([-1.19425070e-01, -3.96321150e-02,  2.50170126e+02, ...,
       -2.70749994e+02, -3.00050000e+02, -3.00150000e+02]), 'epochs': 5}
fscore for training: 0.9025210084033614
fscore against test set 0.972972972972973
```



### Conclusion.

Surprisingly got some pretty good models. In this case, albeit faster and pretty good, my stochastic models were not as good as my batch ones.
My models had 5237 weights. Pretty complex and cool!

My best model had a step size / learning rate of 0.03. It had a max iterations of 2000 with a tolerance of 0.1

It had an fscore of 0.9561484207298375 for the training set, and somehow a 1 for the test set. Woohoo!

Overall this made sense and I learned a lot. Now I understand how binary classification works, how probability gets implemented + sigmoid, as well as stochastic vs. batch.


