# Pytorch | Multiclass | Dropout.
## Matt Collyer
____
### Assignment:

* linear regression no pytorch
 -with L2 regularization
* linear regression pytorch manual (no built in optimizer/loss functions) 
-with L2 regularization
* linear regression pytorch automatic (concise version)
-with L2 regularization
* multiclass classification pytorch manual (no built in optimizer/loss functions)
-with dropout
* multiclass classification pytorch automatic (concise version)
-with dropout

### What I did.
Hey there Justin. Here is what I've got for this unit!
I have implemented L2 Ridge Regression into my linear regression, and have achieved a mild understanding of pytorch. Sweet.

The pytorch tutorial was fairly sraightforward. No major roadblocks. Very simple.

Writing the manual multiclass version was a bit trickier, and the 3 layer manual version with dropout was trickiest. Luckily your walkthrough made the process clear and very understandable. I feel it makes most sense to talk about the manual pytorch 3 layer version for this writeup, as it took the bulk of my time writing and, as I'll show later, yielded the strongest results.

I have been debating how much code I should walk through on this writeup. I've come to the conclusion that it might just be redundant and pointless (especially to you, Justin) to walkthrough most of it, so I've taken just one chunk to look at. 
The following code, I believe, is the most significant difference between the standard manual version and the manual multilayer. 

```python 
def softmax_regression(X, W_h1, W_h2, W_o, drop_h1, drop_h2, is_training = True):
  flat_X = X.reshape((-1, W_h1[0].shape[0])) 
  score = flat_X @ W_h1[0] + W_h1[1]
  H1 = relu(score)
  if is_training:
    H1 = dropout(H1, drop_h1)
  score = H1 @ W_h2[0] + W_h2[1]
  H2 = relu(score)
  if is_training:
    H2 = dropout(H2, drop_h2)
  score = H2 @ W_o[0] + W_o[1]
  return softmax(score)
```
In the softmax regression, we now send in multiple layers of weights + biases. We then walk through each layer, using ReLU (Rectified Linear Unit) to throw out all negatives and randomly (based on given probability) dropping out some neurons in the hidden layers. After the hidden layers we then get our score from the output layer, and finally get the softmax.

### Where I got confused
Well, unrelated, but L2 ridge regression stumped me much longer than dropout did. I would say that all of this manual multiclass / dropout / relu stuff made sense on its own, but combined with learning pytorch it definitely was intimidating. In the end it all made sense.

### What I made
I created 4 models, 2 manual no dropout, 2 manual with dropout.

Params:

```
  Model 1
  step_size = 0.1
  epochs = 3
  
  Model 2
  step_size = 0.01
  epochs = 2


  Model 3
  step_size = 0.1
  epochs = 3
  dropout_l1 = 0.2
  dropout_l2 = 0.5
  
  Model 4
  step_size = 0.01
  epochs = 2
  dropout_l1 = 0.4
  dropout_l2 = 0.6
```

### What I got

Manual models with no hidden layers
```
Manual model 1
Test Error: 
 Accuracy: 82.2%, Avg loss: 0.544177 

Manual model 2
Test Error: 
 Accuracy: 78.9%, Avg loss: 0.647163 
```
Manual models with hidden layers and dropout:

```
Model 3
Test Error: 
 Accuracy: 84.3%, Avg loss: 0.437244 

Model 4   <--- Winner!
Test Error: 
 Accuracy: 85.7%, Avg loss: 0.401186 

```
### Conclusion
Not too bad!
I am very surprised at how much better the multilayer + dropout models performed. My best model had 2 epochs, a step size of 0.01, and 40%, 60% dropout rates for my hidden layers respectively. Surprising how using the params for the 'worse' manual model (2), gave me the best results with dropout. Surprised at how much an effect it can have.

In the end, this all made sense. I now have a grasp of how machine learning libraries are generally structured and how multiclass and performance tweaks like dropout work. On to finish LENET!