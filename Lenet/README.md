# Convolutional Neural Network
## Matt Collyer
____
### Assignment:

* Implement:
  ```python
  def cross_correlation(F : torch.tensor, G : torch.tensor, padding = (0, 0), stride = (1, 1)) -> torch.tensor:

  def pooling(F : torch.tensor, pool_size, pool_type='max', padding = (0, 0), stride = (1, 1)) -> torch.tensor:
  ```
* A working gradient descent to learn a kernel given the input and expected output.

* Path 2 – Automatic Version
-You can write a concise version of a CNN using any of the built in pytorch layers. Make a few variations and test on the fashion data and include your observations/thought process/results in a write up.

![LeNet](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F2000%2F1*1TI1aGBZ4dybR6__DI9dzA.png&f=1&nofb=1)

### What I did.
Implementing the cross correlation and pooling functions were relatively straightforward. I ran into some syntactical obstacles (like with creating a tensor) that blocked me for an embarassing amount of time - but conceptually it all made sense.

My gradient descent came very close to learning the actual kernal.
For the neural network itself I chose path two: writing a concise CNN with pytorchs built-in layers.

I used pytorchs Conv2d layer for convolutions, AvgPool2d for pooling, and ReLU for my activation.

```python
class Lenet(nn.Module):
    def __init__(self, classes):
        super(Lenet, self).__init__()
        self.conv_layers = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=classes),
        )

```
This gave me a lot more comfort in setting up the network. So much easier and cleaner using the concise method- utilizing all of pytorch. 

### Things that helped
These two sources helped me very much
[Implementing Yann LeCun’s LeNet-5 in PyTorch](https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320)

[Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

### Where I got confused
I got confused on layer 5. Some pytorch Lenet examples I found had layer 5 as linear - others convolutional. In the first source I mentioned, there is a good explanation that settled my doubts.

During training, I found my loss to be decreasing slowly. So slow that after 20 epochs I had an accuracy of around 50%. I reached out to you about this and it turns out my learning rate was far too low. I experimented with raising it but I had a fear to raise it over 0.2  as I figured it would just be worse. Lesson learned. I ended up settling on 0.8

### What I made

I messed with my parameters for a while before I found some nice outcomes.

My best model had params:

```python
learning_rate = 0.8
batch_size = 128
epochs = 10
```

### How it went | Conclusion

The model with the above params trained and tested on the fashionMINST dataset gave me the following results- 

Test Error: 
 Accuracy: 85.6%, Avg loss: 0.012677 

Not bad! I tried one with 15 epochs and it did not nearly decrease the loss enough to make it worthwhile. 
I learned to trust my setup of the network and just keep messing with the parameters. Oh, and convolutions ;) 

Thanks!
