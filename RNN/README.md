# Recurrent Neural Network
## Matt Collyer
____
### Assignment:

#### Preprocess
* Tokenize/lowercase the headlines

* Turn words that only appear once in the data to <oov>

* Assign each word a unique ID and turn the headlines into vectors of IDs

#### Neural Network

* Build a NN that:

    -Embeds the words into an embedding vector (Embedding Layer )
    -Runs it through a simple, single layer RNN
    -Takes the hidden state from the RNN and puts through a Linear layer to predict the category of the news article

### What I did.
I did the preprocessing ( only a few changes needed from the binary classification assignment ) and built the RNN.

I took the dataset, removed unwanted columns and split it into three sets - training, test and validation.

Then preprocessed as described above. It was a bit tedious and I ran into some roadblocks - but I got it.

As for the network itself:

```python
class RNN(nn.Module):
    def __init__(self, dict_size, embedding_size, hidden_size, classes):
        super(RNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(dict_size, embedding_size),
            nn.ReLU(),
            nn.RNN(embedding_size, hidden_size)

        )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, classes)
        )    
    def forward(self, x):
        x, _ = self.layers(x)
        x = x[:, -1, :]
        x = self.linear(x)
        pred = F.softmax(x, dim=1)
        return pred
```
I first run this through an embedding layer to compress the input space a bit. Then ReLU, then our RNN layer.
We grab the hidden state from that, do a bit of manipulation and send it into our Linear layer. We then softmax that and return our predictions. 
This is the unique part of this assignment. The loss fn, and general training + testing process is the same.

### Things that helped

Chatting with you
Pytorch docs for [embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN) layers
[This video](https://www.youtube.com/watch?v=0_PgWWmauHk) was kinda helpful.
### Where I got confused
I got confused a bit during preprocess - like how to generate and share a vocab among training + test + validation sets.

The tuple sent out by the RNN - how to grab the hidden state.

The manipulations needed to send that into linear so that we can actually calculate loss.

The sweet spot for params. Still have not found it. Embedding size + hidden layer size. Gah!
### What I made | How it went | Conclusion
I made garbage. That's what I made. 

But for real. No matter what params I tweaked I could not get better than 40% accuracy. I probably did something wrong somewhere! Oh well. Let me know if you see it.

My best model with results:
``Accuracy: 40.5%, Avg loss: 0.020158``
Had params:
```python
learning_rate = 0.5
embedding_size = 300
hidden_size = 50
sentence_length = 10
batch_size = 32
epochs = 10
classes = 4
```
oh well! 

I did learn a lot. The more practice with this stuff the better and I already feel pretty comfortable building concise pytorch networks. 

This is my LAST assignment for all of Bennington. 
GOODBYE BENNINGTON CS I WILL MISS YOU. 
THANK YOU JUSTIN!!! I CAN'T THANK YOU ENOUGH! 
now I must go and be an "adult"... whatever that means