# Bird Species Identification

Bird Identification using a CNN with PyTorch. The densenet161 pretrained model is used and fine tuned to the [dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). 
You can read more about densenet161 [here](https://www.kaggle.com/pytorch/densenet161).

Accuracy achieved is 75%.

The Bird Identifier notebook was used to train the model.
The trained weights were save in a file named model.pt.
The label_image.py file imports the model and identifies the species.
