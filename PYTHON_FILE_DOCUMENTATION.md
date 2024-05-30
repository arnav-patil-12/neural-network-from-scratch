# Python File Documentation

As promised in [README.md](README.md), this document is meant to provide an overview of my code my understanding of it as I followed through the video playlist.

## [layers.py](layers.py)

This was the first file I created, because it's the first one the video creates as well. I began by creating an abstract class ```Layer``` which I will use as the basis for all my layers. The ```NotImplementedError``` is raised to signify that ```Layer``` is an abstract class and should not be initalized as it is.

```
class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
```

### The Linear (or Dense) Layer

To define the dense layer, we must give it an input and output size (as in how many "neurons" feed into it, and how many neurons it feeds into). 
