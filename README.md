# MNIST-Recognition-in-Rust

## Overview
This is a little project idea I had after coming across this video: https://youtu.be/w8yWXqWQYmU and the corresponding kaggle notebook. Originally, my idea was to reimplement the network in rust in a similar fashion, but over time I realized two things:
1. The video and the corresponding kaggle notebook do not actually work, and most likely serve as a "demonstration" of the creators machine learning prowess for those who wouldn't take the time to run the code
2. This would be a cool opportunity to overengineer a little bit for the sake of learning

After realizing those two things, I went back and started refactoring the codebase, which wasn't learning regardless due to my implementation of a broken algorithm. I found this process to be highly rewarding, as it forced me to understand each and every component of my simple one layer network. This allowed me to figure out many of the errors in the network, some of which were:
- My softmax implementation was entirely incorrect
- My ReLU implementation was correct, but I was not using a leaky ReLU, leading to my network exploding
- I was not using the correct weights initialization for each layer, leading to my network beung wildly inconsistent and an accuracy of around 40%
- My lin-alg was not correct, and at one point I was using element-wise multiplication instead of true matrix multiplication

Overall, this was a fun project that showed me how simple neural networks work at a low level, and additionally gave me a chance to try out rust.

## Instructions for Building

Building this project is relatively straightforward, with a few additional steps needed. Most of the building is handled by `cargo`, but some of the configuration needs to be done by hand due to differences in systems.

First, clone the repository:

```
git clone git@github.com:FolliesHandle/rustnist.git
```

Then, cd into the project directory and create the folder for storing the mnist data:
```
cd rustnist/ && mkdir data/
```

Download the MNIST dataset, and put it into `data/`, and rename the files so that they match the formatting below:
```
data:
| t10k-images-idx3-ubyte   
| train-images-idx3-ubyte
| t10k-labels-idx1-ubyte   
| train-labels-idx1-ubyte
```

Finally, make sure that `blas-src` is compiled with the correct BLAS source in `Cargo.toml`. If you do not want to use BLAS, then you can remove the dependency and the feature in `Cargo.toml`. If you do this, you will have to remove the import in `relu.rs`, `dataset.rs`, `softmax.rs`, and `layer.rs`.

## Final Notes

Please feel free to open an issue if anything you see in the repository is bad practice in terms of rust, or if you see any areas of improvement!
