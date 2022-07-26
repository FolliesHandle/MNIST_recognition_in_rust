# MNIST Recognition in Rust - rustnist
 
## Overview
This project, rustnist, takes inspiration from the following video: https://youtu.be/w8yWXqWQYmU and the corresponding kaggle notebook. Originally, I endeavored to reimplement the network in rust in a similar fashion, but over time I realized two things:
1. The video and the corresponding kaggle notebook do not correctly implement a neural network that can classify the mnist dataset, and most likely serve as a "demonstration" of the creators machine learning prowess for those who won’t take the time to run the code
2. This would be a unique opportunity to further my understanding of both machine learning fundamentals and rust fundamentals
After realizing those two things my new goal became learning and implementing a simple neural network from scratch, and to accomplish this I rewrote most algorithms in the network and made the program more modular. I found this process to be highly rewarding, as it forced me to understand each and every component of my simple one-layer network. This allowed me to learn quite a few things, notably:
- The implementation of softmax and its uses
- The implementation of ReLU and Leaky ReLU 
- Weight initialization best practices for different activation functions
- The underlying linear algebra of a neural network
 
Overall, this was a stimulating project that taught me how simple neural networks work at a low level and how to debug a malfunctioning network, and additionally gave me a chance to try out Rust and its ndarray and random functionality.
 
## Instructions for Building
Most of the building is handled by `cargo`, but there is some custom configuration needed to be done for BLAS support, as different OSs/architectures support BLAS through different libraries.
 
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
 
Additionally, make sure that `blas-src` is compiled with the correct BLAS source in `Cargo.toml`. If you do not want to use BLAS, then you can remove the dependency and the feature in `Cargo.toml`. If you do this, you will have to remove the import in `relu.rs`, `dataset.rs`, `softmax.rs`, and `layer.rs`.
 
Finally, the program can be built with:
```
cargo build --release
```
 
## Running
If everything has been built correctly, you will find an executable named `rustnist` in `./target/release/` that can be used to run the program. You are also able to adjust some basic hyperparameters with flags:
```
-a, --alpha <ALPHA>              Learning rate of the network [default: 0.01]
-b, --batch-size <BATCH_SIZE>    Batch size for BGD [default: 100]
-e, --epochs <EPOCHS>            Amount of epochs to train for [default: 1000]
-l, --layer-size <LAYER_SIZE>    # of nodes in the hidden layer [default: 128]
```
The default hyperparameters above should get you an accuracy of around 80%, and feel free to mess around with each parameter as you see fit.
 
 
## Final Notes
The actual implementation of `rustnist` is build to be modular in nature, and one can define additional layers and activations with a minimal amount of effort. This is NOT a neural network library obviously, so do not expect it to blow your mind when you add more layers or create a complex network, but as something to play around with it is definitely fun.
 
To add a layer with a different activation function, such as tanh or sigmoid, you should use `relu.rs` as a functional template for what needs to be implemented. The cliffnotes are:
- Create a struct with a `Layer` member
- Initialize weights in `new()`
- Implement `forward_prop` and `backward_prop` in the struct implementation
- Implement `activate` and `deactivate`
- Make sure your math is sound
And then you can manually add that layer to `model.rs` in the functions `new`, `forward_prop`, `backward_prop`, and `update_params`.
 
Please feel free to open an issue if anything you see in the repository is bad practice in terms of rust, or if you see any areas of improvement!
