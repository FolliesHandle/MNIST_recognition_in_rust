use ndarray::{prelude::Array2, Axis, Zip};
extern crate blas_src;
// trait used to enforce activation and deactivation
pub trait ActivationLayer {
    fn activate(&mut self);
    fn deactivate(&mut self, previous_layer: &Layer);
}

// struct for a layer (loosely defined)
#[derive(Clone)]
pub struct Layer {
    // nodes before activation
    pub preactivation: Array2<f32>,
    // nodes after activation
    pub layer: Array2<f32>,
    // derivative of nodes and their affects on network performance
    pub d_activation: Array2<f32>,
    // weights for each connection in the network
    pub weights: Array2<f32>,
    // derivative of weights
    pub d_weights: Array2<f32>,
    // bias terms for each node
    pub biases: Array2<f32>,
    // derivative of bias terms
    pub d_biases: Array2<f32>,
    // learning_rate of network
    alpha: f32,
    // amount of samples for each forward and backwards pass
    samples: usize,
}

impl Layer {
    // create a new layer with all elements initialized to zero except for samples and alpha
    pub fn new_layer(input: usize, nodes: usize, samples: usize, alpha: f32) -> Layer {
        // preactivation, layer, and derivative of the layer are the same size
        let preactivation = Array2::zeros((nodes, samples));
        let layer = Array2::zeros((nodes, samples));
        let d_activation = Array2::zeros((nodes, samples));
        // weights are used in between layers
        let weights = Array2::<f32>::zeros((nodes, input));
        let d_weights = Array2::<f32>::zeros((nodes, input));
        // biases used between layers
        let biases = Array2::<f32>::zeros((nodes, 1));
        let d_biases = Array2::<f32>::zeros((nodes, 1));

        Layer {
            preactivation,
            layer,
            d_activation,
            weights,
            d_weights,
            biases,
            d_biases,
            alpha,
            samples,
        }
    }

    // create a dummy layer, enabling the layer to be passed into a function despite not needing
    // the rest of the functionality
    pub fn dummy_layer(layer: Array2<f32>) -> Layer {
        Layer {
            preactivation: Array2::<f32>::zeros((1, 1)),
            layer: layer,
            d_activation: Array2::<f32>::zeros((1, 1)),
            weights: Array2::<f32>::zeros((1, 1)),
            d_weights: Array2::<f32>::zeros((1, 1)),
            biases: Array2::<f32>::zeros((1, 1)),
            d_biases: Array2::<f32>::zeros((1, 1)),
            alpha: 0.0,
            samples: 0,
        }
    }

    // calculate preactivation layer for activating by activation layer
    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.preactivation = &self.weights.dot(&previous_layer.layer) + &self.biases;
    }

    // calculate derivative of weights and biases based on derivative of the activation
    pub fn backward_prop(&mut self, previous_layer: &Layer) {
        self.d_weights = self
            .d_activation
            .dot(&previous_layer.layer.t())
            .map(|x| x * 1.0 / self.samples as f32);
        self.d_biases = self
            .d_activation
            .sum_axis(Axis(1))
            .map(|x| x * 1.0 / (self.samples as f32))
            .insert_axis(Axis(1));
    }

    // update weights and biases
    pub fn update_params(&mut self) {
        Zip::from(&mut self.weights)
            .and(&self.d_weights)
            .for_each(|a, b| *a -= self.alpha * *b);
        Zip::from(&mut self.biases)
            .and(&self.d_biases)
            .for_each(|a, b| *a -= self.alpha * *b);
    }

    // create a dummy one hot encoded layer, primarily used for conversion of 1D labels to
    // 2D array that can be used in backwards propogation
    pub fn one_hot(input: &Array2<f32>) -> Layer {
        let mut output_one_hot = Array2::<f32>::zeros((10, input.ncols()));
        for ((_, j), value) in input.indexed_iter() {
            output_one_hot[[*value as usize, j]] = 1f32;
        }
        Layer::dummy_layer(output_one_hot)
    }
}
