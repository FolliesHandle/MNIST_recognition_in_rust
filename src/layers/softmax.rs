use ndarray::{prelude::Array2, Array, Ix2};
use ndarray_rand::{rand_distr::Normal, RandomExt};

use super::layer::{ActivationLayer, Layer};
extern crate blas_src;

// implementation of softmax layer
pub struct Softmax {
    pub layer: Layer,
}

impl Softmax {
    pub fn new(input: usize, nodes: usize, samples: usize, alpha: f32) -> Softmax {
        // normal layer init
        let mut layer = Layer::new_layer(input, nodes, samples, alpha);
        // xavier init for softmax
        layer.weights =
            Array::<f32, Ix2>::random((nodes, input), Normal::new(0.0f32, 1.0f32).unwrap());
        Softmax { layer: layer }
    }

    // normal forward prop
    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.layer.forward_prop(&previous_layer);
        self.activate();
    }

    // backwards prop function
    pub fn backward_prop(&mut self, previous_layer: &Layer, next_layer: &Layer) {
        self.deactivate(&Layer::one_hot(&previous_layer.layer));
        self.layer.backward_prop(&next_layer);
    }
}

impl ActivationLayer for Softmax {
    // actual softmax function implementation
    fn activate(&mut self) {
        let mut row_sums = Array2::<f32>::zeros((1, self.layer.preactivation.ncols()));
        let mut out = Array2::<f32>::zeros(self.layer.preactivation.raw_dim());
        let mut m = f32::NEG_INFINITY;
        // get max in layer
        for item in self.layer.preactivation.iter() {
            if *item > m {
                m = *item;
            }
        }
        // subtract max from each element, and add it to sum of its row
        for ((i, j), item) in self.layer.preactivation.indexed_iter() {
            out[[i, j]] = (*item - m).exp();
            row_sums[[0, j]] += out[[i, j]];
        }
        // divide each element by sum of its row
        for ((i, j), _) in self.layer.preactivation.indexed_iter() {
            out[[i, j]] /= row_sums[[0, j]];
        }
        self.layer.layer = out;
    }

    // calculate derivative of activation
    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation = (&self.layer.layer - &previous_layer.layer).map(|x| x * 2f32);
    }
}
