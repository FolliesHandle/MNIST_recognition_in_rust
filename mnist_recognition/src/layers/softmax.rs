
use ndarray::{prelude::Array2, s, Array, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use super::layer::{Layer, ActivationLayer};


pub struct Softmax {
    pub layer: Layer,
}

impl Softmax {
    pub fn new(
        input: usize,
        nodes: usize,
        samples: usize,
        alpha: f64,
    ) -> Softmax {
        let mut layer = Layer::new_layer(input, nodes, samples, alpha);
        layer.weights = Array::<f64, Ix2>::random(
            (nodes, input),
            Normal::new(0.0f64, 1.0f64/input as f64).unwrap(),
        );
        Softmax { layer: layer }
    }
}

impl ActivationLayer for Softmax {
    fn activate(&mut self) {
        let mut row_sums = Array2::<f64>::zeros((1, self.layer.preactivation.ncols()));
        let mut out = Array2::<f64>::zeros(self.layer.preactivation.raw_dim());
        let mut m = f64::NEG_INFINITY;
        for item in self.layer.preactivation.iter() {
            if *item > m {
                m = *item;
            }
        }
        for ((i, j),item) in self.layer.preactivation.indexed_iter() {
            out[[i,j]] = (*item - m).exp();
            row_sums[[0,j]] += out[[i,j]];
        }
        
        for ((i, j), _) in self.layer.preactivation.indexed_iter() {
            out[[i,j]] /= row_sums[[0,j]];
        }
        self.layer.layer = out;
        
    }

    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation = &self.layer.layer - &previous_layer.layer;
    }
}