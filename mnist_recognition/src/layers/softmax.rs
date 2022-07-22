
use ndarray::{prelude::Array2, s, Array, Ix2};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use super::layer::{Layer, ActivationLayer};
extern crate blas;

pub struct Softmax {
    pub layer: Layer,
}

impl Softmax {
    pub fn new(
        input: usize,
        nodes: usize,
        samples: usize,
        alpha: f32,
    ) -> Softmax {
        let mut layer = Layer::new_layer(input, nodes, samples, alpha);
        layer.weights = Array::<f32, Ix2>::random(
            (nodes, input),
            Normal::new(0.0f32, 1.0f32).unwrap(),
        );
        Softmax { layer: layer }
    }

    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.layer.forward_prop(&previous_layer);
        self.activate();
    }
}

impl ActivationLayer for Softmax {
    fn activate(&mut self) {
        let mut row_sums = Array2::<f32>::zeros((1, self.layer.preactivation.ncols()));
        let mut out = Array2::<f32>::zeros(self.layer.preactivation.raw_dim());
        let mut m = f32::NEG_INFINITY;
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
        self.layer.d_activation = (&self.layer.layer - &previous_layer.layer).map(|x| x*2f32);
    }
}