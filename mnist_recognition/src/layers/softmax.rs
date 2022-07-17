
use ndarray::{prelude::Array2, s};

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
        let layer = Layer::new_layer(input, nodes, samples, alpha);
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
        dbg!(out.slice(s![..,50usize]));
        self.layer.layer = out;
        
    }

    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation = &self.layer.layer - &previous_layer.layer;
    }
}