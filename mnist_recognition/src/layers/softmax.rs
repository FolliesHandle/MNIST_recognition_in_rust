
use ndarray::{prelude::Array2};

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
        let mut m = -f64::INFINITY;
        for item in self.layer.preactivation.iter() {
                if item > &m {
                    m = *item;
                }
        }
        for ((i, j),item) in self.layer.preactivation.indexed_iter() {
            self.layer.layer[[i,j]] = (item - m).exp();
            row_sums[[0,j]] += self.layer.layer[[i,j]];
        }
        
        for ((i, j), _) in self.layer.preactivation.indexed_iter() {
            self.layer.layer[[i,j]] /= row_sums[[0,j]];
        }
    }

    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation = &self.layer.layer - &previous_layer.layer;
    }
}