

use ndarray::{prelude::Array2};
use super::layer::{Layer, ActivationLayer};
pub struct ReLU {
    pub layer: Layer,
    relu_coefficient: f64,
}

impl ReLU {
    pub fn new(
        input: usize,
        nodes: usize,
        samples: usize,
        alpha: f64,
        relu_coefficient: f64,
    ) -> ReLU {
        let layer = Layer::new_layer(input, nodes, samples, alpha);
        ReLU {
            layer: layer,
            relu_coefficient :relu_coefficient,
        }
    }

    fn derivate(&mut self) -> Array2<f64>{
        let mut deriv = Array2::<f64>::zeros(self.layer.layer.raw_dim());
        for ((i, j), item) in self.layer.layer.indexed_iter() {
            if *item > 0.0 {
                deriv[[i, j]] = 1.0;
            } else {
                deriv[[i, j]] = self.relu_coefficient;
            }
        }
        deriv
    }
}

impl ActivationLayer for ReLU {
    fn activate(&mut self) {
        for ((i, j), item) in self.layer.preactivation.indexed_iter() {
            if *item < 0.0 {
                self.layer.layer[[i, j]] = &self.layer.preactivation[[i, j]] * &self.relu_coefficient;
            }
        }
    }
    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation = &previous_layer.weights.t().dot(&previous_layer.d_activation) * &self.derivate();
    }
}
