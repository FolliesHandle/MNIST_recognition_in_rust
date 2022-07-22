

use ndarray::{prelude::Array2, Array, Ix2};
use ndarray_rand::{RandomExt, rand_distr::{Normal}};
use super::layer::{Layer, ActivationLayer};
extern crate blas;

pub struct ReLU {
    pub layer: Layer,
    relu_coefficient: f32,
}

impl ReLU {
    pub fn new(
        input: usize,
        nodes: usize,
        samples: usize,
        alpha: f32,
        relu_coefficient: f32,
    ) -> ReLU {
        let mut layer = Layer::new_layer(input, nodes, samples, alpha);
        layer.weights = Array::<f32, Ix2>::random(
            (nodes, input),
            // Normal::new(0.0f32, (2f32/input as f32).sqrt()).unwrap(),
            Normal::new(nodes as f32, input as f32).unwrap(),
        );
        for item in layer.weights.iter_mut() {
            *item *= (2f32/input as f32).sqrt();
        }
        ReLU {
            layer: layer,
            relu_coefficient :relu_coefficient,
        }
    }

    fn derivate(&mut self) -> Array2<f32>{
        let mut deriv = Array2::<f32>::zeros(self.layer.layer.raw_dim());
        for ((i, j), item) in self.layer.layer.indexed_iter() {
            if *item > 0.0 {
                deriv[[i, j]] = 1.0;
            } else {
                deriv[[i, j]] = self.relu_coefficient;
            }
        }
        deriv
    }
    
    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.layer.forward_prop(&previous_layer);
        self.activate();
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
        self.layer.d_activation = (&previous_layer.weights.t().dot(&previous_layer.d_activation)) * &self.derivate();
    }
}
