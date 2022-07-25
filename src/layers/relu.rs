use super::layer::{ActivationLayer, Layer};
use ndarray::{prelude::Array2, Array, Ix2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
extern crate blas_src;
// implementation of relu with option to make it leaky
pub struct ReLU {
    pub layer: Layer,
    relu_coefficient: f32,
}

impl ReLU {
    // create new relu layer
    pub fn new(
        input: usize,
        nodes: usize,
        samples: usize,
        alpha: f32,
        relu_coefficient: f32,
    ) -> ReLU {
        // create a normal layer
        let mut layer = Layer::new_layer(input, nodes, samples, alpha);
        // init weights with he initialization
        layer.weights = Array::<f32, Ix2>::random(
            (nodes, input),
            Normal::new(0.0f32, 2f32 / input as f32).unwrap(),
        );
        ReLU {
            layer: layer,
            relu_coefficient: relu_coefficient,
        }
    }

    // calculate derivative of relu
    fn derivate(&mut self) -> Array2<f32> {
        let mut deriv = Array2::<f32>::zeros(self.layer.layer.raw_dim());
        for ((i, j), item) in self.layer.layer.indexed_iter() {
            // if the item is greater than 0, relu was activated
            if *item > 0.0 {
                deriv[[i, j]] = 1.0;
            }
            // if the item is 0 or less, it was not activated by relu
            else {
                deriv[[i, j]] = self.relu_coefficient;
            }
        }
        deriv
    }
    // forward prop
    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.layer.forward_prop(&previous_layer);
        self.activate();
    }

    // backwards prop
    pub fn backward_prop(&mut self, previous_layer: &Layer, next_layer: &Layer) {
        self.deactivate(&previous_layer);
        self.layer.backward_prop(&next_layer);
    }
}

impl ActivationLayer for ReLU {
    // activate using relu piecewise function
    fn activate(&mut self) {
        for ((i, j), item) in self.layer.preactivation.indexed_iter() {
            if *item < 0.0 {
                self.layer.layer[[i, j]] =
                    &self.layer.preactivation[[i, j]] * &self.relu_coefficient;
            } else {
                self.layer.layer[[i, j]] = self.layer.preactivation[[i, j]];
            }
        }
    }
    // calculate gradient of the layer
    fn deactivate(&mut self, previous_layer: &Layer) {
        self.layer.d_activation =
            (&previous_layer.weights.t().dot(&previous_layer.d_activation)) * &self.derivate();
    }
}
