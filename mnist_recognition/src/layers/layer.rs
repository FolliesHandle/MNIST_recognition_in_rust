use ndarray::{prelude::Array2, Axis, Zip};
extern crate blas_src;

pub trait ActivationLayer {
    fn activate(&mut self);
    fn deactivate(&mut self, previous_layer: &Layer);
}


#[derive(Clone)]
pub struct Layer {
    pub preactivation: Array2<f32>,
    pub layer: Array2<f32>,
    pub d_activation: Array2<f32>,
    pub weights: Array2<f32>,
    pub d_weights: Array2<f32>,
    pub biases: Array2<f32>,
    pub d_biases: Array2<f32>,
    alpha: f32,
    samples: usize,
}

impl Layer {
    pub fn new_layer(input: usize, nodes: usize, samples: usize, alpha: f32) -> Layer {
        let preactivation = Array2::zeros((nodes, samples));
        let layer = Array2::zeros((nodes, samples));
        let d_activation = Array2::zeros((nodes, samples));
        let weights = Array2::<f32>::zeros(
            (nodes, input),
        );
        let d_weights = Array2::<f32>::zeros((nodes, input));
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

    pub fn forward_prop(&mut self, previous_layer: &Layer) {
        self.preactivation = &self.weights.dot(&previous_layer.layer) + &self.biases;
    }

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

    pub fn update_params(&mut self) {
        Zip::from(&mut self.weights)
            .and(&self.d_weights)
            .for_each(|a, b| *a -= self.alpha * *b);
        Zip::from(&mut self.biases)
            .and(&self.d_biases)
            .for_each(|a, b| *a -= self.alpha * *b);
    }

    pub fn one_hot(input: &Array2<f32>) -> Layer {
        let mut output_one_hot = Array2::<f32>::zeros((10, input.ncols()));
        for ((_, j), value) in input.indexed_iter() {
            output_one_hot[[*value as usize, j]] = 1f32;
        }
        Layer::dummy_layer(output_one_hot)
    }
}
