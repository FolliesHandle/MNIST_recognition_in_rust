mod dataset;
use rand::Rng;

use crate::dataset::Dataset;
use ndarray::prelude::{Array2};


pub struct Model{
    hidden_layer_weights: Array2<f32>,
    hidden_layer_biases: Array2<f32>,
    hidden_layer_preactivation: Array2<f32>,
    output_layer_weights: Array2<f32>,
    output_layer_biases: Array2<f32>,
    output_layer_preactivation: Array2<f32>,
    hidden_layer: Array2<f32>,
    output_layer: Array2<f32>,
    dataset: Dataset,
}

impl Model{
    pub fn new() -> Model {
        let mut rng = rand::thread_rng();
        let dataset = Dataset::new();
        let hidden_layer_weights: Array2<f32> =
            Array2::from_shape_simple_fn((10, 784), || rng.gen::<f32>() - 0.5f32);
        let hidden_layer_biases: Array2<f32> =
            Array2::from_shape_simple_fn((10, 1), || rng.gen::<f32>() - 0.5f32);

        let output_layer_weights: Array2<f32> =
            Array2::from_shape_simple_fn((10, 10), || rng.gen::<f32>() - 0.5f32);
        let output_layer_biases: Array2<f32> =
            Array2::from_shape_simple_fn((10, 1), || rng.gen::<f32>() - 0.5f32);
        
        let hidden_layer_preactivation = &hidden_layer_weights * &dataset.training_data + &hidden_layer_biases;
        let hidden_layer = Model::relu(&self, hidden_layer_preactivation);

        let output_layer_preactivation = &output_layer_weights * &hidden_layer + &output_layer_biases;
        let output_layer = Model::softmax(output_layer_preactivation);
        Model { 
            hidden_layer_weights: hidden_layer_weights, 
            hidden_layer_biases: hidden_layer_biases, 
            hidden_layer_preactivation: hidden_layer_preactivation, 
            output_layer_weights: output_layer_weights, 
            output_layer_biases: output_layer_biases, 
            output_layer_preactivation: output_layer_preactivation, 
            hidden_layer: hidden_layer,
            output_layer: output_layer,
            dataset: dataset, 
        }
    }
    
    fn max_over_zero(&self, nm: f32) -> f32 {
        if nm > 0.0 {
            return nm;
        }
        0.0
    }
    
    fn relu(&self, preactivation_layer: Array2<f32>){
        let mut copy = preactivation_layer;
        for ((i,j),ele) in preactivation_layer.indexed_iter() {
            copy[[i,j]] = self.max_over_zero(*ele);
        }
    }

    fn softmax(&self, preactivate_layer: Array2<f32>) -> Array2<f32> {
        let mut sum: f32 = 0.0;

        for item in preactivate_layer.iter() {
            sum += item.exp();
        }

        let mut out = Array2::<f32>::zeros(preactivate_layer.dim());

        for ((i, j), value) in out.indexed_iter() {
            out[[i, j]] = preactivate_layer[[i,j]].exp() / sum;
        }
        out
    }
    
    fn forward_prop(&self) {
        self.hidden_layer_preactivation = self.hidden_layer_weights * &self.dataset.training_data + self.hidden_layer_biases;

        self.hidden_layer= Model::relu(self.hidden_layer_preactivation);
      
        self.output_layer_preactivation = self.output_layer_weights * &self.hidden_layer + &self.output_layer_biases;

        self.output_layer = Model::softmax(self.output_layer_preactivation);
    }
}