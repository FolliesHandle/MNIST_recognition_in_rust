mod dataset;
use rand::Rng;

use crate::model::dataset::Dataset;
use ndarray::{prelude::{Array2, Array}, Axis};

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
        
        println!("hlw {:?}, ilsize {:?}, hlb {:?}", hidden_layer_weights.dim(), dataset.training_data.dim(), hidden_layer_biases.dim());
        let hidden_layer_preactivation: Array2<f32> = &hidden_layer_weights.dot(&dataset.training_data) + &hidden_layer_biases;
        let hidden_layer: Array2<f32> = Model::relu(&hidden_layer_preactivation);

        let output_layer_preactivation = &output_layer_weights.dot(&hidden_layer) + &output_layer_biases;
        let output_layer  = Model::softmax(&output_layer_preactivation);

        println!("outputsize {:?}", output_layer.dim());
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
    
    fn max_over_zero(nm: f32) -> f32 {
        if nm > 0.0 {
            return nm;
        }
        0.0
    }
    


    fn relu(preactivation_layer: &Array2<f32>) -> Array2<f32>{
        let mut out = Array2::<f32>::zeros(preactivation_layer.raw_dim());
        for ((i,j),ele) in preactivation_layer.indexed_iter() {
            out[[i,j]] = Model::max_over_zero(*ele);
        }
        out
    }

    fn softmax(preactivate_layer: &Array2<f32>) -> Array2<f32> {
        let mut sum: f32 = 0.0;

        for item in preactivate_layer.iter() {
            sum += item.exp();
        }

        let mut out = Array2::<f32>::zeros(preactivate_layer.raw_dim());

        for ((i, j), value) in preactivate_layer.indexed_iter() {
            out[[i, j]] = value.exp() / sum;
        }
        out
    }
    
    fn forward_prop(&mut self) {
        self.hidden_layer_preactivation = &self.hidden_layer_weights * &self.dataset.training_data + &self.hidden_layer_biases;

        self.hidden_layer= Model::relu(&self.hidden_layer_preactivation);
      
        self.output_layer_preactivation = &self.output_layer_weights * &self.hidden_layer + &self.output_layer_biases;

        self.output_layer = Model::softmax(&self.output_layer_preactivation);
    }

    fn calc_output_one_hot(&mut self) -> Array2<f32>{
        let mut output_one_hot = Array2::<f32>::zeros(self.output_layer.raw_dim());

        for i in Array::range(0f32, 9f32, 1f32).iter() {
            for j in self.output_layer.iter() {
                output_one_hot[[*i as usize, *j as usize]] = 1f32;
            }
        }
        output_one_hot.t();
        output_one_hot
    }

    fn derivate_relu(relu_layer: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(relu_layer.raw_dim());
        for ((i,j),item) in relu_layer.indexed_iter() {
            if item > &0.0 {
                out[[i,j]] = 1.;
            } else {
                out[[i,j]] = 0.;
            }
        }
        out
    }

    fn backward_prop(&mut self) -> (Array2<f32>,Array2<f32>,Array2<f32>,Array2<f32>) {
        let size = self.output_layer.len_of(Axis(0)) as f32;
        let one_hot_hidden = self.calc_output_one_hot();
        let output_derivative_preactivation = &self.hidden_layer - &one_hot_hidden;
        let output_derivative_weights = &output_derivative_preactivation.dot(&self.output_layer.t()).map(|x| x * 1./size);
        let output_derivative_biases = &output_derivative_preactivation.sum_axis(Axis(2)).map(|x| x* 1./size);
        let hidden_derivative_preactivation = self.output_layer_weights.t().dot(&output_derivative_preactivation) * Model::derivate_relu(&self.hidden_layer_preactivation);
        let hidden_derivative_weights = &hidden_derivative_preactivation.dot(&self.dataset.training_data.t()).map(|x| x * 1./size);
        let hidden_derivative_biases = &hidden_derivative_preactivation.sum_axis(Axis(2)).map(|x| x* 1./size);

        (
            *output_derivative_weights,
            *output_derivative_biases,
            *hidden_derivative_weights,
            *hidden_derivative_biases
        )
    
    }

}