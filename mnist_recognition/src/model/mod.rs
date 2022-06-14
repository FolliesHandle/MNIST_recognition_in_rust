mod dataset;
use std::ops::Sub;

use rand::Rng;

use crate::model::dataset::Dataset;
use ndarray::{
    prelude::{Array, Array2},
    Array1, Axis,
};

pub struct Model {
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

// Object containing the network and the functions necessary to train and test it on the MNIST dataset
impl Model {
    // Constructor
    // Arguments: None
    // Returns: Model object containing
    //          - Dataset object
    //          - Hidden Layer: weights, biases, nodes
    //          - Output Layer: weights, biases, nodes
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

        let hidden_layer_preactivation: Array2<f32> =
            &hidden_layer_weights.dot(&dataset.training_data) + &hidden_layer_biases;
        let hidden_layer: Array2<f32> = Model::relu(&hidden_layer_preactivation);

        let output_layer_preactivation =
            &output_layer_weights.dot(&hidden_layer) + &output_layer_biases;
        let output_layer = Model::softmax(&output_layer_preactivation);

        println!(
            "outputsize {:?} {:?}",
            output_layer.dim(),
            dataset.training_labels.dim()
        );
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

    // Simple function to facilitate relu activation
    // Arguments: nm: f32 -> number to compare to zero
    // Returns: nm if nm > 0.0, else 0.0
    fn greater_than_zero(nm: f32) -> f32 {
        if nm > 0.0 {
            return nm;
        }
        0.0
    }

    // ReLU Activation Function
    // Arguments: preactivation_layer: &Array2<f32> -> matrix to be activated
    // Returns: Activated matrix with fn greater_than_zero performed elementwise
    fn relu(preactivation_layer: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(preactivation_layer.raw_dim());
        for ((i, j), ele) in preactivation_layer.indexed_iter() {
            out[[i, j]] = Model::greater_than_zero(*ele);
        }
        out
    }

    // Softmax operation for output layer activation
    // Arguments: preactivation_layer: &Array2<f32> -> matrix to be activated
    // Returns: matrix with each row normalized to a probability distribution between [0,1]
    fn softmax(preactivation_layer: &Array2<f32>) -> Array2<f32> {
        let mut sum: f32 = 0.0;

        for item in preactivation_layer.iter() {
            sum += item.exp();
        }

        let mut out = Array2::<f32>::zeros(preactivation_layer.raw_dim());

        for ((i, j), value) in preactivation_layer.indexed_iter() {
            out[[i, j]] = value.exp() / sum;
        }
        out
    }

    // Forward Propogation function for inference/training
    // Modifies variables in place
    fn forward_prop(&mut self, data: &Array2<f32>) {
        // hidden layer => weights * nodes in input layer + biases
        self.hidden_layer_preactivation =
            &self.hidden_layer_weights.dot(data) + &self.hidden_layer_biases;

        // activate hidden layer with ReLU activation
        self.hidden_layer = Model::relu(&self.hidden_layer_preactivation);

        // output layer => weights * nodes in hidden layer + biases
        self.output_layer_preactivation =
            &self.output_layer_weights.dot(&self.hidden_layer) + &self.output_layer_biases;

        // activate output layer with softmax activation
        self.output_layer = Model::softmax(&self.output_layer_preactivation);
    }

    // Creates a one hot array of the output for use in backwards propogation
    // Arguments: none
    // Returns: array representing labels inferred by the output layer
    //          - rows represent detection results from one single "digit"
    //          - each row has a single one in the place of the most probable class
    //            from the probability distribution
    fn calc_output_one_hot(&mut self) -> Array2<f32> {
        let mut output_one_hot = Array2::<f32>::zeros(self.output_layer.raw_dim());

        for i in Array::range(0f32, 9f32, 1f32).iter() {
            for j in self.dataset.training_labels.iter() {
                output_one_hot[[*i as usize, *j as usize]] = 1f32;
            }
        }
        output_one_hot.t();
        output_one_hot
    }

    // Derivative of ReLU activation function above
    // Arguments: relu_layer: &Array32<f32> -> layer that underwent ReLU activation in forward_prop
    // Returns: Array of 1s and 0s, 1s in place of positive values after activation
    fn derivate_relu(relu_layer: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(relu_layer.raw_dim());
        for ((i, j), item) in relu_layer.indexed_iter() {
            if item > &0.0 {
                out[[i, j]] = 1.;
            } else {
                out[[i, j]] = 0.;
            }
        }
        out
    }

    // Backwards propogation of correct labels
    // Modifies variables in place
    fn backward_prop(&mut self, learning_rate: f32) {
        let size = self.output_layer.len_of(Axis(0)) as f32;
        let one_hot_hidden = self.calc_output_one_hot();

        let output_derivative_preactivation = &self.hidden_layer - &one_hot_hidden;
        let output_derivative_weights = &output_derivative_preactivation
            .dot(&self.output_layer.t())
            .map(|x| x * 1. / size);
        let output_derivative_biases = &output_derivative_preactivation
            .sum_axis(Axis(1))
            .map(|x| x * 1. / size)
            .insert_axis(Axis(1));

        let hidden_derivative_preactivation = self
            .output_layer_weights
            .t()
            .dot(&output_derivative_preactivation)
            * Model::derivate_relu(&self.hidden_layer_preactivation);
        let hidden_derivative_weights = &hidden_derivative_preactivation
            .dot(&self.dataset.training_data.t())
            .map(|x| x * 1. / size);
        let hidden_derivative_biases = &hidden_derivative_preactivation
            .sum_axis(Axis(1))
            .map(|x| x * 1. / size)
            .insert_axis(Axis(1));

        self.update_params(
            output_derivative_weights,
            output_derivative_biases,
            hidden_derivative_weights,
            hidden_derivative_biases,
            learning_rate,
        )
    }

    fn update_params(
        &mut self,
        output_derivative_weights: &Array2<f32>,
        output_derivative_biases: &Array2<f32>,
        hidden_derivative_weights: &Array2<f32>,
        hidden_derivative_biases: &Array2<f32>,
        learning_rate: f32,
    ) {
        self.hidden_layer_weights = self
            .hidden_layer_weights
            .clone()
            .sub(&hidden_derivative_weights.map(|x| x * learning_rate));
        println!("hidden_layer_biases {:?} hidden_derivative_biases {:?}", &self.hidden_layer_biases.dim(), &hidden_derivative_biases.dim());
        self.hidden_layer_biases = self
            .hidden_layer_biases
            .clone()
            .sub(&hidden_derivative_biases.map(|x| x * learning_rate));

        self.output_layer_weights = self
            .output_layer_weights
            .clone()
            .sub(&output_derivative_weights.map(|x| x * learning_rate));
        self.output_layer_biases = self
            .output_layer_biases
            .clone()
            .sub(&output_derivative_biases.map(|x| x * learning_rate));
    }

    fn get_predictions(&self) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((1, self.output_layer.ncols()));
        for i in 0..self.output_layer.ncols() {
            let max_index = -1.0;
            let max_value = -1.0;
            for j in 0..self.output_layer.nrows() {
                if self.output_layer[[j, i]] > max_value {
                    let max_index = j;
                    let max_value = self.output_layer[[j, i]];
                }
            }
            out[[0, i]] = max_index;
        }
        out
    }

    fn get_accuracy(&self, predictions: Array2<f32>, ground_truth: &Array2<f32>) -> f32 {
        assert!(predictions.ncols() == ground_truth.ncols());
        let mut sum = 0.0;
        for ((i, j), item) in predictions.indexed_iter() {
            if item == &ground_truth[[i, j]] {
                sum += 1.0;
            }
        }
        let dataset_size: f32 = predictions.ncols() as f32;
        return sum / dataset_size;
    }

    pub fn train(&mut self, learning_rate: f32, iterations: usize) {
        for i in 0..iterations {
            self.forward_prop(&self.dataset.training_data.clone());
            self.backward_prop(learning_rate);
            if i % 10 == 0 {
                println!("Total Iterations: {}", i);
                println!("Accuracy: {}", self.get_accuracy(self.get_predictions(), &self.dataset.training_labels));
            }
        }
    }

    pub fn test(&mut self) {
        println!("\n\nTESTING NETWORK");
        self.forward_prop(&self.dataset.testing_data.clone());
        print!("Accuracy: {}", self.get_accuracy(self.get_predictions(), &self.dataset.training_labels));
    }
    
}
