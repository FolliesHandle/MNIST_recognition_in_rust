mod dataset;
use std::{ops::Sub};

use rand::Rng;

use crate::model::dataset::Dataset;
use ndarray::{
    prelude::{Array2},
    Axis, Array1,
};

pub struct Model {
    hidden_layer_weights: Array2<f64>,
    hidden_layer_biases: Array2<f64>,
    hidden_layer_preactivation: Array2<f64>,
    output_layer_weights: Array2<f64>,
    output_layer_biases: Array2<f64>,
    output_layer_preactivation: Array2<f64>,
    hidden_layer: Array2<f64>,
    output_layer: Array2<f64>,
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
        let hidden_layer_weights: Array2<f64> =
            Array2::from_shape_simple_fn((10, 784), || rng.gen::<f64>() - 0.5f64);
        let hidden_layer_biases: Array2<f64> =
            Array2::from_shape_simple_fn((10, 1), || rng.gen::<f64>() - 0.5f64);

        let output_layer_weights: Array2<f64> =
            Array2::from_shape_simple_fn((10, 10), || rng.gen::<f64>() - 0.5f64);
        let output_layer_biases: Array2<f64> =
            Array2::from_shape_simple_fn((10, 1), || rng.gen::<f64>() - 0.5f64);

        let hidden_layer_preactivation: Array2<f64> =
            &hidden_layer_weights.dot(&dataset.training_data) + &hidden_layer_biases;
        let hidden_layer: Array2<f64> = Model::relu(&hidden_layer_preactivation);

        let output_layer_preactivation =
            &output_layer_weights.dot(&hidden_layer) + &output_layer_biases;
        let output_layer = Model::softmax(&output_layer_preactivation);

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
    // Arguments: nm: f64 -> number to compare to zero
    // Returns: nm if nm > 0.0, else 0.0
    fn greater_than_zero(nm: f64) -> f64 {
        if nm > 0.0 {
            return nm;
        }
        0.0
    }

    // ReLU Activation Function
    // Arguments: preactivation_layer: &Array2<f64> -> matrix to be activated
    // Returns: Activated matrix with fn greater_than_zero performed elementwise
    fn relu(preactivation_layer: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(preactivation_layer.raw_dim());
        for ((i, j), ele) in preactivation_layer.indexed_iter() {
            out[[i, j]] = Model::greater_than_zero(*ele);
        }
        out
    }

    // Softmax operation for output layer activation
    // Arguments: preactivation_layer: &Array2<f64> -> matrix to be activated
    // Returns: matrix with each row normalized to a probability distribution between [0,1]
    fn softmax(preactivation_layer: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(preactivation_layer.raw_dim());
        for i in 0..preactivation_layer.ncols() {
            let mut m = -f64::INFINITY;
            for j in 0..preactivation_layer.nrows() {
                if preactivation_layer[[j, i]] > m {
                    m = preactivation_layer[[j, i]];
                }
            }

            let mut sum = 0.0;

            for j in 0..preactivation_layer.nrows() {
                sum += (preactivation_layer[[j, i]] - m).exp();
            }

            let constant = m + sum.log(1.0_f64.exp());
            for j in 0..preactivation_layer.nrows() {
                out[[j, i]] = (preactivation_layer[[j, i]] - constant).exp();
            }
        }
        out
    }

    fn check_nan(mat: &Array2<f64>) {
        for item in mat.iter() {
            assert!(item.is_finite());
        }
    }

    // Forward Propogation function for inference/training
    // Modifies variables in place
    fn forward_prop(&mut self) {
        // hidden layer => weights * nodes in input layer + biases
        self.hidden_layer_preactivation =
            &self.hidden_layer_weights.dot(&self.dataset.training_data) + &self.hidden_layer_biases;
        Model::check_nan(&self.hidden_layer_preactivation);

        // activate hidden layer with ReLU activation
        self.hidden_layer = Model::relu(&self.hidden_layer_preactivation);
        Model::check_nan(&self.hidden_layer);

        // output layer => weights * nodes in hidden layer + biases
        self.output_layer_preactivation =
            &self.output_layer_weights.dot(&self.hidden_layer) + &self.output_layer_biases;
        Model::check_nan(&self.output_layer_preactivation);

        // activate output layer with softmax activation
        self.output_layer = Model::softmax(&self.output_layer_preactivation);
        Model::check_nan(&self.output_layer);
    }

    // Creates a one hot array of the output for use in backwards propogation
    // Arguments: none
    // Returns: array representing labels inferred by the output layer
    //          - rows represent detection results from one single "digit"
    //          - each row has a single one in the place of the most probable class
    //            from the probability distribution
    fn calc_output_one_hot(&mut self) -> Array2<f64> {
        let mut output_one_hot = Array2::<f64>::zeros((10, self.dataset.training_labels.ncols()));
        for ((_, j), value) in self.dataset.training_labels.indexed_iter() {
                output_one_hot[[*value as usize, j]] = 1f64;
        }
        output_one_hot.t();
        output_one_hot
    }

    // Derivative of ReLU activation function above
    // Arguments: relu_layer: &Array32<f64> -> layer that underwent ReLU activation in forward_prop
    // Returns: Array of 1s and 0s, 1s in place of positive values after activation
    fn derivate_relu(relu_layer: &Array2<f64>, output_layer: &mut Array2<f64>) -> Array2<f64> {
        for ((i, j), item) in relu_layer.indexed_iter() {
            if item <= &0.0 {
                output_layer[[i, j]] = 0.0;
            }
        }
        output_layer.clone()
    }

    // Backwards propogation of correct labels
    // Modifies variables in place
    fn backward_prop(&mut self, learning_rate: f64) {
        let size = self.output_layer.len_of(Axis(1)) as f64;
        let one_hot_hidden = self.calc_output_one_hot();

        let output_derivative_preactivation = &self.hidden_layer - &one_hot_hidden;
        Model::check_nan(&output_derivative_preactivation);
        let output_derivative_weights = &output_derivative_preactivation
            .dot(&self.output_layer.t())
            .map(|x| x * 1. / size);

        Model::check_nan(output_derivative_weights);
        let output_derivative_biases = &output_derivative_preactivation
            .sum_axis(Axis(1))
            .map(|x| x * 1. / size)
            .insert_axis(Axis(1));

        Model::check_nan(output_derivative_biases);
        let hidden_derivative_preactivation = &Model::derivate_relu(
            &self.hidden_layer_preactivation,
            &mut self
                .output_layer_weights
                .t()
                .dot(&output_derivative_preactivation),
        );
        Model::check_nan(&hidden_derivative_preactivation);
        let hidden_derivative_weights = &hidden_derivative_preactivation
            .dot(&self.dataset.training_data.t())
            .map(|x| x * 1. / size);
        Model::check_nan(hidden_derivative_weights);
        let hidden_derivative_biases = &hidden_derivative_preactivation
            .sum_axis(Axis(1))
            .map(|x| x * 1. / size)
            .insert_axis(Axis(1));
        Model::check_nan(hidden_derivative_biases);

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
        output_derivative_weights: &Array2<f64>,
        output_derivative_biases: &Array2<f64>,
        hidden_derivative_weights: &Array2<f64>,
        hidden_derivative_biases: &Array2<f64>,
        learning_rate: f64,
    ) {
        self.hidden_layer_weights = self
            .hidden_layer_weights
            .clone()
            .sub(&hidden_derivative_weights.map(|x| x * learning_rate));
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

    fn get_predictions(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((1, self.output_layer.ncols()));
        for i in 0..self.output_layer.ncols() {
            let mut max_index = 0.0;
            let mut max_value = 0.0;
            for j in 0..self.output_layer.nrows() {
                if self.output_layer[[j, i]] > max_value {
                    max_index = j as f64;
                    max_value = self.output_layer[[j, i]];
                }
            }
            out[[0, i]] = max_index;
        }
        out
    }

    fn get_accuracy(&self, predictions: Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        assert!(predictions.ncols() == ground_truth.ncols());
        let mut sum = 0.0f64;
        let mut label_accuracy = Array2::<i32>::zeros((10, 2));
        for ((i, j), item) in predictions.indexed_iter() {
            if item == &ground_truth[[i, j]] {
                sum += 1.0;
                label_accuracy[[ground_truth[[i,j]] as usize, 0]] += 1;
            }
            label_accuracy[[ground_truth[[i,j]] as usize, 1]] += 1;
        }
        let dataset_size: f64 = predictions.ncols() as f64;

        let mut i = 0;
        for item in label_accuracy.axis_iter(Axis(0)) {
            println!("Digit {}: {} out of {}, {}", i, item[0], item[1], item[0] as f64/item[1] as f64);
            i += 1;
        }
        return sum / dataset_size;
    }

    pub fn train(&mut self, learning_rate: f64, iterations: usize) {
        for i in 0..iterations {
            self.forward_prop();
            self.backward_prop(learning_rate);
            if i % 10 == 0 {
                println!("Total Iterations: {}", i);
                println!(
                    "Accuracy: {}",
                    self.get_accuracy(self.get_predictions(), &self.dataset.training_labels)
                );
            }
        }
    }

    pub fn test(&mut self) {
        println!("\n\nTESTING NETWORK");
        self.forward_prop();
        print!(
            "Accuracy: {}",
            self.get_accuracy(self.get_predictions(), &self.dataset.training_labels)
        );
    }
}
