
use rand::{distributions::Uniform, prelude::Distribution};
use crate::layers::dataset::Dataset;
use ndarray::{prelude::Array2, Axis, Zip};

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
    relu_coefficient: f64,
}

// Object containing the network and the functions necessary to train and test it on the MNIST Dataset
impl Model {
    // Constructor
    // Arguments: None
    // Returns: Model object containing
    //          - Dataset object
    //          - Hidden Layer: weights, biases, nodes
    //          - Output Layer: weights, biases, nodes
    pub fn new() -> Model {
        let rand_range = Uniform::from(-0.5..=0.5);
        let mut rng = rand::thread_rng();
        let dataset = Dataset::new();
        let hidden_layer_weights: Array2<f64> =
            Array2::from_shape_simple_fn((64, 784), || rand_range.sample(&mut rng));
        let hidden_layer_biases: Array2<f64> =
            Array2::from_shape_simple_fn((64, 1), || rand_range.sample(&mut rng));

        let output_layer_weights: Array2<f64> =
            Array2::from_shape_simple_fn((10, 64), || rand_range.sample(&mut rng));
        let output_layer_biases: Array2<f64> =
            Array2::from_shape_simple_fn((10, 1), || rand_range.sample(&mut rng));

        let hidden_layer_preactivation: Array2<f64> =
            Array2::zeros((64, dataset.training_labels.ncols()));
        let hidden_layer: Array2<f64> = Array2::zeros((64, dataset.training_labels.ncols()));

        let output_layer_preactivation = Array2::zeros((10, dataset.training_labels.ncols()));
        let output_layer = Array2::zeros((10, dataset.training_labels.ncols()));

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
            relu_coefficient: 0.0,
        }
    }

    // Simple function to facilitate relu activation
    // Arguments: nm: f64 -> number to compare to zero
    // Returns: nm if nm > 0.0, else 0.0
    fn relu_single(&self, nm: f64) -> f64 {
        if nm > 0.0 {
            return nm;
        }
        self.relu_coefficient * nm
    }

    // ReLU Activation Function
    // Arguments: preactivation_layer: &Array2<f64> -> matrix to be activated
    // Returns: Activated matrix with fn greater_than_zero performed elementwise
    fn relu(&self, preactivation_layer: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(preactivation_layer.raw_dim());
        for ((i, j), ele) in preactivation_layer.indexed_iter() {
            out[[i, j]] = self.relu_single(*ele);
        }
        out
    }

    // Softmax operation for output layer activation
    // Arguments: preactivation_layer: &Array2<f64> -> matrix to be activated
    // Returns: matrix with each row normalized to a probability distribution between [0,1]
    fn softmax(preactivation_layer: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(preactivation_layer.raw_dim());
        let mut out_helper = Array2::<f64>::zeros((1, preactivation_layer.ncols()));
        let mut m = -f64::INFINITY;
        for item in preactivation_layer.iter() {
                if item > &m {
                    m = *item;
                }
        }
        for ((i, j),item) in preactivation_layer.indexed_iter() {
            out[[i,j]] = (item - m).exp();
            out_helper[[0,j]] += out[[i,j]];
        }
        
        for ((i, j), _) in preactivation_layer.indexed_iter() {
            out[[i,j]] /= out_helper[[0,j]];
        }
        out
    }



    // fn check_nan(mat: &Array2<f64>) {
    //     for item in mat.iter() {
    //         assert!(item.is_finite());
    //     }
    // }

    // Forward Propogation function for inference/training
    // Modifies variables in place
    fn forward_prop(&mut self) {
        // hidden layer => weights * nodes in input layer + biases
        self.hidden_layer_preactivation =
            &self.hidden_layer_weights.dot(&self.dataset.training_data) + &self.hidden_layer_biases;

        // activate hidden layer with ReLU activation
        self.hidden_layer = self.relu(&self.hidden_layer_preactivation);

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
    fn calc_output_one_hot(&mut self) -> Array2<f64> {
        let mut output_one_hot = Array2::<f64>::zeros((10, self.dataset.training_labels.ncols()));
        for ((_, j), value) in self.dataset.training_labels.indexed_iter() {
            output_one_hot[[*value as usize, j]] = 1f64;
        }
        output_one_hot
    }

    // Derivative of ReLU activation function above
    // Arguments: relu_layer: &Array32<f64> -> layer that underwent ReLU activation in forward_prop
    // Returns: Array of 1s and 0s, 1s in place of positive values after activation
    fn derivate_relu(&self, relu_layer: &Array2<f64>) -> Array2<f64> {
        let mut output_layer = Array2::<f64>::zeros(relu_layer.raw_dim());
        for ((i, j), item) in relu_layer.indexed_iter() {
            if item > &0.0 {
                output_layer[[i, j]] = 1.0;
            } else {
                output_layer[[i, j]] = self.relu_coefficient;
            }
        }
        output_layer
    }

    // Backwards propogation of correct labels
    // Modifies variables in place
    fn backward_prop(&mut self, learning_rate: f64) {
        let size = self.output_layer.len_of(Axis(1)) as f64;
        let one_hot_hidden = self.calc_output_one_hot();

        let output_derivative_preactivation = &self.output_layer - &one_hot_hidden;

        let output_derivative_weights = &output_derivative_preactivation
            .dot(&self.hidden_layer.t())
            .map(|x| x * 1. / size);

        let output_derivative_biases = &output_derivative_preactivation
            .sum_axis(Axis(1))
            .map(|x| x * 1. / size)
            .insert_axis(Axis(1));

        let hidden_derivative_preactivation = &self
            .output_layer_weights
            .t()
            .dot(&output_derivative_preactivation)
            * &self.derivate_relu(&self.hidden_layer_preactivation);

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

    // fn update_params(
    //     &mut self,
    //     output_derivative_weights: &Array2<f64>,
    //     output_derivative_biases: &Array2<f64>,
    //     hidden_derivative_weights: &Array2<f64>,
    //     hidden_derivative_biases: &Array2<f64>,
    //     learning_rate: f64,
    // ) {
    //     self.hidden_layer_weights = self
    //         .hidden_layer_weights
    //         .clone()
    //         .sub(&hidden_derivative_weights.map(|x| x * learning_rate));
    //     self.hidden_layer_biases = self
    //         .hidden_layer_biases
    //         .clone()
    //         .sub(&hidden_derivative_biases.map(|x| x * learning_rate));

    //     self.output_layer_weights = self
    //         .output_layer_weights
    //         .clone()
    //         .sub(&output_derivative_weights.map(|x| x * learning_rate));
    //     self.output_layer_biases = self
    //         .output_layer_biases
    //         .clone()
    //         .sub(&output_derivative_biases.map(|x| x * learning_rate));
    // }

    fn update_params(
        &mut self,
        output_derivative_weights: &Array2<f64>,
        output_derivative_biases: &Array2<f64>,
        hidden_derivative_weights: &Array2<f64>,
        hidden_derivative_biases: &Array2<f64>,
        learning_rate: f64,
    ) {
        Zip::from(&mut self.hidden_layer_weights)
            .and(hidden_derivative_weights)
            .apply(|a, b| *a -= learning_rate * b);
        Zip::from(&mut self.hidden_layer_biases)
            .and(hidden_derivative_biases)
            .apply(|a, b| *a -= learning_rate * b);
        Zip::from(&mut self.output_layer_weights)
            .and(output_derivative_weights)
            .apply(|a, b| *a -= learning_rate * b);
        Zip::from(&mut self.output_layer_biases)
            .and(output_derivative_biases)
            .apply(|a, b| *a -= learning_rate * b);
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
                label_accuracy[[ground_truth[[i, j]] as usize, 0]] += 1;
            }
            label_accuracy[[ground_truth[[i, j]] as usize, 1]] += 1;
        }
        let dataset_size: f64 = predictions.ncols() as f64;

        let mut i = 0;
        println!("-----------------------------");
        for item in label_accuracy.axis_iter(Axis(0)) {
            println!(
                "Digit {}: {} out of {}, {}",
                i,
                item[0],
                item[1],
                item[0] as f64 / item[1] as f64
            );
            i += 1;
        }
        println!("-----------------------------");
        return sum / dataset_size;
    }

    pub fn train(&mut self, learning_rate: f64, iterations: usize) {
        for i in 0..iterations {
            self.forward_prop();
            self.backward_prop(learning_rate);
            if i % 10 == 0 {
                println!("\n\n-----------------------------");
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
