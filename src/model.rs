use crate::layers::{dataset::Dataset, relu::ReLU, softmax::Softmax};
use ndarray::{prelude::Array2, Axis};

// Used to specify whether or not the training data and forward prop should
// be ran with training or testing data
pub enum CONFIG {
    TRAIN,
    TEST,
}

// 2 layer neural network, with accuracy vector for tracking performance
pub struct Model {
    // dataset struct holding full data and slices
    dataset: Dataset,
    // ReLU layer
    hidden_layer: ReLU,
    // Softmax layer
    output_layer: Softmax,
    // 10 x 2 vector for tracking accuracy
    accuracy: Array2<f32>,
}

impl Model {
    // creates a new layer, letting each layer's constructors handle the initialization
    // layer_size: specifies size of ReLU layer
    // alpha: specifies learning rate of network
    // slice_range: specifies size of each slice
    pub fn new(layer_size: usize, alpha: f32, slice_range: isize) -> Model {
        let mut dataset = Dataset::new(slice_range);
        dataset.set_slice(CONFIG::TRAIN);
        let hidden_layer = ReLU::new(784, layer_size, slice_range as usize, alpha, 0.01);
        let output_layer = Softmax::new(layer_size, 10, slice_range as usize, alpha);
        let accuracy = Array2::<f32>::zeros((10, 2));
        Model {
            dataset,
            hidden_layer,
            output_layer,
            accuracy,
        }
    }

    // forward propogration function, mostly handled in layer
    // configuration: specifies whether or not network is training
    fn forward_prop(&mut self, configuration: CONFIG) {
        match configuration {
            CONFIG::TRAIN => {
                self.hidden_layer
                    .forward_prop(&self.dataset.train_data_slice);
                self.output_layer.forward_prop(&self.hidden_layer.layer);
            }
            CONFIG::TEST => {
                self.hidden_layer
                    .forward_prop(&self.dataset.test_data_slice);
                self.output_layer.forward_prop(&self.hidden_layer.layer);
            }
        }
    }

    // backwards propogation function, mostly handled in layer
    fn backward_prop(&mut self) {
        self.output_layer
            .backward_prop(&self.dataset.train_label_slice, &self.hidden_layer.layer);
        self.hidden_layer
            .backward_prop(&self.output_layer.layer, &self.dataset.train_data_slice);
    }
    // updating of weights and biases
    fn update_params(&mut self) {
        self.output_layer.layer.update_params();
        self.hidden_layer.layer.update_params();
    }

    // gets predictions from softmaxed output layer in label format
    fn get_predictions(&self) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((1, self.output_layer.layer.layer.ncols()));
        // loop for finding indexes of the max value in each row of the output layer
        for i in 0..self.output_layer.layer.layer.ncols() {
            let mut max_index = 0.0;
            let mut max_value = 0.0;
            for j in 0..self.output_layer.layer.layer.nrows() {
                if self.output_layer.layer.layer[[j, i]] > max_value {
                    max_index = j as f32;
                    max_value = self.output_layer.layer.layer[[j, i]];
                }
            }
            out[[0, i]] = max_index;
        }
        out
    }

    // adds batch accuracy to total accuracy measurement per epoch
    fn set_accuracy(&mut self, predictions: Array2<f32>, ground_truth: Array2<f32>) {
        // make sure that predictions and ground_truth are same size
        assert!(predictions.ncols() == ground_truth.ncols());
        for ((i, j), item) in predictions.indexed_iter() {
            if *item == ground_truth[[i, j]] {
                // adds to column 0 if predicition is correct
                self.accuracy[[ground_truth[[i, j]] as usize, 0]] += 1f32;
            }
            // adds all instances to column 1 for accurate count of samples
            self.accuracy[[ground_truth[[i, j]] as usize, 1]] += 1f32;
        }
    }

    // gets accuracy of entire epoch for display
    fn get_accuracy(&mut self) -> f32 {
        let mut i = 0;
        let mut dataset_size = 0f32;
        let mut sum = 0f32;
        println!("-----------------------------");
        for item in self.accuracy.axis_iter(Axis(0)) {
            println!(
                "Digit {}: {} out of {}, {}",
                i,                               // digit
                item[0],                         // total detected
                item[1],                         // total in ground truth
                item[0] as f32 / item[1] as f32  // digit-wise accuracy
            );
            sum += item[0];
            dataset_size += item[1];
            i += 1;
        }
        println!("-----------------------------");
        return sum / dataset_size;
    }

    // train the network, and print accuracy every 10 epochs
    pub fn train(&mut self, epochs: usize) {
        for i in 0..epochs {
            self.dataset.shuffle();
            for _ in
                0..(self.dataset.training_data.layer.ncols() / self.dataset.slice_range as usize)
            {
                // get new dataset slice
                self.dataset.set_slice(CONFIG::TRAIN);
                // forward
                self.forward_prop(CONFIG::TRAIN);
                // calculate gradients
                self.backward_prop();
                // update
                self.update_params();
                // tally accuracy for batch
                self.set_accuracy(
                    self.get_predictions(),
                    self.dataset.train_label_slice.layer.clone(),
                );
            }
            if i % 10 == 0 {
                // print accuracy
                println!("\n\n-----------------------------");
                println!("Total Epochs: {}", i);
                println!("Accuracy: {}", self.get_accuracy(),);
            }
            // reset accuracy for next epoch
            self.accuracy = Array2::<f32>::zeros((10, 2));
        }
    }

    // test network on separate data to cross-validate
    pub fn test(&mut self) {
        println!("\n\nTESTING NETWORK");
        self.accuracy = Array2::<f32>::zeros((10, 2));
        for _ in 0..(self.dataset.testing_data.layer.ncols() / self.dataset.slice_range as usize) {
            // set slice from testing net
            self.dataset.set_slice(CONFIG::TEST);
            // forward
            self.forward_prop(CONFIG::TEST);
            // tally accuracy for batch
            self.set_accuracy(
                self.get_predictions(),
                self.dataset.test_label_slice.layer.clone(),
            );
        }
        // print testing accuracy
        print!("Accuracy: {}", self.get_accuracy(),);
    }
}
