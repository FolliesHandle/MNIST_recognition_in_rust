use crate::layers::{softmax::Softmax, relu::ReLU, dataset::Dataset, layer::{Layer, ActivationLayer}};
use ndarray::{prelude::Array2, Axis};


enum CONFIG {
    TRAIN,
    TEST,
}
pub struct Model {
    dataset: Dataset,
    hidden_layer: ReLU,
    output_layer: Softmax,
}

impl Model {
    pub fn new(alpha: f64) -> Model {
        let dataset = Dataset::new();
        let hidden_layer = ReLU::new(784, 128, dataset.training_data.layer.ncols(), alpha, 0.01);
        let output_layer = Softmax::new(128, 10, dataset.training_data.layer.ncols(), alpha);
        Model {
            dataset: dataset,
            hidden_layer: hidden_layer,
            output_layer: output_layer,
        }
    }
    fn forward_prop(&mut self, configuration: CONFIG) {
        match configuration {
            CONFIG::TRAIN => {
                self.hidden_layer.layer.forward_prop(&self.dataset.training_data);
                self.hidden_layer.activate();
                self.output_layer.layer.forward_prop(&self.hidden_layer.layer);
                self.output_layer.activate();
            }
            CONFIG::TEST => {
                self.hidden_layer.layer.forward_prop(&self.dataset.testing_data);
                self.hidden_layer.activate();
                self.output_layer.layer.forward_prop(&self.hidden_layer.layer);
                self.output_layer.activate();
            }
        }
        
    }

    fn backward_prop(&mut self) {
        self.output_layer.deactivate(&Layer::one_hot(&self.dataset.training_labels.layer));
        self.output_layer.layer.backward_prop(&self.hidden_layer.layer);
        self.hidden_layer.deactivate(&self.output_layer.layer);
        self.hidden_layer.layer.backward_prop(&self.dataset.training_data);
    }
    
    fn update_params(&mut self) {
        self.output_layer.layer.update_params();
        self.hidden_layer.layer.update_params();
    }

    fn get_predictions(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((1, self.output_layer.layer.layer.ncols()));
        for i in 0..self.output_layer.layer.layer.ncols() {
            let mut max_index = 0.0;
            let mut max_value = 0.0;
            for j in 0..self.output_layer.layer.layer.nrows() {
                if self.output_layer.layer.layer[[j, i]] > max_value {
                    max_index = j as f64;
                    max_value = self.output_layer.layer.layer[[j, i]];
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
            if *item == ground_truth[[i, j]] {
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

    pub fn train(&mut self, iterations: usize) {
        for i in 0..iterations {
            self.dataset.randomize();
            self.forward_prop(CONFIG::TRAIN);
            self.backward_prop();
            self.update_params();
            if i % 10 == 0 {
                println!("\n\n-----------------------------");
                println!("Total Iterations: {}", i);
                println!(
                    "Accuracy: {}",
                    self.get_accuracy(self.get_predictions(), &self.dataset.training_labels.layer)
                );
            }
        }
    }

    pub fn test(&mut self) {
        println!("\n\nTESTING NETWORK");
        self.forward_prop(CONFIG::TEST);
        print!(
            "Accuracy: {}",
            self.get_accuracy(self.get_predictions(), &self.dataset.training_labels.layer)
        );
    }
}
