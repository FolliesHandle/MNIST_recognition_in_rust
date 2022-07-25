use crate::layers::{softmax::Softmax, relu::ReLU, dataset::Dataset};
use ndarray::{prelude::Array2, Axis};

pub enum CONFIG {
    TRAIN,
    TEST,
}
pub struct Model {
    dataset: Dataset,
    hidden_layer: ReLU,
    output_layer: Softmax,
    accuracy: Array2<f32>,
}

impl Model {
    pub fn new(alpha: f32, slice_range: isize) -> Model {
        let mut dataset = Dataset::new(slice_range);
        dataset.set_slice(CONFIG::TRAIN);
        let hidden_layer = ReLU::new(784, 128, slice_range as usize, alpha, 0.01);
        let output_layer = Softmax::new(128, 10, slice_range as usize, alpha);
        let accuracy = Array2::<f32>::zeros((10, 2));
        Model {
            dataset,
            hidden_layer,
            output_layer,
            accuracy,
        }
    }
    fn forward_prop(&mut self, configuration: CONFIG) {
        match configuration {
            CONFIG::TRAIN => {
                self.hidden_layer.forward_prop(&self.dataset.train_data_slice);
                self.output_layer.forward_prop(&self.hidden_layer.layer);
            }
            CONFIG::TEST => {
                self.hidden_layer.forward_prop(&self.dataset.test_data_slice);
                self.output_layer.forward_prop(&self.hidden_layer.layer);
            }
        }
        
    }

    fn backward_prop(&mut self) {
        self.output_layer.backward_prop(&self.dataset.train_label_slice, &self.hidden_layer.layer);
        self.hidden_layer.backward_prop(&self.output_layer.layer, &self.dataset.train_data_slice);
    }
    
    fn update_params(&mut self) {
        self.output_layer.layer.update_params();
        self.hidden_layer.layer.update_params();
    }

    fn get_predictions(&self) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((1, self.output_layer.layer.layer.ncols()));
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

    fn set_accuracy(&mut self, predictions: Array2<f32>, ground_truth: Array2<f32>){
        assert!(predictions.ncols() == ground_truth.ncols());
        for ((i, j), item) in predictions.indexed_iter() {
            if *item == ground_truth[[i, j]] {
                self.accuracy[[ground_truth[[i, j]] as usize, 0]] += 1f32;
            }
            self.accuracy[[ground_truth[[i, j]] as usize, 1]] += 1f32;
        }
    }

    fn get_accuracy(&mut self) -> f32 {
        let mut i = 0;
        let mut dataset_size = 0f32;
        let mut sum = 0f32;
        println!("-----------------------------");
        for item in self.accuracy.axis_iter(Axis(0)) {
            println!(
                "Digit {}: {} out of {}, {}",
                i,
                item[0],
                item[1],
                item[0] as f32 / item[1] as f32
            );
            sum += item[0];
            dataset_size += item[1];
            i += 1;
        }
        println!("-----------------------------");
        return sum / dataset_size;
    }

    pub fn train(&mut self, iterations: usize) {
        for i in 0..iterations {
            self.dataset.shuffle();
            for _ in 0..(self.dataset.training_data.layer.ncols()/self.dataset.slice_range as usize) {
                self.dataset.set_slice(CONFIG::TRAIN);
                self.forward_prop(CONFIG::TRAIN);
                self.backward_prop();
                self.update_params();
                self.set_accuracy(self.get_predictions(), self.dataset.train_label_slice.layer.clone());
            }
            if i % 10 == 0 {
                println!("\n\n-----------------------------");
                println!("Total Iterations: {}", i);
                println!(
                    "Accuracy: {}",
                    self.get_accuracy(),
                );
            }
            self.accuracy = Array2::<f32>::zeros((10, 2));
        }
    }

    pub fn test(&mut self) {
        println!("\n\nTESTING NETWORK");
        self.accuracy = Array2::<f32>::zeros((10, 2));
        for _ in 0..(self.dataset.testing_data.layer.ncols() /self.dataset.slice_range as usize) {
            self.dataset.set_slice(CONFIG::TEST);
            self.forward_prop(CONFIG::TEST);
            self.set_accuracy(self.get_predictions(), self.dataset.test_label_slice.layer.clone());
        }
        print!(
            "Accuracy: {}",
            self.get_accuracy(),
        );
    }
}
