use mnist::*;
use ndarray::s;
use ndarray::{Array2, Axis, Slice};
use rand::Rng;

use super::layer::Layer;
use crate::model::CONFIG;

// MNIST dataset converted to layers for interfacing with the model
pub struct Dataset {
    // Full dataset
    pub training_data: Layer,
    pub training_labels: Layer,

    // Full testing set
    pub testing_data: Layer,
    pub testing_labels: Layer,

    // Slice of training set
    pub train_data_slice: Layer,
    pub train_label_slice: Layer,

    // Slice of testing set
    pub test_data_slice: Layer,
    pub test_label_slice: Layer,

    // size of the slices, stored for access in other structs
    pub slice_range: isize,
}

impl Dataset {
    // function shuffles the training data and labels in the same order
    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        let shared_length = self
            .training_data
            .layer
            .index_axis_mut(ndarray::Axis(0), 0)
            .len();
        assert_eq!(
            self.training_data.layer.ncols(),
            self.training_labels.layer.ncols()
        );
        for i in 0..shared_length {
            let next = rng.gen_range(0..shared_length);
            if i != next {
                let (mut index, mut random) = self
                    .training_data
                    .layer
                    .multi_slice_mut((s![.., i], s![.., next]));
                std::mem::swap(&mut index, &mut random);

                let (mut index_2, mut random_2) = self
                    .training_labels
                    .layer
                    .multi_slice_mut((s![.., i], s![.., next]));
                std::mem::swap(&mut index_2, &mut random_2);
            }
        }
    }

    // function converts vectors gotten from MNISTBuilder to Array2<f32>
    pub fn vec_to_array(vector: &Vec<u8>, n: usize, m: usize, data: bool) -> Array2<f32> {
        if data {
            Array2::from_shape_vec((n, m), vector.to_vec())
                .expect("Error converting images to Array3 struct")
                .t()
                .map(|x| *x as f32)
                .map(|x| *x / 256f32)
        } else {
            Array2::from_shape_vec((n, m), vector.to_vec())
                .expect("Error converting images to Array3 struct")
                .t()
                .map(|x| *x as f32)
        }
    }

    // create a new dataset
    pub fn new(slice_range: isize) -> Dataset {
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        }: Mnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(0)
            .test_set_length(20_000)
            .finalize();

        // create each array, and assert that it is not empyt data
        let training_data: Array2<f32> = Dataset::vec_to_array(&trn_img, 50_000, 784, true);
        assert!(!training_data.is_empty());
        let training_labels: Array2<f32> = Dataset::vec_to_array(&trn_lbl, 50_000, 1, false);
        assert!(!training_labels.is_empty());
        let testing_data: Array2<f32> = Dataset::vec_to_array(&tst_img, 20_000, 784, true);
        assert!(!testing_data.is_empty());
        let testing_labels: Array2<f32> = Dataset::vec_to_array(&tst_lbl, 20_000, 1, false);
        assert!(!testing_labels.is_empty());

        // slice of training set
        let train_data_slice = Array2::<f32>::zeros((slice_range as usize, 784));
        let train_label_slice = Array2::<f32>::zeros((slice_range as usize, 1));
        // slice of testing set
        let test_data_slice = Array2::<f32>::zeros((slice_range as usize, 784));
        let test_label_slice = Array2::<f32>::zeros((slice_range as usize, 1));
        Dataset {
            training_data: Layer::dummy_layer(training_data),
            training_labels: Layer::dummy_layer(training_labels),
            testing_data: Layer::dummy_layer(testing_data),
            testing_labels: Layer::dummy_layer(testing_labels),
            train_data_slice: Layer::dummy_layer(train_data_slice),
            train_label_slice: Layer::dummy_layer(train_label_slice),
            test_data_slice: Layer::dummy_layer(test_data_slice),
            test_label_slice: Layer::dummy_layer(test_label_slice),
            slice_range,
        }
    }

    // sets slice variables randomly
    pub fn set_slice(&mut self, mode: CONFIG) {
        let mut rng = rand::thread_rng();
        match mode {
            CONFIG::TRAIN => {
                // generate start and end value
                let mut start = rng.gen_range(0..self.training_data.layer.ncols()) as isize;
                let mut end = &start + &self.slice_range;
                // make sure that index of end exists
                if (start + self.slice_range)
                    >= (self.training_data.layer.ncols() - 1usize) as isize
                {
                    end = &start - &self.slice_range;
                }
                // swaps start and end if end is larger
                if start > end {
                    let temp = start;
                    start = end;
                    end = temp;
                }
                // slices data and converts to a separate array in memory
                self.train_data_slice = Layer::dummy_layer(
                    self.training_data
                        .layer
                        .slice_axis(Axis(1), Slice::new(start, Some(end), 1))
                        .clone()
                        .to_owned(),
                );
                self.train_label_slice = Layer::dummy_layer(
                    self.training_labels
                        .layer
                        .slice_axis(Axis(1), Slice::new(start, Some(end), 1))
                        .clone()
                        .to_owned(),
                );
            }
            CONFIG::TEST => {
                // algorithmically the same as above, just with testing set
                let mut start = rng.gen_range(0..self.testing_data.layer.ncols()) as isize;
                let mut end = &start + &self.slice_range;
                if (start + self.slice_range) >= (self.testing_data.layer.ncols() - 1usize) as isize
                {
                    end = &start - &self.slice_range;
                }
                if start > end {
                    let temp = start;
                    start = end;
                    end = temp;
                }
                self.test_data_slice = Layer::dummy_layer(
                    self.testing_data
                        .layer
                        .slice_axis(Axis(1), Slice::new(start, Some(end), 1))
                        .clone()
                        .to_owned(),
                );
                self.test_label_slice = Layer::dummy_layer(
                    self.testing_labels
                        .layer
                        .slice_axis(Axis(1), Slice::new(start, Some(end), 1))
                        .clone()
                        .to_owned(),
                );
            }
        }
    }
}
