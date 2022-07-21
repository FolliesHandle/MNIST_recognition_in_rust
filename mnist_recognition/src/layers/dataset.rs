use mnist::*;
use ndarray::{Array2};
use rand::{distributions::Uniform, prelude::Distribution};

use super::layer::Layer;
pub struct Dataset {
    // 2D array of flattened images from MNIST Dataset, shuffled
    pub training_data: Layer,
    pub training_labels: Layer,
    pub testing_data: Layer,
    pub testing_labels: Layer,
}

impl Dataset {
    fn shuffle(slices: &mut [&mut Array2<f64>]) {
        if slices.len() > 0 {
            let mut rng = rand::thread_rng();

            let shared_length = slices[0].index_axis_mut(ndarray::Axis(0), 0).len();

            for i in 0..shared_length {
                let next = Uniform::from(i..shared_length);
                let index = next.sample(&mut rng);
                for slice in slices.iter_mut() {
                    let mut row = slice.index_axis_mut(ndarray::Axis(0), 0);
                    row.swap(i, index);
                }
            }
        }
    }

    pub fn randomize(&mut self) -> () {
        Dataset::shuffle(&mut [&mut self.training_data.layer, &mut self.training_labels.layer]);
        Dataset::shuffle(&mut [&mut self.testing_data.layer, &mut self.testing_labels.layer]);
    }

    pub fn f32_vec_to_array(vector: &Vec<f32>, n: usize, m: usize) -> Array2<f64> {
        Array2::from_shape_vec((n, m), vector.to_vec())
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f64)
    }

    pub fn u8_vec_to_array(vector: &Vec<u8>, n: usize, m: usize) -> Array2<f64> {
        Array2::from_shape_vec((n, m), vector.to_vec())
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f64)
    }

    pub fn new() -> Dataset {
        let NormalizedMnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        }: NormalizedMnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(0)
            .test_set_length(10_000)
            .finalize()
            .normalize();

        // Create
        let mut training_data: Array2<f64> = Dataset::f32_vec_to_array(&trn_img, 50_000, 784);

        let mut training_labels: Array2<f64> = Dataset::u8_vec_to_array(&trn_lbl, 50_000, 1);

        let mut testing_data: Array2<f64> = Dataset::f32_vec_to_array(&tst_img, 10_000, 784);

        let mut testing_labels: Array2<f64> = Dataset::u8_vec_to_array(&tst_lbl, 10_000, 1);

        Dataset::shuffle(&mut [&mut training_data, &mut training_labels]);
        Dataset::shuffle(&mut [&mut testing_data, &mut testing_labels]);

        Dataset {
            training_data: Layer::dummy_layer(training_data),
            training_labels: Layer::dummy_layer(training_labels),
            testing_data: Layer::dummy_layer(testing_data),
            testing_labels: Layer::dummy_layer(testing_labels),
        }
    }
}
