use mnist::*;
use ndarray::s;
use ndarray::{Array2};
use rand::Rng;

// use crate::model::CONFIG;

use super::layer::Layer;
// pub struct Slices {
//     data: Layer,
//     label: Layer,
// }

pub struct Dataset {
    // 2D array of flattened images from MNIST Dataset, shuffled
    pub training_data: Layer,
    pub training_labels: Layer,
    pub testing_data: Layer,
    pub testing_labels: Layer,
    // testing_slice: Slices,
    // training_slice: Slices,
}

impl Dataset {
    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        let shared_length = self.training_data.layer.index_axis_mut(ndarray::Axis(0), 0).len();
        assert_eq!(self.training_data.layer.ncols(), self.training_labels.layer.ncols());
        println!("Shuffling Data!");
        for i in 0..shared_length {
            let next = rng.gen_range(0..shared_length);
            if i != next {
                let (mut index, mut random) = self.training_data.layer.multi_slice_mut((s![.., i], s![.., next]));
                std::mem::swap(&mut index, &mut random);

                let (mut index_2, mut random_2) = self.training_labels.layer.multi_slice_mut((s![.., i], s![.., next]));
                std::mem::swap(&mut index_2, &mut random_2);
            }
        }
        
    }

    pub fn vec_to_array(vector: &Vec<u8>, n: usize, m: usize, data: bool) -> Array2<f32> {
        if data {
            Array2::from_shape_vec((n, m), vector.to_vec())
                .expect("Error converting images to Array3 struct")
                .t()
                .map(|x| *x as f32)
                .map(|x| *x / 256f32)
        }    
        else {
            Array2::from_shape_vec((n, m), vector.to_vec())
                .expect("Error converting images to Array3 struct")
                .t()
                .map(|x| *x as f32)
        }
    }

    pub fn new() -> Dataset {
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
            .test_set_length(10_000)
            .finalize();

        // Create
        let training_data: Array2<f32> = Dataset::vec_to_array(&trn_img, 50_000, 784, true);
        assert!(!training_data.is_empty());
        let training_labels: Array2<f32> = Dataset::vec_to_array(&trn_lbl, 50_000, 1, false);
        assert!(!training_labels.is_empty());
        let testing_data: Array2<f32> = Dataset::vec_to_array(&tst_img, 10_000, 784, true);
        assert!(!testing_data.is_empty());
        let testing_labels: Array2<f32> = Dataset::vec_to_array(&tst_lbl, 10_000, 1, false);
        assert!(!testing_labels.is_empty());

        Dataset {
            training_data: Layer::dummy_layer(training_data),
            training_labels: Layer::dummy_layer(training_labels),
            testing_data: Layer::dummy_layer(testing_data),
            testing_labels: Layer::dummy_layer(testing_labels),
        }
    }

    // pub fn set_slice(self, index: isize, mode: CONFIG) {
    //     match mode {
    //         CONFIG::TRAIN => {
    //             self.training_slice.data = Layer::dummy_layer(
    //                 self.training_data
    //                     .layer
    //                     .slice_axis(Axis(1), Slice::new(index, Some(index), 1))
    //                     .to_owned(),
    //             );
    //             self.training_slice.label = Layer::dummy_layer(
    //                 self.training_data
    //                     .layer
    //                     .slice_axis(Axis(1), Slice::new(index, Some(index), 1))
    //                     .to_owned(),
    //             );
    //         }
    //         CONFIG::TEST => {
    //             self.testing_slice.data = Layer::dummy_layer(
    //                 self.training_data
    //                     .layer
    //                     .slice_axis(Axis(1), Slice::new(index, Some(index), 1))
    //                     .to_owned(),
    //             );
    //             self.testing_slice.label = Layer::dummy_layer(
    //                 self.training_data
    //                     .layer
    //                     .slice_axis(Axis(1), Slice::new(index, Some(index), 1))
    //                     .to_owned(),
    //             );
    //         }
    //     }
    // }
}
