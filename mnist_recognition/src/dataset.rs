
use core::slice;

use mnist::*;
use ndarray::{Array2, Data};
use rand::Rng;
pub struct Dataset {
    // 2D array of flattened images from MNIST dataset, shuffled
    training_data: Array2<f32>,
    training_labels: Array2<f32>,
    testing_data: Array2<f32>,
    testing_labels: Array2<f32>,
}

impl Dataset {
    fn shuffle<T: Copy>(slices: &mut [&mut [T]]) {
        if slices.len() > 0{
            let mut rng = rand::thread_rng();
            let shared_length = slices[0].len();
            assert!(slices.iter().all(|s| s.len() == shared_length));
            
            for i in 0..shared_length {
                let next = rng.gen_range(i..shared_length);

                for slice in slices.iter_mut() {
                    let tmp: T = slice[i];
                    slice[i] = slice[next];
                    slice[next] = tmp;
                }
            }
        }
        
    }

    pub fn new() -> Dataset {
        let Mnist {
            mut trn_img,
            mut trn_lbl,
            mut tst_img,
            mut tst_lbl,
            ..
        }: Mnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();
        
        Dataset::shuffle(&mut [&mut trn_img, &mut trn_lbl]);
        Dataset::shuffle(&mut [&mut tst_img, &mut tst_lbl]);
    
        // Create
        let training_data: Array2<f32> = Array2::from_shape_vec((50_000, 784), trn_img)
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f32 / 256.0);
    
        // Convert the returned Mnist struct to Array2 format
        let training_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .t()
            .map(|x| *x as f32);
    
        let testing_data: Array2<f32> = Array2::from_shape_vec((10_000, 784), tst_img)
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f32 / 256.);
    
        let testing_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
            .expect("Error converting testing labels to Array2 struct")
            .t()
            .map(|x| *x as f32);
        Dataset{ 
            training_data: training_data,
            training_labels: training_labels,
            testing_data: testing_data,
            testing_labels: testing_labels,
        }
    }
}

