
use mnist::*;
use ndarray::{Array2, NdIndex, Dim};
use rand::Rng;
pub struct Dataset {
    // 2D array of flattened images from MNIST dataset, shuffled
    pub training_data: Array2<f32>,
    pub training_labels: Array2<f32>,
    pub testing_data: Array2<f32>,
    pub testing_labels: Array2<f32>,
}

impl Dataset {
    fn shuffle(slices: &mut [&mut Array2<f32>]) {
        if slices.len() > 0{
            let mut rng = rand::thread_rng();
            let shared_length = slices[0].index_axis_mut(ndarray::Axis(0), 0).len();
            
            for i in 0..shared_length {
                let next = rng.gen_range(i..shared_length);

                for slice in slices.iter_mut() {
                    let mut row = slice.index_axis_mut(ndarray::Axis(0), 0);
                    row.swap(i, next);
                }
            }
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
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();
        
        
        // Create
        let mut training_data: Array2<f32> = Array2::from_shape_vec((50_000, 784), trn_img)
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f32 / 256.0);
    
        // Convert the returned Mnist struct to Array2 format
        let mut training_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .t()
            .map(|x| *x as f32);
    
        let mut testing_data: Array2<f32> = Array2::from_shape_vec((10_000, 784), tst_img)
            .expect("Error converting images to Array3 struct")
            .t()
            .map(|x| *x as f32 / 256.);
    
        let mut testing_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
            .expect("Error converting testing labels to Array2 struct")
            .t()
            .map(|x| *x as f32);
            
        Dataset::shuffle(&mut [&mut training_data, &mut training_labels]);
        Dataset::shuffle(&mut [&mut testing_data, &mut testing_labels]);
        

        Dataset{ 
            training_data: training_data,
            training_labels: training_labels,
            testing_data: testing_data,
            testing_labels: testing_labels,
        }
    }
}

