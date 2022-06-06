mod dataset;

use crate::dataset::Dataset;

use ndarray::prelude::*;
use rand::prelude::*;

fn max(num: f32) -> f32 {
    if num > 0.0 {
        return num;
    }
    0.0
}

fn relu(preactivation_layer: Array2<f32>) -> Array2<f32> {
    for mut ele in preactivation_layer.iter() {
        ele = &max(*ele);
    }
    preactivation_layer
}

fn forward_prop(
    layer_1_weights: Array2<f32>,
    layer_1_biases: Array2<f32>,
    layer_2_weights: Array2<f32>,
    layer_2_biases: Array2<f32>,
    input_layer: Array2<f32>,
) {
    let layer_1_preactivation: Array2<f32> = &layer_1_weights * &input_layer + &layer_1_biases;
    println!("{:?}", layer_1_preactivation);
    let layer_1_activated: Array2<f32> = relu(layer_1_preactivation);
    println!("{:?}", layer_1_activated);
    let layer_2_preactivation: Array2<f32> = &layer_2_weights * &layer_1_activated + &layer_2_biases;
    println!("{:?}", layer_2_preactivation);
    //let mut layer_2_activated: Array2<f32> = softmax(&mut layer_2_preactivation);
    //println!("{:?}", layer_2_activated);
}

fn main() {
    let dataset = Dataset::new();
    let mut rng = rand::thread_rng();

    let _hidden_layer_1_weights: Array2<f32> =
        Array2::from_shape_simple_fn((10, 784), || rng.gen::<f32>() - 0.5f32);
    let _hidden_layer_1_biases: Array2<f32> =
        Array2::from_shape_simple_fn((10, 1), || rng.gen::<f32>() - 0.5f32);

    let _hidden_layer_2_weights: Array2<f32> =
        Array2::from_shape_simple_fn((10, 10), || rng.gen::<f32>() - 0.5f32);
    let _hidden_layer_2_biases: Array2<f32> =
        Array2::from_shape_simple_fn((10, 1), || rng.gen::<f32>() - 0.5f32);
}
