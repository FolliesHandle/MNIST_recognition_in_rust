
use ndarray::{prelude::Array2, Axis, Zip};

use super::layer::{Layer, ActivationLayer};


pub struct Softmax {
    layer: Layer,
}