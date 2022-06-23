mod model;
mod layers;


use crate::model::Model;

fn main() {
    let mut model = Model::new();
    model.train(0.1, 5000);
    model.test();
}
