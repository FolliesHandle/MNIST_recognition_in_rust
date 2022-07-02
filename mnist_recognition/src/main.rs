mod model;
mod layers;


use crate::model::Model;

fn main() {
    let mut model = Model::new(0.1);
    model.train(5000);
    model.test();
}
