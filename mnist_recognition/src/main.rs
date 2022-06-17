mod model;

use crate::model::Model;

fn main() {
    let mut model = Model::new();
    model.train(0.10, 500);
    model.test();
}
