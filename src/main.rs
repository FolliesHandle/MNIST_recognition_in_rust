mod layers;
mod model;
use clap::Parser;

use crate::model::Model;

// command-line parsing for hyperparameters
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// # of nodes in the hidden layer
    #[clap(short, long, value_parser, default_value_t = 128)]
    layer_size: usize,
    /// Learning rate of the network
    #[clap(short, long, value_parser, default_value_t = 0.01)]
    alpha: f32,
    /// Batch size for BGD
    #[clap(short, long, value_parser, default_value_t = 100)]
    batch_size: isize,
    /// Amount of epochs to train for
    #[clap(short, long, value_parser, default_value_t = 1000)]
    epochs: usize,
}

fn main() {
    let args = Args::parse();
    let mut model = Model::new(args.layer_size, args.alpha, args.batch_size);
    model.train(args.epochs);
    model.test();
}
