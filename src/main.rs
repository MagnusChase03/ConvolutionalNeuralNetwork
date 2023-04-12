mod models;

use models::{layers::Layer, layers};

fn main() {
    
    let mut pooling_layer = layers::PoolingLayer::new((2, 2), (1, 1), (0, 0));

    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0]; 4];
    let output = pooling_layer.forward(&inputs);

    println!("{:?}", output);

}
