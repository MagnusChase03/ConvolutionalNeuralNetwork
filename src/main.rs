mod models;
mod utils;

use models::{layers::Layer, layers};

fn main() {
    
    // let mut pooling_layer = layers::PoolingLayer::new((2, 2), (2, 2), (1, 1));

    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0]; 4];
    // let output = pooling_layer.forward(&inputs);
    // println!("{:?}", output);

    let mut convolution_layer = layers::ConvolutionLayer::new((2, 2), (1, 1), (0, 0));

    // println!("{:?}", convolution_layer.weights);

    let output = convolution_layer.forward(&inputs);

    println!("{:?}", output);

}
