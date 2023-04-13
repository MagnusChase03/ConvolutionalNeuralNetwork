mod models;
mod utils;

use models::layers::Layer;
use models::neural_network::*;

fn main() {
    

    let nn = NeuralNetwork::default();
    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0]; 4];

    let outputs = nn.forward(&inputs);
    println!("{:?}", outputs);
    

}
