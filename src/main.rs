mod models;
mod utils;

use models::neural_network::*;
use utils::functions::log_liklyhood;

fn main() {
    

    let nn = NeuralNetwork::default();
    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]; 5];

    let outputs = nn.forward(&inputs);
    let perc = log_liklyhood(&outputs[0]);
    println!("{:?}", perc);
    

}
