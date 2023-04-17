mod models;
mod utils;

use models::neural_network::*;
use utils::functions::log_liklyhood;

use models::layers::*;

fn main() {
    

    // let nn = NeuralNetwork::default();
    // let inputs = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]; 5];
    let inputs = vec![vec![1.0, 2.0, 3.0]];
    let errors = vec![vec![-0.025, 0.075]];

    // let outputs = nn.forward(&inputs);
    // let perc = log_liklyhood(&outputs[0]);
    // println!("{:?}", perc);

    let mut dd = DenseLayer::new((3, 2));
    let mut output = dd.forward(&inputs);

    println!("{:?}", dd.weights);
    println!("{:?}", output);

    dd.backward(&inputs, &errors);

    output = dd.forward(&inputs);

    println!("{:?}", dd.weights);
    println!("{:?}", output);
    

}
