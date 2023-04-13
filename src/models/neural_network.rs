use crate::models::layers::*;

pub struct NeuralNetwork {

    pub layers: Vec<Box<dyn Layer>>

}

impl NeuralNetwork {

    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {

        NeuralNetwork {layers}

    }

    pub fn default() -> Self {

        let layers: Vec<Box<dyn Layer>> = vec![Box::new(ConvolutionLayer::new((3, 3), (1, 1), (0, 0))),
            Box::new(PoolingLayer::new((2, 2), (2, 2), (0, 0))),
            Box::new(DenseLayer::new((1, 2)))];

        NeuralNetwork::new(layers)

    }

    pub fn forward(&self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

        let mut output = input.clone();
        for l in 0..self.layers.len() {

            output = self.layers[l].forward(&output);

        }

        output

    }

}