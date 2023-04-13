pub fn log_liklyhood(input: &Vec<f64>) -> Vec<f64> {

    let mut output = vec![0.0; input.len()];

    let mut sum = 0.0;
    for i in 0..input.len() {

        sum += input[i];
        
    }

    for i in 0..input.len() {

        output[i] = input[i] / sum;
        
    }

    output

}