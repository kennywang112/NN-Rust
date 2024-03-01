use rand::prelude::*;
use crate::activation_function::sigmoid;

pub fn feedforward() {

    // Parameters
    let input_size: usize = 2;
    let hidden_size: usize = 3;
    let output_size: usize = 1;
    // Weights
    let weights_input_hidden = vec![vec![0.0; hidden_size]; input_size];
    let weights_hidden_output = vec![vec![0.0; output_size]; hidden_size];
    // Bias
    let bias_hidden = rand::thread_rng().gen_range(-1.0..1.0);
    let bias_output = rand::thread_rng().gen_range(-1.0..1.0);

    println!("{:?}", weights_input_hidden);
    println!("{:?}", bias_hidden);

    let input_data = vec![0.5, 0.8];

    // Feedforward
    let mut hidden_layer_input = vec![0.0; hidden_size];
    for i in 0..input_size {
        for j in 0..hidden_size {
            hidden_layer_input[j] += input_data[i] * weights_input_hidden[i][j];
            println!("{:?}", hidden_layer_input[j])
        }
    }
    let hidden_layer_output: Vec<f64> = hidden_layer_input.iter().map(|&x| sigmoid(x + bias_hidden)).collect();

    let mut output_layer_input = vec![0.0; output_size];
    for i in 0..hidden_size {
        for j in 0..output_size {
            output_layer_input[j] += hidden_layer_output[i] * weights_hidden_output[i][j];
        }
    }
    let output_layer_output: Vec<f64> = output_layer_input.iter().map(|&x| sigmoid(x + bias_output)).collect();

    println!("Input: {:?}", input_data);
    println!("Output: {:?}", output_layer_output);
}