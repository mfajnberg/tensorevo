//! Definition of the `Individual` struct.
//!
//! The `Individual` is the centerpiece of the evolutionary process.

use std::fs::read_to_string;
use std::io::Error as IOError;
use std::path::Path;

use num_traits::{ToPrimitive, NumCast};
use serde_json;
use serde::{Deserialize, Serialize};
use log::trace;

use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::cost_function::{quadratic_cost_function, quadratic_cost_function_prime};


/// Neural network with evolutionary methods.
///
/// Derives the `Deserialize` and `Serialize` traits from `serde`, as well as `PartialEq` and `Debug`.
#[derive(Deserialize, Serialize, PartialEq, Debug)]
#[serde(bound = "")]
pub struct Individual<T: Tensor> {
    /// Layers of the neural network ordered from input to output layer.
    layers: Vec<Layer<T>>,

    /// Latest cost calculated from a validation dataset.
    error_validation: Option<f32>,
    // TODO: Cost function as a field?
}

#[derive(Debug)]
pub enum LoadError {
    IO(IOError),
    Deserialization(serde_json::Error),
}

impl From<IOError> for LoadError {
    fn from(err: IOError) -> Self {
        return LoadError::IO(err);
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(err: serde_json::Error) -> Self {
        return LoadError::Deserialization(err);
    }
}


impl<T: Tensor> Individual<T> {
    /// Constructs a new individual from a vector of layers.
    ///
    /// Initial validation error is set to `None`.
    ///
    /// # Arguments
    /// * `layers` - vector of `Layer` structs ordered from input to output layer
    ///
    /// # Returns
    /// New `Individual` with the given layers
    pub fn new(layers: Vec<Layer<T>>) -> Self {
        return Individual { 
            layers, 
            error_validation: None,
        }
    }

    /// Load an individual from a json file
    /// 
    /// # Arguments
    /// * `path` - path to the json file
    /// Todo: Make generic to allow both `Path` and `&str` as input type
    /// 
    /// # Returns
    /// A new `Individual` instance or a `LoadError`
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        return Ok(serde_json::from_str(read_to_string(path)?.as_str())?);
    }

    /// Performs a full forward pass for a given input and returns the network's output.
    ///
    /// # Arguments
    /// * `input` - Tensor with a shape that matches the first/input layer
    ///
    /// # Returns
    /// Output tensor from the last layer
    pub fn forward_pass(&self, input: &T) -> T {
        let mut _weighted_input: T;
        let mut output: T = input.clone();
        for layer in (self.layers).iter() {
            (_weighted_input, output) = layer.feed_forward(&output);
        }
        return output;
    }

    /// Performs backpropagation with a given batch of inputs and desired outputs.
    ///
    /// Based on the book
    /// [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
    /// by Michael Nielsen
    ///
    /// # Arguments
    /// * `batch_inputs` - 2D-tensor where each column is an input sample.
    ///                    Therefore the number of rows must match the number of input neurons.
    /// * `batch_desired_outputs` - 2D-tensor where each column is the corresponding desired output
    ///                             for each input sample/column in the `batch_inputs` tensor.
    ///
    /// # Returns
    /// The gradient of the cost function parameterized by the given inputs and desired outputs.
    // TODO: Refactor if possible
    fn backprop(&self, batch_inputs: &T, batch_desired_outputs: &T) -> (Vec<T>, Vec<T>) {
        let num_layers = self.layers.len();
        let mut nabla_weights = Vec::<T>::new();
        let mut nabla_biases = Vec::<T>::new();
        for layer in &self.layers {
            nabla_weights.push(T::zeros(layer.weights.shape()));
            nabla_biases.push(T::zeros(layer.biases.shape()));
        }
        // Feed forward and save weighted inputs and activations for each layer:
        let mut weighted_inputs = Vec::<T>::new();
        let mut activation: T = batch_inputs.clone();
        let mut activations = Vec::<T>::new();
        activations.push(activation.clone());
        for layer in (self.layers).iter() {
            let weighted_input: T;
            (weighted_input, activation) = layer.feed_forward(&activation);
            weighted_inputs.push(weighted_input);
            activations.push(activation.clone());
        }
        // Go backwards
        let cost_derivative = quadratic_cost_function_prime(&activation, batch_desired_outputs);
        let weighted_input = &weighted_inputs[num_layers - 1];
        let mut delta = cost_derivative.hadamard(&self.layers[num_layers - 1].activation.call_derivative(weighted_input));
        nabla_weights[num_layers - 1] = delta.dot(&activations[num_layers - 2].transpose());
        nabla_biases[num_layers - 1] = delta.sum_axis(1);
        for layer_num in (0..num_layers - 1).rev() {
            let weighted_input = &weighted_inputs[layer_num];
            let sp = self.layers[layer_num].activation.call_derivative(weighted_input);
            delta = self.layers[layer_num + 1].weights.transpose().dot(&delta).hadamard(&sp);
            if layer_num > 0 {
                nabla_weights[layer_num] = delta.dot(&activations[layer_num - 1].transpose());
            } else { // activation of input layer == input
                nabla_weights[layer_num] = delta.dot(&batch_inputs.transpose());
            }
            nabla_biases[layer_num] = delta.sum_axis(1);
        }
        return (nabla_biases, nabla_weights);
    }

    pub fn stochastic_gradient_descent_step(
        &mut self,
        batch_inputs: &T,
        batch_desired_outputs: &T,
        update_factor: T::Element,
    ) {
        let num_layers = self.layers.len();
        let (nabla_biases, nabla_weights) = self.backprop(batch_inputs, batch_desired_outputs);
        for idx in 0..num_layers {
            let weights_update_factor = T::from_num(update_factor, self.layers[idx].weights.shape());
            self.layers[idx].weights = self.layers[idx].weights.sub(
                &weights_update_factor.hadamard(&nabla_weights[idx])
            );
            let biases_update_factor = T::from_num(update_factor, self.layers[idx].biases.shape());
            self.layers[idx].biases = self.layers[idx].biases.sub(
                &biases_update_factor.hadamard(&nabla_biases[idx])
            );
        }
        // TODO: Calculate validation error here instead of in `sgd`
    }

    pub fn stochastic_gradient_descent(
        &mut self,
        training_data: &Vec<(T, T)>,
        validation_data: Option<(&T, &T)>,
        learning_rate: f32,
    ) {
        let (_, batch_size) = training_data[0].0.shape();
        let update_factor = <T::Element as NumCast>::from(learning_rate / batch_size.to_f32().unwrap()).unwrap();
        let num_batches = training_data.len();
        for (i, (batch_inputs, batch_desired_outputs)) in training_data.iter().enumerate() {
            trace!("batch: {}/{}", i+1, num_batches);
            self.stochastic_gradient_descent_step(batch_inputs, batch_desired_outputs, update_factor);
            match validation_data {
                None => {},
                Some((input, desired_output)) => {
                    let output = self.forward_pass(input);
                    self.error_validation = Some(quadratic_cost_function(&output, desired_output)) // todo: update_error()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use ndarray::array;
    use tempfile::NamedTempFile;

    use crate::activation::{Activation, sigmoid, sigmoid_prime, relu, relu_prime};
    use crate::tensor::NDTensor;

    #[test]
    fn test_from_file() -> Result<(), LoadError> {
        let individual_json = r#"{"layers":[
            {"weights":[[0,1],[2,3]],"biases":[[4],[5]],"activation":"sigmoid"},
            {"weights":[[0,-1],[-2,-3]],"biases":[[-4],[-5]],"activation":"relu"}
        ]}"#;
        let individual_expected = Individual{
            layers: vec![
                Layer::new(
                    NDTensor::new(array![[0., 1.], [2., 3.]]),
                    NDTensor::new(array![[4.], [5.]]),
                    Activation::new("sigmoid", sigmoid, sigmoid_prime),
                ),
                Layer::new(
                    NDTensor::new(array![[0., -1.], [-2., -3.]]),
                    NDTensor::new(array![[-4.], [-5.]]),
                    Activation::new("relu", relu, relu_prime),
                ),
            ],
            error_validation: None,
        };
        let mut individual_file = NamedTempFile::new()?;
        individual_file.write_all(individual_json.as_bytes())?;
        let individual_loaded = Individual::<NDTensor<f32>>::from_file(individual_file.path())?;
        assert_eq!(individual_expected, individual_loaded);
        return Ok(());
    }
    
    fn test_sgd() {
        let mut individual = Individual::new(
            vec![
                Layer{
                    weights: NDTensor::from_vec(
                        &vec![
                            vec![1., 0.],
                            vec![0., 1.],
                        ]
                    ),
                    biases: NDTensor::from_vec(
                        &vec![
                            vec![0.],
                            vec![0.],
                        ]
                    ),
                    activation: Activation::<NDTensor<f64>>::from_name("relu"),
                },
                Layer{
                    weights: NDTensor::from_vec(
                        &vec![
                            vec![1., 0.],
                            vec![0., 1.],
                        ]
                    ),
                    biases: NDTensor::from_vec(
                        &vec![
                            vec![0.],
                            vec![0.],
                        ]
                    ),
                    activation: Activation::<NDTensor<f64>>::from_name("relu"),
                },
            ]
        );
        let input1 = NDTensor::from_vec(
            &vec![
                vec![1., 2.],
                vec![1., 3.],
            ]
        );
        let input2 = NDTensor::from_vec(
            &vec![
                vec![4., 1.],
                vec![5., 1.],
            ]
        );
        println!("inputs:\n{}\n{}\n", &input1, &input2);
        let expected_output1 = NDTensor::from_vec(
            &vec![
                vec![10., 5.],
                vec![1., 5.],
            ]
        );
        let expected_output2 = NDTensor::from_vec(
            &vec![
                vec![5., 10.],
                vec![4., 1.],
            ]
        );
        println!("output1 before sgd:\n{}\n", individual.forward_pass(&input1));
        println!("output2 before sgd:\n{}\n", individual.forward_pass(&input2));
        let training_data = vec![(input1, expected_output1), (input2, expected_output2)];
        for i in 0..500 {
            println!("\nepoch: {}", i + 1);
            individual.stochastic_gradient_descent(&training_data, None, 0.01);
            println!("weights 1:\n{}", individual.layers[0].weights);
            println!("biases 1:\n{}", individual.layers[0].biases);
            println!("weights 2:\n{}", individual.layers[1].weights);
            println!("biases 2:\n{}", individual.layers[1].biases);
            println!("\noutput1:\n{}", individual.forward_pass(&training_data[0].0));
            println!("output2:\n{}", individual.forward_pass(&training_data[1].0));
        }
    }
}
