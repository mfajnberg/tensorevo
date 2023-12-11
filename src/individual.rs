//! Definition of the [`Individual`] struct.
//!
//! The `Individual` is the centerpiece of the evolutionary process.

use std::fmt::Debug;
use std::fs::read_to_string;
use std::io::Error as IOError;
use std::path::Path;

use log::trace;
use num_traits::FromPrimitive;
use serde_json;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::cost_function::CostFunction;


/// Neural network with evolutionary methods.
///
/// Derives the [`Deserialize`] and [`Serialize`] traits from [`serde`],
/// as well as [`PartialEq`] and [`Debug`].
#[derive(Deserialize, Serialize, PartialEq, Debug)]
#[serde(bound = "")]
pub struct Individual<T: Tensor> {
    /// Layers of the neural network ordered from input to output layer.
    layers: Vec<Layer<T>>,

    /// Cost function used to calculate the error.
    #[serde(default = "CostFunction::default")]
    cost_function: CostFunction<T>,
}


/// Error that may occur when trying to load an [`Individual`] instance from a file.
#[derive(Debug, Error)]
#[error(transparent)]
pub enum LoadError {
    IO(#[from] IOError),
    Deserialization(#[from] serde_json::Error),
}


impl<T: Tensor> Individual<T> {
    /// Constructs a new individual from a vector of layers.
    ///
    /// # Arguments
    /// * `layers` - vector of [`Layer`] structs ordered from input to output layer
    /// * `cost_function` - self explanatory
    ///
    /// # Returns
    /// New [`Individual`] with the given layers
    pub fn new(layers: Vec<Layer<T>>, cost_function: CostFunction<T>) -> Self {
        Self { layers, cost_function }
    }

    /// Load an individual from a json file
    /// 
    /// # Arguments
    /// * `path` - path to the json file
    /// 
    /// # Returns
    /// A new [`Individual`] instance or a [`LoadError`]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        Ok(serde_json::from_str(read_to_string(path)?.as_str())?)
    }

    /// Performs a full forward pass for a given input and returns the network's output.
    ///
    /// # Arguments
    /// * `input` - [Tensor] with a shape that matches the first/input layer
    ///
    /// # Returns
    /// Output tensor from the last layer
    pub fn forward_pass(&self, input: &T) -> T {
        self.layers.iter().fold(input.clone(), |output, layer| layer.feed_forward(&output).1)
    }

    /// Passes the `input` through the network and returns the intermediate results of each layer.
    ///
    /// # Arguments
    /// * `input` - 2D-tensor where each column is an input sample.
    ///             Therefore the number of rows must match the number of input neurons.
    /// 
    /// # Returns
    /// Two vectors of tensors, the first being the weighted inputs to each layer and the second
    /// being the activations of each layer.
    fn forward_pass_memoized(&self, input: &T) -> (Vec<T>, Vec<T>) {
        let num_layers = self.layers.len();
        let mut weighted_inputs: Vec<T> = Vec::<T>::with_capacity(num_layers);
        let mut activations = Vec::<T>::with_capacity(num_layers);
        let mut activation: T = input.clone();
        let mut weighted_input: T;
        // TODO: See, if we actually need this first item in `backprop` (below).
        //       Consider replacing this loop with `Iterator` methods.
        //       https://github.com/mfajnberg/tensorevo/issues/21
        activations.push(activation.clone());
        for layer in &self.layers {
            (weighted_input, activation) = layer.feed_forward(&activation);
            weighted_inputs.push(weighted_input);
            activations.push(activation.clone());
        }
        return (weighted_inputs, activations);
    }

    /// Performs backpropagation with a given batch of inputs and desired outputs.
    ///
    /// Based on the book
    /// [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
    /// by Michael Nielsen
    ///
    /// # Arguments
    /// * `input` - 2D-tensor where each column is an input sample.
    /// * `desired_output` - 2D-tensor where each column is the corresponding desired output
    ///                      for each input sample/column in the `input` tensor.
    /// 
    /// # Returns
    /// The gradient of the cost function parameterized by the given inputs and desired outputs in reverse order
    /// (starting with the last layer).
    fn backprop(&self, input: &T, desired_output: &T) -> (Vec<T>, Vec<T>) {
        let (weighted_inputs, activations) = self.forward_pass_memoized(input);
        let num_layers = self.layers.len();
        // Initialize vectors to hold weights- and biases-gradients for all layers.
        let mut nabla_weights = Vec::<T>::with_capacity(num_layers);
        let mut nabla_biases = Vec::<T>::with_capacity(num_layers);
        // Weighted input to the activation function of the last layer:
        let weighted_input = weighted_inputs.last().unwrap();
        // Activation of the last layer, i.e. the network's output:
        let output = activations.last().unwrap();
        // Derivative of the last layer's activation function with respect to its weighted input:
        let activation_derivative = self.layers[num_layers - 1].activation.call_derivative(weighted_input);
        // Derivative of the cost function with respect to the last layer's activation:
        let cost_derivative = self.cost_function.call_derivative(output, desired_output);
        // Delta of the last layer:
        let delta = cost_derivative * activation_derivative;
        // Calculate and add the last layer's gradient components first.
        nabla_weights.push(delta.dot(activations[num_layers - 2].transpose()));
        nabla_biases.push(delta.sum_axis(1));
        // Loop over the remaining layer indices in reverse order,
        // i.e. starting with the second to last index (`num_layers - 2`) and ending with `0`.
        for layer_num in (0..num_layers - 1).rev() {
            // Weighted input to the layer:
            let weighted_input = &weighted_inputs[layer_num];
            // Activation of the previous layer:
            let previous_activation = if layer_num > 0 { &activations[layer_num - 1] } else { input };
            // Derivative of the layer's activation function with respect to its weighted input:
            let activation_derivative = self.layers[layer_num].activation.call_derivative(weighted_input);
            // Delta of the layer:
            let delta = self.layers[layer_num + 1].weights.transpose().dot(&delta) * activation_derivative;
            // Calculate and add the layer's gradient components.
            nabla_weights.push(delta.dot(previous_activation.transpose()));
            nabla_biases.push(delta.sum_axis(1));
        }
        (nabla_weights, nabla_biases)
    }

    /// Updates the weights and biases of the individual, 
    /// given a single batch of inputs and corresponding desired outputs.
    ///
    /// # Arguments
    /// * `input` - 2D-tensor where each column is an input sample.
    /// * `desired_output` - 2D-tensor where each column is the corresponding desired output
    ///                      for each input sample/column in the `input` tensor.
    /// * `update_factor` - The learning rate divided by the batch size.
    ///                     Adjusts the rate of change to the individual's weights and biases.
    /// * `validation_data` - An optional tuple of validation inputs and outputs used to update the model's error score.
    pub fn stochastic_gradient_descent_step(
        &mut self,
        input: &T,
        desired_output: &T,
        update_factor: T::Component,
        validation_data: Option<(&T, &T)>,
    ) {
        let (nabla_weights, nabla_biases) = self.backprop(input, desired_output);
        let gradients = nabla_weights.iter().zip(&nabla_biases).rev();
        for (idx, (nw, nb)) in gradients.enumerate() {
            let weights_update_factor = T::from_num(update_factor, self.layers[idx].weights.shape());
            self.layers[idx].weights -= &(weights_update_factor * nw);
            let biases_update_factor = T::from_num(update_factor, self.layers[idx].biases.shape());
            self.layers[idx].biases -= &(biases_update_factor * nb);
        }
        if let Some((input, desired_output)) = validation_data {
            trace!("validation error: {}", self.calculate_error(input, desired_output))
        }
    }

    /// Updates the weights and biases of the individual, given an entire vector of training data.
    /// Also updates the individual's error, if validation data is passed in as well.
    ///
    /// # Arguments
    /// * `training_data` - Vector of tuples of two Tensors each, where the first one is a batch of inputs,
    ///                     and the second one is a corresponding batch of output data.
    /// * `validation_data` - An optional tuple of validation inputs and outputs passed onward to 
    ///                       `stochastic_gradient_descent_step`
    /// * `learning_rate` - Determines the rate of change to the individual's weights and biases during training.
    pub fn stochastic_gradient_descent(
        &mut self,
        training_data: Vec<(&T, &T)>,
        learning_rate: f32,
        validation_data: Option<(&T, &T)>,
    ) {
        let (_, batch_size) = training_data[0].0.shape();
        let update_factor = T::Component::from_f32(learning_rate / batch_size as f32).unwrap();
        let num_batches = training_data.len();
        for (i, (batch_inputs, batch_desired_outputs)) in training_data.iter().enumerate() {
            trace!("batch: {}/{}", i+1, num_batches);
            self.stochastic_gradient_descent_step(batch_inputs, batch_desired_outputs, update_factor, validation_data);
        }
    }

    /// Calculates the model's error given some input and desired output.
    /// 
    /// # Arguments
    /// * `input` - Tensor with a shape that matches the first/input layer
    /// * `desired_output` - The desired output for the given input data
    /// 
    /// # Returns
    /// The error value
    pub fn calculate_error(&self, input: &T, desired_output: &T,) -> f32 {
        self.cost_function.call(&self.forward_pass(input), desired_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use ndarray::array;
    use tempfile::NamedTempFile;

    use crate::activation::{Activation, sigmoid, sigmoid_prime, relu, relu_prime};
    use crate::cost_function::{CostFunction, quadratic, quadratic_prime};

    #[test]
    fn test_from_file() -> Result<(), LoadError> {
        let individual_json = r#"{"layers": [
            {
                "weights":    {"v": 1, "dim": [2,2], "data": [0.0,1.0,2.0,3.0]},
                "biases":     {"v": 1, "dim": [2,1], "data": [4.0,5.0]},
                "activation": "sigmoid"
            },
            {
                "weights":    {"v": 1, "dim": [2,2], "data": [0.0,-1.0,-2.0,-3.0]},
                "biases":     {"v": 1, "dim": [2,1], "data": [-4.0,-5.0]},
                "activation": "relu"
            }
        ]}"#;
        let individual_expected = Individual{
            layers: vec![
                Layer::new(
                    array![[0., 1.], [2., 3.]],
                    array![[4.], [5.]],
                    Activation::new("sigmoid", sigmoid, sigmoid_prime),
                ),
                Layer::new(
                    array![[0., -1.], [-2., -3.]],
                    array![[-4.], [-5.]],
                    Activation::new("relu", relu, relu_prime),
                ),
            ],
            cost_function: CostFunction::new("quadratic", quadratic, quadratic_prime),
        };
        let mut individual_file = NamedTempFile::new()?;
        individual_file.write_all(individual_json.as_bytes())?;
        let individual_loaded = Individual::from_file(individual_file.path())?;
        assert_eq!(individual_expected, individual_loaded);
        Ok(())
    }
} 
