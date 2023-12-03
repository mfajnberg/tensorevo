use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::tensor::Tensor;


#[derive(Deserialize, Serialize, PartialEq, Debug)]
#[serde(bound = "")]
pub struct Layer<T: Tensor> {
    pub weights: T,
    pub biases: T,
    pub activation: Activation<T>
}

impl<T: Tensor> Layer<T> {
    pub fn new(weights: T, biases: T, activation: Activation<T>) -> Self {
        return Self{weights, biases, activation}
    }

    pub fn feed_forward(&self, input: &T) -> (T, T) {
        let weighted_input = &self.weights.dot(input) + &self.biases;
        let activation = self.activation.call(&weighted_input);
        return (weighted_input, activation);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::tensor::TensorBase;

    #[test]
    fn test() {
        let input = Array2::from_array(
            [
                [1.],
                [1.],
            ]
        );
        let layer = Layer{
            weights: Array2::from_array(
                [
                    [1., 0.],
                    [0., 1.],
                ]
            ),
            biases: Array2::from_array(
                [
                    [-1.],
                    [-1.],
                ]
            ),
            activation: Activation::<Array2<f64>>::from_name("sigmoid"),
        };
        let (z, a) = layer.feed_forward(&input);
        println!("{}", z);
        println!("{}", a);
    }    
}
