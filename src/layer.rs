use serde::{Deserialize, Serialize};

use crate::activation::{Activation, deserialize_activation, serialize_activation};
use crate::tensor::Tensor;


#[derive(Deserialize, Serialize)]
#[serde(bound = "")]
pub struct Layer<T: Tensor> {
    pub weights: T,
    pub biases: T,
    #[serde(serialize_with = "serialize_activation")]
    #[serde(deserialize_with = "deserialize_activation")]
    pub activation: Activation<T>
}

impl<T: Tensor> Layer<T> {
    pub fn feed_forward(&self, input: &T) -> (T, T) {
        let weighted_input = self.weights.dot(input).add(&self.biases);
        let activation = self.activation.call(&weighted_input);
        return (weighted_input, activation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::NDTensor;

    #[test]
    fn test() {
        let input = NDTensor::from_vec(
            &vec![
                vec![1.],
                vec![1.],
            ]
        );
        let layer = Layer{
            weights: NDTensor::from_vec(
                &vec![
                    vec![1., 0.],
                    vec![0., 1.],
                ]
            ),
            biases: NDTensor::from_vec(
                &vec![
                    vec![-1.],
                    vec![-1.],
                ]
            ),
            activation: Activation::<NDTensor<f64>>::from_name("sigmoid"),
        };
        let (z, a) = layer.feed_forward(&input);
        println!("{}", z);
        println!("{}", a);
    }    
}
