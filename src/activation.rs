//! Definition of the `Activation` struct and the most common activation functions as well as their
//! derivatives.

use std::cmp::{Eq, PartialEq};
use std::fmt::Debug;

use serde::{Deserialize, Deserializer, Serializer};

use crate::tensor::{Tensor, TensorElement};


/// Convenience struct to store an activation function together with its name and derivative.
///
/// This is used in the `Layer` struct.
/// Facilitates (de-)serialization.
#[derive(Debug, Eq, PartialEq)]
pub struct Activation<T: Tensor> {
    name: String,
    function: fn(&T) -> T,
    derivative: fn(&T) -> T,
}


/// Methods for convenient construction and calling.
impl<T: Tensor> Activation<T> {
    /// Basic constructor to manually define all fields.
    pub fn new(name: &str, function: fn(&T) -> T, derivative: fn(&T) -> T) -> Self {
        return Self{name: name.to_owned(), function, derivative}
    }

    /// Convenience constructor for known/available activation functions.
    ///
    /// Pre-defined functions are determined from hard-coded names:
    /// - `sigmoid`
    /// - `relu`
    pub fn from_name(name: &str) -> Self {
        let function: fn(&T) -> T;
        let derivative: fn(&T) -> T;
        if name == "sigmoid" {
            (function, derivative) = (sigmoid, sigmoid_prime);
        } else if name == "relu" {
            (function, derivative) = (relu, relu_prime);
        } else {
            panic!();
        }
        return Self {
            name: name.to_owned(),
            function,
            derivative,
        }
    }

    /// Proxy for the actual activation function.
    pub fn call(&self, tensor: &T) -> T {
        return (self.function)(tensor);
    }

    /// Proxy for the derivative of the activation function.
    pub fn call_derivative(&self, tensor: &T) -> T {
        return (self.derivative)(tensor);
    }
}


/// Serializes an `Activation` object using a `serde::Serializer`.
///
/// Used in the `Layer` struct to serialize its `activation` field.
pub fn serialize_activation<S, T>(
    activation: &Activation<T>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Tensor,
{
    return serializer.serialize_str(&activation.name);
}


/// Deserializes from a `serde::Deserializer` to an `Activation` object.
///
/// Used in the `Layer` struct to deserialize to its `activation` field.
pub fn deserialize_activation<'de, D, T>(deserializer: D) -> Result<Activation<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Tensor,
{
    let name = String::deserialize(deserializer)?;
    return Ok(Activation::from_name(name.as_str()));
}


/// Reference implementation of the sigmoid activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn sigmoid<T: Tensor>(tensor: &T) -> T {
    return tensor.map(sigmoid_element);
}


/// Reference implementation of the sigmoid activation function.
///
/// Takes a tensor as input and mutates it in place.
pub fn sigmoid_inplace<T: Tensor>(tensor: &mut T) {
    tensor.map_inplace(sigmoid_element);
}


/// Sigmoid function for a scalar/number.
pub fn sigmoid_element<TE: TensorElement>(number: TE) -> TE {
    let one = TE::from_usize(1).unwrap();
    return one / (one + (-number).exp());
}


/// Reference implementation of the derivative of the sigmoid activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn sigmoid_prime<T: Tensor>(tensor: &T) -> T {
    return tensor.map(sigmoid_prime_element);
}


/// Derivative of the sigmoid function for a scalar/number.
pub fn sigmoid_prime_element<TE: TensorElement>(number: TE) -> TE {
    let one = TE::from_usize(1).unwrap();
    return sigmoid_element(number) * (one - sigmoid_element(number));
}


/// Reference implementation of the Rectified Linear Unit (RELU) activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn relu<T: Tensor>(tensor: &T) -> T {
    return tensor.map(relu_element);
}


/// Rectified Linear Unit (RELU) activation function for a scalar/number.
pub fn relu_element<TE: TensorElement>(number: TE) -> TE {
    let zero = TE::from_usize(0).unwrap();
    if number < zero {
        return zero;
    }
    return number;
}


/// Reference implementation of the derivative of the Rectified Linear Unit (RELU) activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn relu_prime<T: Tensor>(tensor: &T) -> T {
    return tensor.map(relu_prime_element);
}


/// Derivative of the Rectified Linear Unit (RELU) function for a scalar/number.
pub fn relu_prime_element<TE: TensorElement>(number: TE) -> TE {
    let zero = TE::from_usize(0).unwrap();
    let one = TE::from_usize(1).unwrap();
    return if number < zero { zero } else { one };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::NDTensor;

    mod test_activation {
        use super::*;

        #[test]
        fn test_from_name() {
            let activation: Activation<NDTensor<f64>> = Activation::from_name("relu");
            let expected_activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            assert_eq!(activation, expected_activation);
        }

        #[test]
        fn test_call() {
            let tensor = NDTensor::from_vec(&vec![vec![-1., 2.]]);
            let activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            let output = activation.call(&tensor);
            let expected_output = NDTensor::from_vec(&vec![vec![0., 2.]]);
            assert_eq!(output, expected_output);
        }

        #[test]
        fn test_call_derivative() {
            let tensor = NDTensor::from_vec(&vec![vec![-1., 2.]]);
            let activation = Activation{
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            let output = activation.call_derivative(&tensor);
            let expected_output = NDTensor::from_vec(&vec![vec![0., 1.]]);
            assert_eq!(output, expected_output);
        }
    }

    #[test]
    fn test_sigmoid_inplace() {
        let mut tensor = NDTensor::from_vec(&vec![vec![0., 36.]]);
        sigmoid_inplace(&mut tensor);
        let expected_tensor = NDTensor::from_vec(&vec![vec![0.5, 0.9999999999999998]]);
        assert_eq!(tensor, expected_tensor);
    }
}