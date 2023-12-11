//! Definition of the `Activation` struct and the most common activation functions as well as their
//! derivatives.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::component::TensorComponent;
use crate::tensor::TensorBase;


type TFunc<T> = fn(&T) -> T;


/// Convenience struct to store an activation function together with its name and derivative.
///
/// This is used in the `Layer` struct.
/// Facilitates (de-)serialization.
#[derive(Debug, Eq, PartialEq)]
pub struct Activation<T: TensorBase> {
    name: String,
    function: TFunc<T>,
    derivative: TFunc<T>,
}


/// Methods for convenient construction and calling.
impl<T: TensorBase> Activation<T> {
    /// Basic constructor to manually define all fields.
    pub fn new<S: Into<String>>(name: S, function: TFunc<T>, derivative: TFunc<T>) -> Self {
        Self { name: name.into(), function, derivative }
    }

    /// Convenience constructor for known/available activation functions.
    ///
    /// Pre-defined functions are determined from hard-coded names:
    /// - `sigmoid`
    /// - `relu`
    pub fn from_name<S: Into<String>>(name: S) -> Self {
        let name: String = name.into();
        let function: TFunc<T>;
        let derivative: TFunc<T>;
        if name == "sigmoid" {
            (function, derivative) = (sigmoid, sigmoid_prime);
        } else if name == "relu" {
            (function, derivative) = (relu, relu_prime);
        } else {
            panic!();
        }
        Self { name, function, derivative }
    }

    /// Proxy for the actual activation function.
    pub fn call(&self, tensor: &T) -> T {
        (self.function)(tensor)
    }

    /// Proxy for the derivative of the activation function.
    pub fn call_derivative(&self, tensor: &T) -> T {
        (self.derivative)(tensor)
    }
}


/// Allows `serde` to serialize `Activation` objects.
impl<T: TensorBase> Serialize for Activation<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.name)
    }
}


/// Allows `serde` to deserialize to `Activation` objects.
impl<'de, T: TensorBase> Deserialize<'de> for Activation<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from_name(String::deserialize(deserializer)?))
    }
}


/// Reference implementation of the sigmoid activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn sigmoid<T: TensorBase>(tensor: &T) -> T {
    tensor.map(sigmoid_component)
}


/// Reference implementation of the sigmoid activation function.
///
/// Takes a tensor as input and mutates it in place.
pub fn sigmoid_inplace<T: TensorBase>(tensor: &mut T) {
    tensor.map_inplace(sigmoid_component);
}


/// Sigmoid function for a scalar/number.
pub fn sigmoid_component<C: TensorComponent>(number: C) -> C {
    let one = C::from_usize(1).unwrap();
    one / (one + (-number).exp())
}


/// Reference implementation of the derivative of the sigmoid activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn sigmoid_prime<T: TensorBase>(tensor: &T) -> T {
    tensor.map(sigmoid_prime_component)
}


/// Derivative of the sigmoid function for a scalar/number.
pub fn sigmoid_prime_component<C: TensorComponent>(number: C) -> C {
    let one = C::from_usize(1).unwrap();
    sigmoid_component(number) * (one - sigmoid_component(number))
}


/// Reference implementation of the Rectified Linear Unit (RELU) activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn relu<T: TensorBase>(tensor: &T) -> T {
    tensor.map(relu_component)
}


/// Rectified Linear Unit (RELU) activation function for a scalar/number.
pub fn relu_component<C: TensorComponent>(number: C) -> C {
    let zero = C::from_usize(0).unwrap();
    if number < zero { zero } else { number }
}


/// Reference implementation of the derivative of the Rectified Linear Unit (RELU) activation function.
///
/// Takes a tensor as input and returns a new tensor.
pub fn relu_prime<T: TensorBase>(tensor: &T) -> T {
    tensor.map(relu_prime_component)
}


/// Derivative of the Rectified Linear Unit (RELU) function for a scalar/number.
pub fn relu_prime_component<C: TensorComponent>(number: C) -> C {
    let zero = C::from_usize(0).unwrap();
    let one = C::from_usize(1).unwrap();
    if number < zero { zero } else { one }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    mod test_activation {
        use super::*;

        #[test]
        fn test_from_name() {
            use ndarray::Array2;

            let activation: Activation<Array2<f64>> = Activation::from_name("relu");
            let expected_activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            assert_eq!(activation, expected_activation);
        }

        #[test]
        fn test_call() {
            let tensor = array![[-1., 2.]];
            let activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            let output = activation.call(&tensor);
            let expected_output = array![[0., 2.]];
            assert_eq!(output, expected_output);
        }

        #[test]
        fn test_call_derivative() {
            let tensor = array![[-1., 2.]];
            let activation = Activation{
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            let output = activation.call_derivative(&tensor);
            let expected_output = array![[0., 1.]];
            assert_eq!(output, expected_output);
        }
    }

    #[test]
    fn test_sigmoid_inplace() {
        let mut tensor = array![[0., 36.]];
        sigmoid_inplace(&mut tensor);
        let expected_tensor = array![[0.5, 0.9999999999999998]];
        assert_eq!(tensor, expected_tensor);
    }
}
