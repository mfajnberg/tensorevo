//! Definition of the [`Activation`] struct and the most common activation functions.

use std::sync::RwLock;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::component::TensorComponent;
use crate::tensor::TensorBase;
use crate::utils::registry::Registry;

pub use crate::utils::registered::Registered;
pub use self::functions::*;


type TFunc<T> = fn(&T) -> T;


/// Convenience struct to store an activation function together with its name and derivative.
///
/// This is used in the [`Layer`] struct.
/// Facilitates (de-)serialization.
///
/// [`Layer`]: crate::layer::Layer
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Activation<T: TensorBase> {
    name: String,
    function: TFunc<T>,
    derivative: TFunc<T>,
}


/// Methods for convenient construction and calling.
impl<T: TensorBase> Activation<T> {
    /// Basic constructor.
    pub fn new(name: impl Into<String>, function: TFunc<T>, derivative: TFunc<T>) -> Self {
        Self { name: name.into(), function, derivative }
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


/// Allows [`serde`] to serialize [`Activation`] objects.
impl<T: 'static + TensorBase> Serialize for Activation<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Registered::serialize_as_key(self, serializer)
    }
}


/// Allows [`serde`] to deserialize to [`Activation`] objects.
impl<'de, T: 'static + TensorBase> Deserialize<'de> for Activation<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Registered::deserialize_from_key(deserializer)
    }
}


/// Provides a static registry of [`Activation<T>`] instances.
///
/// Reference implementations for some common functions (and their derivatives) are available by
/// default under the following keys:
/// - `sigmoid` (see [`sigmoid`] and [`sigmoid_prime`])
/// - `relu` (see [`relu`] and [`relu_prime`])
///
/// Custom instances registered via [`Registered::register`] under those names will replace the
/// corresponding default implementations.
///
/// Registered instances can be retrieved by name via [`Registered::get`].
///
/// # Example
/// ...
impl<T: 'static + TensorBase> Registered<String> for Activation<T> {
    /// Returns a reference to the name provided in the [`Activation::new`] constructor.
    fn key(&self) -> &String {
        &self.name
    }

    /// Registers reference implementations of some common activation functions (and their derivatives).
    ///
    /// This function should not be called directly. It is called once during initialization of the
    /// [`Registered::Registry`] singleton for `Activation<T>`.
    fn registry_post_init(registry_lock: &RwLock<Self::Registry>) {
        let mut registry = registry_lock.write().unwrap();
        let _ = registry.add(
            "sigmoid".to_owned(),
            Self::new("sigmoid".to_owned(), sigmoid, sigmoid_prime),
        );
        let _ = registry.add(
            "relu".to_owned(),
            Self::new("relu".to_owned(), relu, relu_prime),
        );
    }
}


/// Reference implementations of some common activation functions as well as their derivatives.
pub mod functions {
    use super::*;

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
}


#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    mod test_activation {
        use super::*;

        #[test]
        fn test_register_and_from_name() {
            use ndarray::Array2;

            type NDActivation = Activation<Array2<f32>>;

            let name = "foo".to_owned();

            let option = NDActivation::get("foo");
            assert_eq!(option, None);

            // Register under that name for the first time.
            let option = NDActivation::new(name.clone(), relu, relu_prime).register();
            assert_eq!(option, None);

            // Get from registry by name.
            let option = NDActivation::get("foo");
            assert_eq!(
                option,
                Some(Activation { name: name.clone(), function: relu, derivative: relu_prime }),
            );

            // Register different one under the same name. Should return the previous one.
            let option = NDActivation::new(name.clone(), sigmoid, sigmoid_prime).register();
            assert_eq!(
                option,
                Some(Activation { name: name.clone(), function: relu, derivative: relu_prime }),
            );

            // Get the new one from the registry by name.
            let option = NDActivation::get("foo");
            assert_eq!(
                option,
                Some(Activation { name: name.clone(), function: sigmoid, derivative: sigmoid_prime }),
            );

            // Get default `sigmoid` from the registry.
            let option = NDActivation::get("sigmoid");
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }),
            );

            // Replace it with a different `Activation` instance.
            fn identity<T: TensorBase>(t: &T) -> T { t.clone() }
            let option = NDActivation::new("sigmoid", identity, identity).register();
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }),
            );
            let option = NDActivation::get("sigmoid");
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: identity, derivative: identity }),
            );
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

    mod test_functions {
        use super::*;

        #[test]
        fn test_sigmoid_inplace() {
            let mut tensor = array![[0., 36.]];
            sigmoid_inplace(&mut tensor);
            let expected_tensor = array![[0.5, 0.9999999999999998]];
            assert_eq!(tensor, expected_tensor);
        }
    }
}
