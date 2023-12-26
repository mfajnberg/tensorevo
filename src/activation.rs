//! Definition of the [`Activation`] struct and the most common activation functions as well as their
//! derivatives.

use std::collections::HashMap;
use std::sync::RwLock;

use generic_singleton::get_or_init;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::component::TensorComponent;
use crate::tensor::TensorBase;


type ActivationRegistry<T> = HashMap<String, Activation<T>>;
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
    pub fn new<S: Into<String>>(name: S, function: TFunc<T>, derivative: TFunc<T>) -> Self {
        let name: String = name.into();
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


/// Associated functions for registering and retrieving activation functions.
///
/// The activation registry is a singleton.
/// It is static from the moment of initalization and generic over the [`TensorBase`] type.
///
/// # Implementation detail
/// Under the hood the activation registry uses the **[`generic_singleton`]** crate
/// to initialize and access a [`HashMap`] with the names as [`String`] keys and [`Activation`] values.
///
/// [`generic_singleton`]: https://docs.rs/generic_singleton/latest/generic_singleton/
impl<T: TensorBase + 'static> Activation<T> {
    /// Gets a static reference to the activation registry singleton.
    /// The registry is wrapped in a [`RwLock`].
    ///
    /// When called for the first time, the activation registry is initialized first.
    fn get_activation_registry() -> &'static RwLock<ActivationRegistry<T>> {
        get_or_init!(|| RwLock::new(ActivationRegistry::<T>::new()))
    }

    /// Creates and registers a new [`Activation`].
    ///
    /// A clone of the newly created [`Activation`] is added to the activation registry.
    /// Subsequent calls to [`Activation::from_name`] passing the same name will return a clone of it.
    ///
    /// # Arguments:
    /// - `name` - Key under which to register the new activation;
    ///            also passed on to the [`Activation::new`] constructor.
    /// - `function` - Passed on to the [`Activation::new`] constructor.
    /// - `derivative` - Passed on to the [`Activation::new`] constructor.
    ///
    /// # Returns
    /// Clone of the newly created [`Activation`] with the specified parameters.
    pub fn register<S: Into<String>>(name: S, function: TFunc<T>, derivative: TFunc<T>) -> Self {
        let name: String = name.into();
        let registry_lock = Self::get_activation_registry();
        if registry_lock.read().unwrap().contains_key(&name) {
            // TODO: Instead of panicking, make this function return a `Result`.
            panic!("Activation already registered: {}", name);
        }
        let new_activation = Activation::<T>::new(name.clone(), function, derivative);
        registry_lock.write().unwrap().insert(name, new_activation.clone());
        new_activation
    }

    /// Returns a clone of a previously registered [`Activation`] by name.
    ///
    /// Unless custom instances were added before via [`Activation::register`], reference implementations
    /// for the following activation functions will be available by default:
    /// - `sigmoid`
    /// - `relu`
    ///
    /// # Arguments
    /// - `name` - The name/key of the activation to return.
    ///
    /// # Returns
    /// Clone of the [`Activation`] instance with the specified `name`.
    pub fn from_name<S: Into<String>>(name: S) -> Self {
        let name: String = name.into();
        let registry_lock = Self::get_activation_registry();
        // The registry should only be empty the first time this method is called.
        // In that case, fill it with the known/common activation functions.
        if registry_lock.read().unwrap().is_empty() {
            Self::register_common(registry_lock)
        }
        if !registry_lock.read().unwrap().contains_key(&name) {
            // TODO: Instead of panicking, make this function return a `Result`.
            panic!("No such key registered: {}", name)
        }
        registry_lock.read().unwrap().get(&name).unwrap().clone()
    }

    /// Registers reference implementations of some common activation functions.
    ///
    /// Activation functions that will be available by name after calling this method:
    /// - `sigmoid`
    /// - `relu`
    ///
    /// # Arguments
    /// - `registry_lock` - Reference to the activation registry wrapped in a [`RwLock`]
    fn register_common(registry_lock: &RwLock<ActivationRegistry<T>>) {
        // TODO: Consider using `HashMap::try_insert` instead to avoid overwriting any custom
        //       implementations of common activation functions.
        registry_lock.write().unwrap().insert(
            "sigmoid".to_owned(),
            Activation::<T>::new("sigmoid".to_owned(), sigmoid, sigmoid_prime),
        );
        registry_lock.write().unwrap().insert(
            "relu".to_owned(),
            Activation::<T>::new("relu".to_owned(), relu, relu_prime),
        );
    }
}


/// Allows [`serde`] to serialize [`Activation`] objects.
impl<T: TensorBase> Serialize for Activation<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.name)
    }
}


/// Allows [`serde`] to deserialize to [`Activation`] objects.
impl<'de, T: TensorBase + 'static> Deserialize<'de> for Activation<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Activation::from_name(String::deserialize(deserializer)?))
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
        fn test_register_and_from_name() {
            use ndarray::Array2;

            let name = "foo".to_owned();
            let function = relu;
            let derivative = relu_prime;

            // Register:
            let activation = Activation::<Array2<f64>>::register(name.clone(), function, derivative);
            let expected_activation = Activation { name, function, derivative };
            assert_eq!(activation, expected_activation);

            // Get from registry by name:
            let activation = Activation::from_name("foo");
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
