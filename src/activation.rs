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
        get_or_init!(|| {
            let registry_lock = RwLock::new(ActivationRegistry::<T>::new());
            Self::register_common(&registry_lock);
            registry_lock
        })
    }

    /// Registers reference implementations of some common activation functions.
    ///
    /// # Arguments
    /// - `registry_lock` - Reference to the activation registry wrapped in a [`RwLock`]
    fn register_common(registry_lock: &RwLock<ActivationRegistry<T>>) {
        let mut registry = registry_lock.write().unwrap();
        let _ = registry.insert(
            "sigmoid".to_owned(),
            Self::new("sigmoid".to_owned(), sigmoid, sigmoid_prime),
        );
        let _ = registry.insert(
            "relu".to_owned(),
            Self::new("relu".to_owned(), relu, relu_prime),
        );
    }

    /// Creates and registers a new [`Activation`] with the specified parameters.
    ///
    /// The newly created `Activation` instance is added to the activation registry.
    /// Subsequent calls to [`Activation::from_name`] passing the same name will return a clone of it.
    ///
    /// If an `Activation` was already registered under the specified `name`, it will be
    /// replaced by the newly created one.
    ///
    /// # Arguments:
    /// - `name` - Key under which to register the new activation;
    ///            also passed on to the [`Activation::new`] constructor.
    /// - `function` - Passed on to the [`Activation::new`] constructor.
    /// - `derivative` - Passed on to the [`Activation::new`] constructor.
    ///
    /// # Returns
    /// [`None`] if an activation with that name had not been registered before.
    /// Otherwise the new `Activation` instance with the specified parameters is inserted and the old one is returned.
    pub fn register<S: Into<String>>(name: S, function: TFunc<T>, derivative: TFunc<T>) -> Option<Self> {
        let name: String = name.into();
        let registry_lock = Self::get_activation_registry();
        registry_lock.write().unwrap()
                     .insert(name.clone(), Activation::new(name, function, derivative))
    }

    /// Retrieves a clone of a previously registered [`Activation`] by name.
    ///
    /// Reference implementations for the following activation functions are available by default:
    /// - `sigmoid`
    /// - `relu`
    ///
    /// A custom instance registered via [`Activation::register`] under one of those names will
    /// replace the default implementation.
    ///
    ///
    /// # Arguments
    /// - `name` - The name/key of the activation to return.
    ///
    /// # Returns
    /// Clone of the [`Activation`] instance with the specified `name` or [`None`] if none was
    /// registered under that name.
    pub fn from_name<S: Into<String>>(name: S) -> Option<Self> {
        let registry_lock = Self::get_activation_registry();
        registry_lock.read().unwrap()
                     .get(&name.into())
                     .and_then(|activation| Some(activation.clone()))
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
        Ok(Activation::from_name(String::deserialize(deserializer)?).unwrap())
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

            type NDActivation = Activation<Array2<f32>>;

            let name = "foo".to_owned();

            let option = NDActivation::from_name("foo");
            assert_eq!(option, None);

            // Register under that name for the first time.
            let option = NDActivation::register(name.clone(), relu, relu_prime);
            assert_eq!(option, None);

            // Get from registry by name.
            let option = NDActivation::from_name("foo");
            assert_eq!(option, Some(Activation { name: name.clone(), function: relu, derivative: relu_prime }));

            // Register different one under the same name. Should return the previous one.
            let option = NDActivation::register(name.clone(), sigmoid, sigmoid_prime);
            assert_eq!(option, Some(Activation { name: name.clone(), function: relu, derivative: relu_prime }));

            // Get the new one from the registry by name.
            let option = NDActivation::from_name("foo");
            assert_eq!(option, Some(Activation { name: name.clone(), function: sigmoid, derivative: sigmoid_prime }));

            // Get default `sigmoid` from the registry.
            let option = NDActivation::from_name("sigmoid");
            assert_eq!(option, Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }));

            // Replace it with a different `Activation` instance.
            fn identity<T: TensorBase>(t: &T) -> T { t.clone() }
            let option = NDActivation::register("sigmoid", identity, identity);
            assert_eq!(option, Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }));
            let option = NDActivation::from_name("sigmoid");
            assert_eq!(option, Some(Activation { name: "sigmoid".to_owned(), function: identity, derivative: identity }));
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
