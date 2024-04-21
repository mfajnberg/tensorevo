//! Definition of the [`Activation`] struct and the most common activation functions.

use std::sync::RwLock;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::component::TensorComponent;
use crate::tensor::TensorBase;
use crate::utils::registry::Registry;

pub use crate::utils::registered::Registered;
pub use self::functions::*;


type TFunc<T> = fn(&T) -> T;


/// Contains a named activation function together with its derivative.
///
/// Used primarily in the [`Layer`] struct.
///
/// See the [**`Registerd`** trait implementation](#impl-Registered<String>-for-Activation<T>)
/// below for a more general usage example.
///
/// [`Layer`]: crate::layer::Layer
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Activation<T: TensorBase> {
    name: String,
    function: TFunc<T>,
    derivative: TFunc<T>,
}


/// Methods for construction and calling.
///
/// Given some `act: Activation<T>` and a `tensor: impl TensorBase` you can call the activation
/// function simply by doing **`act(&tensor)`** and the derivative via **`act.d(&tensor)`**.
impl<T: TensorBase> Activation<T> {
    /// Basic constructor.
    ///
    /// # Arguments
    /// - `name` - Will be used for [serialization](#impl-Serialize-for-Activation<T>)
    ///            and as the key in the static registry.
    /// - `function` - The actual activation function.
    /// - `derivative` - The derivative of `function`. (Needed for backpropagation.)
    ///
    /// # Returns
    /// A new instance of `Activation<T>` with the provided values.
    pub fn new(name: impl Into<String>, function: TFunc<T>, derivative: TFunc<T>) -> Self {
        Self { name: name.into(), function, derivative }
    }

    /// Proxy for the derivative of the activation function.
    pub fn d(&self, tensor: &T) -> T {
        (self.derivative)(tensor)
    }

    /// Returns the name of the activation function.
    pub fn name(&self) -> &str {
        &self.name
    }
}


/// Makes `Activation<T>` callable by value (i.e. consuming the instance).
///
/// This is mainly implemented because [`FnOnce`] is a supertrait of [`Fn`].
impl<T: TensorBase> FnOnce<(&T,)> for Activation<T> {
    type Output = T;

    /// Proxy for the actual underlying activation function.
    extern "rust-call" fn call_once(self, args: (&T,)) -> Self::Output {
        (self.function)(args.0)
    }
}


/// Makes `Activation<T>` callable by mutable reference.
///
/// This is mainly implemented because [`FnMut`] is a supertrait of [`Fn`].
impl<T: TensorBase> FnMut<(&T,)> for Activation<T> {
    /// Proxy for the actual underlying activation function.
    extern "rust-call" fn call_mut(&mut self, args: (&T,)) -> Self::Output {
        (self.function)(args.0)
    }
}


/// Makes `Activation<T>` callable by immutable reference.
///
/// This allows you to use the call operator `( )` on instances and essentially treat them as functions.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use tensorevo::activation::Activation;
/// use tensorevo::tensor::TensorBase;
///
/// fn zero_function<T: TensorBase>(t: &T) -> T {
///     T::zeros(t.shape::<T::Dim>())
/// }
///
/// fn main() {
///     let a = Activation::new("zero", zero_function, zero_function);
///     let x = array![[1., 2.]];
///     let y = a(&x);
///     assert_eq!(y, array![[0., 0.]]);
/// }
/// ```
///
/// In the example above, activation function and derivative are identical.
/// In practice they are obviously different functions.
/// The call operator always calls the actual activation function.
/// To call the derivative use the [`Activation::d`] method.
impl<T: TensorBase> Fn<(&T,)> for Activation<T> {
    /// Proxy for the actual underlying activation function.
    extern "rust-call" fn call(&self, args: (&T,)) -> Self::Output {
        (self.function)(args.0)
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
    fn deserialize<DE: Deserializer<'de>>(deserializer: DE) -> Result<Self, DE::Error> {
        Registered::deserialize_from_key(deserializer)
    }
}


/// Provides a static registry of [`Activation<T>`] instances.
///
/// Reference implementations for some common functions (and their derivatives) are available by
/// default under the following keys:
/// - `sigmoid` (see [`sigmoid`] and [`sigmoid_prime`])
/// - `relu` (see [`relu`] and [`relu_prime`])
/// - `identity` (see [`identity`] and [`to_one`])
///
/// Custom instances registered via [`Registered::register`] under those names will replace the
/// corresponding default implementations.
///
/// Registered instances can be retrieved by name via [`Registered::get`].
///
/// # Example
///
/// ```rust
/// use ndarray::{Array2, array};
/// use num_traits::One;
///
/// use tensorevo::activation::{Activation, Registered};
/// use tensorevo::tensor::TensorOp;
///
/// fn double<T: TensorOp>(t: &T) -> T {
///     t + t
/// }
///
/// fn to_two<T: TensorOp>(t: &T) -> T {
///     let ones = T::from_num(T::Component::one(), t.shape::<T::Dim>());
///     &ones + &ones
/// }
///
/// fn main() {
///     type T = Array2::<f32>;
///
///     // Register a custom activation function:
///     Activation::<T>::new("double", double, to_two).register();
///
///     // Get a previously registered custom activation function:
///     let activation = Activation::<T>::get("double").unwrap();
///     let input = array![[2., -1.]];
///     assert_eq!(activation(&input), array![[4., -2.]]);
///     assert_eq!(activation.d(&input), array![[2., 2.]]);
///
///     // No activation with the name `foo` was registered:
///     assert_eq!(Activation::<T>::get("foo"), None);
///
///     // Common activation functions are available by default:
///     let relu = Activation::<T>::get("relu").unwrap();
///     assert_eq!(relu(&input), array![[2., 0.]]);
///     assert_eq!(relu.d(&input), array![[1., 0.]]);
/// }
/// ```
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
        let _ = registry.add(
            "identity".to_owned(),
            Self::new("identity".to_owned(), identity, to_one),
        );
    }
}


/// Reference implementations of some common activation functions as well as their derivatives.
pub mod functions {
    use num_traits::One;

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
        let one = C::one();
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
        let one = C::one();
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
        let zero = C::zero();
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
        let zero = C::zero();
        let one = C::one();
        if number < zero { zero } else { one }
    }


    /// Identity function returning a clone of the input `tensor`.
    pub fn identity<T: TensorBase>(tensor: &T) -> T {
        tensor.clone()
    }


    /// Returns a tensor filled with ones with the shape of `tensor`. (Derivative of the identity.)
    pub fn to_one<T: TensorBase>(tensor: &T) -> T {
        T::from_num(T::Component::one(), tensor.shape::<T::Dim>())
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;

    mod test_activation {
        use super::*;

        #[test]
        fn test_new() {
            let activation = Activation::<Array2<f32>>::new("relu", relu, relu_prime);
            assert_eq!(
                activation,
                Activation { name: "relu".to_owned(), function: relu, derivative: relu_prime },
            );
        }

        #[test]
        fn test_d() {
            let tensor = array![[-1., 2.]];
            let activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime,
            };
            let output = activation.d(&tensor);
            let expected_output = array![[0., 1.]];
            assert_eq!(output, expected_output);
        }

        #[test]
        fn test_registered() {
            type NDActivation = Activation<Array2<f32>>;

            // Nothing registered under the key "foo".
            let option = NDActivation::get("foo");
            assert_eq!(option, None);

            // Register under that name for the first time.
            let option = NDActivation::new("foo", relu, relu_prime).register();
            assert_eq!(option, None);

            // Get from registry by name.
            let option = NDActivation::get("foo");
            assert_eq!(
                option,
                Some(Activation { name: "foo".to_owned(), function: relu, derivative: relu_prime }),
            );

            // Register different one under the same name. Should return the previous one.
            let option = NDActivation::new("foo", sigmoid, sigmoid_prime).register();
            assert_eq!(
                option,
                Some(Activation { name: "foo".to_owned(), function: relu, derivative: relu_prime }),
            );

            // Get the new one from the registry by name.
            let option = NDActivation::get("foo");
            assert_eq!(
                option,
                Some(Activation { name: "foo".to_owned(), function: sigmoid, derivative: sigmoid_prime }),
            );

            // Get default `sigmoid` from the registry.
            let option = NDActivation::get("sigmoid");
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }),
            );

            // Replace it with a different `Activation` instance. (For illustrative purposes only.)
            let option = NDActivation::new("sigmoid", identity, to_one).register();
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: sigmoid, derivative: sigmoid_prime }),
            );
            let option = NDActivation::get("sigmoid");
            assert_eq!(
                option,
                Some(Activation { name: "sigmoid".to_owned(), function: identity, derivative: to_one }),
            );
        }

        #[test]
        fn test_fn_traits() {
            let tensor = array![[-1., 2.]];
            let mut activation = Activation {
                name: "relu".to_owned(),
                function: relu,
                derivative: relu_prime
            };
            let expected_output = array![[0., 2.]];

            // Receiver borrowed:
            let output = activation(&tensor);
            assert_eq!(output, expected_output);

            // Receiver mutably borrowed:
            let output = activation.call_mut((&tensor,));
            assert_eq!(output, expected_output);

            // Receiver moved:
            let output = activation.call_once((&tensor,));
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
