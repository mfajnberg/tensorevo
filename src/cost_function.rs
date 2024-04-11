//! Definition of the [`CostFunction`] struct and the most common cost functions.

use std::sync::RwLock;

use num_traits::ToPrimitive;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tensor::{TensorBase, TensorOp};
use crate::utils::registry::Registry;

pub use crate::utils::registered::Registered;
pub use self::functions::*;


type TFunc<T> = fn(&T, &T) -> f32;
type TFuncPrime<T> = fn(&T, &T) -> T;


/// Contains a named cost function together with its derivative.
///
/// Used primarily in the [`Individual`] struct.
///
/// See the [**`Registerd`** trait implementation](#impl-Registered<String>-for-CostFunction<T>)
/// below for a more general usage example.
///
/// [`Individual`]: crate::individual::Individual
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostFunction<T: TensorBase> {
    name: String,
    function: TFunc<T>,
    derivative: TFuncPrime<T>,
}


/// Methods for construction and calling.
impl<T: TensorBase> CostFunction<T> {
    /// Basic constructor.
    ///
    /// # Arguments
    /// - `name` - Will be used for [serialization](#impl-Serialize-for-CostFunction<T>)
    ///            and as the key in the static registry.
    /// - `function` - The actual cost function.
    ///                It should typically take the actual output of a calculation and the desired
    ///                output for that same calculation as `TensorBase` types and return a float.
    /// - `derivative` - The derivative of `function`. (Needed for backpropagation.)
    ///                  Unlike the cost function itself, this should return a `TensorBase` type.
    ///
    /// # Returns
    /// A new instance of `CostFunction<T>` with the provided values.
    pub fn new(name: impl Into<String>, function: TFunc<T>, derivative: TFuncPrime<T>) -> Self {
        Self { name: name.into(), function, derivative }
    }

    /// Proxy for the derivative of the cost function.
    pub fn d(&self, output: &T, desired_output: &T) -> T {
        (self.derivative)(output, desired_output)
    }

    /// Returns the name of the cost function.
    pub fn name(&self) -> &str {
        &self.name
    }
}


/// Returns an instance of the quadratic cost function as the default.
impl<T: 'static + TensorOp> Default for CostFunction<T> {
    fn default() -> Self {
        Self::get("quadratic").unwrap()
    }
}


/// Makes `CostFunction<T>` callable by value (i.e. consuming the instance).
///
/// This is mainly implemented because [`FnOnce`] is a supertrait of [`Fn`].
impl<T: TensorBase> FnOnce<(&T, &T)> for CostFunction<T> {
    type Output = f32;

    /// Proxy for the actual underlying cost function.
    extern "rust-call" fn call_once(self, args: (&T, &T)) -> Self::Output {
        (self.function)(args.0, args.1)
    }
}


/// Makes `CostFunction<T>` callable by mutable reference.
///
/// This is mainly implemented because [`FnMut`] is a supertrait of [`Fn`].
impl<T: TensorBase> FnMut<(&T, &T)> for CostFunction<T> {
    /// Proxy for the actual underlying cost function.
    extern "rust-call" fn call_mut(&mut self, args: (&T, &T)) -> Self::Output {
        (self.function)(args.0, args.1)
    }
}


/// Makes `CostFunction<T>` callable by immutable reference.
///
/// This allows you to use the call operator `( )` on instances and essentially treat them as functions.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use tensorevo::cost_function::CostFunction;
/// use tensorevo::tensor::TensorBase;
///
/// fn zero_function<T: TensorBase>(_t1: &T, _t2: &T) -> f32 {
///     0.
/// }
///
/// fn zero_derivative<T: TensorBase>(t1: &T, _t2: &T) -> T {
///     T::zeros(t1.shape())
/// }
///
/// fn main() {
///     let c = CostFunction::new("zero", zero_function, zero_derivative);
///     let t1 = array![[1., 2.]];
///     let t2 = array![[3., 4.]];
///     let out = c(&t1, &t2);
///     assert_eq!(out, 0.);
/// }
/// ```
///
/// The call operator always calls the actual cost function.
/// To call the derivative use the [`CostFunction::d`] method.
impl<T: TensorBase> Fn<(&T, &T)> for CostFunction<T> {
    /// Proxy for the actual underlying cost function.
    extern "rust-call" fn call(&self, args: (&T, &T)) -> Self::Output {
        (self.function)(args.0, args.1)
    }
}


/// Allows [`serde`] to serialize [`CostFunction`] objects.
impl<T: 'static + TensorOp> Serialize for CostFunction<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Registered::serialize_as_key(self, serializer)
    }
}


/// Allows [`serde`] to deserialize to [`CostFunction`] objects.
impl<'de, T: 'static + TensorOp> Deserialize<'de> for CostFunction<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Registered::deserialize_from_key(deserializer)
    }
}


/// Provides a static registry of [`CostFunction<T>`] instances.
///
/// Reference implementations for some common functions (and their derivatives) are available by
/// default under the following keys:
/// - `quadratic` (see [`quadratic`] and [`quadratic_prime`])
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
/// use tensorevo::cost_function::{CostFunction, Registered};
/// use tensorevo::tensor::TensorOp;
///
///
/// type T = Array2::<f32>;
///
/// // No cost function with the name `foo` was registered:
/// assert_eq!(CostFunction::<T>::get("foo"), None);
///
/// // Common cost functions are available by default:
/// let quadratic = CostFunction::<T>::get("quadratic").unwrap();
/// let x = array![[1., 2.]];
/// let y = array![[1., 0.]];
/// assert_eq!(quadratic(&x, &y), 1.);
/// assert_eq!(quadratic.d(&x, &y), array![[0., 2.]]);
/// ```
impl<T: 'static + TensorOp> Registered<String> for CostFunction<T> {
    /// Returns a reference to the name provided in the [`CostFunction::new`] constructor.
    fn key(&self) -> &String {
        &self.name
    }

    /// Registers reference implementations of some common cost functions (and their derivatives).
    ///
    /// This function should not be called directly. It is called once during initialization of the
    /// [`Registered::Registry`] singleton for `CostFunction<T>`.
    fn registry_post_init(registry_lock: &RwLock<Self::Registry>) {
        let mut registry = registry_lock.write().unwrap();
        let _ = registry.add(
            "quadratic".to_owned(),
            Self::new("quadratic".to_owned(), quadratic, quadratic_prime),
        );
    }
}

/// Reference implementations of some common cost functions as well as their derivatives.
pub mod functions {
    use super::*;

    /// Reference implementation of the quadratic cost function.
    ///
    /// Calculated as the euclidian norm of the difference between actual and desired output divided by
    /// two.
    /// Used primarily during stochastic gradient descent.
    ///
    /// # Arguments
    /// * `output` - The actual output returned by a individual as a tensor.
    /// * `desired_output` - The desired output as a tensor.
    ///
    /// # Returns
    /// The cost as 32 bit float.
    pub fn quadratic<T: TensorOp>(output: &T, desired_output: &T) -> f32 {
        (desired_output - output).norm().to_f32().unwrap() / 2.
    }


    /// Reference implementation of the derivative of the quadratic cost function.
    ///
    /// Simply the difference between actual and desired output.
    /// Used only during stochastic gradient descent.
    ///
    /// # Arguments
    /// * `output` - The actual output returned by a individual as a tensor.
    /// * `desired_output` - The desired output as a tensor.
    ///
    /// # Returns
    /// Another tensor.
    pub fn quadratic_prime<T: TensorOp>(output: &T, desired_output: &T) -> T {
        output - desired_output
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;

    mod test_cost_function {
        use super::*;

        #[test]
        fn test_new() {
            let cost_function = CostFunction::<Array2<f32>>::new(
                "quadratic",
                quadratic,
                quadratic_prime,
            );
            assert_eq!(
                cost_function,
                CostFunction {
                    name: "quadratic".to_owned(),
                    function: quadratic,
                    derivative: quadratic_prime,
                },
            );
        }

        #[test]
        fn test_d() {
            let actual_out  = array![[1., 2.]];
            let desired_out = array![[1., 1.]];
            let cost_func = CostFunction {
                name: "quadratic".to_owned(),
                function: quadratic,
                derivative: quadratic_prime,
            };
            let cost = cost_func.d(&actual_out, &desired_out);
            let expected_cost = array![[0., 1.]];
            assert_eq!(cost, expected_cost);
        }

        #[test]
        fn test_default() {
            let default = CostFunction::<Array2<f32>>::default();
            assert_eq!(
                default,
                CostFunction {
                    name: "quadratic".to_owned(),
                    function: quadratic,
                    derivative: quadratic_prime,
                },
            );
        }

        #[test]
        fn test_fn_traits() {
            let actual_out  = array![[1., 2.]];
            let desired_out = array![[1., 0.]];
            let mut cost_func = CostFunction {
                name: "quadratic".to_owned(),
                function: quadratic,
                derivative: quadratic_prime,
            };
            let expected_cost = 1.;

            // Receiver borrowed:
            let cost = cost_func(&actual_out, &desired_out);
            assert_eq!(cost, expected_cost);

            // Receiver mutably borrowed:
            let cost = cost_func.call_mut((&actual_out, &desired_out));
            assert_eq!(cost, expected_cost);

            // Receiver moved:
            let cost = cost_func.call_once((&actual_out, &desired_out));
            assert_eq!(cost, expected_cost);
        }
    }
}

