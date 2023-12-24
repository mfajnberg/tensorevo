//! Definition of the [`CostFunction`] struct and the most common cost functions as well as their
//! derivatives.

use num_traits::ToPrimitive;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tensor::{TensorBase, TensorOp};


type TFunc<T> = fn(&T, &T) -> f32;
type TFuncPrime<T> = fn(&T, &T) -> T;


/// Convenience struct to store a cost function together with its name and derivative.
///
/// This is used in the [`Individual`] struct.
/// Facilitates (de-)serialization.
///
/// [`Individual`]: crate::individual::Individual
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostFunction<T: TensorBase> {
    name: String,
    function: TFunc<T>,
    derivative: TFuncPrime<T>,
}


/// Methods for convenient construction and calling.
impl<T: TensorBase> CostFunction<T> {
    /// Basic constructor to manually define all fields.
    pub fn new<S: Into<String>>(name: S, function: TFunc<T>, derivative: TFuncPrime<T>) -> Self {
        Self { name: name.into(), function, derivative }
    }

    /// Convenience constructor for known/available cost functions.
    ///
    /// Pre-defined functions are determined from hard-coded names:
    /// - [`quadratic`]
    pub fn from_name<S, TO>(name: S) -> CostFunction<TO>
    where
        S:  Into<String>,
        TO: TensorOp,
    {
        let name: String = name.into();
        let function: TFunc<TO>;
        let derivative: TFuncPrime<TO>;
        if name == "quadratic" {
            (function, derivative) = (quadratic, quadratic_prime);
        } else {
            panic!();
        }
        CostFunction { name, function, derivative }
    }

    /// Proxy for the actual cost function.
    pub fn call(&self, output: &T, desired_output: &T) -> f32 {
        (self.function)(output, desired_output)
    }

    /// Proxy for the derivative of the cost function.
    pub fn call_derivative(&self, output: &T, desired_output: &T) -> T {
        (self.derivative)(output, desired_output)
    }
}


impl<T: TensorOp> Default for CostFunction<T> {
    fn default() -> Self {
        Self::from_name("quadratic")
    }
}


/// Allows [`serde`] to serialize [`CostFunction`] objects.
impl<T: TensorBase> Serialize for CostFunction<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.name)
    }
}


/// Allows [`serde`] to deserialize to [`CostFunction`] objects.
impl<'de, T: TensorOp> Deserialize<'de> for CostFunction<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from_name(String::deserialize(deserializer)?))
    }
}


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
