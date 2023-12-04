//! Definition of the `TensorComponent` and related trait.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Neg;

use num_traits::{Float, FromPrimitive, NumAssign, NumCast};


/// Exponentiation with the base `e` (Euler's number).
pub trait Exp {
    /// Result type of the exponentiation.
    type Output;

    /// Raises `e` to the power of `self`.
    fn exp(&self) -> Self::Output;
}


/// Exponentiation to any floating point or integer power.
pub trait Pow {
    /// Result type of the exponentiation.
    type Output;

    /// Raises `self` to the power of `rhs` (floating point number).
    fn powf(&self, rhs: impl Float) -> Self::Output;

    /// Raises `self` to the power of `rhs` (integer).
    fn powi(&self, rhs: i32) -> Self::Output;

    /// Returns the square root of `self`.
    fn sqrt(&self) -> Self::Output {
        self.powf(0.5)
    }
}


/// Blanket implementation of `Exp` for any type that implements `Float`.
impl<F: Float> Exp for F {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        F::exp(*self)
    } 
}


/// Blanket implementation of `Pow` for any type that implements `Float`.
impl<F: Float> Pow for F {
    type Output = Self;

    fn powf(&self, rhs: impl Float) -> Self::Output {
        F::powf(*self, F::from(rhs).unwrap())
    }

    fn powi(&self, rhs: i32) -> Self::Output {
        F::powi(*self, rhs)
    }

    /// Returns the square root of `self` via the provided `Float::sqrt` method.
    fn sqrt(&self) -> Self::Output {
        F::sqrt(*self)
    }
}


/// Trait that must be implemented by any type in order to be usable as a `Tensor` component.
pub trait TensorComponent:
    // https://doc.rust-lang.org/rust-by-example/scope/lifetime/static_lifetime.html#trait-bound
    'static
    + Copy
    + Debug
    + Display
    + Exp<Output = Self>
    + FromPrimitive
    + Neg<Output = Self>
    + NumAssign
    + NumCast
    + PartialEq
    + PartialOrd
    + Pow<Output = Self>
    + Sum
{}


/// Generic implementation of the trait for any type that satisfies the `TensorComponent` bounds.
///
/// Since `Exp` and `Pow` are implemented for all `Float` types above, those are automatically covered
/// and will implement `TensorComponent`.
impl<N> TensorComponent for N
where N:
    'static
    + Copy
    + Debug
    + Display
    + Exp<Output = Self>
    + FromPrimitive
    + Neg<Output = Self>
    + NumAssign
    + NumCast
    + PartialEq
    + PartialOrd
    + Pow<Output = Self>
    + Sum
{}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp() {
        let exponent = 0.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 1.);
        let exponent = 1.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 2.718281828459045);
    }

    #[test]
    fn test_pow() {
        let base = 8.;
        let result = Pow::powi(&base, 2);
        assert_eq!(result, 64.);
        let result = Pow::powf(&base, 1./3.);
        assert_eq!(result, 2.);
        let base = 64.;
        let result = Pow::sqrt(&base);
        assert_eq!(result, 8.);
    }

    /// Tests that the built-in float types fully implement `TensorComponent`.
    #[test]
    fn test_tensor_component_impl() {
        fn some_generic_func<C: TensorComponent>(_: C) {}
        some_generic_func(0f32);
        some_generic_func(1f64);
        // Not implemented for integer types yet. The following would not pass the compiler:
        // some_generic_func(0);
    }
}
