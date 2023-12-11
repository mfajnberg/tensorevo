//! Definition of the [`TensorComponent`] and related traits.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Neg;

use num_traits::real::Real;
use num_traits::{FromPrimitive, NumAssign, NumCast, Signed, ToPrimitive};


/// Exponentiation with the base `e` (Euler's number).
pub trait Exp {
    /// Result type of the exponentiation.
    type Output;

    /// Raises `e` to the power of `self`.
    fn exp(&self) -> Self::Output;
}


/// Exponentiation to any real number or integer power.
pub trait Pow {
    /// Result type of the exponentiation to a real number power.
    type Output;

    /// Raises `self` to the power of `exponent` (real number).
    fn powf(&self, exponent: impl Real) -> Self::Output;

    /// Raises `self` to the power of `exponent` (integer).
    fn powi(&self, exponent: i32) -> Self;

    /// Returns the square root of `self`.
    fn sqrt(&self) -> Self::Output {
        self.powf(0.5)
    }
}


/// Wrapper around the built-in `f32::exp` method.
impl Exp for f32 {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        f32::exp(*self)
    }
}


/// Wrapper around the built-in `f64::exp` method.
impl Exp for f64 {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        f64::exp(*self)
    } 
}


/// Raise `e` to the a power of `isize` type and round down.
impl Exp for isize {
    type Output = Self;

    /// Converts `self` to an `f64`, performs the exponentiation and converts back (rounding down).
    fn exp(&self) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().exp()
        ).unwrap()
    }
}


/// Wrapper around the built-in `f32` methods `powf`, `powi` and `sqrt`.
impl Pow for f32 {
    type Output = Self;

    fn powf(&self, exponent: impl Real) -> Self::Output {
        f32::powf(*self, <f32 as NumCast>::from(exponent).unwrap())
    }

    fn powi(&self, exponent: i32) -> Self {
        f32::powi(*self, exponent)
    }

    fn sqrt(&self) -> Self::Output {
        f32::sqrt(*self)
    }
}


/// Wrapper around the built-in `f64` methods `powf`, `powi` and `sqrt`.
impl Pow for f64 {
    type Output = Self;

    fn powf(&self, exponent: impl Real) -> Self::Output {
        f64::powf(*self, <f64 as NumCast>::from(exponent).unwrap())
    }

    fn powi(&self, exponent: i32) -> Self {
        f64::powi(*self, exponent)
    }

    fn sqrt(&self) -> Self::Output {
        f64::sqrt(*self)
    }
}


/// Perform exponentiation, rounding down where no integer is returned.
impl Pow for isize {
    type Output = Self;

    /// Converts both `self` and `exponent` to `f64`, performs exponentiation, and converts back
    /// (rounding down).
    fn powf(&self, exponent: impl Real) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().powf(exponent.to_f64().unwrap())
        ).unwrap()
    }

    /// Converts both `self` and `exponent` to `f64`, performs exponentiation, and converts back
    /// (rounding down).
    fn powi(&self, exponent: i32) -> Self {
        self.pow(exponent.to_u32().unwrap())
    }

    /// Converts `self` to `f64`, takes the square root, and converts back (rounding down).
    fn sqrt(&self) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().sqrt()
        ).unwrap()
    }
}


/// Trait that must be implemented by any type in order to be usable as a [`Tensor`] component.
///
/// [`Tensor`]: crate::tensor::Tensor
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
    + Signed
    + Sum
{}


/// Generic implementation of the trait for any type that satisfies the [`TensorComponent`] bounds.
///
/// Since [`Exp`] and [`Pow`] are implemented for [`f32`], [`f64`] and [`isize`],
/// those are automatically covered and will implement [`TensorComponent`].
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
    + Signed
    + Sum
{}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp() {
        let exponent: f32 = 0.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 1.);

        let exponent: f64 = 1.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 2.718281828459045);

        let exponent: isize = 1;
        let result = exponent.exp();
        assert_eq!(result, 2);
    }

    #[test]
    fn test_pow() {
        let base: f32 = 8.;
        let result = Pow::powi(&base, 2);
        assert_eq!(result, 64.);

        let result = Pow::powf(&base, 1./3.);
        assert_eq!(result, 2.);

        let base: f64 = 64.;
        let result = Pow::sqrt(&base);
        assert_eq!(result, 8.);

        let base: isize = 64;
        let result = base.sqrt();
        assert_eq!(result, 8);

        let result = result.powf(1./3.);
        assert_eq!(result, 2);

        let result = result.powi(2);
        assert_eq!(result, 4);
    }

    /// Tests that the built-in types fully implement `TensorComponent`.
    #[test]
    fn test_tensor_component_impl() {
        fn some_generic_func<C: TensorComponent>(_: C) {}
        some_generic_func(0f32);
        some_generic_func(1f64);
        some_generic_func(-1isize);
        // Not implemented for other signed integer types yet. The following would not pass the compiler:
        // some_generic_func(1i32);
    }
}
