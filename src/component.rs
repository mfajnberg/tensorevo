//! Definition of the [`TensorComponent`] trait.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Neg;

use num_traits::{FromPrimitive, NumAssign, NumCast, Signed};

use crate::ops::{Exp, Pow};


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

