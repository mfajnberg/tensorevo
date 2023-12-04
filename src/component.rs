//! Definition of the `TensorComponent` trait.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Neg;

use num_traits::{Float, FromPrimitive, NumAssign, NumCast};


pub trait Exp {
    type Output;

    fn exp(&self) -> Self::Output;
}


pub trait Pow {
    type Output;

    fn powf(&self, rhs: impl Float) -> Self::Output;

    fn powi(&self, rhs: i32) -> Self::Output;

    fn sqrt(&self) -> Self::Output {
        self.powf(0.5)
    }
}


impl<F: Float> Exp for F {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        F::exp(*self)
    } 
}


impl<F: Float> Pow for F {
    type Output = Self;

    fn powf(&self, rhs: impl Float) -> Self::Output {
        F::powf(*self, F::from(rhs).unwrap())
    }

    fn powi(&self, rhs: i32) -> Self::Output {
        F::powi(*self, rhs)
    }

    fn sqrt(&self) -> Self::Output {
        F::sqrt(*self)
    }
}


/// Tensor component.
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


/// Generic implementation of the trait for any type that satisfies the `TensorComponent` bounds:
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

