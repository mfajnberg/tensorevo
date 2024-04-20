//! Definition of the [`TensorBase`] trait, as well as some useful trait aliases.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::FromPrimitive;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::component::TensorComponent;
use crate::dimension::{Dim1, Dim2, Dim3, Dimension, HasHigherDimension, HasLowerDimension};
use crate::ops::{Dot, Norm};


/// Basic methods of multi-dimensional arrays ("tensors").
pub trait TensorBase:
    Clone
    + Debug
    // Component access and assignment via subscript:
    + Index<Self::Dim, Output = Self::Component>
    + IndexMut<Self::Dim>
    + PartialEq
    + Sized
{
    /// The type of every component of the tensor.
    type Component: TensorComponent;

    /// The dimensionality/index type of the tensor.
    type Dim: Dimension + HasHigherDimension + HasLowerDimension;
    
    /// Creates a tensor of the specified `shape` with all zero components.
    fn zeros(shape: impl Into<Self::Dim>) -> Self {
        Self::from_num(Self::Component::from_usize(0).unwrap(), shape)
    }
    
    /// Creates a tensor of the specified `shape` with all components equal to `num`.
    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self;

    /// Creates a tensor of the specified `shape` from items from the provided `iterable`.
    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>;

    /// Returns the shape of the tensor.
    fn shape<S: From<Self::Dim>>(&self) -> S;
    
    /// Returns the transpose of itself as a new tensor.
    fn transpose(&self) -> Self;

    /// Returns all the tensor's components as a slice in logical order.
    fn as_slice(&self) -> &[Self::Component];

    /// Append a `tensor` of lower dimensionality along the specified `axis`.
    ///
    /// Panics if `axis` is out of bounds.
    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower>;

    /// Calls function `f` on each component and returns the result as a new tensor.
    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component;
    
    /// Calls function `f` on each component mutating the tensor in place.
    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component;

    /// Return an iterator of indexes and references to the components of the tensor.
    fn iter_indexed<IDX: From<Self::Dim>>(&self) -> impl Iterator<Item = (IDX, &Self::Component)>;

    /// Return an iterator of indexes and mutable references to the components of the tensor.
    fn iter_indexed_mut<IDX: From<Self::Dim>>(&mut self) -> impl Iterator<Item = (IDX, &mut Self::Component)>;

    /// Returns an iterator of references to the components of the tensor (in logical order).
    fn iter(&self) -> impl Iterator<Item = &Self::Component>;

    /// Returns the sum of all sub-tensors along the specified `axis` as a new tensor of a lower dimensionality.
    ///
    /// Panics if `axis` is out of bounds.
    fn sum_axis<T>(&self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower>;

    /// Returns the sum of all sub-tensors along the specified `axis` as a new tensor of the same dimensionality.
    ///
    /// Panics if `axis` is out of bounds.
    fn sum_axis_same_dim(&self, axis: usize) -> Self;

    /// Transforms the tensor into one with an additional axis/dimension.
    ///
    /// Panics if `axis` is out of bounds.
    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher>;
}


/// Calculate norms of a tensor.
/// [`TensorBase`] combined with basic mathematical operations.
pub trait TensorOp =
where 
    for<'a> Self:
        TensorBase +
        // Componentwise addition (left-hand side moved):
        Add<Output = Self> +
        Add<&'a Self, Output = Self> +
        // Componentwise addition-assignment:
        AddAssign<&'a Self> +
        // Componentwise division (left-hand side moved):
        Div<Output = Self> +
        Div<&'a Self, Output = Self> +
        // Componentwise division-assignment:
        DivAssign<&'a Self> +
        // Dot product:
        Dot<Output = Self> +
        Dot<&'a Self, Output = Self> +
        // Componentwise multiplication (left-hand side moved):
        Mul<Output = Self> +
        Mul<&'a Self, Output = Self> +
        // Componentwise multiplication-assignment:
        MulAssign<&'a Self> +
        // Negation (moving):
        Neg<Output = Self> +
        // Norm:
        Norm +
        // Componentwise subtraction (left-hand side moved):
        Sub<Output = Self> +
        Sub<&'a Self, Output = Self> +
        // Componentwise subtraction-assignment:
        SubAssign<&'a Self>,
    for<'a> &'a Self:
        // Componentwise addition (left-hand side borrowed):
        Add<Output = Self> +
        Add<Self, Output = Self> +
        // Componentwise division (left-hand side borrowed):
        Div<Output = Self> +
        Div<Self, Output = Self> +
        // Componentwise multiplication (left-hand side borrowed):
        Mul<Output = Self> +
        Mul<Self, Output = Self> +
        // Negation (borrowing):
        Neg<Output = Self> +
        // Componentwise subtraction (left-hand side borrowed):
        Sub<Output = Self> +
        Sub<Self, Output = Self>;


/// Owned [`TensorBase`] combined with `serde` (de-)serialization.
pub trait TensorSerde = 'static + TensorBase + DeserializeOwned + Serialize;


/// [`TensorOp`] and [`TensorSerde`].
pub trait Tensor = TensorOp + TensorSerde;


pub trait Tensor1 = Tensor<Dim = Dim1>;

pub trait Tensor2 = Tensor<Dim = Dim2>;

pub trait Tensor3 = Tensor<Dim = Dim3>;


#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that `Array2` fully implements `TensorOp`.
    /// This test function does not need assertions because we just want to make sure that
    /// the compiler allows `test_tensor_op_traits` to be called with `Array2` arguments.
    #[test]
    fn test_tensor_op_array2() {
        use ndarray::array;

        let tensor_a = array![
            [1., 2.],
            [3., 4.]
        ];
        let tensor_b = array![
            [ 0., -1.],
            [-2., -3.]
        ];
        test_tensor_op_traits(tensor_a, tensor_b);
    }

    /// Tests that the `TensorOp` trait alias covers the expected traits.
    /// This function does not need assertions and does not need to be run as a test
    /// because it just needs to pass the compiler.
    fn test_tensor_op_traits<T: TensorOp<Dim = Dim2>>(mut t1: T, t2: T) {
        // Negation (borrowing & moving):
        let t3 = -&t1;
        let _ = -t3;
        // Dot product (borrowing and moving right-hand side):
        let t4 = t1.dot(&t2);
        t1.dot(t4);
        // Operation-assignment (borrowed right-hand side only):
        t1 += &t2;
        t1 /= &t2;
        t1 *= &t2;
        t1 -= &t2;
        // Basic arithmetic operations:
        // Borrow x Borrow:
        let t5 = &t1 + &t2;
        let t6 = &t1 - &t2;
        let t7 = &t1 * &t2;
        let t8 = &t1 / &t2;
        // Borrow x Move:
        let t9  = &t1 + t5.clone();
        let t10 = &t1 - t6.clone();
        let t11 = &t1 * t7.clone();
        let t12 = &t1 / t8.clone();
        // Move x Borrow:
        let t13 = t5 + &t2;
        let t14 = t6 - &t2;
        let t15 = t7 * &t2;
        let t16 = t8 / &t2;
        // Move x Move:
        let _ = t9  + t10;
        let _ = t11 - t12;
        let _ = t13 * t14;
        let _ = t15 / t16;
        // Index access to and re-assignment of components:
        let x = t1[[0, 0]];
        t1[[1, 0]] = x;
    }
}    

