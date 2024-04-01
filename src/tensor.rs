//! Definition of the [`TensorBase`] and related traits, as well as some trait aliases combining them.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use ndarray::{Array1, Array2, ArrayView, Axis};
use num_traits::FromPrimitive;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::component::TensorComponent;
use crate::dimension::{Dim1, Dim2, Dimension, HasHigherDimension, HasLowerDimension};
use crate::ops::{Dot, Norm};


/// Basic methods of multi-dimensional arrays ("tensors").
pub trait TensorBase:
    Clone
    + Debug
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

    /// Returns the shape of the tensor as a tuple of unsigned integers.
    fn shape<S: From<Self::Dim>>(&self) -> S;
    
    /// Returns the transpose of itself as a new tensor.
    fn transpose(&self) -> Self;

    fn as_slice(&self) -> &[Self::Component];

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower>;

    /// Calls function `f` on each component and returns the result as a new tensor.
    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component;
    
    /// Calls function `f` on each component mutating the tensor in place.
    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component;
    
    fn indexed_iter(&self) -> impl Iterator<Item = (Self::Dim, &Self::Component)>;

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = (Self::Dim, &mut Self::Component)>;

    fn iter(&self) -> impl Iterator<Item = &Self::Component>;

    /// Returns the sum of all rows (0) or columns (1) as a new tensor.
    fn sum_axis<T>(&self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower>;

    fn sum_axis_same_dim(&self, axis: usize) -> Self;

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher>;
}


/// Calculate norms of a tensor.
/// [`TensorBase`] combined with basic mathematical operations and indexing.
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
        // Component access and assignment via subscript:
        Index<<Self as TensorBase>::Dim, Output = <Self as TensorBase>::Component> +
        IndexMut<<Self as TensorBase>::Dim> +
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


impl<C: TensorComponent> TensorBase for Vec<C> {
    type Component = C;
    type Dim = Dim1;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        vec![num; shape.into()]
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        iterable.into_iter().take(shape.into()).collect()
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        self.len().into()
    }

    fn transpose(&self) -> Self {
        self.clone()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Vec::as_slice(self)
    }

    fn append<T>(&mut self, _axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Vec::push(self, tensor.as_slice()[0])
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component {
        self.iter().copied().map(f).collect()
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component {
        // Not actually in-place at all, but allocating a whole new vector before replacing.
        // It is hard to do this without unsafe code.
        *self = self.map(f);
    }

    fn indexed_iter(&self) -> impl Iterator<Item = (Self::Dim, &Self::Component)> {
        <&'_ Vec<Self::Component>>::into_iter(self).enumerate()
    }

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = (Self::Dim, &mut Self::Component)> {
        <&'_ mut Vec<Self::Component>>::into_iter(self).enumerate()
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        <&'_ Vec<Self::Component>>::into_iter(self)
    }

    fn sum_axis<T>(&self, _axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        T::from_num(self.iter().copied().sum(), ())
    }

    fn sum_axis_same_dim(&self, _axis: usize) -> Self {
        vec![self.iter().copied().sum()]
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let shape = if axis == 0 {
            [1, self.len()]
        } else if axis == 1 {
            [self.len(), 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}


/// Implementation of [`TensorBase`] for `ndarray::Array1`.
impl<C: TensorComponent> TensorBase for Array1<C> {
    type Component = C;
    type Dim = Dim1;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        Array1::from_elem(shape.into(), num)
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        Array1::from_iter(iterable.into_iter().take(shape.into()))
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        S::from(self.dim())
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Array1::as_slice(self).unwrap()
    }

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array1::push(self, Axis(axis), ArrayView::from_shape((), tensor.as_slice()).unwrap()).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn indexed_iter(&self) -> impl Iterator<Item = (Self::Dim, &Self::Component)> {
        Array1::<C>::indexed_iter(self).map(|(idx, component)| (idx.into(), component))
    }

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = (Self::Dim, &mut Self::Component)> {
        Array1::<C>::indexed_iter_mut(self).map(|(idx, component)| (idx.into(), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array1::<C>::iter(self)
    }

    fn sum_axis<T>(&self, _axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        T::from_num(Array1::sum(self), ())
    }

    fn sum_axis_same_dim(&self, axis: usize) -> Self {
        Array1::sum_axis(self, Axis(axis)).insert_axis(Axis(axis))
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let shape = if axis == 0 {
            [1, self.len()]
        } else if axis == 1 {
            [self.len(), 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}


/// Implementation of [`TensorBase`] for `ndarray::Array2`.
impl<C: TensorComponent> TensorBase for Array2<C> {
    type Component = C;
    type Dim = Dim2;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        Array2::from_elem(shape.into(), num)
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        let iterator: &mut dyn Iterator<Item = Self::Component> = &mut iterable.into_iter();
        Array2::from_shape_simple_fn(shape.into(), || iterator.next().unwrap())
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        S::from(self.dim().into())
    }
    
    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Array2::as_slice(self).unwrap()
    }

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array2::push(self, Axis(axis), ArrayView::from(tensor.as_slice())).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn indexed_iter(&self) -> impl Iterator<Item = (Self::Dim, &Self::Component)> {
        Array2::<C>::indexed_iter(self).map(|(idx, component)| (idx.into(), component))
    }

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = (Self::Dim, &mut Self::Component)> {
        Array2::<C>::indexed_iter_mut(self).map(|(idx, component)| (idx.into(), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array2::<C>::iter(self)
    }

    fn sum_axis<T>(&self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        let axis_sum_iterator = Array2::axis_iter(self, Axis(axis)).map(|subview| subview.sum());
        T::from_iter(axis_sum_iterator, Array2::len_of(self, Axis(axis)))
    }

    fn sum_axis_same_dim(&self, axis: usize) -> Self {
        Array2::sum_axis(self, Axis(axis)).insert_axis(Axis(axis))
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let (rows, cols) = Array2::dim(&self);
        let shape = if axis == 0 {
            [1, rows, cols]
        } else if axis == 1 {
            [rows, 1, cols]
        } else if axis == 2 {
            [rows, cols, 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    mod test_tensor_base_array2 {
        use super::*;

        #[test]
        fn test_zeros() {
            let tensor: Array2<f32> = TensorBase::zeros([1, 3]);
            let expected = array![[0., 0., 0.]];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_from_num() {
            let tensor = Array2::from_num(3.14, [2, 2]);
            let expected = array![
                [3.14, 3.14],
                [3.14, 3.14]
            ];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_shape() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let shape: Dim2 = TensorBase::shape(&tensor);
            assert_eq!(shape, [2, 3]);
        }

        #[test]
        fn test_transpose() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let result = tensor.transpose();
            let expected = array![
                [0., 3.],
                [1., 4.],
                [2., 5.]
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map() {
            fn double<C: TensorComponent>(x: C) -> C {
                return x * C::from_usize(2).unwrap();
            }
    
            let tensor = array![
                [0., 1.],
                [2., 3.]
            ];
            let result = TensorBase::map(&tensor, double);
            let expected = array![
                [0., 2.],
                [4., 6.]
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map_inplace() {
            fn halve<C: TensorComponent>(x: C) -> C {
                return x / C::from_usize(2).unwrap();
            }
    
            let mut tensor = array![
                [0., -2.],
                [4., -6.]
            ];
            TensorBase::map_inplace(&mut tensor, halve);
            let expected = array![
                [0., -1.],
                [2., -3.]
            ];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_sum_axis() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let result: Array1<f64> = TensorBase::sum_axis(&tensor, 0);
            let expected = array![3., 5., 7.];
            assert_eq!(result, expected);

            let result: Array1<f64> = TensorBase::sum_axis(&tensor, 1);
            let expected = array![
                3. ,
                12.
            ];
            assert_eq!(result, expected);
        }
    }

    /// Tests that `Array2` fully implements `TensorOp`.
    /// This test function does not need assertions because we just want to make sure that
    /// the compiler allows `test_tensor_op_traits` to be called with `Array2` arguments.
    #[test]
    fn test_tensor_op_array2() {
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

