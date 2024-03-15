//! Definition of the [`TensorBase`] and related traits, as well as some trait aliases combining them.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use ndarray::{Array2, Axis, ArrayView};
use num_traits::FromPrimitive;
use num_traits::real::Real;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::component::TensorComponent;


/// Basic methods of multi-dimensional arrays ("tensors").
pub trait TensorBase:
    Clone
    + Debug
    + Display
    + PartialEq
    + Sized
{
    /// The type of every component of the tensor.
    type Component: TensorComponent;
    
    /// Creates a tensor of the specified `shape` with all zero components.
    fn zeros(shape: (usize, usize)) -> Self {
        Self::from_num(Self::Component::from_usize(0).unwrap(), shape)
    }
    
    /// Creates a tensor of the specified `shape` with all components equal to `num`.
    fn from_num(num: Self::Component, shape: (usize, usize)) -> Self;

    fn from_iter<I, J>(iterator: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Component>;
    
    /// Returns the shape of the tensor as a tuple of unsigned integers.
    fn shape(&self) -> (usize, usize);
    
    /// Returns a vector of vectors of the tensor's components.
    fn to_vec(&self) -> Vec<Vec<Self::Component>>;
    
    /// Returns the transpose of itself as a new tensor.
    fn transpose(&self) -> Self;

    /// Append a row to the tensor.
    fn append_row(&mut self, row: &[Self::Component]);

    /// Append a column to the tensor.
    fn append_column(&mut self, column: &[Self::Component]);

    /// Calls function `f` on each component and returns the result as a new tensor.
    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component;
    
    /// Calls function `f` on each component mutating the tensor in place.
    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component;
    
    fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &Self::Component)>;

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut Self::Component)>;

    fn iter(&self) -> impl Iterator<Item = &Self::Component>;

    /// Returns the sum of all rows (0) or columns (1) as a new tensor.
    fn sum_axis(&self, axis: usize) -> Self;
}


/// Dot product aka. matrix multiplication.
pub trait Dot<Rhs = Self> {
    /// The output type of the multiplication operation.
    type Output;

    /// Returns the dot product of `self` on the left and `rhs` on the right.
    fn dot(&self, rhs: Rhs) -> Self::Output;
}


/// Calculate norms of a tensor.
pub trait Norm {
    /// Output type of any norm method.
    type Output: TensorComponent;

    /// Returns the supremum norm.
    fn norm_max(&self) -> Self::Output;

    /// Returns the p-norm for any `p` >= 1.
    fn norm_p(&self, p: impl Real) -> Self::Output;

    /// Returns the l1-norm (manhattan norm).
    fn norm_1(&self) -> Self::Output {
        self.norm_p(1.)
    }

    /// Returns the l2-norm (euclidian norm).
    fn norm_2(&self) -> Self::Output {
        self.norm_p(2.)
    }

    /// Alias for `norm_2`.
    fn norm(&self) -> Self::Output {
        self.norm_2()
    }
}


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
        Index<[usize; 2], Output = <Self as TensorBase>::Component> +
        IndexMut<[usize; 2]> +
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


/// Implementation of [`TensorBase`] for `ndarray::Array2`.
impl<C: TensorComponent> TensorBase for Array2<C> {
    type Component = C;

    fn from_num(num: Self::Component, shape: (usize, usize)) -> Self {
        Self::from_elem(shape, num)
    }

    fn from_iter<I, J>(iterator: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = Self::Component>,
    {
        let mut data = Vec::new();
        let mut num_rows: usize = 0;
        let mut num_columns: Option<usize> = None;
        for row in iterator {
            num_rows += 1;
            let row_vec: Vec<Self::Component> = row.into_iter().collect();
            match num_columns {
                Some(n) => {
                    if n != row_vec.len() { panic!() }
                },
                None => num_columns = Some(row_vec.len()),
            }
            data.extend_from_slice(&row_vec);
        }
        Array2::from_shape_vec(
            (num_rows, num_columns.unwrap_or(0)),
            data,
        ).unwrap()
    }
    
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    
    fn to_vec(&self) -> Vec<Vec<Self::Component>> {
        self.rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect()
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn append_row(&mut self, row: &[Self::Component]) {
        self.push_row(ArrayView::from(row)).unwrap()
    }

    fn append_column(&mut self, column: &[Self::Component]) {
        self.push_column(ArrayView::from(column)).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn indexed_iter(&self) -> impl Iterator<Item = ((usize, usize), &Self::Component)> {
        Array2::<C>::indexed_iter(self)
    }

    fn indexed_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut Self::Component)> {
        Array2::<C>::indexed_iter_mut(self)
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array2::<C>::iter(self)
    }

    fn sum_axis(&self, axis: usize) -> Self {
        self.sum_axis(Axis(axis)).insert_axis(Axis(axis))
    }
}


/// Implementation of `Dot` for `ndarray::Array2` (right-hand side moved).
impl<C: TensorComponent> Dot for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}


/// Implementation of `Dot` for `ndarray::Array2` (right-hand side borrowed).
impl<C: TensorComponent> Dot<&Array2<C>> for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.dot(rhs)
    }
}


/// Implementation of `Norm` for `ndarray::Array2`.
impl<P: TensorComponent> Norm for Array2<P> {
    type Output = P;

    /// Returns the largest absolute value of all array components.
    fn norm_max(&self) -> Self::Output {
        self.iter().fold(
            P::zero(),
            |largest, component| {
                let absolute = component.abs();
                if largest > absolute {
                    largest
                } else {
                    absolute
                }
            }
        )
    }

    /// Converts `p` to `f32` before calculating the norm.
    ///
    /// Panics, if `p` is less than 1.
    fn norm_p(&self, p: impl Real) -> Self::Output {
        let pf32 = p.to_f32().unwrap();
        if pf32 < 1. { panic!("P-norm undefined for p < 1") }
        self.iter()
            .map(|component| component.abs().powf(p))
            .sum::<P>()
            .powf(1./pf32)
    }

    /// Sums the absolute values of all array components.
    fn norm_1(&self) -> Self::Output {
        self.iter()
            .map(|component| component.abs())
            .sum()
    }

    /// Takes the square root of the squares of all array components.
    fn norm_2(&self) -> Self::Output {
        self.iter()
            .map(|component| component.abs().powf(2.))
            .sum::<P>()
            .sqrt()
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
            let tensor: Array2<f32> = TensorBase::zeros((1, 3));
            let expected = array![[0., 0., 0.]];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_from_num() {
            let tensor = Array2::from_num(3.14, (2, 2));
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
            let shape = TensorBase::shape(&tensor);
            assert_eq!(shape, (2, 3));
        }

        #[test]
        fn test_to_vec() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let vector = TensorBase::to_vec(&tensor);
            let expected = vec![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            assert_eq!(vector, expected);
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
            let result = TensorBase::sum_axis(&tensor, 0);
            let expected = array![[3., 5., 7.]];
            assert_eq!(result, expected);

            let result = TensorBase::sum_axis(&tensor, 1);
            let expected = array![
                [3. ],
                [12.]
            ];
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_dot_array2() {
        let tensor_a = array![
            [0., 1., 2.],
            [3., 2., 1.]
        ];
        let tensor_b = array![
            [-1., -2.],
            [-3., -2.],
            [-1.,  0.]
        ];
        let result = Dot::dot(&tensor_a, tensor_b);
        let expected = array![
            [ -5.,  -2.],
            [-10., -10.]
        ];
        assert_eq!(result, expected);

        let result2 = Dot::dot(&result, &result);
        let expected2 = array![
            [45., 30.],
            [150., 120.]
        ];
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_norm_array2() {
        let tensor = array![
            [ 0.,  1.,  2.],
            [-3., -1.,  1.]
        ];
        let result = tensor.norm_max();
        assert_eq!(result, 3.);

        let result = tensor.norm_p(80.).round();
        assert_eq!(result, 3.);

        let result = tensor.norm_1();
        assert_eq!(result, 8.);

        let result = tensor.norm_2();
        assert_eq!(result, 4.);
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
    fn test_tensor_op_traits<T: TensorOp>(mut t1: T, t2: T) {
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

