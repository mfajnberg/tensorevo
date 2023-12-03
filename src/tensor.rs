//! Definitions of the `Tensor` and `TensorElement` trait and the `NDTensor` reference implementation.

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops;

use ndarray::{Array2, Axis};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Deserializer, Serialize};
use serde::de::DeserializeOwned;


/// Tensor Element
// TODO: Consider seprating some of those supertraits.
pub trait TensorElement:
    // https://doc.rust-lang.org/rust-by-example/scope/lifetime/static_lifetime.html#trait-bound
    'static
    + Debug
    + DeserializeOwned
    + Display
    // https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html
    + Float
    // https://docs.rs/num-traits/latest/num_traits/cast/trait.FromPrimitive.html
    + FromPrimitive
    + Serialize
    + Sum
{}


/// Generic implementation of the trait for any type that satisfies the `TensorElement` bounds:
impl<N> TensorElement for N
where N:
    'static
    + Debug
    + DeserializeOwned
    + Display
    + Float
    + FromPrimitive
    + Serialize
    + Sum
{}


/// Types that support required linear algebra operations
/// Can be created from vectors, json, or a number
/// Can map functions
pub trait TensorBase:
    Clone
    + Debug
    + Display
    + PartialEq
    + Sized
    // TODO: Add all relevant operator overloading traits from `std::ops`.
    //       https://github.com/mfajnberg/tensorevo/issues/5
{
    type Element: TensorElement;  // associated type must implement the `TensorElement` trait
    
    /// Creates a `Tensor` from an array of arrays of `TensorElement`
    fn from_array<const M: usize, const N: usize>(a: [[Self::Element; N]; M]) -> Self;

    /// Creates a `Tensor` from a vector of vectors of `TensorElement`
    fn from_vec(v: Vec<Vec<Self::Element>>) -> Self;
    
    /// Creates a `Tensor` from a number
    fn from_num(num: Self::Element, shape: (usize, usize)) -> Self;
    
    /// Creates a `Tensor` with only zeros given a 2d shape
    // TODO: Provide a default implementation using `from_num`.
    fn zeros(shape: (usize, usize)) -> Self;
    
    /// Returns the 2d shape of self as a tuple
    fn shape(&self) -> (usize, usize);
    
    /// Returns a vector of vectors of the tensor's elements
    fn to_vec(&self) -> Vec<Vec<Self::Element>>;
    
    /// Returns the transpose of itself as a new tensor
    fn transpose(&self) -> Self;

    /// Maps a function to each element and returns the result as a new tensor
    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Element) -> Self::Element;
    
    /// Maps a function to each element in place
    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Element) -> Self::Element;
    
    /// Returns the norm of the tensor
    fn vec_norm(&self) -> Self::Element;

    /// Returns the sum of all rows (0) or columns (1) as a new tensor
    fn sum_axis(&self, axis: usize) -> Self;
}


pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(&self, rhs: Rhs) -> Self::Output;
}


pub trait TensorOp =
where 
    for<'a> Self:
        TensorBase +
        Dot<Output = Self> +
        Dot<&'a Self, Output = Self> +
        ops::AddAssign<&'a Self> +
        ops::Add<&'a Self, Output = Self> +
        ops::DivAssign<&'a Self> +
        ops::Div<&'a Self, Output = Self> +
        ops::MulAssign<&'a Self> +
        ops::Mul<&'a Self, Output = Self> +
        ops::SubAssign<&'a Self> +
        ops::Sub<&'a Self, Output = Self>,
    for<'a> &'a Self:
        ops::Add<Output = Self> +
        ops::Div<Output = Self> +
        ops::Mul<Output = Self> +
        ops::Sub<Output = Self>;


pub trait TensorSerde = TensorBase + DeserializeOwned + Serialize;


pub trait Tensor = TensorOp + TensorSerde;


/// Takes a serde `Deserializer` and constructs a `Tensor` object from it.
///
/// Until we can do `impl<T> Deserialize for T where T: Tensor`,
/// we need this function as a workaround to use in concrete implementations
/// of `Tensor` (see `Deserialize` implementation for `NDTensor` below).
pub fn deserialize_to_tensor<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: TensorBase,
{
    let v = Vec::<Vec<T::Element>>::deserialize(deserializer)?;
    return Ok(T::from_vec(v));
}



// TEMP.
fn array2_from_iter<I, J, TE>(i: I) -> Array2<TE>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = TE>,
    TE: TensorElement,
{
    let mut data = Vec::new();
    let mut num_rows: usize = 0;
    let mut num_columns: Option<usize> = None;
    for row in i {
        num_rows += 1;
        let row_vec: Vec<TE> = row.into_iter().collect();
        match num_columns {
            Some(n) => {
                if n != row_vec.len() { panic!() }
            },
            None => num_columns = Some(row_vec.len()),
        }
        data.extend_from_slice(&row_vec);
    }
    return Array2::from_shape_vec((num_rows, num_columns.unwrap_or(0)), data).unwrap();
}


impl<TE: TensorElement> TensorBase for Array2<TE> {
    type Element = TE;

    fn from_array<const M: usize, const N: usize>(a: [[TE; N]; M]) -> Self {
        array2_from_iter(a)
    }

    fn from_vec(v: Vec<Vec<Self::Element>>) -> Self {
        array2_from_iter(v)
    }

    fn from_num(num: Self::Element, shape: (usize, usize)) -> Self {
        Self::from_elem(shape, num)
    }
    
    fn zeros(shape: (usize, usize)) -> Self {
        Self::zeros(shape)
    }
    
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    
    fn to_vec(&self) -> Vec<Vec<Self::Element>> {
        let mut output = Vec::<Vec<Self::Element>>::new();
        for row in self.rows() {
            output.push(row.to_vec());
        }
        return output;
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(TE) -> TE {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(TE) -> TE {
        self.mapv_inplace(f)
    }

    fn vec_norm(&self) -> Self::Element {
        self.iter().map(|x| x.powi(2)).sum::<Self::Element>().sqrt()
    }

    fn sum_axis(&self, axis: usize) -> Self {
        self.sum_axis(Axis(axis)).insert_axis(Axis(axis))
    }
}


impl<TE: TensorElement> Dot for Array2<TE> {
    type Output = Self;

    fn dot(&self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}


impl<TE: TensorElement> Dot<&Array2<TE>> for Array2<TE> {
    type Output = Self;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.dot(rhs)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    mod test_ndtensor {
        use super::*;

        fn double<TE: TensorElement>(x: TE) -> TE {
            return x * TE::from_usize(2).unwrap();
        }
    
        fn halve<TE: TensorElement>(x: TE) -> TE {
            return x / TE::from_usize(2).unwrap();
        }
    
        #[test]
        fn test_map() {
            let tensor = Array2::from_array([[0., 1.], [2., 3.]]);
            let result = TensorBase::map(&tensor, double);
            let expected = Array2::from_array([[0., 2.], [4., 6.]]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map_inplace() {
            let mut tensor = Array2::from_array([[0., -2.], [4., -6.]]);
            TensorBase::map_inplace(&mut tensor, halve);
            let expected = Array2::from_array([[0., -1.], [2., -3.]]);
            assert_eq!(tensor, expected);
        }
    
        #[test]
        fn test_dot() {
            let tensor_a = Array2::from_array([[0., 1., 2.], [3., 2., 1.]]);
            let tensor_b = Array2::from_array([[-1., -2.], [-3., -2.], [-1., 0.]]);
            let result = Dot::dot(&tensor_a, tensor_b);
            let expected = Array2::from_array([[-5., -2.], [-10., -10.]]);
            assert_eq!(result, expected);
            let result2 = result.dot(&result);
            let expected2 = Array2::from_array([[45., 30.], [150., 120.]]);
            assert_eq!(result2, expected2);
        }
    }

    /// Tests that the `TensorOp` trait alias covers the expected traits.
    /// Also tests that `NDTensor` fully implements `TensorOp`.
    /// Only for the compiler, doesn't need to be executed as a test.
    #[allow(dead_code)]
    fn test_tensor_op() {
        fn some_generic_func<T: TensorOp>(mut t1: T, t2: T) {
            t1 += &t2;
            t1 /= &t2;
            t1 *= &t2;
            t1 -= &t2;
            let _t5 = &t1 + &t2;
            let _t6 = &t1 - &t2;
            let _t7 = &t1 * &t2;
            let _t8 = &t1 / &t2;
            t1.dot(&t2);
            t1.dot(t2);
        }
        let tensor_a = Array2::from_array([[1., 2.], [3., 4.]]);
        let tensor_b = Array2::from_array([[0., -1.], [-2., -3.]]);
        some_generic_func(tensor_a, tensor_b);
    }
}    

