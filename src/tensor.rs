//! Definitions of the `Tensor` and `TensorElement` trait and the `NDTensor` reference implementation.

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::iter::Sum;
use std::ops;

use ndarray::{Array2, Axis};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
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

    /// Creates a Tensor from a Vec of TensorElements
    fn from_vec(v: &[Vec<Self::Element>]) -> Self;
    
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
        ops::AddAssign +
        ops::AddAssign<&'a Self> +
        ops::DivAssign +
        ops::DivAssign<&'a Self> +
        ops::MulAssign +
        ops::MulAssign<&'a Self> +
        ops::SubAssign +
        ops::SubAssign<&'a Self>,
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
    return Ok(T::from_vec(&v));
}


/// Reference `Tensor` implementation using `ndarray` data structures
///
/// Data is currently always stored as a 2D array!
#[derive(Clone, Debug, PartialEq)]
pub struct NDTensor<TE: TensorElement> {
    // https://docs.rs/ndarray/latest/ndarray/type.Array2.html
    data: Array2<TE>,
}


/// Custom `NDTensor`-specific methods not defined on the `Tensor` trait.
impl<TE: TensorElement> NDTensor<TE> {
    /// Simple constructor to directly pass the data as an `ndarray`.
    pub fn new(data: Array2<TE>) -> Self {
        return Self{data}
    }
}


/// Allows `serde` to deserialize to `NDTensor` objects.
impl<'de, TE: TensorElement> Deserialize<'de> for NDTensor<TE> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        return deserialize_to_tensor(deserializer);
    }
}


/// Allows `serde` to serialize `NDTensor` objects.
impl<TE: TensorElement> Serialize for NDTensor<TE> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let vec = &self.to_vec();
        return vec.serialize(serializer);
    }
}


/// Generic implementation of `Tensor` for `NDTensor` in terms of any `TensorElement`.
impl<TE: TensorElement> TensorBase for NDTensor<TE> {
    type Element = TE;

    fn from_vec(v: &[Vec<Self::Element>]) -> Self {
        let mut data = Vec::new();
        let num_columns = v.first().map_or(0, |row| row.len());
        let num_rows = v.len();
        for row in v {
            data.extend_from_slice(row);
        }
        let arr = Array2::from_shape_vec((num_rows, num_columns), data);
        return Self {data: arr.unwrap()};
    }

    fn from_num(num: Self::Element, shape: (usize, usize)) -> Self {
        return Self {data: Array2::<TE>::from_elem(shape, num)};
    }
    
    fn zeros(shape: (usize, usize)) -> Self {
        return Self {data: Array2::<TE>::zeros(shape)}
    }
    
    fn shape(&self) -> (usize, usize) {
        return self.data.dim();
    }
    
    fn to_vec(&self) -> Vec<Vec<Self::Element>> {
        let mut output = Vec::<Vec<Self::Element>>::new();
        for row in (self.data).rows() {
            output.push(row.to_vec());
        }
        return output;
    }

    fn transpose(&self) -> Self {
        return Self {data: self.data.t().to_owned()}
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(TE) -> TE {
        return Self {data: self.data.mapv(f)}
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(TE) -> TE {
        self.data.mapv_inplace(f)
    }

    fn vec_norm(&self) -> Self::Element {
        return self.data.iter().map(|x| x.powi(2)).sum::<Self::Element>().sqrt();
    }

    fn sum_axis(&self, axis: usize) -> Self {
        return Self {data: self.data.sum_axis(Axis(axis)).insert_axis(Axis(axis))};
    }
}


/// Allow printing `NDTensor` objects via `{}`
impl<TE: TensorElement> Display for NDTensor<TE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        return write!(f, "{}", self.data);
    }
}


impl<TE: TensorElement> Dot for NDTensor<TE> {
    type Output = Self;

    fn dot(&self, rhs: Self) -> Self::Output {
        Self {data: self.data.dot(&rhs.data)}
    }
}


impl<TE: TensorElement> Dot<&NDTensor<TE>> for NDTensor<TE> {
    type Output = Self;

    fn dot(&self, rhs: &Self) -> Self::Output {
        Self {data: self.data.dot(&rhs.data)}
    }
}


// Arithmetic trait implementations from `std::ops`.
// TODO: Write a macro for those.


impl<TE: TensorElement> ops::Add for &NDTensor<TE> {
    type Output = NDTensor<TE>;

    fn add(self, rhs: Self) -> Self::Output {
        NDTensor {data: &self.data + &rhs.data}
    }
}


impl<TE: TensorElement> ops::AddAssign for NDTensor<TE> {
    fn add_assign(&mut self, rhs: Self) {
        *self = NDTensor {data: &self.data + &rhs.data};
    }
}


impl<TE: TensorElement> ops::AddAssign<&NDTensor<TE>> for NDTensor<TE> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = NDTensor {data: &self.data + &rhs.data};
    }
}


impl<TE: TensorElement> ops::Div for &NDTensor<TE> {
    type Output = NDTensor<TE>;

    fn div(self, rhs: Self) -> Self::Output {
        NDTensor {data: &self.data / &rhs.data}
    }
}


impl<TE: TensorElement> ops::DivAssign for NDTensor<TE> {
    fn div_assign(&mut self, rhs: Self) {
        *self = NDTensor {data: &self.data / &rhs.data};
    }
}


impl<TE: TensorElement> ops::DivAssign<&NDTensor<TE>> for NDTensor<TE> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = NDTensor {data: &self.data / &rhs.data};
    }
}


impl<TE: TensorElement> ops::Mul for &NDTensor<TE> {
    type Output = NDTensor<TE>;

    fn mul(self, rhs: Self) -> Self::Output {
        NDTensor {data: &self.data * &rhs.data}
    }
}


impl<TE: TensorElement> ops::MulAssign for NDTensor<TE> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = NDTensor {data: &self.data * &rhs.data};
    }
}


impl<TE: TensorElement> ops::MulAssign<&NDTensor<TE>> for NDTensor<TE> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = NDTensor {data: &self.data * &rhs.data};
    }
}


impl<TE: TensorElement> ops::Sub for &NDTensor<TE> {
    type Output = NDTensor<TE>;

    fn sub(self, rhs: Self) -> Self::Output {
        NDTensor {data: &self.data - &rhs.data}
    }
}


impl<TE: TensorElement> ops::SubAssign for NDTensor<TE> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = NDTensor {data: &self.data - &rhs.data};
    }
}


impl<TE: TensorElement> ops::SubAssign<&NDTensor<TE>> for NDTensor<TE> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = NDTensor {data: &self.data - &rhs.data};
    }
}


#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    fn double<TE: TensorElement>(x: TE) -> TE {
        return x * TE::from_usize(2).unwrap();
    }

    fn halve<TE: TensorElement>(x: TE) -> TE {
        return x / TE::from_usize(2).unwrap();
    }

    fn double_tensor<T: TensorBase>(tensor: &T) -> T {
        return tensor.map(double);
    }
    
    fn halve_tensor_inplace<T: TensorBase>(tensor: &mut T) {
        tensor.map_inplace(halve)
    }

    #[test]
    fn test_dot() {
        let tensor_a = NDTensor::from_vec(&vec![vec![0., 1., 2.], vec![3., 2., 1.]]);
        let tensor_b = NDTensor::from_vec(&vec![vec![-1., -2.], vec![-3., -2.], vec![-1., 0.]]);
        let result = tensor_a.dot(tensor_b);
        let expected = NDTensor::from_vec(&vec!(vec![-5., -2.], vec![-10., -10.]));
        assert_eq!(result, expected);
        let result2 = result.dot(&result);
        let expected2 = NDTensor::from_vec(&vec!(vec![45., 30.], vec![150., 120.]));
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_add() {
        let tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![0., -1.], vec![-2., -3.]));
        let result = &tensor_a + &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 1.], vec![1., 1.]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_assign() {
        let mut tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![0., -1.], vec![-2., -3.]));
        tensor_a += &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 1.], vec![1., 1.]));
        assert_eq!(tensor_a, expected);
        tensor_a += tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 0.], vec![-1., -2.]));
        assert_eq!(tensor_a, expected);
    }

    #[test]
    fn test_div() {
        let tensor_a = NDTensor::from_vec(&vec!(vec![2., 4.], vec![6., 8.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![2., -1.], vec![-2., -4.]));
        let result = &tensor_a / &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., -4.], vec![-3., -2.]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_div_assign() {
        let mut tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![1., -1.], vec![-1., -2.]));
        tensor_a /= &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., -2.], vec![-3., -2.]));
        assert_eq!(tensor_a, expected);
        tensor_a /= tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 1.]));
        assert_eq!(tensor_a, expected);
    }

    #[test]
    fn test_mul() {
        let tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![-1., -2.], vec![-3., -4.]));
        let result = &tensor_a * &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![-1., -4.], vec![-9., -16.]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![0., -1.], vec![-2., -3.]));
        tensor_a *= &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![0., -2.], vec![-6., -12.]));
        assert_eq!(tensor_a, expected);
        tensor_a *= tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![0., 2.], vec![12., 36.]));
        assert_eq!(tensor_a, expected);
    }

    #[test]
    fn test_sub() {
        let tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![-1., -2.], vec![-3., -4.]));
        let result = &tensor_a - &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![2., 4.], vec![6., 8.]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sub_assign() {
        let mut tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![0., -1.], vec![-2., -3.]));
        tensor_a -= &tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 3.], vec![5., 7.]));
        assert_eq!(tensor_a, expected);
        tensor_a -= tensor_b;
        let expected = NDTensor::from_vec(&vec!(vec![1., 4.], vec![7., 10.]));
        assert_eq!(tensor_a, expected);
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
            let t5 = &t1 + &t2;
            let t6 = &t1 - &t2;
            let t7 = &t1 * &t2;
            let t8 = &t1 / &t2;
            t1 += t5;
            t1 += t6;
            t1 += t7;
            t1 += t8;
            t1.dot(&t2);
            t1.dot(t2);
        }
        let tensor_a = NDTensor::from_vec(&vec!(vec![1., 2.], vec![3., 4.]));
        let tensor_b = NDTensor::from_vec(&vec!(vec![0., -1.], vec![-2., -3.]));
        some_generic_func(tensor_a, tensor_b);
    }
}    

