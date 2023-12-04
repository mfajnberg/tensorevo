//! Definition of the `Tensor` trait.

use std::fmt::{Debug, Display};
use std::ops;

use ndarray::{Array2, Axis};
use num_traits::FromPrimitive;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::component::TensorComponent;


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
    type Component: TensorComponent;  // associated type must implement the `TensorComponent` trait
    
    /// Creates a `Tensor` with only zeros given a 2d shape
    fn zeros(shape: (usize, usize)) -> Self {
        Self::from_num(Self::Component::from_usize(0).unwrap(), shape)
    }
    
    /// Creates a `Tensor` from a number
    fn from_num(num: Self::Component, shape: (usize, usize)) -> Self;
    
    /// Returns the 2d shape of self as a tuple
    fn shape(&self) -> (usize, usize);
    
    /// Returns a vector of vectors of the tensor's components
    fn to_vec(&self) -> Vec<Vec<Self::Component>>;
    
    /// Returns the transpose of itself as a new tensor
    fn transpose(&self) -> Self;

    /// Calls a function for each component and returns the result as a new tensor
    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component;
    
    /// Calls a function for each component in place
    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component;
    
    /// Returns the norm of the tensor
    fn vec_norm(&self) -> Self::Component;

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
        ops::Add<Output = Self> +
        ops::Add<&'a Self, Output = Self> +
        ops::DivAssign<&'a Self> +
        ops::Div<Output = Self> +
        ops::Div<&'a Self, Output = Self> +
        ops::MulAssign<&'a Self> +
        ops::Mul<Output = Self> +
        ops::Mul<&'a Self, Output = Self> +
        ops::SubAssign<&'a Self> +
        ops::Sub<Output = Self> +
        ops::Sub<&'a Self, Output = Self>,
    for<'a> &'a Self:
        ops::Add<Output = Self> +
        ops::Add<Self, Output = Self> +
        ops::Div<Output = Self> +
        ops::Div<Self, Output = Self> +
        ops::Mul<Output = Self> +
        ops::Mul<Self, Output = Self> +
        ops::Sub<Output = Self> +
        ops::Sub<Self, Output = Self>;


pub trait TensorSerde = TensorBase + DeserializeOwned + Serialize;


pub trait Tensor = TensorOp + TensorSerde;


impl<C: TensorComponent> TensorBase for Array2<C> {
    type Component = C;

    fn from_num(num: Self::Component, shape: (usize, usize)) -> Self {
        Self::from_elem(shape, num)
    }
    
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    
    fn to_vec(&self) -> Vec<Vec<Self::Component>> {
        let mut output = Vec::<Vec<Self::Component>>::new();
        for row in self.rows() {
            output.push(row.to_vec());
        }
        return output;
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    // TODO: Define separate `Norm` trait
    //       https://github.com/mfajnberg/tensorevo/issues/20
    fn vec_norm(&self) -> Self::Component {
        self.iter().map(|x| x.powi(2)).sum::<Self::Component>().sqrt()
    }

    fn sum_axis(&self, axis: usize) -> Self {
        self.sum_axis(Axis(axis)).insert_axis(Axis(axis))
    }
}


impl<C: TensorComponent> Dot for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}


impl<C: TensorComponent> Dot<&Array2<C>> for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: &Self) -> Self::Output {
        self.dot(rhs)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    mod test_tensor_ndarray {
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
            let expected = array![[3.14, 3.14], [3.14, 3.14]];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_shape() {
            let tensor = array![[0., 1., 2.], [3., 4., 5.]];
            let shape = TensorBase::shape(&tensor);
            assert_eq!(shape, (2, 3));
        }

        #[test]
        fn test_to_vec() {
            let tensor = array![[0., 1., 2.], [3., 4., 5.]];
            let vector = tensor.to_vec();
            let expected = vec![[0., 1., 2.], [3., 4., 5.]];
            assert_eq!(vector, expected);
        }

        #[test]
        fn test_transpose() {
            let tensor = array![[0., 1., 2.], [3., 4., 5.]];
            let result = tensor.transpose();
            let expected = array![[0., 3.], [1., 4.], [2., 5.]];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map() {
            fn double<C: TensorComponent>(x: C) -> C {
                return x * C::from_usize(2).unwrap();
            }
    
            let tensor = array![[0., 1.], [2., 3.]];
            let result = TensorBase::map(&tensor, double);
            let expected = array![[0., 2.], [4., 6.]];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map_inplace() {
            fn halve<C: TensorComponent>(x: C) -> C {
                return x / C::from_usize(2).unwrap();
            }
    
            let mut tensor = array![[0., -2.], [4., -6.]];
            TensorBase::map_inplace(&mut tensor, halve);
            let expected = array![[0., -1.], [2., -3.]];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_sum_axis() {
            let tensor = array![[0., 1., 2.], [3., 4., 5.]];
            let result = TensorBase::sum_axis(&tensor, 0);
            let expected = array![[3., 5., 7.]];
            assert_eq!(result, expected);
            let result = TensorBase::sum_axis(&tensor, 1);
            let expected = array![[3.], [12.]];
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_dot_array() {
        let tensor_a = array![[0., 1., 2.], [3., 2., 1.]];
        let tensor_b = array![[-1., -2.], [-3., -2.], [-1., 0.]];
        let result = Dot::dot(&tensor_a, tensor_b);
        let expected = array![[-5., -2.], [-10., -10.]];
        assert_eq!(result, expected);
        let result2 = Dot::dot(&result, &result);
        let expected2 = array![[45., 30.], [150., 120.]];
        assert_eq!(result2, expected2);
    }

    /// Tests that the `TensorOp` trait alias covers the expected traits.
    /// Also tests that `Array2` fully implements `TensorOp`.
    /// Only for the compiler, doesn't need to be executed as a test.
    #[allow(dead_code)]
    fn test_tensor_op() {
        fn some_generic_func<T: TensorOp>(mut t1: T, t2: T) {
            t1 += &t2;
            t1 /= &t2;
            t1 *= &t2;
            t1 -= &t2;
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
            t1.dot(&t2);
            t1.dot(t2);
        }
        let tensor_a = array![[1., 2.], [3., 4.]];
        let tensor_b = array![[0., -1.], [-2., -3.]];
        some_generic_func(tensor_a, tensor_b);
    }
}    

