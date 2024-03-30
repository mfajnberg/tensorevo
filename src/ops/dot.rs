use ndarray::{Array2, Array1};

use crate::component::TensorComponent;


/// Dot product aka. matrix multiplication.
pub trait Dot<Rhs = Self> {
    /// The output type of the multiplication operation.
    type Output;

    /// Returns the dot product of `self` on the left and `rhs` on the right.
    fn dot(&self, rhs: Rhs) -> Self::Output;
}


/// Dot product of an [`Array2`] with another [`Array2`] on the right-hand side (moved).
impl<C: TensorComponent> Dot for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: Self) -> Self::Output {
        Array2::dot(self, &rhs)
    }
}


/// Dot product of an [`Array2`] with `&`[`Array2`] on the right-hand side (borrowed).
impl<C: TensorComponent> Dot<&Self> for Array2<C> {
    type Output = Self;

    fn dot(&self, rhs: &Self) -> Self::Output {
        Array2::dot(self, rhs)
    }
}


/// Dot product of an [`Array2`] with [`Array1`] on the right-hand side (moved).
impl<C: TensorComponent> Dot<Array1<C>> for Array2<C> {
    type Output = Array1<C>;

    fn dot(&self, rhs: Array1<C>) -> Self::Output {
        Array2::dot(self, &rhs)
    }
}


/// Dot product of an [`Array2`] with `&`[`Array1`] on the right-hand side (borrowed).
impl<C: TensorComponent> Dot<&Array1<C>> for Array2<C> {
    type Output = Array1<C>;

    fn dot(&self, rhs: &Array1<C>) -> Self::Output {
        Array2::dot(self, rhs)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

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
}
