use ndarray::Array2;
use num_traits::real::Real;

use crate::component::TensorComponent;


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
}

