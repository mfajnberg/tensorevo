use num_traits::{FromPrimitive, ToPrimitive};


/// Exponentiation with the base `e` (Euler's number).
pub trait Exp {
    /// Result type of the exponentiation.
    type Output;

    /// Raises `e` to the power of `self`.
    fn exp(&self) -> Self::Output;
}


/// Wrapper around the built-in `f32::exp` method.
impl Exp for f32 {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        f32::exp(*self)
    }
}


/// Wrapper around the built-in `f64::exp` method.
impl Exp for f64 {
    type Output = Self;

    fn exp(&self) -> Self::Output {
        f64::exp(*self)
    } 
}


/// Raise `e` to the a power of `isize` type and round down.
impl Exp for isize {
    type Output = Self;

    /// Converts `self` to an `f64`, performs the exponentiation and converts back (rounding down).
    fn exp(&self) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().exp()
        ).unwrap()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp() {
        let exponent: f32 = 0.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 1.);

        let exponent: f64 = 1.;
        let result = Exp::exp(&exponent);
        assert_eq!(result, 2.718281828459045);

        let exponent: isize = 1;
        let result = exponent.exp();
        assert_eq!(result, 2);
    }
}

