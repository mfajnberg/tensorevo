use num_traits::real::Real;
use num_traits::{FromPrimitive, NumCast, ToPrimitive};


/// Exponentiation to any real number or integer power.
pub trait Pow {
    /// Result type of the exponentiation to a real number power.
    type Output;

    /// Raises `self` to the power of `exponent` (real number).
    fn powf(&self, exponent: impl Real) -> Self::Output;

    /// Raises `self` to the power of `exponent` (integer).
    fn powi(&self, exponent: i32) -> Self;

    /// Returns the square root of `self`.
    fn sqrt(&self) -> Self::Output {
        self.powf(0.5)
    }
}


/// Wrapper around the built-in `f32` methods `powf`, `powi` and `sqrt`.
impl Pow for f32 {
    type Output = Self;

    fn powf(&self, exponent: impl Real) -> Self::Output {
        f32::powf(*self, <f32 as NumCast>::from(exponent).unwrap())
    }

    fn powi(&self, exponent: i32) -> Self {
        f32::powi(*self, exponent)
    }

    fn sqrt(&self) -> Self::Output {
        f32::sqrt(*self)
    }
}


/// Wrapper around the built-in `f64` methods `powf`, `powi` and `sqrt`.
impl Pow for f64 {
    type Output = Self;

    fn powf(&self, exponent: impl Real) -> Self::Output {
        f64::powf(*self, <f64 as NumCast>::from(exponent).unwrap())
    }

    fn powi(&self, exponent: i32) -> Self {
        f64::powi(*self, exponent)
    }

    fn sqrt(&self) -> Self::Output {
        f64::sqrt(*self)
    }
}


/// Perform exponentiation, rounding down where no integer is returned.
impl Pow for isize {
    type Output = Self;

    /// Converts both `self` and `exponent` to `f64`, performs exponentiation, and converts back
    /// (rounding down).
    fn powf(&self, exponent: impl Real) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().powf(exponent.to_f64().unwrap())
        ).unwrap()
    }

    /// Converts both `self` and `exponent` to `f64`, performs exponentiation, and converts back
    /// (rounding down).
    fn powi(&self, exponent: i32) -> Self {
        self.pow(exponent.to_u32().unwrap())
    }

    /// Converts `self` to `f64`, takes the square root, and converts back (rounding down).
    fn sqrt(&self) -> Self::Output {
        Self::from_f64(
            self.to_f64().unwrap().sqrt()
        ).unwrap()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow() {
        let base: f32 = 8.;
        let result = Pow::powi(&base, 2);
        assert_eq!(result, 64.);

        let result = Pow::powf(&base, 1./3.);
        assert_eq!(result, 2.);

        let base: f64 = 64.;
        let result = Pow::sqrt(&base);
        assert_eq!(result, 8.);

        let base: isize = 64;
        let result = base.sqrt();
        assert_eq!(result, 8);

        let result = result.powf(1./3.);
        assert_eq!(result, 2);

        let result = result.powi(2);
        assert_eq!(result, 4);
    }
}

