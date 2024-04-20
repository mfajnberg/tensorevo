//! Traits for tensor dimensionality/indexing.


/// Trait for types that express the dimensions of something.
///
/// Required for the [`TensorBase`]`::Dim` type.
///
/// [`TensorBase`]: crate::tensor::TensorBase
pub trait Dimension {
    /// The number of dimensions expressed by the type.
    const N: usize;
}

/// Trait for `Dimension` types that have an associated lower dimension type.
///
/// Required for the [`TensorBase`]`::Dim` type.
///
/// [`TensorBase`]: crate::tensor::TensorBase
pub trait HasLowerDimension: Dimension {
    /// The associated lower dimension type.
    ///
    /// For consistency, that type should point back to the implementing type. (See below)
    type Lower: Dimension + HasHigherDimension;
}

/// Trait for `Dimension` types that have an associated higher dimension type.
///
/// Required for the [`TensorBase`]`::Dim` type.
///
/// [`TensorBase`]: crate::tensor::TensorBase
pub trait HasHigherDimension: Dimension {
    /// The associated higher dimension type.
    ///
    /// For consistency, that type should point back to the implementing type. (See below)
    type Higher: Dimension + HasLowerDimension;
}


/// Alias for the 0-dimension reference type.
pub type Dim0 = ();

/// Alias for the 1-dimension reference type.
pub type Dim1 = usize;

/// Alias for the 2-dimension reference type.
pub type Dim2 = [usize; 2];

/// Alias for the 3-dimension reference type.
pub type Dim3 = [usize; 3];

/// Alias for the 4-dimension reference type.
pub type Dim4 = [usize; 4];


impl Dimension for Dim0 {
    const N: usize = 0;
}

impl HasHigherDimension for Dim0 {
    type Higher = Dim1;
}

impl Dimension for Dim1 {
    const N: usize = 1;
}

impl HasLowerDimension for Dim1 {
    type Lower = Dim0;
}

impl HasHigherDimension for Dim1 {
    type Higher = Dim2;
}

impl Dimension for Dim2 {
    const N: usize = 2;
}

impl HasLowerDimension for Dim2 {
    type Lower = Dim1;
}

impl HasHigherDimension for Dim2 {
    type Higher = Dim3;
}

impl Dimension for Dim3 {
    const N: usize = 3;
}

impl HasLowerDimension for Dim3 {
    type Lower = Dim2;
}

impl HasHigherDimension for Dim3 {
    type Higher = Dim4;
}

impl Dimension for Dim4 {
    const N: usize = 4;
}

impl HasLowerDimension for Dim4 {
    type Lower = Dim3;
}

