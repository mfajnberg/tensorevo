//! Traits for tensor dimensionality/indexing.


pub trait Dimension {
    const N: usize;
}

pub trait HasLowerDimension {
    type Lower: HasHigherDimension + Dimension;
}

pub trait HasHigherDimension {
    type Higher: HasLowerDimension + Dimension;
}


pub type Dim0 = ();

pub type Dim1 = usize;

pub type Dim2 = [usize; 2];

pub type Dim3 = [usize; 3];


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

