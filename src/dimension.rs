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


impl Dimension for () {
    const N: usize = 0;
}

impl HasHigherDimension for () {
    type Higher = usize;
}

impl Dimension for usize {
    const N: usize = 1;
}

impl HasLowerDimension for usize {
    type Lower = ();
}

impl HasHigherDimension for usize {
    type Higher = [usize; 2];
}

impl Dimension for [usize; 2] {
    const N: usize = 2;
}

impl HasLowerDimension for [usize; 2] {
    type Lower = usize;
}

impl HasHigherDimension for [usize; 2] {
    type Higher = [usize; 3];
}

impl Dimension for [usize; 3] {
    const N: usize = 3;
}

impl HasLowerDimension for [usize; 3] {
    type Lower = [usize; 2];
}

