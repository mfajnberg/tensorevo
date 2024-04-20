//! Implementations of the [`TensorBase`] trait on a few foreign types.

use ndarray::{Array1, Array2, Array3, ArrayView, Axis};

use crate::component::TensorComponent;
use crate::dimension::{Dim1, Dim2, Dim3, HasHigherDimension, HasLowerDimension};
use crate::tensor::TensorBase;


/// Implementation of [`TensorBase`] for [`Vec`].
impl<C: TensorComponent> TensorBase for Vec<C> {
    type Component = C;
    type Dim = Dim1;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        vec![num; shape.into()]
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        iterable.into_iter().take(shape.into()).collect()
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        self.len().into()
    }

    fn transpose(&self) -> Self {
        self.clone()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Vec::as_slice(self)
    }

    fn append<T>(&mut self, _axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Vec::push(self, tensor.as_slice()[0])
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(Self::Component) -> Self::Component {
        self.iter().copied().map(f).collect()
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(Self::Component) -> Self::Component {
        // Not actually in-place at all, but allocating a whole new vector before replacing.
        // It is hard to do this without unsafe code.
        *self = self.map(f);
    }

    fn iter_indexed<IDX: From<Self::Dim>>(&self) -> impl Iterator<Item = (IDX, &Self::Component)> {
        <&'_ Vec<Self::Component>>::into_iter(self).enumerate().map(|(idx, component)| (idx.into(), component))
    }

    fn iter_indexed_mut<IDX: From<Self::Dim>>(&mut self) -> impl Iterator<Item = (IDX, &mut Self::Component)> {
        <&'_ mut Vec<Self::Component>>::into_iter(self).enumerate().map(|(idx, component)| (idx.into(), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        self[..].iter()
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Component> {
        self[..].iter_mut()
    }

    fn iter_axis<T>(&self, axis: usize) -> impl Iterator<Item = T>
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        if axis != 0 {
            panic!("Axis {axis} out of bounds!")
        }
        self.iter().map(|component| T::from_num(*component, ()))
    }

    fn sum_axis<T>(&self, _axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        T::from_num(self.iter().copied().sum(), ())
    }

    fn sum_axis_same_dim(&self, _axis: usize) -> Self {
        vec![self.iter().copied().sum()]
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let shape = if axis == 0 {
            [1, self.len()]
        } else if axis == 1 {
            [self.len(), 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}


/// Implementation of [`TensorBase`] for [`ndarray::Array1`].
impl<C: TensorComponent> TensorBase for Array1<C> {
    type Component = C;
    type Dim = Dim1;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        Array1::from_elem(shape.into(), num)
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        Array1::from_iter(iterable.into_iter().take(shape.into()))
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        S::from(self.dim())
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Array1::as_slice(self).unwrap()
    }

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array1::push(self, Axis(axis), ArrayView::from_shape((), tensor.as_slice()).unwrap()).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn iter_indexed<IDX: From<Self::Dim>>(&self) -> impl Iterator<Item = (IDX, &Self::Component)> {
        Array1::<C>::indexed_iter(self).map(|(idx, component)| (idx.into(), component))
    }

    fn iter_indexed_mut<IDX: From<Self::Dim>>(&mut self) -> impl Iterator<Item = (IDX, &mut Self::Component)> {
        Array1::<C>::indexed_iter_mut(self).map(|(idx, component)| (idx.into(), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array1::<C>::iter(self)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Component> {
        Array1::<C>::iter_mut(self)
    }

    fn iter_axis<T>(&self, axis: usize) -> impl Iterator<Item = T>
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array1::<C>::axis_iter(self, Axis(axis))
                    .map(|subview| T::from_iter(subview.iter().map(|c| *c), subview.dim()))
    }

    fn sum_axis<T>(&self, _axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        T::from_num(Array1::sum(self), ())
    }

    fn sum_axis_same_dim(&self, axis: usize) -> Self {
        Array1::sum_axis(self, Axis(axis)).insert_axis(Axis(axis))
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let shape = if axis == 0 {
            [1, self.len()]
        } else if axis == 1 {
            [self.len(), 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}


/// Implementation of [`TensorBase`] for [`ndarray::Array2`].
impl<C: TensorComponent> TensorBase for Array2<C> {
    type Component = C;
    type Dim = Dim2;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        Array2::from_elem(shape.into(), num)
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        let iterator: &mut dyn Iterator<Item = Self::Component> = &mut iterable.into_iter();
        Array2::from_shape_simple_fn(shape.into(), || iterator.next().unwrap())
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        S::from(self.dim().into())
    }
    
    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Array2::as_slice(self).unwrap()
    }

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array2::push(self, Axis(axis), ArrayView::from(tensor.as_slice())).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn iter_indexed<IDX: From<Self::Dim>>(&self) -> impl Iterator<Item = (IDX, &Self::Component)> {
        Array2::<C>::indexed_iter(self).map(|(idx, component)| (IDX::from(idx.into()), component))
    }

    fn iter_indexed_mut<IDX: From<Self::Dim>>(&mut self) -> impl Iterator<Item = (IDX, &mut Self::Component)> {
        Array2::<C>::indexed_iter_mut(self).map(|(idx, component)| (IDX::from(idx.into()), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array2::<C>::iter(self)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Component> {
        Array2::<C>::iter_mut(self)
    }

    fn iter_axis<T>(&self, axis: usize) -> impl Iterator<Item = T>
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array2::<C>::axis_iter(self, Axis(axis))
                    .map(|subview| T::from_iter(subview.iter().map(|c| *c), subview.dim()))
    }

    fn sum_axis<T>(&self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        let other_axis = if axis == 0 {
            Axis(1)
        } else if axis == 1 {
            Axis(0)
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        let axis_sum_iterator = Array2::axis_iter(self, other_axis).map(|subview| subview.sum());
        T::from_iter(axis_sum_iterator, Array2::len_of(self, other_axis))
    }

    fn sum_axis_same_dim(&self, axis: usize) -> Self {
        Array2::sum_axis(self, Axis(axis)).insert_axis(Axis(axis))
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let (rows, cols) = Array2::dim(&self);
        let shape = if axis == 0 {
            [1, rows, cols]
        } else if axis == 1 {
            [rows, 1, cols]
        } else if axis == 2 {
            [rows, cols, 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), shape)
    }
}


/// Implementation of [`TensorBase`] for [`ndarray::Array3`].
impl<C: TensorComponent> TensorBase for Array3<C> {
    type Component = C;
    type Dim = Dim3;

    fn from_num(num: Self::Component, shape: impl Into<Self::Dim>) -> Self {
        Array3::from_elem(shape.into(), num)
    }

    fn from_iter<I, S>(iterable: I, shape: S) -> Self
    where
        I: IntoIterator<Item = Self::Component>,
        S: Into<Self::Dim>
    {
        let iterator: &mut dyn Iterator<Item = Self::Component> = &mut iterable.into_iter();
        Array3::from_shape_simple_fn(shape.into(), || iterator.next().unwrap())
    }

    fn shape<S: From<Self::Dim>>(&self) -> S {
        S::from(self.dim().into())
    }
    
    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn as_slice(&self) -> &[Self::Component] {
        Array3::as_slice(self).unwrap()
    }

    fn append<T>(&mut self, axis: usize, tensor: &T)
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array3::push(
            self,
            Axis(axis),
            ArrayView::from_shape(
                tensor.shape::<Dim2>(),
                tensor.as_slice()
            ).unwrap()
        ).unwrap()
    }

    fn map<F>(&self, f: F) -> Self
    where F: FnMut(C) -> C {
        self.mapv(f)
    }

    fn map_inplace<F>(&mut self, f: F)
    where F: FnMut(C) -> C {
        self.mapv_inplace(f)
    }

    fn iter_indexed<IDX: From<Self::Dim>>(&self) -> impl Iterator<Item = (IDX, &Self::Component)> {
        Array3::<C>::indexed_iter(self).map(|(idx, component)| (IDX::from(idx.into()), component))
    }

    fn iter_indexed_mut<IDX: From<Self::Dim>>(&mut self) -> impl Iterator<Item = (IDX, &mut Self::Component)> {
        Array3::<C>::indexed_iter_mut(self).map(|(idx, component)| (IDX::from(idx.into()), component))
    }

    fn iter(&self) -> impl Iterator<Item = &Self::Component> {
        Array3::<C>::iter(self)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Self::Component> {
        Array3::<C>::iter_mut(self)
    }

    fn iter_axis<T>(&self, axis: usize) -> impl Iterator<Item = T>
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        Array3::<C>::axis_iter(self, Axis(axis))
                    .map(|subview| T::from_iter(subview.iter().map(|c| *c), subview.dim()))
    }

    fn sum_axis<T>(&self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasLowerDimension>::Lower> {
        // Determine the shape of the resulting 2-dimensional tensor.
        let (x, y, z) = self.dim();
        let mut result = if axis == 0 {
            T::zeros((y, z))
        } else if axis == 1 {
            T::zeros((x, z))
        } else if axis == 2 {
            T::zeros((x, y))
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        // Look at each subview (matrix) along the specified axis.
        // Their shapes should all be equal to the shape of the result matrix.
        for subview in Array3::axis_iter(self, Axis(axis)) {
            // Add components in the same row and column along the specified axis.
            for ((row, col), component) in result.iter_indexed_mut() {
                *component += subview[[row, col]]
            }
        }
        result
    }

    fn sum_axis_same_dim(&self, axis: usize) -> Self {
        Array3::sum_axis(self, Axis(axis)).insert_axis(Axis(axis))
    }

    fn insert_axis<T>(self, axis: usize) -> T
    where T: TensorBase<Component = Self::Component, Dim = <Self::Dim as HasHigherDimension>::Higher> {
        let (x, y, z) = Array3::dim(&self);
        let new_shape = if axis == 0 {
            [1, x, y, z]
        } else if axis == 1 {
            [x, 1, y, z]
        } else if axis == 2 {
            [x, y, 1, z]
        } else if axis == 3 {
            [x, y, z, 1]
        } else {
            panic!("Axis {axis} out of bounds!")
        };
        T::from_iter(self.iter().copied(), new_shape)
    }
}


#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    // TODO: Add tests for `Vec`, `Array1`, and `Array3` implementations.

    mod test_tensor_base_array2 {
        use super::*;

        // TODO: Write missing tests for new methods.

        #[test]
        fn test_zeros() {
            let tensor: Array2<f32> = TensorBase::zeros([1, 3]);
            let expected = array![[0., 0., 0.]];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_from_num() {
            let tensor = Array2::from_num(3.14, [2, 2]);
            let expected = array![
                [3.14, 3.14],
                [3.14, 3.14]
            ];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_shape() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let shape: Dim2 = TensorBase::shape(&tensor);
            assert_eq!(shape, [2, 3]);
        }

        #[test]
        fn test_transpose() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let result = tensor.transpose();
            let expected = array![
                [0., 3.],
                [1., 4.],
                [2., 5.]
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map() {
            fn double<C: TensorComponent>(x: C) -> C {
                return x * C::from_usize(2).unwrap();
            }
    
            let tensor = array![
                [0., 1.],
                [2., 3.]
            ];
            let result = TensorBase::map(&tensor, double);
            let expected = array![
                [0., 2.],
                [4., 6.]
            ];
            assert_eq!(result, expected);
        }

        #[test]
        fn test_map_inplace() {
            fn halve<C: TensorComponent>(x: C) -> C {
                return x / C::from_usize(2).unwrap();
            }
    
            let mut tensor = array![
                [0., -2.],
                [4., -6.]
            ];
            TensorBase::map_inplace(&mut tensor, halve);
            let expected = array![
                [0., -1.],
                [2., -3.]
            ];
            assert_eq!(tensor, expected);
        }

        #[test]
        fn test_sum_axis() {
            let tensor = array![
                [0., 1., 2.],
                [3., 4., 5.]
            ];
            let result: Array1<f64> = TensorBase::sum_axis(&tensor, 0);
            let expected = array![3., 5., 7.];
            assert_eq!(result, expected);

            let result: Array1<f64> = TensorBase::sum_axis(&tensor, 1);
            let expected = array![
                3. ,
                12.
            ];
            assert_eq!(result, expected);
        }
    }
}

