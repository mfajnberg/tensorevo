//! Definition of the [`Registry`] trait and related trait aliases.

use std::collections::HashMap;
use std::hash::Hash;


/// Trait alias for types that can be used as keys in a [`Registry`] type.
pub trait RegistryKey = 'static + Clone + Eq + Hash + Send + Sync;

/// Trait alias for types that can be used as values in a [`Registry`] type.
pub trait RegistryValue = 'static + Send + Sync;


/// Types that can function as a registry.
///
/// Add values under specific keys, potentially replacing older values and retrieve references to
/// previously added values.
pub trait Registry<K, V>: Default + Send + Sized + Sync
where
    K: RegistryKey,
    V: RegistryValue,
{
    /// Adds a new key-value-pair to the `Registry` type instance.
    ///
    /// A reference to the `value` will be accessible via [`Registry::get_ref`] by passing `key`.
    ///
    /// # Arguments
    /// - `key` - Key under which to store the `value` in the registry.
    /// - `value` - Value to store in the registry under `key`.
    ///
    /// # Returns
    /// [`None`] if nothing was previously added under the specified `key`.
    /// Otherwise the previous value is replaced and returned.
    fn add(&mut self, key: K, value: V) -> Option<V>;

    /// Retrieves a reference to the value previously registered under `key`.
    ///
    /// # Arguments
    /// - `key` - Key under which the original value was added to the registry.
    ///
    /// # Returns
    /// Reference to the value stored in the registry under `key` or [`None`] if no value was
    /// registered under the specified `key`.
    fn get_ref(&self, key: &K) -> Option<&V>;
}


/// Trivial implementation of [`Registry`] for [`HashMap`].
impl<K, V> Registry<K, V> for HashMap<K, V>
where
    K: RegistryKey,
    V: RegistryValue,
{
    /// Proxy for the [`HashMap::insert`] method.
    fn add(&mut self, key: K, value: V) -> Option<V> {
        HashMap::<K, V>::insert(self, key, value)
    }

    /// Proxy for the [`HashMap::get`] method.
    fn get_ref(&self, key: &K) -> Option<&V> {
        HashMap::<K, V>::get(&self, key)
    }
}

