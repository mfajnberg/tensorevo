//! Definition of the [`Registered`] trait.

use std::collections::HashMap;
use std::sync::RwLock;

use generic_singleton::get_or_init;
use serde::{Deserializer, Serialize, Serializer};
use serde::de::DeserializeOwned;

use crate::utils::registry::{Registry, RegistryKey, RegistryValue};


/// Add instances to a static [`Registry`] under an arbitrary key and retrieve them later.
///
/// The registry is a generic singleton. It is static from the moment of initialization and generic
/// over the key type. This means that different key types will use different registries. Likewise,
/// each type implementing [`Registered`] will have its own registry.
///
/// A practical use case is facilitating deserialization to function pointers from names.
/// This trait is implemented for [`Activation`] and [`CostFunction`].
///
/// # Example
///
/// Say we have a struct with a function pointer field that we want to serialize and deserialize.
/// For convenience we store a name for that function in a separate field. We can use that name as
/// the registry key for a struct pointing to our function.
///
/// Implementing the [`serde`] traits is made very easy with the
/// [`Registered::deserialize_from_key`]/[`Registered::serialize_as_key`] convenience functions.
///
/// ```rust
/// use serde::{Deserialize, Deserializer, Serialize, Serializer};
/// use tensorevo::utils::registered::Registered;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct NamedFunction {
///     name: String,
///     function: fn() -> f32,
/// }
///
/// impl Registered<String> for NamedFunction {
///     fn key(&self) -> &String {
///         &self.name
///     }
/// }
///
/// impl<'de> Deserialize<'de> for NamedFunction {
///     fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
///         Registered::deserialize_from_key(deserializer)
///     }
/// }
///
/// impl Serialize for NamedFunction {
///     fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
///         Registered::serialize_as_key(self, serializer)
///     }
/// }
///
/// fn zero() -> f32 { 0. }
///
/// fn main() {
///     let named_zero = NamedFunction { name: "zero".to_owned(), function: zero };
///     named_zero.register();  // This step is required for deserialization.
///     let serialized = serde_json::to_string(&named_zero).unwrap();
///     assert_eq!(&serialized, "\"zero\"");
///     let deserialized: NamedFunction = serde_json::from_str(&serialized).unwrap();
///     assert_eq!(deserialized, named_zero);
/// }
/// ```
///
/// # Implementation detail
/// Under the hood the registry uses the **[`generic_singleton`]** crate to initialize and access
/// a [`Registry`] type ([`HashMap`] by default) with `K` type keys and `Self` type values.
///
/// [`generic_singleton`]: https://docs.rs/generic_singleton/latest/generic_singleton/
/// [`Activation`]: crate::activation::Activation
/// [`CostFunction`]: crate::cost_function::CostFunction
pub trait Registered<K>: Clone + RegistryValue
where
    K: DeserializeOwned + RegistryKey + Serialize,
{
    /// Associated [`Registry`] type to use for storing instances of `Self`.
    ///
    /// [`HashMap<K, Self>`] by default.
    type Registry: Registry<K, Self> = HashMap<K, Self>;

    /// Returns the key (as a reference) under which the instance can be registered.
    fn key(&self) -> &K;

    /// Called by the default implementation of [`Registered::get_registry`] upon initialization.
    ///
    /// This means by default it will be called at most **once** for any `Registered<K>` type.
    /// Can be used to e.g. automatically fill the registry with initial instances.
    ///
    /// This function should probably not be called directly.
    ///
    /// # Arguments
    /// - `registry_lock` - Associated registry singleton wrapped in a borrowed [`RwLock`].
    #[allow(unused_variables)]
    fn registry_post_init(registry_lock: &RwLock<Self::Registry>) {}

    /// Returns a reference to the associated registry singleton.
    ///
    /// When called for the first time on a specific `Registry<K>` type, the associated
    /// [`Registered::Registry`] is initialized and [`Registered::registry_post_init`] is called with it.
    /// Repeated calls simply return the registry singleton.
    ///
    /// This function should probably not be called directly.
    /// Use [`Registered::register`] and [`Registered::get`] instead.
    ///
    /// # Returns
    /// Reference to the associated static registry singleton wrapped in a [`RwLock`].
    fn get_registry() -> &'static RwLock<Self::Registry> { 
        get_or_init!(|| { 
            let registry_lock = RwLock::new(Self::Registry::default()); 
            Self::registry_post_init(&registry_lock); 
            registry_lock 
        }) 
    }

    /// Adds a clone of the instance to the associated registry under the key returned by the
    /// [`Registered::key`] implementation.
    ///
    /// Subsequent calls to [`Registered::get`] passing the instance's key will
    /// return a clone of that instance.
    ///
    /// Initializes the associated registry singleton first, if necessary.
    ///
    /// # Returns
    /// [`None`] if nothing was registered under the instance's key.
    /// Otherwise the previous instance is replaced and returned.
    fn register(&self) -> Option<Self> {
        let registry_lock = Self::get_registry();
        registry_lock.write().unwrap()
                     .add(self.key().clone(), self.clone())
    }

    /// Retrieves a clone of a previously registered instance.
    ///
    /// Initializes the associated registry singleton first, if necessary.
    ///
    /// # Arguments
    /// - `key` - Key under which the original instance was registered.
    ///
    /// # Returns
    /// Clone of the instance with the specified `key` or [`None`] if none was registered under
    /// that key.
    fn get(key: impl Into<K>) -> Option<Self> {
        let registry_lock = Self::get_registry();
        registry_lock.read().unwrap()
                     .get_ref(&key.into())
                     .cloned()
    }

    /// Convenience method for implementing [`serde::Serialize`] for types that implement
    /// [`Registered`] such that instances are serialized through their keys.
    fn serialize_as_key<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.key().serialize(serializer)
    }

    /// Convenience function for implementing [`serde::Deserialize`] for types that implement
    /// [`Registered`] such that instances are deserialized from their names.
    fn deserialize_from_key<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de>
    {
        Ok(Self::get(K::deserialize(deserializer)?).unwrap().clone())
    }
}

