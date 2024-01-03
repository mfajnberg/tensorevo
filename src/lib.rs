//! Rust library for creating, training and evolving neural networks.

#![feature(associated_type_defaults, fn_traits, iter_array_chunks, map_try_insert, trait_alias, unboxed_closures)]

pub mod activation;
pub mod component;
pub mod cost_function;
pub mod individual;
pub mod layer;
pub mod population;
pub mod tensor;
pub mod utils;

