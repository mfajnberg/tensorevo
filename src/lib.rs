//! Rust library for creating, training and evolving neural networks.

#![feature(associated_type_defaults, fn_traits, iter_array_chunks, map_try_insert, trait_alias, unboxed_closures)]
#![cfg_attr(test, feature(proc_macro_hygiene))]

pub mod activation;
pub mod component;
pub mod cost_function;
pub mod dimension;
pub mod evolution;
pub mod individual;
pub mod layer;
pub mod ops;
pub mod tensor;
pub mod utils;
pub mod world;

