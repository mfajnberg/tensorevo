//! Rust library for creating, training and evolving neural networks.

#![feature(associated_type_defaults, iter_array_chunks, map_try_insert, trait_alias)]

pub mod activation;
pub mod component;
pub mod cost_function;
pub mod individual;
pub mod layer;
pub mod population;
pub mod tensor;
pub mod utils;

