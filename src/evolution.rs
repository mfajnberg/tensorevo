pub mod crossover;
pub mod init;
pub mod mutation;
pub mod procreation;
pub mod selection;
pub mod speciation;

pub use procreation::procreate;
pub use selection::select;
pub use speciation::get_species;

