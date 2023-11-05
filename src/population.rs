use std::collections::HashMap;

use crate::species::Species;
use crate::individual::Individual;
use crate::tensor::Tensor;


type SelectFunc<T> = for<'a> fn(&Population<T>, &'a Species<T>) -> Vec<(&'a Individual<T>, &'a Individual<T>)>;
type ProcreateFunc<T> = fn(&Individual<T>, &Individual<T>) -> Individual<T>;
type DetermineSpeciesKey<T> = fn(&Individual<T>) -> String;


pub struct Population<T: Tensor> {
    species: HashMap<String, Species<T>>,
    kill_weak_and_select_parents: SelectFunc<T>,
    procreate_pair: ProcreateFunc<T>, // crossover, mutation
    determine_species_key: DetermineSpeciesKey<T>,
}


impl<T: Tensor> Population<T> {
    pub fn new(
        selection_function: SelectFunc<T>,
        procreation_function: ProcreateFunc<T>,
        determine_key_function: DetermineSpeciesKey<T>,
    ) -> Self {
        return Population {
            species: HashMap::new(),
            kill_weak_and_select_parents: selection_function,
            procreate_pair: procreation_function,
            determine_species_key: determine_key_function,
        };
    }
    
    // mutates sets of individuals within one species
    pub fn apply_selection(&mut self, species_key: String) {
        let current_species = self.species.get(&species_key).unwrap();
        let pairs = (self.kill_weak_and_select_parents)(self, current_species);

        let new_individuals: Vec<Individual<T>> = pairs.iter().map(
            |(parent1, parent2)| (self.procreate_pair)(parent1, parent2)
        ).collect();

        for new_individual in new_individuals {
            let key = (self.determine_species_key)(&new_individual);
            match self.species.get_mut(&key) {
                Some(existing_species) => existing_species.add(new_individual),
                _ => {
                    let mut new_species = Species::<T>::new();
                    new_species.add(new_individual);
                    self.species.insert(key, new_species);
                }
            }
        }
    }
}
