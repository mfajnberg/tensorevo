use std::collections::HashMap;

use crate::species::Species;
use crate::individual::Individual;
use crate::tensor::Tensor;


type SelectFunc<T> = for<'a> fn(&Population<T>, &'a mut Species<T>) -> Vec<(&'a Individual<T>, &'a Individual<T>)>;
type ProcreateFunc<T> = fn(&Individual<T>, &Individual<T>) -> Individual<T>;
type DetermineSpeciesKey<T> = fn(&Individual<T>) -> String;


pub struct Population<T: Tensor> {
    species: HashMap<String, Species<T>>,
    kill_weak_and_select_parents: SelectFunc<T>,
    procreate_pair: ProcreateFunc<T>, // crossover, mutation
    determine_species_key: DetermineSpeciesKey<T>,
    pub validation_data: (T, T),
}


impl<T: Tensor> Population<T> {
    pub fn new(
        kill_weak_and_select_parents: SelectFunc<T>,
        procreate_pair: ProcreateFunc<T>,
        determine_species_key: DetermineSpeciesKey<T>,
        validation_data: (T, T),
    ) -> Self {
        Self {
            species: HashMap::new(),
            kill_weak_and_select_parents,
            procreate_pair,
            determine_species_key,
            validation_data
        }
    }
    
    // mutates sets of individuals within one species
    pub fn apply_selection(&mut self, species_key: String) {
        let mut current_species = self.species.get_mut(&species_key).unwrap();
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

fn selection_function<'a, T: Tensor>(population: &Population<T>, species: &'a mut Species<T>) -> Vec<(&'a Individual<T>, &'a Individual<T>)> {
    species.sort_by_error(&population.validation_data.0, &population.validation_data.1);
    let num_individuals = species.size();
    species.drain_individuals(num_individuals/2, num_individuals);
    // generate pairs of parents from the vector of fit individuals
}