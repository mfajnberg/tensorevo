use rand::Rng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::component::TensorComponent;


pub fn init_weight<C: TensorComponent>(rng: &mut ThreadRng, can_zero: bool) -> Option<C> {
    if can_zero && *[true, false].choose(rng).unwrap() {
        Some(C::zero())
    } else {
        C::from_f32(rng.gen_range(0f32..1f32))
    }
}

