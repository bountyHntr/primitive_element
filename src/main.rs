use std::sync::Arc;
use std::time::Instant;

use primitive_element::{FiniteField, FiniteFieldElement};

fn main() {
    let now = Instant::now();

    let modulus = Some(vec![
        1, 1039, 1865, 6877, 1184, 5629, 6068, 4121, 2674, 4716, 4765, 2699, 3088, 3887, 6598, 230,
        2642, 172, 1539, 4669, 6149, 1045, 6676, 599, 3229, 7251, 3049, 4748, 3940, 1579, 7053,
        3689, 6494,
    ]);
    let field = Arc::new(FiniteField::new(7919, 32, modulus));
    let primitive = FiniteFieldElement::primitive_multithreaded(field, None);

    let elapsed = now.elapsed();

    println!("{} ({:.2?})", primitive, elapsed);
}
