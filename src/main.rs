use primitive_element::FiniteField;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let modulus = Some(vec![
        1, 121, 1816, 3235, 3611, 3201, 1630, 4008, 981, 1218, 4168,
    ]);
    let field = FiniteField::new(4231, 10, modulus);
    let primitive = field.primitive_element();

    let elapsed = now.elapsed();

    println!("{} ({:.2?})", primitive, elapsed);
}
