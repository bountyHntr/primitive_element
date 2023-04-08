use primitive_element::FiniteField;

fn main() {
    let now = std::time::Instant::now();

    let modulus = Some(vec![1, 2049, 1194, 2386, 459, 1980, 2554, 3295, 1898, 2831, 311]);
    let field = FiniteField::new(4231, 10, modulus);

    let primitive = field.primitive_element();
    
    let elapsed = now.elapsed();

    println!("{:?}", primitive.coefficients());
    println!("Elapsed: {:.2?}", elapsed);
}
