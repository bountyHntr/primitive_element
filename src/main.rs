use std::sync::Arc;
use std::time::Instant;

use primitive_element::{FiniteField, FiniteFieldElement};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
    /// The characteristic of the field
    prime_number: u128,

    /// The power to which the characteristic of the field is raised to obtain an extension (prime_number^extension_power)
    #[arg(short, long, default_value_t = 1)]
    extension_power: usize,

    /// Irreducible monic polynomial modulo which the field extension is constructed in the form [(coefficient, degree of x)...]
    /// e.g. x^4 + 2*x + 3 enter as "(1,4); (2,1); (3,0)"
    #[arg(short, long, value_delimiter = ';')]
    modulus: Option<Vec<String>>,

    /// Flag enabling multi-threaded calculation mode
    #[arg(short, long, default_value_t = false)]
    use_multiple_threads: bool,

    /// Number of threads used in multithreaded mode
    #[arg(short, long)]
    n_threads: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let modulus = match args.modulus.as_ref() {
        Some(modulus) => parse_modulus(modulus),
        None => None,
    };

    let now = Instant::now();

    let field = Arc::new(FiniteField::new(
        args.prime_number,
        args.extension_power,
        modulus,
    ));

    let primitive = if args.use_multiple_threads {
        FiniteFieldElement::primitive_multithreaded(field, args.n_threads)
    } else {
        FiniteFieldElement::primitive(field)
    };

    let time_elapsed = now.elapsed();

    println!("{} ({:.2?})", primitive, time_elapsed);
}

fn parse_modulus(modulus: &[String]) -> Option<Vec<u128>> {
    if modulus.len() == 0 {
        return Some(vec![]);
    }

    let coefficients: Vec<u128> = parse_coefficient(&modulus[0]);

    let mut parsed_modulus = vec![0u128; (coefficients[1] + 1) as usize];
    parsed_modulus[0] = coefficients[0];

    let max_power = parsed_modulus.len() - 1;
    for i in 1..modulus.len() {
        let coefficients = parse_coefficient(&modulus[i]);

        let x_power = coefficients[1] as usize;
        if x_power > max_power {
            panic!(
                "invalid irreducible polynomial: power of x in {} coefficient: {}; max power: {}",
                i + 1,
                x_power,
                max_power
            );
        }

        let idx = max_power - x_power;
        parsed_modulus[idx] = coefficients[0];
    }

    Some(parsed_modulus)
}

fn parse_coefficient(coefficient: &str) -> Vec<u128> {
    let pattern: &[_] = &[' ', '(', ')', '\t', '\n'];

    coefficient
        .trim_matches(pattern)
        .split(',')
        .map(|number| {
            number.parse().expect(&format!(
                "coefficients {} contains non-numeric values",
                coefficient
            ))
        })
        .collect()
}
