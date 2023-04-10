use std::sync::Arc;
use std::time::Instant;

use primitive_element::{FiniteField, FiniteFieldElement};

use clap::Parser;

/// Ilia Koltsa <bountyhntr1337@gmail.com>.
/// Tool for searching for a random primitive element of a field of the form GF(p^n).
/// Example of usage:
/// #1. For a field modulo 11: `primitive_element 11`;
/// #2.1. For a field GF(5569^5) modulo x^5 + 3*x + 5556: `primitive_element -e 5 -m "(1,5); (3,1); (5556,0)" 5569`;
/// #2.2. For calculations using 2 system threads: `primitive_element -e 5 -m "(1,5); (3,1); (5556,0)" -u -n 2 5569`.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The characteristic of the field
    prime_number: u128,

    /// The power to which the characteristic of the field is raised to obtain an extension (prime_number^extension_power)
    #[arg(short, long, default_value_t = 1, requires = "modulus")]
    extension_power: usize,

    /// Irreducible monic polynomial modulo which the field extension is constructed in the form [(coefficient, degree of x)...]
    /// e.g. x^4 + 2*x + 3 enter as "(1,4); (2,1); (3,0)"
    #[arg(short, long, value_delimiter = ';', requires = "extension_power")]
    modulus: Option<Vec<String>>,

    /// Flag enabling multi-threaded calculation mode
    #[arg(short, long, default_value_t = false)]
    use_multiple_threads: bool,

    /// Number of threads used in multithreaded mode
    #[arg(short, long, default_value_t = 4, requires = "use_multiple_threads")]
    n_threads: usize,
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
        FiniteFieldElement::primitive_multithreaded(field, Some(args.n_threads))
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
