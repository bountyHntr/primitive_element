use std::collections::VecDeque;
use std::fmt::Display;
use std::ops::{Add, Mul, Sub};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use rand::distributions::Uniform;
use rand::Rng;

use prime_factorization::Factorization;

use threadpool::ThreadPool;

#[derive(Debug, PartialEq, Eq)]
pub struct FiniteField {
    prime_number: u128,
    extension_power: usize,
    modulus: Vec<u128>,

    powers_by_factors: Vec<u128>,
}

impl FiniteField {
    pub fn new(prime_number: u128, extension_power: usize, modulus: Option<Vec<u128>>) -> Self {
        assert!(
            extension_power > 0,
            "extension power must be positive number"
        );
        assert!(prime_number > 1, "prime number must be greater than 1");

        let multiplicative_order = prime_number.pow(extension_power as u32) - 1;

        let modulus = match modulus {
            Some(poly) => poly,
            None => vec![1, multiplicative_order],
        };

        assert_eq!(
            modulus.len() - 1,
            extension_power,
            "invalid irreducible polynomial for given extension power"
        );
        assert_eq!(modulus[0], 1, "irreducible polynomial must be monic");

        let mut factors = Factorization::run(multiplicative_order).factors;
        factors.dedup();

        let powers_by_factors = factors
            .into_iter()
            .map(|factor| multiplicative_order / factor)
            .collect();

        FiniteField {
            prime_number,
            extension_power,
            modulus,
            powers_by_factors,
        }
    }

    pub fn prime_number(&self) -> u128 {
        self.prime_number
    }

    pub fn extension_power(&self) -> usize {
        self.extension_power
    }

    pub fn modulus(&self) -> &[u128] {
        &self.modulus
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FiniteFieldElement {
    field: Arc<FiniteField>,
    coefficients: VecDeque<u128>,
}

impl FiniteFieldElement {
    const N_WORKERS: usize = 4;

    pub fn new(field: Arc<FiniteField>, coefficients: Vec<u128>) -> FiniteFieldElement {
        assert_eq!(
            field.extension_power,
            coefficients.len(),
            "invalid number of coefficients: {}; field extension power: {}",
            coefficients.len(),
            field.extension_power
        );

        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| coefficient % field.prime_number)
            .collect();

        FiniteFieldElement {
            field,
            coefficients,
        }
    }

    pub fn zero(field: Arc<FiniteField>) -> FiniteFieldElement {
        FiniteFieldElement {
            field: field.clone(),
            coefficients: VecDeque::from(vec![0u128; field.extension_power]),
        }
    }

    pub fn identity(field: Arc<FiniteField>) -> FiniteFieldElement {
        let mut ff_element = Self::zero(field);
        *ff_element.coefficients.back_mut().unwrap() = 1;

        ff_element
    }

    pub fn is_identity(&self) -> bool {
        !self
            .coefficients
            .iter()
            .take(self.coefficients.len() - 1)
            .any(|&coef| coef != 0)
            && self.coefficients.back() == Some(&1)
    }

    pub fn is_zero(&self) -> bool {
        !self.coefficients.iter().any(|&coef| coef != 0)
    }

    pub fn coefficients(&self) -> &VecDeque<u128> {
        &self.coefficients
    }

    pub fn pow(self, mut power: u128) -> Self {
        let mut result = Self::identity(self.field.clone());
        let mut square = self;

        while power != 0 {
            if power & 1 == 1 {
                result = result * &square;
            }

            power = power >> 1;
            square = square.clone() * &square;
        }

        result
    }

    pub fn random(field: Arc<FiniteField>) -> FiniteFieldElement {
        let range = Uniform::new(0, field.prime_number);
        let mut rng = rand::thread_rng();

        let coefficients: VecDeque<u128> = (0..field.extension_power)
            .map(|_| rng.sample(&range))
            .collect();

        FiniteFieldElement {
            field,
            coefficients,
        }
    }

    pub fn primitive(field: Arc<FiniteField>) -> FiniteFieldElement {
        'outer: loop {
            let random_element = Self::random(field.clone());

            if random_element.is_zero() {
                continue;
            }

            for &power in &field.powers_by_factors {
                if random_element.clone().pow(power).is_identity() {
                    continue 'outer;
                }
            }

            return random_element;
        }
    }

    pub fn primitive_multithreaded(
        field: Arc<FiniteField>,
        n_workers: Option<usize>,
    ) -> FiniteFieldElement {
        let n_workers = match n_workers {
            Some(n) => n,
            None => Self::N_WORKERS,
        };

        let found = Arc::new(AtomicBool::new(false));
        let pool = ThreadPool::new(n_workers);

        let (tx, rx) = mpsc::channel();

        for _ in 0..n_workers {
            let tx = tx.clone();
            let found = found.clone();
            let field = field.clone();

            pool.execute(move || {
                'outer: while !found.load(Ordering::SeqCst) {
                    let random_element = Self::random(field.clone());

                    if random_element.is_zero() {
                        continue;
                    }

                    for &power in &field.powers_by_factors {
                        if found.load(Ordering::SeqCst) {
                            return;
                        }

                        if random_element.clone().pow(power).is_identity() {
                            continue 'outer;
                        }
                    }

                    found.store(true, Ordering::SeqCst);
                    tx.send(random_element).unwrap();
                    return;
                }
            })
        }

        pool.join();
        rx.recv().unwrap()
    }
}

impl Add<&Self> for FiniteFieldElement {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "elements do not belong to the same field"
        );

        let prime_number = self.field.prime_number();

        for i in 0..self.coefficients.len() {
            if rhs.coefficients[i] != 0 {
                self.coefficients[i] = (self.coefficients[i] + rhs.coefficients[i]) % prime_number;
            }
        }

        self
    }
}

impl Sub<&Self> for FiniteFieldElement {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "elements do not belong to the same field"
        );

        let prime_number = self.field.prime_number();

        for i in 0..self.coefficients.len() {
            if rhs.coefficients[i] == 0 {
                continue;
            }

            if self.coefficients[i] < rhs.coefficients[i] {
                self.coefficients[i] += prime_number
            }

            self.coefficients[i] = self.coefficients[i] - rhs.coefficients[i];
        }

        self
    }
}

impl Mul<u128> for FiniteFieldElement {
    type Output = Self;

    fn mul(mut self, rhs: u128) -> Self::Output {
        let prime_number = self.field.prime_number();

        for i in 0..self.coefficients.len() {
            self.coefficients[i] = (self.coefficients[i] * rhs) % prime_number;
        }

        self
    }
}

impl Mul<&Self> for FiniteFieldElement {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "elements do not belong to the same field"
        );

        let mut result = Self::zero(self.field.clone());
        let modulus = Self::new(self.field.clone(), self.field.modulus()[1..].to_owned());
        let max_power = rhs.coefficients.len() - 1;

        for i in 0..=max_power {
            let power = max_power - i;
            let mut tmp = self.clone() * rhs.coefficients[i];

            for _ in 0..power {
                // left shift
                let coef = tmp.coefficients.pop_front().unwrap();
                tmp.coefficients.push_back(0);

                for _ in 0..coef {
                    tmp = tmp - &modulus;
                }
            }

            result = result + &tmp;
        }

        result
    }
}

impl Display for FiniteFieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_power = self.coefficients.len() - 1;

        for i in 0..max_power {
            if self.coefficients[i] != 0 {
                write!(f, "{}*x^{} + ", self.coefficients[i], max_power - i)?;
            }
        }

        write!(f, "{}", self.coefficients.back().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_element() {
        let modulus = Some(vec![1, 1, 0, 0, 2]); // x^4 + x^3 + 2
        let field = Arc::new(FiniteField::new(3, 4, modulus));
        let field_element = FiniteFieldElement::random(field.clone());

        let coefs = field_element.coefficients();
        assert_eq!(coefs.len(), field.extension_power() as usize);

        let prime_number = field.prime_number();
        for i in 0..coefs.len() {
            assert!(coefs[i] < prime_number);
        }
    }

    #[test]
    #[should_panic]
    fn test_ff_element_invalid_num_of_coefficient() {
        let modulus = Some(vec![1, 1, 0, 0, 2]); // x^4 + x^3 + 2
        let field = Arc::new(FiniteField::new(3, 4, modulus));

        FiniteFieldElement::new(field, vec![1, 1, 0]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_extension_power() {
        FiniteField::new(3, 0, None);
    }

    #[test]
    #[should_panic]
    fn test_invalid_prime_number() {
        FiniteField::new(1, 2, None);
    }

    #[test]
    fn test_add() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = Arc::new(FiniteField::new(3, 2, modulus));

        let el1 = FiniteFieldElement::new(field.clone(), vec![2, 0]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![2, 1]);

        let expected_result = FiniteFieldElement::new(field, vec![1, 1]);
        assert_eq!(el1 + &el2, expected_result);
    }

    #[test]
    fn test_add_prime_field() {
        let field = Arc::new(FiniteField::new(7, 1, None));
        let el1 = FiniteFieldElement::new(field.clone(), vec![6]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![3]);

        let expected_result = FiniteFieldElement::new(field, vec![2]);
        assert_eq!(el1 + &el2, expected_result);
    }

    #[test]
    fn test_sub() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = Arc::new(FiniteField::new(3, 2, modulus));

        let el1 = FiniteFieldElement::new(field.clone(), vec![2, 0]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![1, 1]);

        let expected_result = FiniteFieldElement::new(field, vec![1, 2]);
        assert_eq!(el1 - &el2, expected_result);
    }

    #[test]
    fn test_mul_by_scalar() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = Arc::new(FiniteField::new(3, 2, modulus));
        let el = FiniteFieldElement::new(field.clone(), vec![2, 1]);

        let expected_result = FiniteFieldElement::new(field, vec![1, 2]);
        assert_eq!(el * 2, expected_result);
    }

    #[test]
    fn test_mul_by_ff_element() {
        let field = Arc::new(FiniteField::new(3, 2, Some(vec![1, 0, 1])));

        let el1 = FiniteFieldElement::new(field.clone(), vec![2, 2]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![2, 0]);

        let expected_result = FiniteFieldElement::new(field.clone(), vec![1, 2]);
        assert_eq!(el1 * &el2, expected_result);

        let el1 = FiniteFieldElement::new(field.clone(), vec![2, 2]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![2, 1]);

        let expected_result = FiniteFieldElement::new(field.clone(), vec![0, 1]);
        assert_eq!(el1 * &el2, expected_result);

        let field = Arc::new(FiniteField::new(
            4231,
            10,
            Some(vec![
                1, 2049, 1194, 2386, 459, 1980, 2554, 3295, 1898, 2831, 311,
            ]),
        ));

        let el1 = FiniteFieldElement::new(
            field.clone(),
            vec![4108, 1006, 1002, 973, 2776, 1231, 740, 4221, 1494, 1640],
        );
        let el2 = FiniteFieldElement::new(
            field.clone(),
            vec![3461, 3711, 3786, 3325, 284, 3477, 522, 1690, 539, 632],
        );

        let expected_result = FiniteFieldElement::new(
            field,
            vec![613, 1441, 2609, 3956, 4054, 922, 1799, 3469, 3759, 2220],
        );
        assert_eq!(el1 * &el2, expected_result);
    }

    #[test]
    fn test_mul_prime_field() {
        let field = Arc::new(FiniteField::new(7, 1, None));
        let el1 = FiniteFieldElement::new(field.clone(), vec![6]);
        let el2 = FiniteFieldElement::new(field.clone(), vec![3]);

        let expected_result = FiniteFieldElement::new(field, vec![4]);
        assert_eq!(el1 * &el2, expected_result);
    }

    #[test]
    fn test_pow() {
        let field = Arc::new(FiniteField::new(3, 2, Some(vec![1, 0, 1])));

        let el = FiniteFieldElement::new(field.clone(), vec![2, 2]);
        let expected_result = FiniteFieldElement::new(field.clone(), vec![0, 2]);
        assert_eq!(el.pow(4), expected_result);

        let el = FiniteFieldElement::new(field.clone(), vec![0, 2]);
        let expected_result = FiniteFieldElement::new(field, vec![0, 1]);
        assert_eq!(el.pow(2), expected_result);

        let field = Arc::new(FiniteField::new(
            4231,
            10,
            Some(vec![
                1, 2049, 1194, 2386, 459, 1980, 2554, 3295, 1898, 2831, 311,
            ]),
        ));

        let el = FiniteFieldElement::new(
            field.clone(),
            vec![403, 2783, 3190, 2185, 2879, 318, 3562, 1792, 2847, 1826],
        );
        let expected_result = FiniteFieldElement::new(
            field.clone(),
            vec![4108, 1006, 1002, 973, 2776, 1231, 740, 4221, 1494, 1640],
        );
        assert_eq!(el.pow(23234234), expected_result);

        let el = FiniteFieldElement::new(
            field.clone(),
            vec![403, 2783, 3190, 2185, 2879, 318, 3562, 1792, 2847, 1826],
        );
        let expected_result = FiniteFieldElement::new(
            field,
            vec![2600, 1009, 2476, 1944, 1322, 3033, 910, 576, 240, 2985],
        );
        assert_eq!(el.pow(234234928348923984923982438), expected_result);
    }

    #[test]
    fn test_is_identity() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = Arc::new(FiniteField::new(3, 2, modulus));

        let el = FiniteFieldElement::new(field.clone(), vec![0, 1]);
        assert!(el.is_identity());

        let el = FiniteFieldElement::new(field, vec![0, 0]);
        assert!(!el.is_identity());
    }

    #[test]
    fn test_is_zero() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = Arc::new(FiniteField::new(3, 2, modulus));

        let el = FiniteFieldElement::new(field.clone(), vec![0, 0]);
        assert!(el.is_zero());

        let el = FiniteFieldElement::new(field, vec![0, 1]);
        assert!(!el.is_zero());
    }
}
