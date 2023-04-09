use std::collections::VecDeque;
use std::fmt::Display;
use std::ops::{Add, Mul, Sub};

use rand::distributions::Uniform;
use rand::Rng;

use prime_factorization::Factorization;

#[derive(Debug, PartialEq, Eq)]
pub struct FiniteField {
    prime_number: u128,
    extension_degree: usize,
    modulus: Vec<u128>,

    powers_by_factors: Vec<u128>,
}

impl FiniteField {
    pub fn new(prime_number: u128, extension_degree: usize, modulus: Option<Vec<u128>>) -> Self {
        assert!(
            extension_degree > 0,
            "extension degree must be positive number"
        );
        assert!(prime_number > 1, "prime number must be greater than 1");

        let multiplicative_order = prime_number.pow(extension_degree as u32) - 1;

        let modulus = match modulus {
            Some(poly) => poly,
            None => vec![1, multiplicative_order],
        };

        assert_eq!(
            modulus.len() - 1,
            extension_degree,
            "invalid modulus for given extension degree"
        );
        assert_eq!(modulus[0], 1, "modulus must be monic polynomial");

        let mut factors = Factorization::run(multiplicative_order).factors;
        factors.dedup();

        let powers_by_factors = factors
            .into_iter()
            .map(|factor| multiplicative_order / factor)
            .collect();

        FiniteField {
            prime_number,
            extension_degree,
            modulus,
            powers_by_factors,
        }
    }

    pub fn prime_number(&self) -> u128 {
        self.prime_number
    }

    pub fn extension_degree(&self) -> usize {
        self.extension_degree
    }

    pub fn modulus(&self) -> &[u128] {
        &self.modulus
    }

    pub fn init_element(&self, coefficients: Vec<u128>) -> FiniteFieldElement {
        assert_eq!(
            self.extension_degree,
            coefficients.len(),
            "invalid number of coefficients: {}; field extension degree: {}",
            coefficients.len(),
            self.extension_degree
        );

        let coefficients = coefficients
            .into_iter()
            .map(|coefficient| coefficient % self.prime_number)
            .collect();

        FiniteFieldElement {
            field: self,
            coefficients,
        }
    }

    pub fn zero_element(&self) -> FiniteFieldElement {
        FiniteFieldElement {
            field: self,
            coefficients: VecDeque::from(vec![0u128; self.extension_degree]),
        }
    }

    pub fn identity_element(&self) -> FiniteFieldElement {
        let mut ff_element = self.zero_element();
        *ff_element.coefficients.back_mut().unwrap() = 1;

        ff_element
    }

    pub fn random_element(&self) -> FiniteFieldElement {
        let range = Uniform::new(0, self.prime_number);
        let mut rng = rand::thread_rng();

        let coefficients: VecDeque<u128> = (0..self.extension_degree)
            .map(|_| rng.sample(&range))
            .collect();

        FiniteFieldElement {
            field: self,
            coefficients,
        }
    }

    pub fn primitive_element(&self) -> FiniteFieldElement {
        'outer: loop {
            let random_element = self.random_element();

            if random_element.is_zero() {
                continue;
            }

            for &power in &self.powers_by_factors {
                if random_element.clone().pow(power).is_identity() {
                    continue 'outer;
                }
            }

            return random_element;
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FiniteFieldElement<'a> {
    field: &'a FiniteField,
    coefficients: VecDeque<u128>,
}

impl<'a> FiniteFieldElement<'a> {
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
        let mut result = self.field.identity_element();
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
}

impl<'a> Add<&Self> for FiniteFieldElement<'a> {
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

impl<'a> Sub<&Self> for FiniteFieldElement<'a> {
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

impl<'a> Mul<u128> for FiniteFieldElement<'a> {
    type Output = Self;

    fn mul(mut self, rhs: u128) -> Self::Output {
        let prime_number = self.field.prime_number();

        for i in 0..self.coefficients.len() {
            self.coefficients[i] = (self.coefficients[i] * rhs) % prime_number;
        }

        self
    }
}

impl<'a> Mul<&Self> for FiniteFieldElement<'a> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "elements do not belong to the same field"
        );

        let mut result = self.field.zero_element();
        let modulus = self
            .field
            .init_element(self.field.modulus()[1..].to_owned());
        let max_degree = rhs.coefficients.len() - 1;

        for i in 0..=max_degree {
            let degree = max_degree - i;
            let mut tmp = self.clone() * rhs.coefficients[i];

            for _ in 0..degree {
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

impl Display for FiniteFieldElement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_degree = self.coefficients.len() - 1;
        for i in 0..max_degree {
            write!(f, "{}*x^{} + ", self.coefficients[i], max_degree - i)?;
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
        let field = FiniteField::new(3, 4, modulus);
        let field_element = field.random_element();

        let coefs = field_element.coefficients();
        assert_eq!(coefs.len(), field.extension_degree() as usize);

        let prime_number = field.prime_number();
        for i in 0..coefs.len() {
            assert!(coefs[i] < prime_number);
        }
    }

    #[test]
    #[should_panic]
    fn test_ff_element_invalid_num_of_coefficient() {
        let modulus = Some(vec![1, 1, 0, 0, 2]); // x^4 + x^3 + 2
        let field = FiniteField::new(3, 4, modulus);

        field.init_element(vec![1, 1, 0]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_extension_degree() {
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
        let field = FiniteField::new(3, 2, modulus);

        let el1 = field.init_element(vec![2, 0]);
        let el2 = field.init_element(vec![2, 1]);

        let expected_result = field.init_element(vec![1, 1]);
        assert_eq!(el1 + &el2, expected_result);
    }

    #[test]
    fn test_add_prime_field() {
        let field = FiniteField::new(7, 1, None);
        let el1 = field.init_element(vec![6]);
        let el2 = field.init_element(vec![3]);

        let expected_result = field.init_element(vec![2]);
        assert_eq!(el1 + &el2, expected_result);
    }

    #[test]
    fn test_sub() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = FiniteField::new(3, 2, modulus);

        let el1 = field.init_element(vec![2, 0]);
        let el2 = field.init_element(vec![1, 1]);

        let expected_result = field.init_element(vec![1, 2]);
        assert_eq!(el1 - &el2, expected_result);
    }

    #[test]
    fn test_mul_by_scalar() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = FiniteField::new(3, 2, modulus);
        let el = field.init_element(vec![2, 1]);

        let expected_result = field.init_element(vec![1, 2]);
        assert_eq!(el * 2, expected_result);
    }

    #[test]
    fn test_mul_by_ff_element() {
        let field = FiniteField::new(3, 2, Some(vec![1, 0, 1]));

        let el1 = field.init_element(vec![2, 2]);
        let el2 = field.init_element(vec![2, 0]);

        let expected_result = field.init_element(vec![1, 2]);
        assert_eq!(el1 * &el2, expected_result);

        let el1 = field.init_element(vec![2, 2]);
        let el2 = field.init_element(vec![2, 1]);

        let expected_result = field.init_element(vec![0, 1]);
        assert_eq!(el1 * &el2, expected_result);

        let field = FiniteField::new(
            4231,
            10,
            Some(vec![
                1, 2049, 1194, 2386, 459, 1980, 2554, 3295, 1898, 2831, 311,
            ]),
        );

        let el1 = field.init_element(vec![
            4108, 1006, 1002, 973, 2776, 1231, 740, 4221, 1494, 1640,
        ]);
        let el2 = field.init_element(vec![3461, 3711, 3786, 3325, 284, 3477, 522, 1690, 539, 632]);

        let expected_result = field.init_element(vec![
            613, 1441, 2609, 3956, 4054, 922, 1799, 3469, 3759, 2220,
        ]);
        assert_eq!(el1 * &el2, expected_result);
    }

    #[test]
    fn test_mul_prime_field() {
        let field = FiniteField::new(7, 1, None);
        let el1 = field.init_element(vec![6]);
        let el2 = field.init_element(vec![3]);

        let expected_result = field.init_element(vec![4]);
        assert_eq!(el1 * &el2, expected_result);
    }

    #[test]
    fn test_pow() {
        let field = FiniteField::new(3, 2, Some(vec![1, 0, 1]));

        let el = field.init_element(vec![2, 2]);
        let expected_result = field.init_element(vec![0, 2]);
        assert_eq!(el.pow(4), expected_result);

        let el = field.init_element(vec![0, 2]);
        let expected_result = field.init_element(vec![0, 1]);
        assert_eq!(el.pow(2), expected_result);

        let field = FiniteField::new(
            4231,
            10,
            Some(vec![
                1, 2049, 1194, 2386, 459, 1980, 2554, 3295, 1898, 2831, 311,
            ]),
        );

        let el = field.init_element(vec![
            403, 2783, 3190, 2185, 2879, 318, 3562, 1792, 2847, 1826,
        ]);
        let expected_result = field.init_element(vec![
            4108, 1006, 1002, 973, 2776, 1231, 740, 4221, 1494, 1640,
        ]);
        assert_eq!(el.pow(23234234), expected_result);

        let el = field.init_element(vec![
            403, 2783, 3190, 2185, 2879, 318, 3562, 1792, 2847, 1826,
        ]);
        let expected_result = field.init_element(vec![
            2600, 1009, 2476, 1944, 1322, 3033, 910, 576, 240, 2985,
        ]);
        assert_eq!(el.pow(234234928348923984923982438), expected_result);
    }

    #[test]
    fn test_is_identity() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = FiniteField::new(3, 2, modulus);

        let el = field.init_element(vec![0, 1]);
        assert!(el.is_identity());

        let el = field.init_element(vec![0, 0]);
        assert!(!el.is_identity());
    }

    #[test]
    fn test_is_zero() {
        let modulus = Some(vec![1, 0, 1]); // x^2 + 1
        let field = FiniteField::new(3, 2, modulus);

        let el = field.init_element(vec![0, 0]);
        assert!(el.is_zero());

        let el = field.init_element(vec![0, 1]);
        assert!(!el.is_zero());
    }
}
