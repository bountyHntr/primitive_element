use std::ops::{Add, Sub, Mul};
use std::fmt::Display;

use rand::distributions::Uniform;
use rand::Rng;

#[derive(Debug, PartialEq, Eq)]
pub struct FiniteField {
    prime_number: u128,
    extension_degree: u16,
    modulus: Vec<u128>,
}

impl FiniteField {
    pub fn new(prime_number: u128, extension_degree: u16, modulus: Vec<u128>) -> Self {
        assert_eq!(modulus.len() - 1, extension_degree as usize);
        assert_eq!(modulus[0], 1); // must be monic polynomial

        FiniteField{
            prime_number,
            extension_degree,
            modulus,
        }
    }

    pub fn prime_number(&self) -> u128 {
        self.prime_number
    }

    pub fn extension_degree(&self) -> u16 {
        self.extension_degree
    }

    pub fn modulus(&self) -> &[u128] {
        &self.modulus
    }

    pub fn random_element(&self) -> FiniteFieldElement {
        let range = Uniform::new(0, self.prime_number);
        let mut rng = rand::thread_rng();

        let coefficients: Vec<u128> = (0..self.extension_degree)
            .map(|_| rng.sample(&range))
            .collect();


        FiniteFieldElement { field: self, coefficients }
    } 
}

impl Display for FiniteField {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       todo!()
   } 
}

#[derive(Debug, PartialEq, Eq)]
pub struct FiniteFieldElement<'a> {
    field: &'a FiniteField,
    coefficients: Vec<u128>,
}

impl<'a> FiniteFieldElement<'a> {
    pub fn new(field: &'a FiniteField) -> FiniteFieldElement<'a> {
        FiniteFieldElement { 
            field: field,
            coefficients: vec![0u128; field.extension_degree() as usize],
        }
    }

    pub fn build(field: &'a FiniteField, coefficients: Vec<u128>) -> FiniteFieldElement<'a> {
        assert_eq!(field.extension_degree() as usize, coefficients.len());

        let prime_number = field.prime_number();
        for i in 0..coefficients.len() {
            assert!(coefficients[i] < prime_number);
        }
    
        FiniteFieldElement { field, coefficients }
    }

    pub fn coefficients(&self) -> &[u128] {
        &self.coefficients
    }

    pub fn pow(self, degree: usize) -> Self {
        todo!()
    }
}

impl <'a> Add for FiniteFieldElement<'a> {
    type Output = FiniteFieldElement<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        assert_eq!(lhs.field, rhs.field);

        let prime_number = lhs.field.prime_number();

        for i in 0..lhs.coefficients.len() {
            lhs.coefficients[i] = (lhs.coefficients[i] + rhs.coefficients[i]) % prime_number;
        }

        lhs
    }
}

impl <'a> Sub for FiniteFieldElement<'a> {
    type Output = FiniteFieldElement<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        assert_eq!(lhs.field, rhs.field);

        let prime_number = lhs.field.prime_number();

        for i in 0..lhs.coefficients.len() {
            if lhs.coefficients[i] < rhs.coefficients[i] {
                lhs.coefficients[i] += prime_number
            }

            lhs.coefficients[i] = (lhs.coefficients[i] - rhs.coefficients[i]) % prime_number;
        }

        lhs
    }
}

impl <'a> Mul for FiniteFieldElement<'a> {
    type Output = FiniteFieldElement<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Display for FiniteFieldElement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_element() {
        let modulus = vec![1, 1, 0, 0, 2]; // x^4 + x^3 + 2

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
    fn test_ff_element_large_coefficient() {
        let modulus = vec![1, 1, 0, 0, 2]; // x^4 + x^3 + 2
        let field = FiniteField::new(3, 4, modulus);

        FiniteFieldElement::build(&field, vec![1, 1, 4, 0]);
    }

    #[test]
    #[should_panic]
    fn test_ff_element_invalid_num_of_coefficient() {
        let modulus = vec![1, 1, 0, 0, 2]; // x^4 + x^3 + 2
        let field = FiniteField::new(3, 4, modulus);

        FiniteFieldElement::build(&field, vec![1, 1, 0]);
    }

    #[test]
    fn test_add() {
        let modulus = vec![1, 0, 1]; // x^2 + 1 
        let field = FiniteField::new(3, 2, modulus);

        let el1 = FiniteFieldElement::build(&field, vec![2, 0]); 
        let el2 = FiniteFieldElement::build(&field, vec![2, 1]);

        let expected_result = FiniteFieldElement::build(&field, vec![1, 1]);

        assert_eq!(el1 + el2, expected_result);
    }

    #[test]
    fn test_sub() {
        let modulus = vec![1, 0, 1]; // x^2 + 1 
        let field = FiniteField::new(3, 2, modulus);

        let el1 = FiniteFieldElement::build(&field, vec![2, 0]); 
        let el2 = FiniteFieldElement::build(&field, vec![1, 1]);

        let expected_result = FiniteFieldElement::build(&field, vec![1, 2]);

        assert_eq!(el1 - el2, expected_result);
    }

    #[test]
    fn test_mul() {
        let modulus = vec![1, 0, 1]; // x^2 + 1 
        let field = FiniteField::new(3, 2, modulus);

        let el1 = FiniteFieldElement::build(&field, vec![2, 2]); 
        let el2 = FiniteFieldElement::build(&field, vec![2, 0]);

        let expected_result = FiniteFieldElement::build(&field, vec![1, 2]);
        assert_eq!(el1 + el2, expected_result);

        let el1 = FiniteFieldElement::build(&field, vec![2, 2]); 
        let el2 = FiniteFieldElement::build(&field, vec![2, 1]);

        let expected_result = FiniteFieldElement::build(&field, vec![0, 1]);
        assert_eq!(el1 + el2, expected_result);
    }
}