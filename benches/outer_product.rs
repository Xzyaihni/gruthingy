#![feature(test)]

use std::iter;

use test::bench::Bencher;

use gruthingy::neural_network::containers::*;

extern crate test;


const ROWS: usize = 11000;
const COLUMNS: usize = 128;

fn random_value(_x: usize, _y: usize) -> f32
{
    fastrand::f32()
}

fn random_values_vec(len: usize) -> Vec<f32>
{
    iter::repeat_with(|| random_value(0, 0)).take(len).collect()
}

#[bench]
fn outer_product_owned(bencher: &mut Bencher)
{
    let mut out = nalgebra::DMatrix::from_fn(ROWS, COLUMNS, random_value);
    let lhs = nalgebra::DVector::from_fn(ROWS, random_value);
    let rhs = nalgebra::DVector::from_fn(COLUMNS, random_value);

    bencher.iter(move ||
    {
        test::black_box(&mut out);

        out.ger(1.0, &lhs, &rhs, 0.0);

        out.clone()
    });
}

#[bench]
fn outer_product_ywrapper(bencher: &mut Bencher)
{
    let mut out_values = random_values_vec(ROWS * COLUMNS);
    let lhs_values = random_values_vec(ROWS);
    let rhs_values = random_values_vec(COLUMNS);

    bencher.iter(move ||
    {
        test::black_box(&mut out_values);

        let out = LayerTypeMut::from_data(&mut out_values, ROWS, COLUMNS);
        let lhs = LayerTypeVectorRef::from_data(&lhs_values, ROWS, 1);
        let rhs = LayerTypeVectorRef::from_data(&rhs_values, COLUMNS, 1);

        out.outer_product_into(lhs, rhs);

        out_values.clone()
    });
}
