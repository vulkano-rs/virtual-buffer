#![feature(test)]

extern crate test;

use std::hint::black_box;
use test::Bencher;

const ITERATIONS: usize = 100_000;

#[bench]
fn push_vec(b: &mut Bencher) {
    b.iter(|| {
        let mut vec = virtual_buffer::vec::Vec::new(ITERATIONS);
        black_box(&vec);

        for _ in black_box(0..ITERATIONS) {
            vec.push(black_box(0usize));
            black_box(&vec);
        }
    });
}

#[bench]
fn push_std_vec(b: &mut Bencher) {
    b.iter(|| {
        let mut vec = std::vec::Vec::new();
        black_box(&vec);

        for _ in black_box(0..ITERATIONS) {
            vec.push(black_box(0usize));
            black_box(&vec);
        }
    });
}
