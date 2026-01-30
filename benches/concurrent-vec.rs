#![feature(test)]

extern crate test;

use std::{hint::black_box, sync::RwLock, thread};
use test::Bencher;

const ITERATIONS: usize = 100_000;
const THREADS: usize = 10;

#[bench]
fn push_contended_concurrent_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec = virtual_buffer::concurrent::vec::Vec::new(ITERATIONS);
        black_box(&vec);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        vec.push(black_box(0usize));
                        black_box(&vec);
                    }
                });
            }
        });
    });
}

#[bench]
fn push_contended_rwlock_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec = RwLock::new(std::vec::Vec::new());
        black_box(&vec);

        thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    for _ in black_box(0..ITERATIONS / THREADS) {
                        vec.write().unwrap().push(black_box(0usize));
                        black_box(&vec);
                    }
                });
            }
        });
    });
}

#[bench]
fn push_uncontended_concurrent_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec = virtual_buffer::concurrent::vec::Vec::new(ITERATIONS);
        black_box(&vec);

        for _ in black_box(0..ITERATIONS) {
            vec.push(black_box(0usize));
            black_box(&vec);
        }
    });
}

#[bench]
fn push_uncontended_rwlock_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec = RwLock::new(std::vec::Vec::new());
        black_box(&vec);

        for _ in black_box(0..ITERATIONS) {
            vec.write().unwrap().push(black_box(0usize));
            black_box(&vec);
        }
    });
}
