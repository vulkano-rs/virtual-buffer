use std::{sync::Barrier, thread};
use virtual_buffer::vec::Vec;

#[test]
fn push_stress() {
    const ITERATIONS: usize = if cfg!(miri) { 10 } else { 1_000 };
    const THREADS: usize = 8;

    let vec = &Vec::new(ITERATIONS);
    let barrier = &Barrier::new(THREADS);

    thread::scope(|s| {
        for i in 0..THREADS {
            s.spawn(move || {
                barrier.wait();

                for n in 1..=ITERATIONS / THREADS {
                    vec.push(i * (ITERATIONS / THREADS) + n);
                }
            });
        }
    });

    assert_eq!(
        vec.into_iter().sum::<usize>(),
        (ITERATIONS + 1) * ITERATIONS / 2,
    );
}

#[test]
fn push_mut_immut() {
    let mut vec = Vec::new(5);

    vec.push_mut(Box::new(1));

    for x in 2..5 {
        vec.push(Box::new(x));
    }

    vec.push_mut(Box::new(5));

    assert_eq!(vec, [1, 2, 3, 4, 5].map(Box::new));
}
