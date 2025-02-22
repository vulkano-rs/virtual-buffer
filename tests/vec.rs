use std::thread;
use virtual_buffer::vec::Vec;

#[cfg(not(miri))]
const ITERATIONS: u64 = 128 * 1024;
#[cfg(miri)]
const ITERATIONS: u64 = 1024;

#[test]
fn push_stress() {
    const THREADS: u64 = 8;

    let vec = Vec::new(ITERATIONS.try_into().unwrap());

    thread::scope(|s| {
        let vec = &vec;

        for i in 0..THREADS {
            s.spawn(move || {
                for n in 1..=ITERATIONS / THREADS {
                    vec.push(i * (ITERATIONS / THREADS) + n);
                }
            });
        }
    });

    assert_eq!(
        vec.into_iter().sum::<u64>(),
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
