use std::thread;
use virtual_buffer::vec::Vec;

#[cfg(not(miri))]
const ITERATIONS: u64 = 1024 * 1024;
#[cfg(miri)]
const ITERATIONS: u64 = 1024;

#[test]
fn push_1_thread() {
    const THREADS: u64 = 1;

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
fn push_2_threads() {
    const THREADS: u64 = 2;

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
fn push_4_threads() {
    const THREADS: u64 = 4;

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
fn push_8_threads() {
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
