# Version 2.0.0 (2026-02-06)

### Breaking changes

- The MSRV is now 1.87.0.
- `virtual_buffer::vec::Vec` is bye bye. This is because it tried to be 3 things, failing at all 3,
  proving to be worthless. In its stead, there are 3 new vector types:
  - `virtual_buffer::vec::Vec` is a vector much like the standard library vector, that is, not
    concurrent.
  - `virtual_buffer::concurrent::vec::Vec` is a concurrent vector, but unlike in the previous
    version, lock-free and with some memory overhead.
  - `virtual_buffer::concurrent::vec::RawVec` is also a lock-free concurrent vector, but without
    memory overhead and with some unsafety instead.
- A vector with a ZST element can no longer have a capacity greater than the `max_capacity`.
- The vectors now use the builder pattern for construction in order to avoid a combinatorial
  explosion of constructor functions.

### Other changes

- The vectors now support an element alignment greater than the page size.
- Added various `reserve` methods to the vectors.
- Added the option to create a vector with a given capacity.
- Added the option to configure the vectors' growth strategy.
- Added the option to allocate a header before the vectors' elements. This can also be used for
  overaligning the elements by setting the header's size to zero, only using its alignment.
- `IntoIter`s' `UnwindSafe` implementations are correct now.
- `Allocation::new` accepts a size of zero now.
- The `std` feature is no longer needed to get `Error` implementations for the error types.
- The new `alloc` feature, also enabled by the `std` feature, is now sufficient to get
  implementations making use of `alloc::borrow`.

# Version 1.0.3 (2025-02-22)

- Fixed `Vec`'s implicit `Sync` implementation which was unsound, allowing you to send a `!Send`
  type to another thread.
- Fixed `Vec::push_mut` mixed with `Vec::push` being unsound.

# Version 1.0.2 (2025-02-15)

- Fixed `Vec::push` unsoundly setting the length.

# Version 1.0.1 (2024-06-29)

- Fixed `Vec` being covariant rather than invariant.
- Fixed a compilation error on vxworks.
