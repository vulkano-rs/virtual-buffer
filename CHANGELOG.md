# Version 1.0.3 (2025-02-22)

- Fixed `Vec`'s implicit `Sync` implementation which was unsound, allowing you to send a `!Send`
  type to another thread.
- Fixed `Vec::push_mut` mixed with `Vec::push` being unsound.

# Version 1.0.2 (2025-02-15)

- Fixed `Vec::push` unsoundly setting the length.

# Version 1.0.1 (2024-06-29)

- Fixed `Vec` being covariant rather than invariant.
- Fixed a compilation error on vxworks.
