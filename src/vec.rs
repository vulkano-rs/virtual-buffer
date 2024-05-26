//! A concurrent, in-place growable vector.

use self::TryReserveErrorKind::{AllocError, CapacityOverflow};
use super::{align_up, page_size, Allocation};
use alloc::borrow::Cow;
use core::{
    borrow::{Borrow, BorrowMut},
    cmp, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice::{self, SliceIndex},
    sync::atomic::{
        AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};
use std::{error::Error, io, sync::Mutex};

/// A concurrent, in-place growable vector.
///
/// This type behaves identically to the standard library `Vec` except that it is guaranteed to
/// never reallocate, and as such can support concurrent reads while also supporting growth.
/// However, growth is not concurrent. There can be at most one writer.
pub struct Vec<T> {
    allocation: Allocation,
    max_capacity: usize,
    capacity: AtomicUsize,
    len: AtomicUsize,
    len_lock: Mutex<()>,
    marker: PhantomData<T>,
}

impl<T> Vec<T> {
    /// Creates a new `Vec`.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity. The capacity can be excessively huge, as none of the memory
    /// is [committed] until you push elements into the vector. The vector grows similarly to the
    /// standard library vector, but instead of reallocating, it commits more memory.
    ///
    /// # Panics
    ///
    /// Panics if the alignment of `T` is greater than the [page size].
    ///
    /// [committed]: super#committing
    /// [page size]: super#pages
    #[must_use]
    pub fn new(max_capacity: usize) -> Self {
        handle_reserve(Self::try_new(max_capacity))
    }

    /// Creates a new `Vec`.
    ///
    /// This function behaves the same as [`new`] except that it doesn't panic when allocation
    /// fails.
    ///
    /// # Errors
    ///
    /// Returns an error when the operating system returns an error.
    ///
    /// # Panics
    ///
    /// Panics if the alignment of `T` is greater than the [page size].
    ///
    /// [`new`]: Self::new
    /// [page size]: super#pages
    pub fn try_new(max_capacity: usize) -> Result<Self, TryReserveError> {
        assert!(mem::align_of::<T>() <= page_size());

        let size = align_up(
            max_capacity
                .checked_mul(mem::size_of::<T>())
                .ok_or(CapacityOverflow)?,
            page_size(),
        );

        #[allow(clippy::cast_possible_wrap)]
        if size > isize::MAX as usize {
            return Err(CapacityOverflow.into());
        }

        if size == 0 {
            return Ok(Self::dangling(max_capacity));
        }

        let allocation = Allocation::new(size).map_err(AllocError)?;

        Ok(Vec {
            allocation,
            max_capacity,
            capacity: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            len_lock: Mutex::new(()),
            marker: PhantomData,
        })
    }

    /// Creates a dangling `Vec`.
    ///
    /// This is useful as a placeholder value to defer allocation until later or if no allocation
    /// is needed.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity.
    #[inline]
    #[must_use]
    pub const fn dangling(max_capacity: usize) -> Self {
        let allocation = Allocation::dangling(mem::align_of::<T>());

        Vec {
            allocation,
            max_capacity,
            capacity: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            len_lock: Mutex::new(()),
            marker: PhantomData,
        }
    }

    /// Returns a slice of the entire vector.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let len = self.len.load(Acquire);

        // SAFETY: The modifier of `self.len` ensures that it is only done after writing the new
        // elements and that said writes have been synchronized. The `Acquire` ordering above
        // synchronizes with the `Release` ordering when setting the len, making sure that the
        // write is visible here.
        unsafe { slice::from_raw_parts(self.as_ptr(), len) }
    }

    /// Returns a mutable slice of the entire vector.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len_mut();

        // SAFETY: The modifier of `self.len` ensures that it is only done after writing the new
        // elements and that said writes have been synchronized. The mutable reference ensures
        // synchronization in this case.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }

    /// Returns a pointer to the vector's buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.allocation.ptr().cast()
    }

    /// Returns a mutable pointer to the vector's buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.ptr().cast()
    }

    /// Returns the total number of elements the vector can hold without [committing] more memory.
    ///
    /// [committing]: super#committing
    #[inline(always)]
    #[must_use]
    pub fn capacity(&self) -> usize {
        if T::IS_ZST {
            usize::MAX
        } else {
            self.capacity.load(Relaxed)
        }
    }

    #[inline(always)]
    fn capacity_mut(&mut self) -> usize {
        if T::IS_ZST {
            usize::MAX
        } else {
            *self.capacity.get_mut()
        }
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Relaxed)
    }

    #[inline(always)]
    fn len_mut(&mut self) -> usize {
        *self.len.get_mut()
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an element to the end of the vector. Returns the index of the inserted element.
    #[inline]
    pub fn push(&self, value: T) -> usize {
        let _len_guard = match self.len_lock.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        };

        let len = self.len();

        if len >= self.capacity() {
            self.reserve_for_push();
        }

        // SAFETY: We made sure the index is in bounds above. We hold a lock on the length, which
        // means that no other threads can be attempting to write to this same index.
        unsafe { self.as_ptr().cast_mut().add(len).write(value) };

        // SAFETY: We have written the element, and synchronize said write with any future readers
        // by using the `Release` ordering.
        unsafe { self.len.store(len + 1, Release) };

        len
    }

    /// Appends an element to the end of the vector. Returns the index of the inserted element.
    #[inline]
    pub fn push_mut(&mut self, value: T) -> usize {
        let len = self.len_mut();

        if len >= self.capacity_mut() {
            self.reserve_for_push();
        }

        // SAFETY: We made sure the index is in bounds above.
        unsafe { self.as_mut_ptr().add(len).write(value) };

        // SAFETY: We have written the element.
        unsafe { *self.len.get_mut() += 1 };

        len
    }

    #[inline(never)]
    fn reserve_for_push(&self) {
        handle_reserve(self.grow_amortized(1));
    }

    // TODO: What's there to amortize over? It should be linear growth.
    fn grow_amortized(&self, additional: usize) -> Result<(), TryReserveError> {
        debug_assert!(additional > 0);

        if T::IS_ZST {
            return Err(CapacityOverflow.into());
        }

        let required_capacity = self.len().checked_add(additional).ok_or(CapacityOverflow)?;

        if required_capacity > self.max_capacity {
            return Err(CapacityOverflow.into());
        }

        let new_capacity = cmp::max(self.capacity() * 2, required_capacity);
        let new_capacity = cmp::max(new_capacity, page_size() / mem::size_of::<T>());
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

        grow(
            &self.allocation,
            &self.capacity,
            new_capacity,
            mem::size_of::<T>(),
        )
    }
}

#[inline(never)]
fn grow(
    allocation: &Allocation,
    capacity: &AtomicUsize,
    new_capacity: usize,
    element_size: usize,
) -> Result<(), TryReserveError> {
    let old_capacity = capacity.load(Relaxed);

    let page_size = page_size();

    let old_size = old_capacity * element_size;
    let new_size = new_capacity
        .checked_mul(element_size)
        .ok_or(CapacityOverflow)?;

    if new_size > allocation.size() {
        return Err(CapacityOverflow.into());
    }

    let old_size = align_up(old_size, page_size);
    let new_size = align_up(new_size, page_size);
    let ptr = allocation.ptr().wrapping_add(old_size);
    let size = new_size - old_size;

    allocation.commit(ptr, size).map_err(AllocError)?;

    let _ = allocation.prefault(ptr, size);

    capacity.store(new_capacity, Relaxed);

    Ok(())
}

#[inline]
fn handle_reserve<T>(res: Result<T, TryReserveError>) -> T {
    match res.map_err(|e| e.kind) {
        Ok(x) => x,
        Err(CapacityOverflow) => capacity_overflow(),
        Err(AllocError(err)) => handle_alloc_error(err),
    }
}

#[inline(never)]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

// Dear Clippy, `io::Error` is 8 bytes.
#[allow(clippy::needless_pass_by_value)]
#[cold]
fn handle_alloc_error(err: io::Error) -> ! {
    panic!("allocation failed: {err}");
}

impl<T> AsRef<Vec<T>> for Vec<T> {
    #[inline]
    fn as_ref(&self) -> &Vec<T> {
        self
    }
}

impl<T> AsMut<Vec<T>> for Vec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<T> {
        self
    }
}

impl<T> AsRef<[T]> for Vec<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> AsMut<[T]> for Vec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> Borrow<[T]> for Vec<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T> BorrowMut<[T]> for Vec<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> Deref for Vec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for Vec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        let elements = ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len_mut());

        // SAFETY: This is the drop implementation. This would be the place to drop the elements.
        unsafe { elements.drop_in_place() };
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<&[U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<&mut [U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&mut [U]) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for &[T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for &mut [T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<[U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for [T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        *self == **other
    }
}

impl<T: PartialEq<U> + Clone, U> PartialEq<Vec<U>> for Cow<'_, [T]> {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&[U; N]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&[U; N]) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Vec<T> {}

impl<T> Extend<T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for elem in iter {
            self.push_mut(elem);
        }
    }
}

impl<'a, T: Copy> Extend<&'a T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for &elem in iter {
            self.push_mut(elem);
        }
    }
}

impl<T: Hash> Hash for Vec<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state);
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for Vec<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<T: PartialOrd> PartialOrd for Vec<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

/// Error that can happen when trying to [reserve] or [commit] memory for a [`Vec`].
///
/// [reserve]: super#reserving
/// [commit]: super#committing
#[derive(Debug)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

impl From<TryReserveErrorKind> for TryReserveError {
    #[inline]
    fn from(kind: TryReserveErrorKind) -> Self {
        TryReserveError { kind }
    }
}

#[derive(Debug)]
enum TryReserveErrorKind {
    CapacityOverflow,
    AllocError(io::Error),
}

impl fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            CapacityOverflow => f.write_str(
                "memory allocation failed because the computed capacity exceeded the collection's \
                maximum",
            ),
            AllocError(_) => f.write_str(
                "memory allocation failed because the operating system returned an error",
            ),
        }
    }
}

impl Error for TryReserveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            TryReserveErrorKind::CapacityOverflow => None,
            TryReserveErrorKind::AllocError(err) => Some(err),
        }
    }
}

trait SizedTypeProperties: Sized {
    const IS_ZST: bool = mem::size_of::<Self>() == 0;
}

impl<T> SizedTypeProperties for T {}
