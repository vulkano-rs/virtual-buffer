//! A low-level, concurrent, in-place growable vector.

use super::{RawVec, RawVecInner};
use crate::{
    Allocation, SizedTypeProperties, is_aligned,
    vec::{GrowthStrategy, TryReserveError, handle_error},
};
use core::{
    alloc::Layout, cmp, fmt, iter::FusedIterator, marker::PhantomData, mem::ManuallyDrop,
    panic::UnwindSafe, ptr, slice,
};

/// Used to build a new vector.
///
/// This struct is created by the [`builder`] method on [`RawVec`].
///
/// [`builder`]: RawVec::builder
#[derive(Clone, Copy, Debug)]
pub struct VecBuilder {
    max_capacity: usize,
    capacity: usize,
    growth_strategy: GrowthStrategy,
    header_layout: Layout,
}

impl VecBuilder {
    #[inline]
    pub(super) const fn new(max_capacity: usize) -> Self {
        VecBuilder {
            max_capacity,
            capacity: 0,
            growth_strategy: GrowthStrategy::new(),
            header_layout: Layout::new::<()>(),
        }
    }

    /// The built `RawVec` will have the minimum capacity required for `capacity` elements.
    ///
    /// The capacity can be greater due to the alignment to the [page size].
    ///
    /// [page size]: crate#pages
    #[inline]
    pub const fn capacity(&mut self, capacity: usize) -> &mut Self {
        self.capacity = capacity;

        self
    }

    /// The built `RawVec` will have the given `growth_strategy`.
    ///
    /// # Panics
    ///
    /// Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    #[inline]
    #[track_caller]
    pub const fn growth_strategy(&mut self, growth_strategy: GrowthStrategy) -> &mut Self {
        growth_strategy.validate();

        self.growth_strategy = growth_strategy;

        self
    }

    /// The built `RawVec` will have a header with the given `header_layout`.
    ///
    /// `header_layout.size()` bytes will be allocated before the start of the vector's elements,
    /// with the start aligned to `header_layout.align()`. You can use [`RawVec::as_ptr`] and
    /// offset backwards to access the header.
    ///
    /// # Panics
    ///
    /// Panics if `header_layout` is not padded to its alignment.
    #[inline]
    #[track_caller]
    pub const fn header(&mut self, header_layout: Layout) -> &mut Self {
        assert!(is_aligned(header_layout.size(), header_layout.align()));

        self.header_layout = header_layout;

        self
    }

    /// Builds the `RawVec`.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Panics
    ///
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if the `capacity` is greater than the `max_capacity`.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub unsafe fn build<T>(&self) -> RawVec<T> {
        match unsafe { self.try_build() } {
            Ok(vec) => vec,
            Err(err) => handle_error(err),
        }
    }

    /// Tries to build the `RawVec`, returning an error when allocation fails.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if the `capacity` is greater than the `max_capacity`.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [reserving]: crate#reserving
    pub unsafe fn try_build<T>(&self) -> Result<RawVec<T>, TryReserveError> {
        Ok(RawVec {
            inner: unsafe {
                RawVecInner::new(
                    self.max_capacity,
                    self.capacity,
                    self.growth_strategy,
                    self.header_layout,
                    size_of::<T>(),
                    align_of::<T>(),
                )
            }?,
            marker: PhantomData,
        })
    }
}

/// An iterator that moves out of a vector.
///
/// This struct is created by the [`into_iter`] method on [`RawVec`].
///
/// [`into_iter`]: RawVec::into_iter
pub struct IntoIter<T> {
    start: *const T,
    end: *const T,
    #[allow(dead_code)]
    allocation: Allocation,
    marker: PhantomData<T>,
}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Send> Send for IntoIter<T> {}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Sync> Sync for IntoIter<T> {}

// We own the collection, so this should be no different than for `RawVec`.
impl<T: UnwindSafe> UnwindSafe for IntoIter<T> {}

impl<T> IntoIter<T> {
    #[inline]
    pub(super) fn new(vec: RawVec<T>) -> Self {
        let mut vec = ManuallyDrop::new(vec);

        // SAFETY: `vec` is wrapped in a `ManuallyDrop` such that a double-free can't happen even
        // if a panic was possible below.
        let allocation = unsafe { ptr::read(&vec.inner.allocation) };

        let start = vec.as_mut_ptr();

        let len = cmp::min(vec.len_mut(), vec.capacity_mut());

        let end = if T::IS_ZST {
            start.cast::<u8>().wrapping_add(len).cast::<T>()
        } else {
            // SAFETY: The ownership synchronizes with setting the capacity, making sure that the
            // newly committed memory is visible here. The constructor of `RawVec` must ensure that
            // `T` is zeroable.
            unsafe { start.add(len) }
        };

        IntoIter {
            start,
            end,
            allocation,
            marker: PhantomData,
        }
    }

    /// Returns the remaining items of this iterator as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.start, self.len()) }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.start.cast_mut(), self.len()) }
    }
}

impl<T> AsRef<[T]> for IntoIter<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for IntoIter<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        let elements = ptr::slice_from_raw_parts_mut(self.start.cast_mut(), self.len());

        // SAFETY: We own the collection, and it is being dropped, which ensures that the elements
        // can't be accessed again.
        unsafe { elements.drop_in_place() };
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        let ptr = if T::IS_ZST {
            self.end = self.end.cast::<u8>().wrapping_sub(1).cast::<T>();

            self.start
        } else {
            let old = self.start;

            // SAFETY: We checked that there are still elements remaining above.
            self.start = unsafe { old.add(1) };

            old
        };

        // SAFETY: We own the collection, and have just incremented the `start` pointer such that
        // this element can't be accessed again.
        Some(unsafe { ptr.read() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        let ptr = if T::IS_ZST {
            self.end = self.end.cast::<u8>().wrapping_sub(1).cast::<T>();

            self.start
        } else {
            // SAFETY: We checked that there are still elements remaining above.
            self.end = unsafe { self.end.sub(1) };

            self.end
        };

        // SAFETY: We own the collection, and have just decremented the `end` pointer such that
        // this element can't be accessed again.
        Some(unsafe { ptr.read() })
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    #[inline]
    fn len(&self) -> usize {
        if T::IS_ZST {
            self.end.addr().wrapping_sub(self.start.addr())
        } else {
            // SAFETY:
            // - By our invariant, `self.end` is always greater than or equal to `self.start`.
            // - `start` and `end` were both created from the same object in `IntoIter::new`.
            // - `RawVec::new` ensures that the allocation size doesn't exceed `isize::MAX` bytes.
            // - We know that the allocation doesn't wrap around the address space.
            unsafe { self.end.offset_from_unsigned(self.start) }
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}
