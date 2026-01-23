//! A low-level, concurrent, in-place growable vector.

use super::RawVec;
use crate::{Allocation, SizedTypeProperties};
use core::{cmp, fmt, iter::FusedIterator, marker::PhantomData, mem::ManuallyDrop, ptr, slice};

/// An iterator that moves out of a vector.
///
/// This struct is created by the [`into_iter`] method on [`RawVec`].
///
/// [`into_iter`]: RawVec::into_iter
pub struct IntoIter<T> {
    _allocation: Allocation,
    start: *const T,
    end: *const T,
    marker: PhantomData<T>,
}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Send> Send for IntoIter<T> {}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Sync> Sync for IntoIter<T> {}

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
            _allocation: allocation,
            start,
            end,
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
            // We know that the return value is positive because by our invariant, `self.end` is
            // always greater or equal to `self.start`.
            #[allow(clippy::cast_sign_loss)]
            // SAFETY:
            // * `start` and `end` were both created from the same object in `IntoIter::new`.
            // * `RawVec::new` ensures that the allocation size doesn't exceed `isize::MAX` bytes.
            // * We know that the allocation doesn't wrap around the address space.
            unsafe {
                self.end.offset_from(self.start) as usize
            }
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}
