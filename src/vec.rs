//! An in-place growable vector.

use self::TryReserveErrorKind::{AllocError, CapacityOverflow};
use crate::{align_up, is_aligned, page_size, Allocation, Error, SizedTypeProperties};
use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp, fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, Index, IndexMut},
    panic::UnwindSafe,
    ptr,
    slice::{self, SliceIndex},
};

/// An in-place growable vector.
///
/// This type behaves identically to the standard library `Vec` except that it is guaranteed to
/// never reallocate.
pub struct Vec<T> {
    inner: VecInner,
    marker: PhantomData<T>,
}

struct VecInner {
    elements: *mut (),
    max_capacity: usize,
    capacity: usize,
    len: usize,
    allocation: Allocation,
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
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [committed]: crate#committing
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub fn new(max_capacity: usize) -> Self {
        Self::with_header(max_capacity, Layout::new::<()>())
    }

    /// Creates a new `Vec`.
    ///
    /// Like [`new`], except additionally allocating a header. `header_layout.size()` bytes will be
    /// allocated before the start of the vector's elements, with the start aligned to
    /// `header_layout.align()`. You can use [`as_ptr`] and offset backwards to access the header.
    ///
    /// # Panics
    ///
    /// - Panics if `header_layout` is not padded to its alignment.
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [`new`]: Self::try_new
    /// [`as_ptr`]: Self::as_ptr
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub fn with_header(max_capacity: usize, header_layout: Layout) -> Self {
        match Self::try_with_header(max_capacity, header_layout) {
            Ok(vec) => vec,
            Err(err) => handle_error(err),
        }
    }

    /// Creates a new `Vec`.
    ///
    /// Like [`new`], except returning an error when allocation fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`new`]: Self::new
    /// [reserving]: crate#reserving
    pub fn try_new(max_capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_header(max_capacity, Layout::new::<()>())
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`with_header`], except returning an error when allocation fails.
    ///
    /// # Panics
    ///
    /// - Panics if `header_layout` is not padded to its alignment.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`with_header`]: Self::with_header
    /// [`as_ptr`]: Self::as_ptr
    /// [reserving]: crate#reserving
    #[track_caller]
    pub fn try_with_header(
        max_capacity: usize,
        header_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        Ok(Vec {
            inner: unsafe {
                VecInner::try_with_header(
                    max_capacity,
                    header_layout,
                    size_of::<T>(),
                    align_of::<T>(),
                )
            }?,
            marker: PhantomData,
        })
    }

    /// Creates a dangling `Vec`.
    ///
    /// This is useful as a placeholder value to defer allocation until later or if no allocation
    /// is needed.
    #[inline]
    #[must_use]
    pub const fn dangling() -> Self {
        Vec {
            inner: VecInner::dangling(align_of::<T>()),
            marker: PhantomData,
        }
    }

    /// Returns a slice of the entire vector.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns a mutable slice of the entire vector.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a pointer to the vector's buffer.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.inner.elements.cast()
    }

    /// Returns a mutable pointer to the vector's buffer.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.elements.cast()
    }

    /// Returns the maximum capacity that was used when creating the `Vec`.
    #[inline]
    #[must_use]
    pub fn max_capacity(&self) -> usize {
        self.inner.max_capacity
    }

    /// Returns the total number of elements the vector can hold without [committing] more memory.
    ///
    /// [committing]: crate#committing
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        if T::IS_ZST {
            usize::MAX
        } else {
            self.inner.capacity
        }
    }

    /// Returns the number of elements in the vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an element to the end of the vector.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, value: T) {
        let len = self.len();

        if len == self.capacity() {
            self.grow_one();
        }

        // SAFETY: We made sure that the index is in bounds above.
        unsafe { self.as_mut_ptr().add(len).write(value) };

        // SAFETY: We have written the element above.
        unsafe { self.inner.len += 1 };
    }

    #[inline(never)]
    #[track_caller]
    fn grow_one(&mut self) {
        if let Err(err) = unsafe { self.inner.grow_amortized(1, size_of::<T>()) } {
            handle_error(err);
        }
    }
}

impl VecInner {
    #[track_caller]
    unsafe fn try_with_header(
        max_capacity: usize,
        header_layout: Layout,
        elem_size: usize,
        elem_align: usize,
    ) -> Result<Self, TryReserveError> {
        assert!(is_aligned(header_layout.size(), header_layout.align()));

        let size = max_capacity
            .checked_mul(elem_size)
            .ok_or(CapacityOverflow)?;

        #[allow(clippy::cast_possible_wrap)]
        if size > isize::MAX as usize {
            return Err(CapacityOverflow.into());
        }

        if size == 0 && header_layout.size() == 0 {
            return Ok(Self::dangling(elem_align));
        }

        // This can't overflow because `Layout`'s size can't exceed `isize::MAX`.
        let elements_offset = align_up(header_layout.size(), elem_align);

        // This can't overflow because `elements_offset` can be at most `1 << 63` and `size` can be
        // at most `isize::MAX`, totaling `usize::MAX`.
        let size = elements_offset + size;

        let page_size = page_size();
        let size = align_up(size, page_size);

        if size == 0 {
            return Err(CapacityOverflow.into());
        }

        let align = cmp::max(header_layout.align(), elem_align);

        // The minimum additional size required to fulfill the alignment requirement of the header
        // and element. The virtual memory allocation is only guaranteed to be aligned to the page
        // size, so if the alignment is greater than the page size, we must pad the allocation.
        let align_size = cmp::max(align, page_size) - page_size;

        let size = align_size.checked_add(size).ok_or(CapacityOverflow)?;

        let allocation = Allocation::new(size).map_err(AllocError)?;
        let aligned_ptr = allocation.ptr().map_addr(|addr| align_up(addr, align));
        let initial_size = align_up(elements_offset, page_size);

        if initial_size != 0 {
            allocation
                .commit(aligned_ptr, initial_size)
                .map_err(AllocError)?;
        }

        let elements = aligned_ptr.wrapping_add(elements_offset).cast();

        let capacity = if elem_size == 0 {
            0
        } else {
            (initial_size - elements_offset) / elem_size
        };

        Ok(VecInner {
            elements,
            max_capacity,
            capacity,
            len: 0,
            allocation,
        })
    }

    #[inline]
    const fn dangling(elem_align: usize) -> Self {
        let allocation = Allocation::dangling(elem_align);

        VecInner {
            elements: allocation.ptr().cast(),
            max_capacity: 0,
            capacity: 0,
            len: 0,
            allocation,
        }
    }

    // TODO: What's there to amortize over? It should be linear growth.
    unsafe fn grow_amortized(
        &mut self,
        additional: usize,
        elem_size: usize,
    ) -> Result<(), TryReserveError> {
        debug_assert!(additional > 0);

        if elem_size == 0 {
            return Err(CapacityOverflow.into());
        }

        let required_capacity = self.len.checked_add(additional).ok_or(CapacityOverflow)?;

        if required_capacity > self.max_capacity {
            return Err(CapacityOverflow.into());
        }

        let old_capacity = self.capacity;
        let page_size = page_size();

        let new_capacity = cmp::max(old_capacity * 2, required_capacity);
        let new_capacity = cmp::max(new_capacity, page_size / elem_size);
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

        let elements_offset = self.elements.addr() - self.allocation.ptr().addr();

        // These can't overflow because the size is already allocated.
        let old_size = align_up(elements_offset + old_capacity * elem_size, page_size);
        let new_size = align_up(elements_offset + new_capacity * elem_size, page_size);

        let ptr = self.allocation.ptr().wrapping_add(old_size);
        let size = new_size - old_size;

        self.allocation.commit(ptr, size).map_err(AllocError)?;

        self.capacity = new_capacity;

        Ok(())
    }
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

impl<T: Clone> Clone for Vec<T> {
    #[inline]
    fn clone(&self) -> Self {
        let mut vec = Vec::new(self.inner.max_capacity);

        for elem in self {
            vec.push(elem.clone());
        }

        vec
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
        let elements = ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len());

        // SAFETY: We own the collection, and it is being dropped, which ensures that the elements
        // can't be accessed again.
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

#[cfg(feature = "alloc")]
impl<T: PartialEq<U> + Clone, U> PartialEq<Vec<U>> for alloc::borrow::Cow<'_, [T]> {
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
            self.push(elem);
        }
    }
}

impl<'a, T: Copy> Extend<&'a T> for Vec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for &elem in iter {
            self.push(elem);
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

impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;

    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;

    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
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

/// An iterator that moves out of a vector.
///
/// This struct is created by the [`into_iter`] method on [`Vec`].
///
/// [`into_iter`]: Vec::into_iter
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

// We own the collection, so this should be no different than for `Vec`.
impl<T: UnwindSafe> UnwindSafe for IntoIter<T> {}

impl<T> IntoIter<T> {
    #[inline]
    fn new(vec: Vec<T>) -> Self {
        let mut vec = ManuallyDrop::new(vec);

        // SAFETY: `vec` is wrapped in a `ManuallyDrop` such that a double-free can't happen even
        // if a panic was possible below.
        let allocation = unsafe { ptr::read(&vec.inner.allocation) };

        let start = vec.as_mut_ptr();

        let end = if T::IS_ZST {
            start.cast::<u8>().wrapping_add(vec.len()).cast::<T>()
        } else {
            // SAFETY: The modifier of `vec.inner.len` ensures that it is only done after writing
            // the new elements.
            unsafe { start.add(vec.len()) }
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
            // We know that the return value is positive because by our invariant, `self.end` is
            // always greater or equal to `self.start`.
            #[allow(clippy::cast_sign_loss)]
            // SAFETY:
            // * `start` and `end` were both created from the same object in `IntoIter::new`.
            // * `Vec::new` ensures that the allocation size doesn't exceed `isize::MAX` bytes.
            // * We know that the allocation doesn't wrap around the address space.
            unsafe {
                self.end.offset_from(self.start) as usize
            }
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}

#[cold]
#[track_caller]
pub(crate) fn handle_error(err: TryReserveError) -> ! {
    match err.kind {
        CapacityOverflow => capacity_overflow(),
        AllocError(err) => handle_alloc_error(err),
    }
}

#[inline(never)]
#[track_caller]
pub(crate) fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

// Dear Clippy, `Error` is 4 bytes.
#[allow(clippy::needless_pass_by_value)]
#[cold]
#[inline(never)]
#[track_caller]
pub(crate) fn handle_alloc_error(err: Error) -> ! {
    panic!("allocation failed: {err}");
}

/// Error that can happen when trying to [reserve] or [commit] memory for a vector.
///
/// [reserve]: crate#reserving
/// [commit]: crate#committing
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
pub(crate) enum TryReserveErrorKind {
    CapacityOverflow,
    AllocError(Error),
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

impl core::error::Error for TryReserveError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match &self.kind {
            TryReserveErrorKind::CapacityOverflow => None,
            TryReserveErrorKind::AllocError(err) => Some(err),
        }
    }
}
