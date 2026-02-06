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

/// Used to create a new vector.
///
/// This struct is created by the [`builder`] method on [`Vec`].
///
/// [`builder`]: Vec::builder
#[derive(Clone, Copy, Debug)]
pub struct VecBuilder {
    max_capacity: usize,
    capacity: usize,
    growth_strategy: GrowthStrategy,
    header_layout: Layout,
}

impl VecBuilder {
    #[inline]
    const fn new(max_capacity: usize) -> Self {
        VecBuilder {
            max_capacity,
            capacity: 0,
            growth_strategy: GrowthStrategy::new(),
            header_layout: Layout::new::<()>(),
        }
    }

    /// The built `Vec` will have the minimum capacity required for `capacity` elements.
    ///
    /// The capacity can be greater due to the alignment to the [page size].
    ///
    /// [page size]: crate#pages
    #[inline]
    pub const fn capacity(&mut self, capacity: usize) -> &mut Self {
        self.capacity = capacity;

        self
    }

    /// The built `Vec` will have the given `growth_strategy`.
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

    /// The built `Vec` will have a header with the given `header_layout`.
    ///
    /// `header_layout.size()` bytes will be allocated before the start of the vector's elements,
    /// with the start aligned to `header_layout.align()`. You can use [`Vec::as_ptr`] and offset
    /// backwards to access the header.
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

    /// Builds the `Vec`.
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
    pub fn build<T>(&self) -> Vec<T> {
        match self.try_build() {
            Ok(vec) => vec,
            Err(err) => handle_error(err),
        }
    }

    /// Tries to build the `Vec`, returning an error when allocation fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if the `capacity` is greater than the `max_capacity`.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [reserving]: crate#reserving
    pub fn try_build<T>(&self) -> Result<Vec<T>, TryReserveError> {
        Ok(Vec {
            inner: unsafe {
                VecInner::new(
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

/// An in-place growable vector.
///
/// This type behaves similarly to the standard library `Vec` except that it is guaranteed to never
/// reallocate. The vector grows similarly to the standard library vector, but instead of
/// reallocating, it commits more memory.
///
/// If you don't specify a [growth strategy], exponential growth with a growth factor of 2 is used,
/// which is the same strategy that the standard library `Vec` uses.
///
/// [growth strategy]: GrowthStrategy
pub struct Vec<T> {
    inner: VecInner,
    marker: PhantomData<T>,
}

struct VecInner {
    elements: *mut (),
    capacity: usize,
    len: usize,
    max_capacity: usize,
    growth_strategy: GrowthStrategy,
    allocation: Allocation,
}

// SAFETY: `Vec` is an owned collection, which makes it safe to send to another thread as long as
// its element is safe to send to another a thread.
unsafe impl<T: Send> Send for Vec<T> {}

// SAFETY: `Vec` doesn't allow modifications through a shared reference, so a shared `Vec` can't be
// used to send elements to another thread. However, `Vec` allows getting a reference to any
// element from any thread. Therefore, it is safe to share `Vec` between threads as long as the
// element is shareable.
unsafe impl<T: Sync> Sync for Vec<T> {}

impl Vec<()> {
    /// Returns a builder for a new `Vec`.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity. The capacity can be excessively huge, as none of the memory
    /// is [committed] until you push elements into the vector or reserve capacity.
    ///
    /// This function is implemented on `Vec<()>` so that you don't have to import the `VecBuilder`
    /// type too. The type parameter has no relation to the built `Vec`.
    ///
    /// [committed]: crate#committing
    #[inline]
    #[must_use]
    pub const fn builder(max_capacity: usize) -> VecBuilder {
        VecBuilder::new(max_capacity)
    }
}

impl<T> Vec<T> {
    /// Creates a new `Vec`.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity. The capacity can be excessively huge, as none of the memory
    /// is [committed] until you push elements into the vector or reserve capacity.
    ///
    /// See the [`builder`] function for more creation options.
    ///
    /// # Panics
    ///
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [committed]: crate#committing
    /// [`builder`]: Self::builder
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub fn new(max_capacity: usize) -> Self {
        VecBuilder::new(max_capacity).build()
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
    ///
    /// This method is guaranteed to never materialize a reference to the underlying data. This,
    /// coupled with the fact that the vector never reallocates, means that the returned pointer
    /// stays valid until the vector is dropped.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.inner.elements.cast()
    }

    /// Returns a mutable pointer to the vector's buffer.
    ///
    /// This method is guaranteed to never materialize a reference to the underlying data. This,
    /// coupled with the fact that the vector never reallocates, means that the returned pointer
    /// stays valid until the vector is dropped.
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
        self.inner.capacity
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
        unsafe { self.inner.len = len + 1 };
    }

    #[cold]
    #[inline(never)]
    #[track_caller]
    fn grow_one(&mut self) {
        unsafe { self.inner.grow_one(size_of::<T>()) };
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// Not to be confused with [reserving virtual memory]; this method is named so as to match the
    /// standard library vector.
    ///
    /// The new capacity is at least `self.len() + additional`. If this capacity is below the one
    /// calculated using the [growth strategy], the latter is used. Does nothing if the capacity is
    /// already sufficient.
    ///
    /// # Panics
    ///
    /// - Panics if `self.len() + additional` would exceed `self.max_capacity()`.
    /// - Panics if [committing] the new capacity fails.
    ///
    /// [reserving virtual memory]: crate#reserving
    /// [growth strategy]: GrowthStrategy
    /// [committing]: crate#committing
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
        unsafe { self.inner.reserve(additional, size_of::<T>()) };
    }

    /// Reserves the minimum capacity required for `additional` more elements.
    ///
    /// Not to be confused with [reserving virtual memory]; this method is named so as to match the
    /// standard library vector.
    ///
    /// The new capacity is at least `self.len() + additional`. The [growth strategy] is ignored,
    /// but the new capacity can still be greater due to the alignment to the [page size]. Does
    /// nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// - Panics if `self.len() + additional` would exceed `self.max_capacity()`.
    /// - Panics if [committing] the new capacity fails.
    ///
    /// [reserving virtual memory]: crate#reserving
    /// [growth strategy]: GrowthStrategy
    /// [page size]: crate#pages
    /// [committing]: crate#committing
    #[track_caller]
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe { self.inner.reserve_exact(additional, size_of::<T>()) };
    }

    /// Tries to reserve capacity for at least `additional` more elements, returning an error on
    /// failure.
    ///
    /// Not to be confused with [reserving virtual memory]; this method is named so as to match the
    /// standard library vector.
    ///
    /// The new capacity is at least `self.len() + additional`. If this capacity is below the one
    /// calculated using the [growth strategy], the latter is used. Does nothing if the capacity is
    /// already sufficient.
    ///
    /// # Errors
    ///
    /// - Returns an error if `self.len() + additional` would exceed `self.max_capacity()`.
    /// - Returns an error if [committing] the new capacity fails.
    ///
    /// [reserving virtual memory]: crate#reserving
    /// [growth strategy]: GrowthStrategy
    /// [committing]: crate#committing
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.inner.try_reserve(additional, size_of::<T>()) }
    }

    /// Tries to reserve the minimum capacity required for `additional` more elements, returning an
    /// error on failure.
    ///
    /// Not to be confused with [reserving virtual memory]; this method is named so as to match the
    /// standard library vector.
    ///
    /// The new capacity is at least `self.len() + additional`. The [growth strategy] is ignored,
    /// but the new capacity can still be greater due to the alignment to the [page size]. Does
    /// nothing if the capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// - Returns an error if `self.len() + additional` would exceed `self.max_capacity()`.
    /// - Returns an error if [committing] the new capacity fails.
    ///
    /// [reserving virtual memory]: crate#reserving
    /// [growth strategy]: GrowthStrategy
    /// [page size]: crate#pages
    /// [committing]: crate#committing
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.inner.try_reserve_exact(additional, size_of::<T>()) }
    }
}

impl VecInner {
    unsafe fn new(
        max_capacity: usize,
        capacity: usize,
        growth_strategy: GrowthStrategy,
        header_layout: Layout,
        elem_size: usize,
        elem_align: usize,
    ) -> Result<Self, TryReserveError> {
        let size = max_capacity
            .checked_mul(elem_size)
            .ok_or(CapacityOverflow)?;

        #[allow(clippy::cast_possible_wrap)]
        if size > isize::MAX as usize || capacity > max_capacity {
            return Err(CapacityOverflow.into());
        }

        if size == 0 && header_layout.size() == 0 {
            let allocation = Allocation::dangling(elem_align);

            return Ok(VecInner {
                elements: allocation.ptr().cast(),
                capacity: max_capacity,
                len: 0,
                max_capacity,
                growth_strategy,
                allocation,
            });
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

        // This can't overflow because the size is already allocated.
        let initial_size = align_up(elements_offset + capacity * elem_size, page_size);

        if initial_size != 0 {
            allocation
                .commit(aligned_ptr, initial_size)
                .map_err(AllocError)?;
        }

        let elements = aligned_ptr.wrapping_add(elements_offset).cast();

        let capacity = if elem_size == 0 {
            max_capacity
        } else {
            (initial_size - elements_offset) / elem_size
        };

        Ok(VecInner {
            elements,
            capacity,
            len: 0,
            max_capacity,
            growth_strategy,
            allocation,
        })
    }

    #[inline]
    const fn dangling(elem_align: usize) -> Self {
        let allocation = Allocation::dangling(elem_align);

        VecInner {
            elements: allocation.ptr().cast(),
            capacity: 0,
            len: 0,
            max_capacity: 0,
            growth_strategy: GrowthStrategy::new(),
            allocation,
        }
    }

    #[inline]
    #[track_caller]
    unsafe fn reserve(&mut self, additional: usize, elem_size: usize) {
        if self.needs_to_grow(additional) {
            unsafe { self.reserve_slow(additional, elem_size) };
        }
    }

    #[cold]
    #[track_caller]
    unsafe fn reserve_slow(&mut self, additional: usize, elem_size: usize) {
        if let Err(err) = unsafe { self.grow(additional, elem_size) } {
            handle_error(err);
        }
    }

    #[track_caller]
    unsafe fn reserve_exact(&mut self, additional: usize, elem_size: usize) {
        if let Err(err) = unsafe { self.try_reserve_exact(additional, elem_size) } {
            handle_error(err);
        }
    }

    unsafe fn try_reserve(
        &mut self,
        additional: usize,
        elem_size: usize,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(additional) {
            unsafe { self.grow(additional, elem_size) }
        } else {
            Ok(())
        }
    }

    unsafe fn try_reserve_exact(
        &mut self,
        additional: usize,
        elem_size: usize,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(additional) {
            unsafe { self.grow_exact(additional, elem_size) }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn needs_to_grow(&self, additional: usize) -> bool {
        additional > self.capacity - self.len
    }

    #[track_caller]
    unsafe fn grow_one(&mut self, elem_size: usize) {
        if let Err(err) = unsafe { self.grow(1, elem_size) } {
            handle_error(err);
        }
    }

    unsafe fn grow(&mut self, additional: usize, elem_size: usize) -> Result<(), TryReserveError> {
        unsafe { self.grow_inner(additional, false, elem_size) }
    }

    unsafe fn grow_exact(
        &mut self,
        additional: usize,
        elem_size: usize,
    ) -> Result<(), TryReserveError> {
        unsafe { self.grow_inner(additional, true, elem_size) }
    }

    unsafe fn grow_inner(
        &mut self,
        additional: usize,
        exact: bool,
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

        let mut new_capacity = required_capacity;

        if !exact {
            new_capacity = cmp::max(new_capacity, self.growth_strategy.grow(old_capacity));
            new_capacity = cmp::min(new_capacity, self.max_capacity);
        }

        let elements_offset = self.elements.addr() - self.allocation.ptr().addr();

        let page_size = page_size();

        // These can't overflow because the size is already allocated.
        let old_size = align_up(elements_offset + old_capacity * elem_size, page_size);
        let new_size = align_up(elements_offset + new_capacity * elem_size, page_size);

        let ptr = self.allocation.ptr().wrapping_add(old_size);
        let size = new_size - old_size;

        self.allocation.commit(ptr, size).map_err(AllocError)?;

        let new_capacity = (new_size - elements_offset) / elem_size;
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

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
            // SAFETY:
            // - By our invariant, `self.end` is always greater than or equal to `self.start`.
            // - `start` and `end` were both created from the same object in `IntoIter::new`.
            // - `Vec::new` ensures that the allocation size doesn't exceed `isize::MAX` bytes.
            // - We know that the allocation doesn't wrap around the address space.
            unsafe { self.end.offset_from_unsigned(self.start) }
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}

/// The strategy to employ when growing a vector.
///
/// Note that arithmetic overflow is not a concern for any of the strategies: if the final result
/// of the calculation (in infinite precision) exceeds `max_capacity`, `max_capacity` is used as
/// the new capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GrowthStrategy {
    /// The current capacity is multiplied by `numerator`, then divided by `denominator`.
    ///
    /// If the new capacity results in the last committed [page] having spare room for more
    /// elements, those elements are added to the new capacity.
    ///
    /// [page]: crate#pages
    Exponential {
        /// The number to multiply the current capacity by. Must be greater than `denominator`.
        numerator: usize,

        /// The number to divide the multiplied number by. Must be nonzero.
        denominator: usize,
    },

    /// `elements` is added to the current capacity.
    ///
    /// If the new capacity results in the last committed [page] having spare room for more
    /// elements, those elements are added to the new capacity.
    ///
    /// [page]: crate#pages
    Linear {
        /// The number of elements to add to the current capacity. Must be nonzero.
        elements: usize,
    },
}

impl Default for GrowthStrategy {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl GrowthStrategy {
    /// Returns the default growth strategy: exponential growth with a growth factor of 2.
    #[inline]
    pub const fn new() -> Self {
        Self::Exponential {
            numerator: 2,
            denominator: 1,
        }
    }

    #[track_caller]
    pub(crate) const fn validate(&self) {
        match *self {
            Self::Exponential {
                numerator,
                denominator,
            } => {
                assert!(numerator > denominator);
                assert!(denominator != 0);
            }
            Self::Linear { elements } => {
                assert!(elements != 0);
            }
        }
    }

    pub(crate) fn grow(&self, old_capacity: usize) -> usize {
        match *self {
            GrowthStrategy::Exponential {
                numerator,
                denominator,
            } => saturating_mul_div(old_capacity, numerator, denominator),
            GrowthStrategy::Linear { elements } => old_capacity.saturating_add(elements),
        }
    }
}

#[cfg(target_pointer_width = "64")]
type DoubleUsize = u128;
#[cfg(target_pointer_width = "32")]
type DoubleUsize = u64;
#[cfg(target_pointer_width = "16")]
type DoubleUsize = u32;

fn saturating_mul_div(val: usize, numerator: usize, denominator: usize) -> usize {
    (val as DoubleUsize * numerator as DoubleUsize / denominator as DoubleUsize)
        .try_into()
        .unwrap_or(usize::MAX)
}

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
    pub(crate) kind: TryReserveErrorKind,
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
