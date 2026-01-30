//! A concurrent, in-place growable vector.

use crate::{
    align_up, is_aligned, page_size,
    vec::{
        capacity_overflow, handle_error, GrowthStrategy, TryReserveError,
        TryReserveErrorKind::{AllocError, CapacityOverflow},
    },
    Allocation, SizedTypeProperties,
};
use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cell::UnsafeCell,
    cmp, fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    panic::RefUnwindSafe,
    ptr,
    slice::{self, SliceIndex},
    sync::atomic::{
        AtomicBool, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
};

pub mod raw;

/// A concurrent, in-place growable vector.
///
/// This type behaves similarly to the standard library `Vec` except that it is guaranteed to never
/// reallocate, and as such can support concurrent reads while also supporting growth. The vector
/// grows similarly to the standard library vector, but instead of reallocating, it commits more
/// memory.
///
/// All operations are lock-free. In order to support this, each element consists of a `T` and an
/// `AtomicBool` used to denote whether the element has been initialized. That means that there is
/// a memory overhead equal to `align_of::<T>()`. You can avoid this overhead if you have a byte of
/// memory to spare in your `T` or by packing the bit into an existing atomic field in `T` by using
/// [`RawVec`].
pub struct Vec<T> {
    /// ```compile_fail,E0597
    /// let vec = virtual_buffer::concurrent::vec::Vec::<&'static str>::new(1);
    /// {
    ///     let s = "oh no".to_owned();
    ///     vec.push(&s);
    /// }
    /// dbg!(vec);
    /// ```
    inner: RawVec<Slot<T>>,
}

// SAFETY: `Vec` is an owned collection, which makes it safe to send to another thread as long as
// its element is safe to send to another a thread.
unsafe impl<T: Send> Send for Vec<T> {}

// SAFETY: `Vec` allows pushing through a shared reference, which allows a shared `Vec` to be used
// to send elements to another thread. Additionally, `Vec` allows getting a reference to any
// element from any thread. Therefore, it is safe to share `Vec` between threads as long as the
// element is both sendable and shareable.
unsafe impl<T: Send + Sync> Sync for Vec<T> {}

impl<T> Vec<T> {
    /// Creates a new `Vec`.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity. The capacity can be excessively huge, as none of the memory
    /// is [committed] until you push elements into the vector.
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
        Vec {
            inner: unsafe { RawVec::new(max_capacity) },
        }
    }

    /// Creates a new `Vec`.
    ///
    /// Like [`new`], except allowing you to specify the growth strategy.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [`new`]: Self::new
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub fn with_growth_strategy(max_capacity: usize, growth_strategy: GrowthStrategy) -> Self {
        Vec {
            inner: unsafe { RawVec::with_growth_strategy(max_capacity, growth_strategy) },
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
        Ok(Vec {
            inner: unsafe { RawVec::try_new(max_capacity) }?,
        })
    }

    /// Creates a new `Vec`.
    ///
    /// Like [`with_growth_strategy`], except returning an error when allocation fails.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`with_growth_strategy`]: Self::with_growth_strategy
    /// [reserving]: crate#reserving
    #[track_caller]
    pub fn try_with_growth_strategy(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
    ) -> Result<Self, TryReserveError> {
        Ok(Vec {
            inner: unsafe { RawVec::try_with_growth_strategy(max_capacity, growth_strategy) }?,
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
            inner: RawVec::dangling(),
        }
    }

    /// Returns the maximum capacity that was used when creating the `Vec`.
    #[inline]
    #[must_use]
    pub fn max_capacity(&self) -> usize {
        self.inner.max_capacity()
    }

    /// Returns the total number of elements the vector can hold without [committing] more memory.
    ///
    /// [committing]: crate#committing
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns the number of elements in the vector.
    ///
    /// This number may [`capacity`], doesn't corresopond to the number of initialized elements,
    /// and also doesn't synchronize with setting the capacity.
    ///
    /// [`capacity`]: Self::capacity
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// This counts elements that failed to be pushed due to an error.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Appends an element to the end of the vector. Returns the index of the inserted element.
    #[inline]
    #[track_caller]
    pub fn push(&self, value: T) -> usize {
        self.push_with(|_| value)
    }

    /// Appends an element to the end of the vector.
    ///
    /// `f` is called with the index of the element and the result is written to the element.
    /// Returns the index of the element.
    #[inline]
    #[track_caller]
    pub fn push_with(&self, f: impl FnOnce(usize) -> T) -> usize {
        let (index, slot) = self.inner.push();

        // SAFETY: `RawVec::push` guarantees that the slot is unique for each push.
        unsafe { &mut *slot.value.get() }.write(f(index));

        // SAFETY: We have written the element above.
        unsafe { slot.occupied.store(true, Release) };

        index
    }

    /// Appends an element to the end of the vector. Returns the index of the inserted element.
    #[inline]
    #[track_caller]
    pub fn push_mut(&mut self, value: T) -> usize {
        self.push_with_mut(|_| value)
    }

    /// Appends an element to the end of the vector.
    ///
    /// `f` is called with the index of the element and the result is written to the element.
    /// Returns the index of the element.
    #[inline]
    #[track_caller]
    pub fn push_with_mut(&mut self, f: impl FnOnce(usize) -> T) -> usize {
        let (index, slot) = self.inner.push_mut();

        slot.value.get_mut().write(f(index));

        // SAFETY: We have written the element above.
        unsafe { *slot.occupied.get_mut() = true };

        index
    }

    /// Returns a reference to an element of the vector.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.as_capacity().get(index)?.value()
    }

    /// Returns a mutable reference to an element of the vector.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.as_mut_capacity().get_mut(index)?.value_mut()
    }

    /// Returns a reference to an element of the vector without doing any checks.
    ///
    /// # Safety
    ///
    /// `index` must have been previously acquired through [`push`] or [`push_mut`] on `self`.
    ///
    /// [`push`]: Self::push
    /// [`push_mut`]: Self::push_mut
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.inner.get_unchecked(index) };

        slot.occupied.load(Acquire);

        // SAFETY: The caller must ensure that the slot has been initialized. The `Acquire`
        // ordering above synchronizes with the `Release` ordering in `push`, making sure that the
        // write is visible here.
        unsafe { slot.value_unchecked() }
    }

    /// Returns a mutable reference to an element of the vector without doing any checks.
    ///
    /// # Safety
    ///
    /// `index` must have been previously acquired through [`push`] or [`push_mut`] on `self`.
    ///
    /// [`push`]: Self::push
    /// [`push_mut`]: Self::push_mut
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: The caller must ensure that the index is in bounds.
        let slot = unsafe { self.inner.get_unchecked_mut(index) };

        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { slot.value_unchecked_mut() }
    }

    /// Returns an iterator that yields references to elements of the vector.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    /// Returns an iterator that yields mutable references to elements of the vector.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            inner: self.inner.iter_mut(),
        }
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

impl<T: Clone> Clone for Vec<T> {
    #[inline]
    fn clone(&self) -> Self {
        Vec {
            inner: self.inner.clone(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for Vec<T> {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        self.inner == other.inner
    }
}

impl<T: PartialEq<U>, U> PartialEq<&[U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        *self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<&mut [U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&mut [U]) -> bool {
        *self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for &[T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for &mut [T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U> PartialEq<[U]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        let len = self.len();

        if len != other.len() {
            return false;
        }

        #[allow(clippy::needless_range_loop)]
        for index in 0..len {
            let Some(elem) = self.get(index) else {
                return false;
            };

            if *elem != other[index] {
                return false;
            }
        }

        true
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for [T] {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        let len = other.len();

        if len != self.len() {
            return false;
        }

        #[allow(clippy::needless_range_loop)]
        for index in 0..len {
            let Some(elem) = other.get(index) else {
                return false;
            };

            if self[index] != *elem {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "alloc")]
impl<T: PartialEq<U> + Clone, U> PartialEq<Vec<U>> for alloc::borrow::Cow<'_, [T]> {
    #[inline]
    fn eq(&self, other: &Vec<U>) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        *self == *other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&[U; N]> for Vec<T> {
    #[inline]
    fn eq(&self, other: &&[U; N]) -> bool {
        *self == *other.as_slice()
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
        Hash::hash(&*self.inner, state);
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output {
        if let Some(value) = self.get(index) {
            value
        } else {
            invalid_index(index)
        }
    }
}

impl<T> IndexMut<usize> for Vec<T> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if let Some(value) = self.get_mut(index) {
            value
        } else {
            invalid_index(index)
        }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn invalid_index(index: usize) -> ! {
    panic!("index {index} is out of bounds or not yet initialized");
}

impl<'a, T> IntoIterator for &'a Vec<T> {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vec<T> {
    type Item = &'a mut T;

    type IntoIter = IterMut<'a, T>;

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
        PartialOrd::partial_cmp(&self.inner, &other.inner)
    }
}

impl<T: Ord> Ord for Vec<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.inner, &other.inner)
    }
}

struct Slot<T> {
    occupied: AtomicBool,
    value: UnsafeCell<MaybeUninit<T>>,
}

// While `Slot` does contain interior mutability, this is never exposed to the user and no
// invariants can be broken.
impl<T: RefUnwindSafe> RefUnwindSafe for Slot<T> {}

impl<T> Slot<T> {
    #[inline(always)]
    fn value(&self) -> Option<&T> {
        if self.occupied.load(Acquire) {
            // SAFETY: We checked that the slot has been initialized above. The `Acquire` ordering
            // above synchronizes with the `Release` ordering in `push`, making sure that the write
            // is visible here.
            Some(unsafe { self.value_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn value_mut(&mut self) -> Option<&mut T> {
        if *self.occupied.get_mut() {
            // SAFETY: We checked that the slot has been initialized above.
            Some(unsafe { self.value_unchecked_mut() })
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn value_unchecked(&self) -> &T {
        // SAFETY: The caller must ensure that access to the cell's inner value is synchronized.
        let value = unsafe { &*self.value.get() };

        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { value.assume_init_ref() }
    }

    #[inline(always)]
    unsafe fn value_unchecked_mut(&mut self) -> &mut T {
        // SAFETY: The caller must ensure that the slot has been initialized.
        unsafe { self.value.get_mut().assume_init_mut() }
    }
}

impl<T: Clone> Clone for Slot<T> {
    #[inline]
    fn clone(&self) -> Self {
        let value = self.value();

        Slot {
            occupied: AtomicBool::new(value.is_some()),
            value: UnsafeCell::new(if let Some(value) = value {
                MaybeUninit::new(value.clone())
            } else {
                MaybeUninit::uninit()
            }),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.value(), f)
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        if *self.occupied.get_mut() {
            // SAFETY: We checked that the slot has been initialized above. We own the value, and
            // the slot is being dropped, which ensures that the value can't be accessed again.
            unsafe { self.value.get_mut().assume_init_drop() };
        }
    }
}

impl<T: PartialEq<U>, U> PartialEq<Slot<U>> for Slot<T> {
    #[inline]
    fn eq(&self, other: &Slot<U>) -> bool {
        match (self.value(), other.value()) {
            (Some(value), Some(other_value)) => *value == *other_value,
            (Some(_), None) => false,
            (None, Some(_)) => false,
            (None, None) => true,
        }
    }
}

impl<T: Eq> Eq for Slot<T> {}

impl<T: Hash> Hash for Slot<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value().hash(state);
    }
}

impl<T: PartialOrd> PartialOrd for Slot<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&self.value(), &other.value())
    }
}

impl<T: Ord> Ord for Slot<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.value(), &other.value())
    }
}

/// An iterator that yields references to elements of a vector.
///
/// This struct is created by the [`iter`] method on [`Vec`].
///
/// [`iter`]: Vec::iter
pub struct Iter<'a, T> {
    inner: slice::Iter<'a, Slot<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let slot = self.inner.next()?;

            if let Some(value) = slot.value() {
                break Some(value);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

impl<T> DoubleEndedIterator for Iter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let slot = self.inner.next_back()?;

            if let Some(value) = slot.value() {
                break Some(value);
            }
        }
    }
}

impl<T> FusedIterator for Iter<'_, T> {}

/// An iterator that yields mutable references to elements of a vector.
///
/// This struct is created by the [`iter_mut`] method on [`Vec`].
///
/// [`iter_mut`]: Vec::iter_mut
pub struct IterMut<'a, T> {
    inner: slice::IterMut<'a, Slot<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let slot = self.inner.next()?;

            if let Some(value) = slot.value_mut() {
                break Some(value);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

impl<T> DoubleEndedIterator for IterMut<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let slot = self.inner.next_back()?;

            if let Some(value) = slot.value_mut() {
                break Some(value);
            }
        }
    }
}

impl<T> FusedIterator for IterMut<'_, T> {}

/// An iterator that moves out of a vector.
///
/// This struct is created by the [`into_iter`] method on [`Vec`].
///
/// [`into_iter`]: Vec::into_iter
pub struct IntoIter<T> {
    start: *mut (),
    end: *mut (),
    #[allow(dead_code)]
    allocation: Allocation,
    marker: PhantomData<T>,
}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Send> Send for IntoIter<T> {}

// SAFETY: We own the collection, and synchronization to it is ensured using mutable references.
unsafe impl<T: Sync> Sync for IntoIter<T> {}

impl<T> IntoIter<T> {
    #[inline]
    pub(super) fn new(vec: Vec<T>) -> Self {
        let mut vec = ManuallyDrop::new(vec.inner);

        // SAFETY: `vec` is wrapped in a `ManuallyDrop` such that a double-free can't happen even
        // if a panic was possible below.
        let allocation = unsafe { ptr::read(&vec.inner.allocation) };

        let start = vec.as_mut_ptr();

        let len = cmp::min(vec.len_mut(), vec.capacity_mut());

        // SAFETY: The ownership synchronizes with setting the capacity, making sure that the newly
        // committed memory is visible here.
        let end = unsafe { start.add(len) };

        IntoIter {
            start: start.cast(),
            end: end.cast(),
            allocation,
            marker: PhantomData,
        }
    }

    #[inline]
    fn start(&self) -> *mut Slot<T> {
        self.start.cast()
    }

    #[inline]
    fn end(&self) -> *mut Slot<T> {
        self.end.cast()
    }

    #[inline]
    fn len(&self) -> usize {
        // We know that the return value is positive because by our invariant, `self.end` is always
        // greater or equal to `self.start`.
        #[allow(clippy::cast_sign_loss)]
        // SAFETY:
        // * `start` and `end` were both created from the same object in `IntoIter::new`.
        // * `Vec::new` ensures that the allocation size doesn't exceed `isize::MAX` bytes.
        // * We know that the allocation doesn't wrap around the address space.
        unsafe {
            self.end().offset_from(self.start()) as usize
        }
    }

    fn as_slice(&self) -> &[Slot<T>] {
        unsafe { slice::from_raw_parts(self.start(), self.len()) }
    }
}

impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Elements<'a, T>(&'a [Slot<T>]);

        impl<T: fmt::Debug> fmt::Debug for Elements<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list()
                    .entries(self.0.iter().flat_map(Slot::value))
                    .finish()
            }
        }

        f.debug_tuple("IntoIter")
            .field(&Elements(self.as_slice()))
            .finish()
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        let elements = ptr::slice_from_raw_parts_mut(self.start(), self.len());

        // SAFETY: We own the collection, and it is being dropped, which ensures that the elements
        // can't be accessed again.
        unsafe { elements.drop_in_place() };
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.start == self.end {
                break None;
            }

            let start = self.start();

            // SAFETY: We checked that there are still elements remaining above.
            self.start = unsafe { start.add(1) }.cast();

            // SAFETY: Same as the previous.
            let slot = unsafe { &*start };

            if let Some(value) = slot.value() {
                // SAFETY: We own the collection, and have just decremented the `end` pointer such
                // that this element can't be accessed again.
                break Some(unsafe { ptr::read(value) });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.len()))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.start == self.end {
                break None;
            }

            // SAFETY: We checked that there are still elements remaining above.
            let end = unsafe { self.end().sub(1) };

            self.end = end.cast();

            // SAFETY: Same as the previous.
            let slot = unsafe { &*end };

            if let Some(value) = slot.value() {
                // SAFETY: We own the collection, and have just decremented the `end` pointer such
                // that this element can't be accessed again.
                break Some(unsafe { ptr::read(value) });
            }
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}

/// A low-level, concurrent, in-place growable vector.
///
/// This is meant to be a low-level primitive used to implement data structures such as the
/// concurrent [`Vec`]. Unlike the concurrent `Vec`, this type stores `T`s directly, and as such
/// doesn't have any memory overhead and derefs to `[T]`. Methods such as [`Vec::get`] are not
/// provided because you should use the slice equivalent. This added flexibility comes at the cost
/// of a bit of unsafety: constructing a `RawVec<T>`, you must ensure that `T` is zeroable, and
/// pushing and accessing elements requires you to handle initialization of the element yourself.
pub struct RawVec<T> {
    inner: RawVecInner,
    marker: PhantomData<T>,
}

struct RawVecInner {
    elements: *mut (),
    capacity: AtomicUsize,
    len: AtomicUsize,
    max_capacity: usize,
    growth_strategy: GrowthStrategy,
    allocation: Allocation,
}

impl<T> RawVec<T> {
    /// Creates a new `RawVec`.
    ///
    /// `max_capacity` is the maximum capacity the vector can ever have. The vector is guaranteed
    /// to never exceed this capacity. The capacity can be excessively huge, as none of the memory
    /// is [committed] until you push elements into the vector.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
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
    pub unsafe fn new(max_capacity: usize) -> Self {
        unsafe { Self::with_header(max_capacity, Layout::new::<()>()) }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`new`], except allowing you to specify the growth strategy.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [`new`]: Self::new
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub unsafe fn with_growth_strategy(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
    ) -> Self {
        unsafe {
            Self::with_growth_strategy_and_header(
                max_capacity,
                growth_strategy,
                Layout::new::<()>(),
            )
        }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`new`], except additionally allocating a header. `header_layout.size()` bytes will be
    /// allocated before the start of the vector's elements, with the start aligned to
    /// `header_layout.align()`. You can use [`as_ptr`] and offset backwards to access the header.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
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
    pub unsafe fn with_header(max_capacity: usize, header_layout: Layout) -> Self {
        unsafe {
            Self::with_growth_strategy_and_header(
                max_capacity,
                GrowthStrategy::default(),
                header_layout,
            )
        }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`with_growth_strategy`] and [`with_header`] combined.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    /// - Panics if `header_layout` is not padded to its alignment.
    /// - Panics if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Panics if [reserving] the allocation returns an error.
    ///
    /// [`with_growth_strategy`]: Self::try_new
    /// [`with_header`]: Self::with_header
    /// [reserving]: crate#reserving
    #[must_use]
    #[track_caller]
    pub unsafe fn with_growth_strategy_and_header(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
        header_layout: Layout,
    ) -> Self {
        match unsafe {
            Self::try_with_growth_strategy_and_header(max_capacity, growth_strategy, header_layout)
        } {
            Ok(vec) => vec,
            Err(err) => handle_error(err),
        }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`new`], except returning an error when allocation fails.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`new`]: Self::new
    /// [reserving]: crate#reserving
    pub unsafe fn try_new(max_capacity: usize) -> Result<Self, TryReserveError> {
        unsafe { Self::try_with_header(max_capacity, Layout::new::<()>()) }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`with_growth_strategy`], except returning an error when allocation fails.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`with_growth_strategy`]: Self::with_growth_strategy
    /// [reserving]: crate#reserving
    #[track_caller]
    pub unsafe fn try_with_growth_strategy(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
    ) -> Result<Self, TryReserveError> {
        unsafe {
            Self::try_with_growth_strategy_and_header(
                max_capacity,
                growth_strategy,
                Layout::new::<()>(),
            )
        }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`with_header`], except returning an error when allocation fails.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
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
    /// [reserving]: crate#reserving
    #[track_caller]
    pub unsafe fn try_with_header(
        max_capacity: usize,
        header_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        unsafe {
            Self::try_with_growth_strategy_and_header(
                max_capacity,
                GrowthStrategy::default(),
                header_layout,
            )
        }
    }

    /// Creates a new `RawVec`.
    ///
    /// Like [`with_growth_strategy_and_header`], except returning an error when allocation fails.
    ///
    /// # Safety
    ///
    /// `T` must be zeroable.
    ///
    /// # Panics
    ///
    /// - Panics if `growth_strategy` isn't valid per the documentation of [`GrowthStrategy`].
    /// - Panics if `header_layout` is not padded to its alignment.
    ///
    /// # Errors
    ///
    /// - Returns an error if the `max_capacity` would exceed `isize::MAX` bytes.
    /// - Returns an error if [reserving] the allocation returns an error.
    ///
    /// [`with_growth_strategy_and_header`]: Self::with_growth_strategy_and_header
    /// [reserving]: crate#reserving
    #[track_caller]
    pub unsafe fn try_with_growth_strategy_and_header(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
        header_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        Ok(RawVec {
            inner: unsafe {
                RawVecInner::new(
                    max_capacity,
                    growth_strategy,
                    header_layout,
                    size_of::<T>(),
                    align_of::<T>(),
                )
            }?,
            marker: PhantomData,
        })
    }

    /// Creates a dangling `RawVec`.
    ///
    /// This is useful as a placeholder value to defer allocation until later or if no allocation
    /// is needed.
    #[inline]
    #[must_use]
    pub const fn dangling() -> Self {
        RawVec {
            inner: RawVecInner::dangling(align_of::<T>()),
            marker: PhantomData,
        }
    }

    /// Returns a slice of the entire vector.
    ///
    /// This returns a slice with a length equal to the vector's capacity.
    #[inline]
    #[must_use]
    pub fn as_capacity(&self) -> &[T] {
        // SAFETY: The `Acquire` ordering in `RawVec::capacity` synchronizes with the `Release`
        // ordering when setting the capacity, making sure that the newly committed memory is
        // visible here. The constructor of `RawVec` must ensure that `T` is zeroable.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.capacity()) }
    }

    /// Returns a mutable slice of the entire vector.
    ///
    /// This returns a slice with a length equal to the vector's capacity.
    #[inline]
    #[must_use]
    pub fn as_mut_capacity(&mut self) -> &mut [T] {
        // SAFETY: The mutable reference synchronizes with setting the capacity, making sure that
        // the newly committed memory is visible here. The constructor of `RawVec` must ensure that
        // `T` is zeroable.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity_mut()) }
    }

    /// Returns a slice of the entire vector.
    ///
    /// This returns a slice with a length equal to the minimum of the vector's length and its
    /// capacity. As such, it is slightly less efficient than [`as_capacity`].
    ///
    /// [`as_capacity`]: Self::as_capacity
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let len = cmp::min(self.len(), self.capacity());

        // SAFETY: The `Acquire` ordering in `RawVec::capacity` synchronizes with the `Release`
        // ordering when setting the capacity, making sure that the newly committed memory is
        // visible here. The constructor of `RawVec` must ensure that `T` is zeroable.
        unsafe { slice::from_raw_parts(self.as_ptr(), len) }
    }

    /// Returns a mutable slice of the entire vector.
    ///
    /// This returns a slice with a length equal to the minimum of the vector's length and its
    /// capacity. As such, it is slightly less efficient than [`as_mut_capacity`].
    ///
    /// [`as_mut_capacity`]: Self::as_mut_capacity
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = cmp::min(self.len_mut(), self.capacity_mut());

        // SAFETY: The mutable reference synchronizes with setting the capacity, making sure that
        // the newly committed memory is visible here. The constructor of `RawVec` must ensure that
        // `T` is zeroable.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
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

    /// Returns the maximum capacity that was used when creating the `RawVec`.
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
            self.inner.capacity.load(Acquire)
        }
    }

    #[inline]
    fn capacity_mut(&mut self) -> usize {
        if T::IS_ZST {
            usize::MAX
        } else {
            *self.inner.capacity.get_mut()
        }
    }

    /// Returns the number of elements in the vector.
    ///
    /// This number may exceed [`capacity`]. It also doesn't synchronize with setting the capacity.
    /// As such, this must not be used in conjunction with [`as_ptr`] to unsafely access any
    /// element unless you compared the length to the capacity.
    ///
    /// [`capacity`]: Self::capacity
    /// [`as_ptr`]: Self::as_ptr
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len.load(Relaxed)
    }

    #[inline]
    fn len_mut(&mut self) -> usize {
        *self.inner.len.get_mut()
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// This counts elements that failed to be pushed due to an error.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an element to the end of the vector.
    ///
    /// Returns the index of the inserted element as well as the element itself. The element is
    /// zeroed; you must initialize it yourself.
    #[inline]
    #[track_caller]
    pub fn push(&self) -> (usize, &T) {
        if T::IS_ZST {
            let mut len = self.inner.len.load(Relaxed);

            loop {
                if len == usize::MAX {
                    capacity_overflow();
                }

                match self
                    .inner
                    .len
                    .compare_exchange(len, len + 1, Relaxed, Relaxed)
                {
                    Ok(_) => {
                        // SAFETY: `T` is a ZST, which means that all elements live at the same
                        // address. The constructor of `RawVec` must ensure that `T` is zeroable.
                        let slot = unsafe { &*self.as_ptr() };

                        // SAFETY: We reserved an index by incrementing `self.inner.len`, which
                        // means that no other threads can be calling `init` with the same index.
                        break (len, slot);
                    }
                    Err(new_len) => len = new_len,
                }
            }
        } else {
            // This cannot overflow because `self.inner.max_capacity` can never exceed `isize::MAX`
            // bytes, and because `RawVec::grow_one` decrements the length back if the length
            // overshot `self.inner.max_capacity`.
            let len = self.inner.len.fetch_add(1, Relaxed);

            if len >= self.inner.capacity.load(Acquire) {
                unsafe { self.grow_one(len) };
            }

            // SAFETY: We made sure that the index is in bounds above. The `Acquire` ordering above
            // synchronizes with the `Release` ordering when setting the capacity, making sure that
            // the newly committed memory is visible here. The constructor of `RawVec` must ensure
            // that `T` is zeroable.
            let slot = unsafe { &*self.as_ptr().add(len) };

            (len, slot)
        }
    }

    #[cold]
    #[inline(never)]
    #[track_caller]
    unsafe fn grow_one(&self, len: usize) {
        if let Err(err) = unsafe { self.inner.grow(len, 1, size_of::<T>()) } {
            if len >= self.inner.max_capacity {
                // This can't overflow because `grow_one` must be called after the length was
                // incremented. It is sound to decrement because it's impossible for an element to
                // be initialized while the length exceeds `self.inner.max_capacity`. Note that
                // this is *not* the case when `grow_amortized` fails in general: it can happen
                // that committing more memory fails on this thread but succeeds on another,
                // resulting in us decrementing the length while it is below `self.inner.capacity`,
                // which would lead to two threads getting the same index in `push`. That's why we
                // must not decrement the length unless it overshot `self.inner.max_capacity`. We
                // decrement the length in order to prevent an overflow in `push`.
                self.inner.len.fetch_sub(1, Relaxed);
            }

            handle_error(err);
        }
    }

    /// Appends an element to the end of the vector.
    ///
    /// Returns the index of the inserted element as well as the element itself. The element is
    /// zeroed; you must initialize it yourself.
    #[inline]
    #[track_caller]
    pub fn push_mut(&mut self) -> (usize, &mut T) {
        let len = self.len_mut();

        if len >= self.capacity_mut() {
            unsafe { self.grow_one_mut() };
        }

        *self.inner.len.get_mut() = len + 1;

        // SAFETY: We made sure the index is in bounds above. The mutable reference synchronizes
        // with setting the capacity, making sure that the newly committed memory is visible here.
        // The constructor of `RawVec` must ensure that `T` is zeroable.
        let slot = unsafe { &mut *self.as_mut_ptr().add(len) };

        (len, slot)
    }

    #[cold]
    #[inline(never)]
    #[track_caller]
    unsafe fn grow_one_mut(&mut self) {
        let len = self.len_mut();

        if let Err(err) = unsafe { self.inner.grow(len, 1, size_of::<T>()) } {
            handle_error(err);
        }
    }
}

impl RawVecInner {
    #[track_caller]
    unsafe fn new(
        max_capacity: usize,
        growth_strategy: GrowthStrategy,
        header_layout: Layout,
        elem_size: usize,
        elem_align: usize,
    ) -> Result<Self, TryReserveError> {
        growth_strategy.validate();

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

        Ok(RawVecInner {
            elements,
            capacity: AtomicUsize::new(capacity),
            len: AtomicUsize::new(0),
            max_capacity,
            growth_strategy,
            allocation,
        })
    }

    #[inline]
    const fn dangling(elem_align: usize) -> Self {
        let allocation = Allocation::dangling(elem_align);

        RawVecInner {
            elements: allocation.ptr().cast(),
            capacity: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            max_capacity: 0,
            growth_strategy: GrowthStrategy::Exponential {
                numerator: 2,
                denominator: 1,
            },
            allocation,
        }
    }

    unsafe fn grow(
        &self,
        len: usize,
        additional: usize,
        elem_size: usize,
    ) -> Result<(), TryReserveError> {
        debug_assert!(additional > 0);

        if elem_size == 0 {
            return Err(CapacityOverflow.into());
        }

        let required_capacity = len.checked_add(additional).ok_or(CapacityOverflow)?;

        if required_capacity > self.max_capacity {
            return Err(CapacityOverflow.into());
        }

        let old_capacity = self.capacity.load(Acquire);

        if old_capacity >= required_capacity {
            // Another thread beat us to it.
            return Ok(());
        }

        let new_capacity = self.growth_strategy.grow(old_capacity);
        let new_capacity = cmp::max(new_capacity, required_capacity);
        let new_capacity = cmp::min(new_capacity, self.max_capacity);

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

        if let Err(capacity) =
            self.capacity
                .compare_exchange(old_capacity, new_capacity, Release, Acquire)
        {
            // We lost the race, but the winner must have updated the capacity like we wanted to.
            debug_assert!(capacity >= new_capacity);
        }

        Ok(())
    }
}

impl<T> AsRef<RawVec<T>> for RawVec<T> {
    #[inline]
    fn as_ref(&self) -> &RawVec<T> {
        self
    }
}

impl<T> AsMut<RawVec<T>> for RawVec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut RawVec<T> {
        self
    }
}

impl<T> AsRef<[T]> for RawVec<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> AsMut<[T]> for RawVec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> Borrow<[T]> for RawVec<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T> BorrowMut<[T]> for RawVec<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T: Clone> Clone for RawVec<T> {
    #[inline]
    fn clone(&self) -> Self {
        let mut vec = unsafe { RawVec::new(self.inner.max_capacity) };

        for elem in self {
            let (_, slot) = vec.push_mut();
            *slot = elem.clone();
        }

        vec
    }
}

impl<T: fmt::Debug> fmt::Debug for RawVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> Deref for RawVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for RawVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        let len = cmp::min(self.len_mut(), self.capacity_mut());
        let elements = ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), len);

        // SAFETY: We own the collection, and it is being dropped, which ensures that the elements
        // can't be accessed again.
        unsafe { elements.drop_in_place() };
    }
}

impl<T: PartialEq<U>, U> PartialEq<RawVec<U>> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &RawVec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<&[U]> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &&[U]) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<&mut [U]> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &&mut [U]) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<RawVec<U>> for &[T] {
    #[inline]
    fn eq(&self, other: &RawVec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<RawVec<U>> for &mut [T] {
    #[inline]
    fn eq(&self, other: &RawVec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U> PartialEq<[U]> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U> PartialEq<RawVec<U>> for [T] {
    #[inline]
    fn eq(&self, other: &RawVec<U>) -> bool {
        *self == **other
    }
}

#[cfg(feature = "alloc")]
impl<T: PartialEq<U> + Clone, U> PartialEq<RawVec<U>> for alloc::borrow::Cow<'_, [T]> {
    #[inline]
    fn eq(&self, other: &RawVec<U>) -> bool {
        **self == **other
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        **self == *other
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&[U; N]> for RawVec<T> {
    #[inline]
    fn eq(&self, other: &&[U; N]) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for RawVec<T> {}

impl<T> Extend<T> for RawVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for elem in iter {
            let (_, slot) = self.push_mut();
            *slot = elem;
        }
    }
}

impl<'a, T: Copy> Extend<&'a T> for RawVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for &elem in iter {
            let (_, slot) = self.push_mut();
            *slot = elem;
        }
    }
}

impl<T: Hash> Hash for RawVec<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state);
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for RawVec<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for RawVec<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<'a, T> IntoIterator for &'a RawVec<T> {
    type Item = &'a T;

    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut RawVec<T> {
    type Item = &'a mut T;

    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> IntoIterator for RawVec<T> {
    type Item = T;

    type IntoIter = raw::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        raw::IntoIter::new(self)
    }
}

impl<T: PartialOrd> PartialOrd for RawVec<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: Ord> Ord for RawVec<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}
