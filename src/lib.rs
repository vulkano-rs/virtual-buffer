//! This crate provides a cross-platform API for dealing with buffers backed by raw virtual memory.
//!
//! Apart from providing protection and isolation between processes, paging, and memory mapped
//! hardware, virtual memory serves to solve another critical issue: the issue of the virtual
//! buffer. It allows us to [reserve] a range of memory only in the process's virtual address
//! space, without actually [committing] any of the memory. This can be used to create a buffer
//! that's infinitely growable and shrinkable *in-place*, without wasting any physical memory, nor
//! even overcommitting any memory. It can also be used to create sparse data structures that don't
//! overcommit memory.
//!
//! The property of growing in-place is very valuable when reallocation is impossible, for example
//! because the data structure needs to be concurrent or otherwise pinned. It may also be of use
//! for single-threaded use cases if reallocation is too expensive (say, tens to hundreds of MB).
//! However, it's probably easier to use something like [`Vec::with_capacity`] in that case.
//!
//! See also [the `vec` module] for an implementation of a concurrent vector.
//!
//! # Reserving
//!
//! Reserving memory involves allocating a range of virtual address space, such that other
//! allocations within the same process can't reserve any of the same virtual address space for
//! anything else. Memory that has been reserved has zero memory cost, however, it can't be
//! accessed. In order to access any of the pages, you will have to commit them first.
//!
//! # Committing
//!
//! A range of reserved memory can be committed to make it accessible. Memory that has been freshly
//! committed doesn't use up any physical memory. It merely counts towards overcommitment, which
//! may increase the likelihood of being OOM-killed, and may take up space for page tables and may
//! use some space in the page file. A committed page is only ever backed by a physical page after
//! being written to for the first time (being "faulted"), or when it was [prefaulted].
//!
//! Committed memory can be committed again without issue, so there is no need to keep track of
//! which pages have been committed in order to safely commit some of them.
//!
//! # Decommitting
//!
//! A range of committed memory can be decommitted, making it inaccessible again, and releasing any
//! physical memory that may have been used for them back to the operating system. Decommitted
//! memory is still reserved.
//!
//! Reserved but uncommitted memory can be decommitted without issue, so there is no need to keep
//! track of which pages have been committed in order to safely decommit some of them.
//!
//! # Unreserving
//!
//! Memory that is unreserved is available for new allocations to reserve again.
//!
//! Committed memory can be unreserved without needing to be decommitted first. However, it's not
//! possible to unreserve a range of reserved memory, only the entire allocation.
//!
//! # Prefaulting
//!
//! By default, each committed page is only ever backed by physical memory after it was first
//! written to. Since this happens for every page, and can be slightly costly due to the overhead
//! of a context switch, operating systems provide a way to *prefault* multiple pages at once.
//!
//! # Pages
//!
//! A page refers to the granularity at which the processor's Memory Management Unit operates and
//! varies between processor architectures. As such, virtual memory operations can only affect
//! ranges that are aligned to the *page size*.
//!
//! # Cargo features
//!
//! | Feature | Description                                       |
//! |---------|---------------------------------------------------|
//! | std     | Enables the use of `std::error` and `std::borrow` |
//!
//! [reserve]: self#reserving
//! [committing]: self#committing
//! [`Vec::with_capacity`]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.with_capacity
//! [the `vec` module]: self::vec
//! [prefaulted]: self#prefaulting

#![allow(
    unused_unsafe,
    clippy::doc_markdown,
    clippy::inline_always,
    clippy::unused_self
)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(unix, windows)))]
compile_error!("unsupported platform");

#[cfg(unix)]
use self::unix as sys;
#[cfg(windows)]
use self::windows as sys;
use core::{fmt, mem};

pub mod vec;

/// An allocation backed by raw virtual memory, giving you the power to directly manipulate the
/// pages within it.
///
/// See also [the crate-level documentation] for more information about virtual memory.
///
/// [the crate-level documentation]: self
pub struct Allocation {
    inner: sys::Allocation,
}

impl Allocation {
    /// Allocates a new region in the process's virtual address space.
    ///
    /// `size` is the size to [reserve] in bytes. This number can be excessively huge, as none of
    /// the memory is [committed] until you call [`commit`]. The memory is [unreserved] when the
    /// `Allocation` is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system returns an error.
    ///
    /// # Panics
    ///
    /// - Panics if `size` is not aligned to the [page size].
    /// - Panics if `size` is zero.
    ///
    /// [reserve]: self#reserving
    /// [page size]: self#pages
    /// [committed]: self#committing
    /// [unreserved]: self#unreserving
    /// [`commit`]: Self::commit
    pub fn new(size: usize) -> Result<Self> {
        assert!(is_aligned(size, page_size()));
        assert_ne!(size, 0);

        let inner = sys::Allocation::new(size)?;

        Ok(Allocation { inner })
    }

    /// Creates a dangling `Allocation`, that is, an allocation with a dangling pointer and zero
    /// size.
    ///
    /// This is useful as a placeholder value to defer allocation until later or if no allocation
    /// is needed.
    ///
    /// `alignment` is the alignment of the allocation's pointer, and must be a power of two.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a power of two.
    #[inline]
    #[must_use]
    pub const fn dangling(alignment: usize) -> Allocation {
        let inner = sys::Allocation::dangling(alignment);

        Allocation { inner }
    }

    /// Returns the pointer to the beginning of the allocation.
    ///
    /// The returned pointer is always valid, including [dangling allocations], for reads and
    /// writes of [`size()`] bytes in the sense that it can never lead to undefined behavior.
    /// However, doing a read or write access to [pages] that have not been [committed] will result
    /// in the process receiving SIGSEGV / STATUS_ACCESS_VIOLATION.
    ///
    /// The pointer must not be accessed after `self` has been dropped.
    ///
    /// [dangling allocations]: Self::dangling
    /// [pages]: self#pages
    /// [committed]: self#committing
    /// [`size()`]: Self::size
    #[inline(always)]
    #[must_use]
    pub const fn ptr(&self) -> *mut u8 {
        self.inner.ptr().cast()
    }

    /// Returns the size that was used to [allocate] `self`.
    ///
    /// [allocate]: Self::new
    #[inline(always)]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.inner.size()
    }

    /// [Commits] the given region of memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system returns an error.
    ///
    /// # Panics
    ///
    /// - Panics if the allocation is [dangling].
    /// - Panics if `ptr` and `size` denote a region that is out of bounds of the allocation.
    /// - Panics if `ptr` and/or `size` is not aligned to the [page size].
    /// - Panics if `size` is zero.
    ///
    /// [Commits]: self#committing
    /// [dangling]: Self::dangling
    /// [page size]: self#pages
    pub fn commit(&self, ptr: *mut u8, size: usize) -> Result<()> {
        self.check_range(ptr, size);

        // SAFETY: Enforced by the `check_range` call above.
        unsafe { self.inner.commit(ptr.cast(), size) }
    }

    /// [Decommits] the given region of memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system returns an error.
    ///
    /// # Panics
    ///
    /// - Panics if the allocation is [dangling].
    /// - Panics if `ptr` and `size` denote a region that is out of bounds of the allocation.
    /// - Panics if `ptr` and/or `size` is not aligned to the [page size].
    /// - Panics if `size` is zero.
    ///
    /// [Decommits]: self#decommitting
    /// [dangling]: Self::dangling
    /// [page size]: self#pages
    pub fn decommit(&self, ptr: *mut u8, size: usize) -> Result<()> {
        self.check_range(ptr, size);

        // SAFETY: Enforced by the `check_range` call above.
        unsafe { self.inner.decommit(ptr.cast(), size) }
    }

    /// [Prefaults] the given region of memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system returns an error.
    ///
    /// # Panics
    ///
    /// - Panics if the allocation is [dangling].
    /// - Panics if `ptr` and `size` denote a region that is out of bounds of the allocation.
    /// - Panics if `ptr` and/or `size` is not aligned to the [page size].
    /// - Panics if `size` is zero.
    ///
    /// [Prefaults]: self#prefaulting
    /// [dangling]: Self::dangling
    /// [page size]: self#pages
    pub fn prefault(&self, ptr: *mut u8, size: usize) -> Result<()> {
        self.check_range(ptr, size);

        // SAFETY: Enforced by the `check_range` call above.
        unsafe { self.inner.prefault(ptr.cast(), size) }
    }

    #[inline(never)]
    fn check_range(&self, ptr: *mut u8, size: usize) {
        assert_ne!(self.size(), 0, "the allocation is dangling");
        assert_ne!(size, 0);

        let allocated_range = addr(self.ptr())..addr(self.ptr()) + self.size();
        let requested_range = addr(ptr)..addr(ptr).checked_add(size).unwrap();
        assert!(allocated_range.start <= requested_range.start);
        assert!(requested_range.end <= allocated_range.end);

        let page_size = page_size();
        assert!(is_aligned(addr(ptr), page_size));
        assert!(is_aligned(size, page_size));
    }
}

impl fmt::Debug for Allocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

/// Returns the [page size] of the system.
///
/// The value is cached globally and very fast to retrieve.
///
/// [page size]: self#pages
#[inline(always)]
#[must_use]
pub fn page_size() -> usize {
    sys::page_size()
}

/// Returns the smallest value greater or equal to `val` that is a multiple of `alignment`. Returns
/// zero on overflow.
///
/// You may use this together with [`page_size`] to align your regions for committing/decommitting.
///
/// `alignment` must be a power of two (which implies that it must be non-zero).
#[inline(always)]
#[must_use]
pub const fn align_up(val: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());

    val.wrapping_add(alignment - 1) & !(alignment - 1)
}

/// Returns the largest value smaller or equal to `val` that is a multiple of `alignment`.
///
/// You may use this together with [`page_size`] to align your regions for committing/decommitting.
///
/// `alignment` must be a power of two (which implies that it must be non-zero).
#[inline(always)]
#[must_use]
pub const fn align_down(val: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());

    val & !(alignment - 1)
}

fn is_aligned(val: usize, alignment: usize) -> bool {
    debug_assert!(alignment.is_power_of_two());

    val & (alignment - 1) == 0
}

/// The type returned by the various [`Allocation`] methods.
pub type Result<T, E = Error> = ::core::result::Result<T, E>;

/// Represents an OS error that can be returned by the various [`Allocation`] methods.
#[derive(Debug)]
pub struct Error {
    code: i32,
}

impl Error {
    /// Returns the OS error that this error represents.
    #[inline]
    #[must_use]
    pub fn as_raw_os_error(&self) -> i32 {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        sys::format_error(self.code, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(unix)]
mod unix {
    #![allow(non_camel_case_types)]

    use super::{without_provenance_mut, Result};
    use core::{
        ffi::{c_char, c_int, c_void, CStr},
        fmt,
        ptr::{self, NonNull},
        str,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[derive(Debug)]
    pub struct Allocation {
        ptr: NonNull<c_void>,
        size: usize,
    }

    // SAFETY: It is safe to send `Allocation::ptr` to another thread because the user would have
    // to use unsafe code themself by dereferencing it.
    unsafe impl Send for Allocation {}

    // SAFETY: It is safe to share `Allocation::ptr` between threads because the user would have to
    // use unsafe code themself by dereferencing it.
    unsafe impl Sync for Allocation {}

    impl Allocation {
        pub fn new(size: usize) -> Result<Self> {
            // Miri doesn't support protections other than read/write.
            #[cfg(not(miri))]
            let prot = libc::PROT_NONE;
            #[cfg(miri)]
            let prot = libc::PROT_READ | libc::PROT_WRITE;

            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

            // SAFETY: Enforced by the fact that we are passing in a null pointer as the address,
            // so that no existing mappings can be affected in any way.
            let ptr = unsafe { libc::mmap(ptr::null_mut(), size, prot, flags, -1, 0) };

            result(ptr != libc::MAP_FAILED)?;

            Ok(Allocation {
                ptr: NonNull::new(ptr).unwrap(),
                size,
            })
        }

        #[inline]
        pub const fn dangling(alignment: usize) -> Self {
            assert!(alignment.is_power_of_two());

            Allocation {
                // SAFETY: We checked that `alignment` is a power of two, which means it must be
                // non-zero.
                ptr: unsafe { NonNull::new_unchecked(without_provenance_mut(alignment).cast()) },
                size: 0,
            }
        }

        #[inline(always)]
        pub const fn ptr(&self) -> *mut c_void {
            self.ptr.as_ptr()
        }

        #[inline(always)]
        pub const fn size(&self) -> usize {
            self.size
        }

        #[cfg(not(miri))]
        pub unsafe fn commit(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected and that `ptr` is aligned
            // to the page size. As for this allocation, the only way to access it is by unsafely
            // dererencing its pointer, where the user has the responsibility to make sure that
            // that is valid.
            result(unsafe { libc::mprotect(ptr, size, libc::PROT_READ | libc::PROT_WRITE) } == 0)
        }

        #[cfg(miri)]
        pub unsafe fn commit(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            // Committing memory has no effect on the operational semantics, so there's nothing for
            // Miri to test anyway except hitting a segmentation fault which is perfectly defined
            // behavior.
            Ok(())
        }

        #[cfg(not(miri))]
        pub unsafe fn decommit(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            // God forbid this be one syscall :ferrisPensive:

            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected and that `ptr` is aligned
            // to the page size. As for this allocation, the only way to access it is by unsafely
            // dererencing its pointer, where the user has the responsibility to make sure that
            // that is valid.
            result(unsafe { libc::madvise(ptr, size, libc::MADV_DONTNEED) } == 0)?;

            // SAFETY: Same as the previous.
            result(unsafe { libc::mprotect(ptr, size, libc::PROT_NONE) } == 0)?;

            Ok(())
        }

        #[cfg(miri)]
        pub unsafe fn decommit(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            // Decommitting memory has no effect on the operational semantics, so there's nothing
            // for Miri to test anyway except hitting a segmentation fault which is perfectly
            // defined behavior.
            Ok(())
        }

        #[cfg(not(miri))]
        pub unsafe fn prefault(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected and that `ptr` is aligned
            // to the page size. This call is otherwise purely an optimization hint and can't
            // change program behavior.
            result(unsafe { libc::madvise(ptr, size, libc::MADV_WILLNEED) } == 0)
        }

        #[cfg(miri)]
        pub unsafe fn prefault(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            Ok(())
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            if self.size != 0 {
                // SAFETY: It is the responsibility of the user who is unsafely derefercing the
                // allocation's pointer to ensure that those accesses don't happen after the
                // allocation has been dropped. We know the pointer and its size is valid because
                // we allocated it.
                unsafe { libc::munmap(self.ptr(), self.size) };
            }
        }
    }

    #[inline(always)]
    pub fn page_size() -> usize {
        static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

        #[cold]
        fn page_size_slow() -> usize {
            let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) }).unwrap();
            PAGE_SIZE.store(page_size, Ordering::Relaxed);

            page_size
        }

        let cached = PAGE_SIZE.load(Ordering::Relaxed);

        if cached != 0 {
            cached
        } else {
            page_size_slow()
        }
    }

    fn result(condition: bool) -> Result<()> {
        if condition {
            Ok(())
        } else {
            Err(super::Error { code: errno() })
        }
    }

    #[cfg(not(target_os = "vxworks"))]
    fn errno() -> i32 {
        unsafe { *errno_location() as i32 }
    }

    #[cfg(target_os = "vxworks")]
    fn errno() -> i32 {
        unsafe { libc::errnoGet() as i32 }
    }

    pub fn format_error(errnum: i32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0 as c_char; 128];

        let res = unsafe {
            libc::strerror_r(errnum as c_int, buf.as_mut_ptr(), buf.len() as libc::size_t)
        };

        assert!(res >= 0, "strerror_r failure");

        let buf = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_bytes();

        let s = str::from_utf8(buf).unwrap_or_else(|err| {
            // SAFETY: The `from_utf8` call above checked that `err.valid_up_to()` bytes are valid.
            unsafe { str::from_utf8_unchecked(&buf[..err.valid_up_to()]) }
        });

        f.write_str(s)
    }

    extern "C" {
        #[cfg(not(target_os = "vxworks"))]
        #[cfg_attr(
            any(
                target_os = "linux",
                target_os = "emscripten",
                target_os = "fuchsia",
                target_os = "l4re",
                target_os = "hurd",
                target_os = "dragonfly"
            ),
            link_name = "__errno_location"
        )]
        #[cfg_attr(
            any(
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "android",
                target_os = "redox",
                target_env = "newlib"
            ),
            link_name = "__errno"
        )]
        #[cfg_attr(
            any(target_os = "solaris", target_os = "illumos"),
            link_name = "___errno"
        )]
        #[cfg_attr(target_os = "nto", link_name = "__get_errno_ptr")]
        #[cfg_attr(
            any(target_os = "freebsd", target_vendor = "apple"),
            link_name = "__error"
        )]
        #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
        #[cfg_attr(target_os = "aix", link_name = "_Errno")]
        fn errno_location() -> *mut c_int;
    }
}

#[cfg(windows)]
mod windows {
    #![allow(non_camel_case_types, non_snake_case)]

    use super::{without_provenance_mut, Result};
    use core::{
        ffi::c_void,
        fmt, mem,
        ptr::{self, NonNull},
        str,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[derive(Debug)]
    pub struct Allocation {
        ptr: NonNull<c_void>,
        size: usize,
    }

    // SAFETY: It is safe to send `Allocation::ptr` to another thread because the user would have
    // to use unsafe code themself by dereferencing it.
    unsafe impl Send for Allocation {}

    // SAFETY: It is safe to share `Allocation::ptr` between threads because the user would have to
    // use unsafe code themself by dereferencing it.
    unsafe impl Sync for Allocation {}

    impl Allocation {
        pub fn new(size: usize) -> Result<Self> {
            // Miri doesn't support protections other than read/write.
            #[cfg(not(miri))]
            let protect = PAGE_NOACCESS;
            #[cfg(miri)]
            let protect = PAGE_READWRITE;

            // SAFETY: Enforced by the fact that we are passing in a null pointer as the address,
            // so that no existing mappings can be affected in any way.
            let ptr = unsafe { VirtualAlloc(ptr::null_mut(), size, MEM_RESERVE, protect) };

            result(!ptr.is_null())?;

            Ok(Allocation {
                ptr: NonNull::new(ptr).unwrap(),
                size,
            })
        }

        #[inline]
        pub const fn dangling(alignment: usize) -> Self {
            assert!(alignment.is_power_of_two());

            Allocation {
                // SAFETY: We checked that `alignment` is a power of two, which means it must be
                // non-zero.
                ptr: unsafe { NonNull::new_unchecked(without_provenance_mut(alignment).cast()) },
                size: 0,
            }
        }

        #[inline(always)]
        pub const fn ptr(&self) -> *mut c_void {
            self.ptr.as_ptr()
        }

        #[inline(always)]
        pub const fn size(&self) -> usize {
            self.size
        }

        #[cfg(not(miri))]
        pub unsafe fn commit(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected. As for this allocation,
            // the only way to access it is by unsafely dererencing its pointer, where the user
            // has the responsibility to make sure that that is valid.
            result(!unsafe { VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) }.is_null())
        }

        #[cfg(miri)]
        pub unsafe fn commit(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            // Committing memory has no effect on the operational semantics, so there's nothing for
            // Miri to test anyway except hitting a segmentation fault which is perfectly defined
            // behavior.
            Ok(())
        }

        #[cfg(not(miri))]
        pub unsafe fn decommit(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected. As for this allocation,
            // the only way to access it is by unsafely dererencing its pointer, where the user
            // has the responsibility to make sure that that is valid.
            result(unsafe { VirtualFree(ptr, size, MEM_DECOMMIT) } != 0)
        }

        #[cfg(miri)]
        pub unsafe fn decommit(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            // Decommitting memory has no effect on the operational semantics, so there's nothing
            // for Miri to test anyway except hitting a segmentation fault which is perfectly
            // defined behavior.
            Ok(())
        }

        #[cfg(all(not(miri), not(target_vendor = "win7")))]
        pub unsafe fn prefault(&self, ptr: *mut c_void, size: usize) -> Result<()> {
            let entry = WIN32_MEMORY_RANGE_ENTRY {
                VirtualAddress: ptr,
                NumberOfBytes: size,
            };

            // SAFETY: The caller must guarantee that `ptr` and `size` are in bounds of the
            // allocation such that no other allocations can be affected. We are targeting our own
            // process, the pointer points to a valid and initialized memory location above, and
            // the size of 1 is correct as there is only one entry. This call is otherwise purely
            // an optimization hint and can't change program behavior.
            result(unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0) } != 0)
        }

        #[cfg(any(miri, target_vendor = "win7"))]
        pub unsafe fn prefault(&self, _ptr: *mut c_void, _size: usize) -> Result<()> {
            Ok(())
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            if self.size != 0 {
                // SAFETY: It is the responsibility of the user who is unsafely derefercing the
                // allocation's pointer to ensure that those accesses don't happen after the
                // allocation has been dropped. We know that the pointer is valid because we
                // allocated it, and we are passing in 0 as the size as required for `MEM_RELEASE`.
                unsafe { VirtualFree(self.ptr(), 0, MEM_RELEASE) };
            }
        }
    }

    #[inline(always)]
    pub fn page_size() -> usize {
        static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

        #[cold]
        fn page_size_slow() -> usize {
            // SAFETY: `SYSTEM_INFO` is composed only of primitive types.
            let mut system_info = unsafe { mem::zeroed() };
            // SAFETY: The pointer points to a valid memory location above.
            unsafe { GetSystemInfo(&mut system_info) };
            let page_size = usize::try_from(system_info.dwPageSize).unwrap();
            PAGE_SIZE.store(page_size, Ordering::Relaxed);

            page_size
        }

        let cached = PAGE_SIZE.load(Ordering::Relaxed);

        if cached != 0 {
            cached
        } else {
            page_size_slow()
        }
    }

    fn result(condition: bool) -> Result<()> {
        if condition {
            Ok(())
        } else {
            Err(super::Error { code: errno() })
        }
    }

    fn errno() -> i32 {
        unsafe { GetLastError() as i32 }
    }

    pub fn format_error(mut errnum: i32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u16; 2048];
        let mut module = ptr::null_mut();
        let mut flags = 0;

        // NTSTATUS errors may be encoded as HRESULT, which may returned from
        // GetLastError. For more information about Windows error codes, see
        // `[MS-ERREF]`: https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/0642cb2f-2075-4469-918c-4441e69c548a
        if (errnum & FACILITY_NT_BIT as i32) != 0 {
            // format according to https://support.microsoft.com/en-us/help/259693
            const NTDLL_DLL: &[u16] = &[
                'N' as _, 'T' as _, 'D' as _, 'L' as _, 'L' as _, '.' as _, 'D' as _, 'L' as _,
                'L' as _, 0,
            ];

            module = unsafe { GetModuleHandleW(NTDLL_DLL.as_ptr()) };

            if !module.is_null() {
                errnum ^= FACILITY_NT_BIT as i32;
                flags = FORMAT_MESSAGE_FROM_HMODULE;
            }
        }

        let res = unsafe {
            FormatMessageW(
                flags | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                module,
                errnum as u32,
                0,
                buf.as_mut_ptr(),
                buf.len() as u32,
                ptr::null(),
            ) as usize
        };

        if res == 0 {
            // Sometimes FormatMessageW can fail e.g., system doesn't like 0 as langId,
            let fm_err = errno();
            return write!(
                f,
                "OS Error {errnum} (FormatMessageW() returned error {fm_err})",
            );
        }

        let mut output_len = 0;
        let mut output = [0u8; 2048];

        for c in char::decode_utf16(buf[..res].iter().copied()) {
            let Ok(c) = c else {
                return write!(
                    f,
                    "OS Error {errnum} (FormatMessageW() returned invalid UTF-16)",
                );
            };

            let len = c.len_utf8();

            if len > output.len() - output_len {
                break;
            }

            c.encode_utf8(&mut output[output_len..]);
            output_len += len;
        }

        // SAFETY: The `encode_utf8` calls above were used to encode valid UTF-8.
        let s = unsafe { str::from_utf8_unchecked(&output[..output_len]) };

        f.write_str(s)
    }

    windows_targets::link!("kernel32.dll" "system" fn GetSystemInfo(
        lpSystemInfo: *mut SYSTEM_INFO,
    ));

    windows_targets::link!("kernel32.dll" "system" fn VirtualAlloc(
        lpAddress: *mut c_void,
        dwSize: usize,
        flAllocationType: u32,
        flProtect: u32,
    ) -> *mut c_void);

    windows_targets::link!("kernel32.dll" "system" fn VirtualFree(
        lpAddress: *mut c_void,
        dwSize: usize,
        dwFreeType: u32,
    ) -> i32);

    #[cfg(not(target_vendor = "win7"))]
    windows_targets::link!("kernel32.dll" "system" fn GetCurrentProcess() -> HANDLE);

    #[cfg(not(target_vendor = "win7"))]
    windows_targets::link!("kernel32.dll" "system" fn PrefetchVirtualMemory(
        hProcess: HANDLE,
        NumberOfEntries: usize,
        VirtualAddresses: *const WIN32_MEMORY_RANGE_ENTRY,
        Flags: u32,
    ) -> i32);

    windows_targets::link!("kernel32.dll" "system" fn GetLastError() -> u32);

    windows_targets::link!("kernel32.dll" "system" fn FormatMessageW(
        dwFlags: u32,
        lpSource: *const c_void,
        dwMessageId: u32,
        dwLanguageId: u32,
        lpBuffer: *mut u16,
        nSize: u32,
        arguments: *const *const i8,
    ) -> u32);

    windows_targets::link!("kernel32.dll" "system" fn GetModuleHandleW(
        lpModuleName: *const u16,
    ) -> HMODULE);

    #[repr(C)]
    struct SYSTEM_INFO {
        wProcessorArchitecture: u16,
        wReserved: u16,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut c_void,
        lpMaximumApplicationAddress: *mut c_void,
        dwActiveProcessorMask: usize,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    const MEM_COMMIT: u32 = 1 << 12;
    const MEM_RESERVE: u32 = 1 << 13;
    const MEM_DECOMMIT: u32 = 1 << 14;
    const MEM_RELEASE: u32 = 1 << 15;

    const PAGE_NOACCESS: u32 = 1 << 0;
    const PAGE_READWRITE: u32 = 1 << 2;

    #[cfg(not(target_vendor = "win7"))]
    type HANDLE = isize;

    #[cfg(not(target_vendor = "win7"))]
    #[repr(C)]
    struct WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: *mut c_void,
        NumberOfBytes: usize,
    }

    const FACILITY_NT_BIT: u32 = 1 << 28;

    const FORMAT_MESSAGE_FROM_HMODULE: u32 = 1 << 11;
    const FORMAT_MESSAGE_FROM_SYSTEM: u32 = 1 << 12;
    const FORMAT_MESSAGE_IGNORE_INSERTS: u32 = 1 << 9;

    type HMODULE = *mut c_void;
}

// TODO: Replace this with `<*const u8>::addr` once we release a breaking version.
#[allow(clippy::transmutes_expressible_as_ptr_casts)]
fn addr(ptr: *const u8) -> usize {
    // SAFETY: `*const u8` and `usize` have the same layout.
    unsafe { mem::transmute::<*const u8, usize>(ptr) }
}

// TODO: Replace this with `ptr::without_provenance_mut` once we release a breaking version.
#[allow(integer_to_ptr_transmutes, clippy::useless_transmute)]
const fn without_provenance_mut(addr: usize) -> *mut u8 {
    // SAFETY: `usize` and `*mut u8` have the same layout.
    unsafe { mem::transmute::<usize, *mut u8>(addr) }
}
