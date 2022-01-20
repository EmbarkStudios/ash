use crate::vk;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::mem::{size_of, align_of};
use std::mem::MaybeUninit;
use std::os::raw::c_void;
use std::{io, slice};

/// [`Align`] handles dynamic alignment. The is useful for dynamic uniform buffers where
/// the alignment might be different. For example a 4x4 f32 matrix has a size of 64 bytes
/// but the min alignment for a dynamic uniform buffer might be 256 bytes. A slice of `&[Mat4x4<f32>]`
/// has a memory layout of `[[64 bytes], [64 bytes], [64 bytes]]`, but it might need to have a memory
/// layout of `[[256 bytes], [256 bytes], [256 bytes]]`.
/// [`Align::copy_from_slice`] will copy a slice of `&[T]` directly into the host memory without
/// an additional allocation and with the correct alignment.
#[derive(Debug, Clone)]
pub struct Align<T> {
    ptr: *mut c_void,
    elem_size: usize,
    size: usize,
    _m: PhantomData<T>,
}

impl<T: Copy> Align<T> {
    /// # Safety
    /// - `self.ptr` *must* be valid and point to a section of memory >= `self.size`.
    pub unsafe fn copy_from_slice(&mut self, slice: &[T]) {
        if self.elem_size == size_of::<T>() {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr as *mut T, slice.len().min(self.size / self.elem_size));
        } else {
            for (i, val) in self.iter_mut().enumerate().take(slice.len()) {
                val.write(slice[i]);
            }
        }
    }
}

fn calc_padding(adr: usize, align: usize) -> usize {
    (align - adr % align) % align
}

impl<T> Align<T> {
    /// - `ptr` must be non-null and point to a valid section of memory >= `size`.
    /// - `size` must be <= `isize::MAX`
    /// - `size` must be aligned to `alignment`
    /// - `alignment` must be greater or equal to `align_of::<T>()`
    /// - `alignment` must be within usize range
    pub fn new(ptr: *mut c_void, alignment: vk::DeviceSize, size: vk::DeviceSize) -> Self {
        assert!(size <= isize::MAX as vk::DeviceSize, "size > isize::MAX");
        let alignment = alignment as usize;
        let size = size as usize;
        let padding = calc_padding(size_of::<T>(), alignment);
        let elem_size = size_of::<T>() + padding;
        assert!(calc_padding(size, alignment) == 0, "size must be aligned");
        assert!(alignment >= align_of::<T>(), "alignment must be greater or equal to align of T");
        Self {
            ptr,
            elem_size,
            size,
            _m: PhantomData,
        }
    }

    /// # Safety
    /// - `self.ptr` *must* be valid and point to a section of memory >= `self.size` until the iterator is dropped.
    pub unsafe fn iter_mut(&mut self) -> AlignIter<T> {
        AlignIter {
            current: 0,
            align: self,
        }
    }
}

#[derive(Debug)]
pub struct AlignIter<'a, T: 'a> {
    align: &'a mut Align<T>,
    current: usize,
}

impl<'a, T: 'a> Iterator for AlignIter<'a, T> {
    type Item = &'a mut MaybeUninit<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.align.size {
            return None;
        }
        // SAFETY: `align.size` is guaranteed to be <= isize::MAX and the above garantees `self.current` stays <= align.size.
        let ptr = unsafe { (self.align.ptr as *mut u8).add(self.current) } as *mut T;
        self.current += self.align.elem_size;
        // SAFETY: MaybeUninit has same layout as T, and in order to obtain an `AlignIter` we guarantee that `ptr` is still valid
        // until it is dropped.
        let maybe_uninit: &'a mut MaybeUninit<T> = unsafe { core::mem::transmute(ptr) };
        Some(maybe_uninit)
    }
}

/// Decode SPIR-V from bytes.
///
/// This function handles SPIR-V of arbitrary endianness gracefully, and returns correctly aligned
/// storage.
///
/// # Examples
/// ```no_run
/// // Decode SPIR-V from a file
/// let mut file = std::fs::File::open("/path/to/shader.spv").unwrap();
/// let words = ash::util::read_spv(&mut file).unwrap();
/// ```
/// ```
/// // Decode SPIR-V from memory
/// const SPIRV: &[u8] = &[
///     // ...
/// #   0x03, 0x02, 0x23, 0x07,
/// ];
/// let words = ash::util::read_spv(&mut std::io::Cursor::new(&SPIRV[..])).unwrap();
/// ```
pub fn read_spv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
    let size = x.seek(io::SeekFrom::End(0))?;
    if size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length not divisible by 4",
        ));
    }
    if size > usize::max_value() as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
    }
    let words = (size / 4) as usize;
    // https://github.com/MaikKlein/ash/issues/354:
    // Zero-initialize the result to prevent read_exact from possibly
    // reading uninitialized memory.
    let mut result = vec![0u32; words];
    x.seek(io::SeekFrom::Start(0))?;
    x.read_exact(unsafe { slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, words * 4) })?;
    const MAGIC_NUMBER: u32 = 0x0723_0203;
    if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
        for word in &mut result {
            *word = word.swap_bytes();
        }
    }
    if result.is_empty() || result[0] != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input missing SPIR-V magic number",
        ));
    }
    Ok(result)
}
