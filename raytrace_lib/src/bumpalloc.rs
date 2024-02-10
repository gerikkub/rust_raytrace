
//use std::collections::HashSet;
use std::cell::{UnsafeCell, RefCell};
use std::alloc::{Allocator, Layout, AllocError};
use std::ptr::NonNull;
use std::alloc;
use log::debug;

struct BumpAllocatorInner {
    mem: UnsafeCell<NonNull<[u8]>>,
    mem_size: usize,
    last_idx: usize,
    alloc_count: usize
}

pub struct BumpAllocator {
    internal: RefCell<BumpAllocatorInner>
}

impl BumpAllocator {
    #[allow(dead_code)]
    pub fn new(size: usize) -> BumpAllocator {
        BumpAllocator {
            internal: RefCell::new( BumpAllocatorInner {
                mem: UnsafeCell::new(alloc::Global.allocate_zeroed(Layout::from_size_align(size, 8).unwrap()).unwrap()),
                mem_size: size,
                last_idx:  0,
                alloc_count: 0
            })
        }
    }
}

unsafe impl Allocator for BumpAllocator {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {

        let mut internal = self.internal.borrow_mut();

        let new_start = (internal.last_idx + layout.align() - 1) & !(layout.align() - 1);
        let new_end = new_start + layout.size();

        debug!("New End: {} Size: {} Request: {:?}", new_end, internal.mem_size, layout);
        if new_end > internal.mem_size {
            return Err(AllocError);
        }

        internal.last_idx = new_end;
        internal.alloc_count += 1;

        let new_ptr = internal.mem.get_mut().as_mut_ptr().wrapping_add(new_start);
        Ok(NonNull::slice_from_raw_parts(NonNull::new(new_ptr).unwrap(), layout.size()))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {

        let mut internal = self.internal.borrow_mut();

        assert!(internal.alloc_count > 0);
        internal.alloc_count -= 1;
    }
}

impl BumpAllocator {
 
    #[allow(dead_code)]
    pub fn reset(&self) {

        let mut internal = self.internal.borrow_mut();

        assert!(internal.alloc_count == 0);

        internal.last_idx = 0;
    }

    #[allow(dead_code)]
    pub fn clear(&self) {

        let mut internal = self.internal.borrow_mut();

        assert!(internal.alloc_count == 0);

        internal.last_idx = 0;
        unsafe {
            internal.mem.get_mut().as_mut_ptr().write_bytes(0, internal.mem_size);
        }
    }

    #[allow(dead_code)]
    pub fn destroy(&self) {

        let mut internal = self.internal.borrow_mut();

        assert!(internal.alloc_count == 0);

        internal.mem_size = 0;
        unsafe {
            alloc::Global.deallocate(NonNull::new(internal.mem.get_mut().as_mut_ptr()).unwrap(), Layout::from_size_align(internal.mem_size, 8).unwrap());
        }
    }
}