#![feature(portable_simd)]
#![feature(slice_pattern)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(non_null_convenience)]

pub mod raytrace;
pub mod progress;
pub mod obj_parser;
mod bitset;
mod bumpalloc;

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
