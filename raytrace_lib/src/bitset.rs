
use core::cmp;
use std::ops;

#[derive(Clone, Debug)]
pub struct BitSet {
    bits: Vec<u64>,
    max: usize,
    bit_count: usize
}

#[derive(Debug)]
pub struct BitSetIterator<'a> {
    bitset: &'a BitSet,
    curr_word: u64,
    iter_word: usize,
    steps: usize
}

impl BitSet {
    pub fn new(maxval: usize) -> BitSet {
        BitSet {
            bits: vec![0; (maxval + 63)/64],
            max: maxval,
            bit_count: 0
        }
    }

    pub fn insert(&mut self, val: u64) {
        let word = (val / 64) as usize;
        let bit = val % 64;
        let preword = self.bits[word];
        self.bits[word] |= 1 << bit;
        if preword != self.bits[word] {
            self.bit_count += 1;
        }
    }

    pub fn remove(&mut self, val: u64) {
        let word = (val / 64) as usize;
        let bit = val % 64;
        let preword = self.bits[word];
        self.bits[word] &= !(1 << bit);
        if preword != self.bits[word] {
            self.bit_count -= 1;
        }
    }

    pub fn iter<'a>(&'a self) -> BitSetIterator<'a> {
        BitSetIterator {
            bitset: &self,
            curr_word: self.bits[0],
            iter_word: 0,
            steps: 0
        }
    }

    pub fn len(&self) -> usize {
        self.bit_count
    }
}

impl ExactSizeIterator for BitSetIterator<'_> {

}

impl Iterator for BitSetIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_word == 0 {
            self.iter_word += 1;

            for w in &self.bitset.bits[self.iter_word..] {
                if *w != 0 {
                    self.curr_word = *w;
                    break;
                } else {
                    self.iter_word += 1;
                }
            }
        }

        if self.curr_word != 0 {
            let ret = self.curr_word.trailing_zeros() as u64 + (self.iter_word as u64) * 64;
            self.curr_word &= !(1 << self.curr_word.trailing_zeros());
            self.steps += 1;
            Some(ret)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bitset.bit_count - self.steps;
        (n, Some(n))
    }
}

impl Extend<u64> for BitSet {

    fn extend<T: IntoIterator<Item=u64>>(&mut self, iter: T) {
        for i in iter {
            self.insert(i);
        }
    }
}

impl Extend<usize> for BitSet {

    fn extend<T: IntoIterator<Item=usize>>(&mut self, iter: T) {
        for i in iter {
            self.insert(i as u64);
        }
    }
}

impl<'a> Extend<&'a usize> for BitSet {

    fn extend<T: IntoIterator<Item=&'a usize>>(&mut self, iter: T) {
        for i in iter {
            self.insert(*i as u64);
        }
    }
}

// TODO: Need to update with count
// impl ops::BitOr for BitSet {
//     type Output = Self;

//     fn bitor(self, rhs: Self) -> BitSet {
//         let mut n = BitSet::new(cmp::max(self.max, rhs.max));
//         let (sm_set, bg_set) = if self.max < rhs.max {
//             (&self, &rhs) 
//         } else {
//             (&rhs, &self) 
//         };

//         for idx in 0..sm_set.max {
//             n.bits[idx] = sm_set.bits[idx] | bg_set.bits[idx];
//         }

//         for idx in sm_set.max..bg_set.max {
//             n.bits[idx] = bg_set.bits[idx];
//         }
//         n
//     }
// }

#[cfg(test)]
mod tests {
    use crate::bitset::BitSet;

    #[test]
    fn basic() {
        let mut b = BitSet::new(100);
        b.insert(2);
        b.insert(50);
        b.insert(51);
        b.insert(75);
        b.insert(76);
        b.insert(77);
        b.insert(99);
        b.remove(2);
        b.remove(76);
        let mut b_iter = b.iter();
        assert_eq!(b_iter.next().unwrap(), 50);
        assert_eq!(b_iter.next().unwrap(), 51);
        assert_eq!(b_iter.next().unwrap(), 75);
        assert_eq!(b_iter.next().unwrap(), 77);
        assert_eq!(b_iter.next().unwrap(), 99);
        assert_eq!(b_iter.next().is_none(), true);

        b.remove(75);
        b.insert(0);

        let mut b2_iter = b.iter();
        assert_eq!(b2_iter.next().unwrap(), 0);
        assert_eq!(b2_iter.next().unwrap(), 50);
        assert_eq!(b2_iter.next().unwrap(), 51);
        assert_eq!(b2_iter.next().unwrap(), 77);
        assert_eq!(b2_iter.next().unwrap(), 99);
    }
}