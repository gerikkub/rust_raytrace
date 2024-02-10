
#[derive(Clone, Debug)]
pub struct BitSet {
    bits: Vec<u64>,
    words: Vec<u64>,
    max: usize,
    bit_count: i64
}

#[derive(Debug)]
pub struct BitSetIterator<'a> {
    bitset: &'a BitSet,
    curr_word: u64,
    iter_word: usize,
    steps: usize
}

impl BitSet {
    #[allow(dead_code)]
    pub fn new(maxval: usize) -> BitSet {
        let max_aligned = maxval.next_multiple_of(64*64);
        BitSet {
            bits: vec![0; max_aligned/64],
            words: vec![0; max_aligned/(64*64)],
            max: maxval,
            bit_count: 0
        }
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, val: u64) {
        let word = (val / 64) as usize;
        let bit = val % 64;
        let preword = self.bits[word];
        self.bits[word] |= 1 << bit;
        if preword != self.bits[word] {
            self.bit_count += 1;
        }

        let word_word = word / 64;
        let word_bit = word % 64;
        self.words[word_word]  |= 1 << word_bit;
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, val: u64) {
        let word = (val / 64) as usize;
        let bit = val % 64;
        let preword = self.bits[word];
        self.bits[word] &= !(1 << bit);
        if preword != self.bits[word] {
            self.bit_count -= 1;
        }

        if self.bits[word] == 0 {
            let word_word = word / 64;
            let word_bit = word % 64;
            self.words[word_word]  &= !(1 << word_bit);
        }

    }

    #[allow(dead_code)]
    pub fn iter<'a>(&'a self) -> BitSetIterator<'a> {
        assert!(self.bit_count >= 0);
        BitSetIterator {
            bitset: &self,
            curr_word: self.bits[0],
            iter_word: 0,
            steps: 0
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        assert!(self.bit_count >= 0);
        self.bit_count as usize
    }

    #[allow(dead_code)]
    pub fn update_bitcount(&mut self) {
        self.bit_count = 0;
        for w in &self.bits {
            self.bit_count += w.count_ones() as i64;
        }
    }

    #[allow(dead_code)]
    pub fn orwith(&mut self, o: &BitSet) {

        assert!(self.max >= o.max);

        if o.bit_count == 0 {
            return;
        }

        for idx in 0..o.words.len() {
            self.words[idx] |= o.words[idx];
        }

        for idx in 0..o.bits.len() {
            self.bits[idx] |= o.bits[idx];
        }

        self.bit_count = -1;
    }
}

impl ExactSizeIterator for BitSetIterator<'_> {

}

impl Iterator for BitSetIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {

        if self.bitset.bit_count == 0 {
            return None;
        }

        if self.curr_word == 0 {

            loop {
                self.iter_word += 1;
                if self.iter_word >= self.bitset.bits.len() {
                    return None;
                }
                let iter_word_word = self.iter_word / 64;
                let iter_word_bit = self.iter_word % 64;

                if self.bitset.words[iter_word_word] & (1 << iter_word_bit) != 0{
                    break;
                }
            }

            self.curr_word = self.bitset.bits[self.iter_word];
            assert!(self.curr_word != 0);
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
        assert!(self.bitset.bit_count >= 0);
        let n = self.bitset.bit_count as usize - self.steps;
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

        println!("{:?}", b);

        assert_eq!(b.len(), 5);
        let mut b_iter = b.iter();
        assert_eq!(b_iter.next().unwrap(), 50);
        assert_eq!(b_iter.next().unwrap(), 51);
        assert_eq!(b_iter.next().unwrap(), 75);
        assert_eq!(b_iter.next().unwrap(), 77);
        assert_eq!(b_iter.next().unwrap(), 99);
        assert_eq!(b_iter.next().is_none(), true);

        b.remove(75);
        b.insert(0);

        println!("{:?}", b);
        assert_eq!(b.len(), 5);
        let mut b2_iter = b.iter();
        assert_eq!(b2_iter.next().unwrap(), 0);
        assert_eq!(b2_iter.next().unwrap(), 50);
        assert_eq!(b2_iter.next().unwrap(), 51);
        assert_eq!(b2_iter.next().unwrap(), 77);
        assert_eq!(b2_iter.next().unwrap(), 99);
    }
}