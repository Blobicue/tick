//! Codepoint flags from llama.cpp's own frozen tables (generated into
//! unicode_data.rs) — the exact categories the binary we are replacing
//! uses, so unicode-version drift is impossible by construction.

use crate::unicode_data::{RANGES_FLAGS, WS_SET};

pub const F_UNDEFINED: u16 = 0x001;
pub const F_NUMBER: u16 = 0x002;
pub const F_LETTER: u16 = 0x004;
pub const F_SEPARATOR: u16 = 0x008;
pub const F_ACCENT: u16 = 0x010;
pub const F_PUNCT: u16 = 0x020;
pub const F_SYMBOL: u16 = 0x040;
pub const F_CONTROL: u16 = 0x080;
pub const F_WHITESPACE: u16 = 0x100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flags(pub u16);

impl Flags {
    #[inline]
    pub fn letter(self) -> bool {
        self.0 & F_LETTER != 0
    }
    #[inline]
    pub fn accent(self) -> bool {
        self.0 & F_ACCENT != 0
    }
    /// letter or combining mark: the gpt-4o letter-run class
    #[inline]
    pub fn letter_ish(self) -> bool {
        self.0 & (F_LETTER | F_ACCENT) != 0
    }
    #[inline]
    pub fn number(self) -> bool {
        self.0 & F_NUMBER != 0
    }
    #[inline]
    pub fn whitespace(self) -> bool {
        self.0 & F_WHITESPACE != 0
    }
}

/// Category flags for one codepoint: binary search over the range
/// table, then OR in the explicit whitespace set.
pub fn flags_of(cpt: u32) -> Flags {
    let mut bits = 0u16;
    let mut lo = 0usize;
    let mut hi = RANGES_FLAGS.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if RANGES_FLAGS[mid].0 <= cpt {
            bits = RANGES_FLAGS[mid].1;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if WS_SET.binary_search(&cpt).is_ok() {
        bits |= F_WHITESPACE;
    }
    Flags(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert!(flags_of('A' as u32).letter());
        assert!(flags_of('a' as u32).letter());
        assert!(flags_of('3' as u32).number());
        assert!(flags_of(' ' as u32).whitespace());
        assert!(flags_of('\t' as u32).whitespace());
        assert!(flags_of(0x0301).accent()); // combining acute
        assert!(flags_of(0x4E2D).letter()); // CJK
        assert!(!flags_of(0x41041).letter()); // unassigned plane
        assert!(flags_of(0x2028).whitespace()); // line separator
    }
}
