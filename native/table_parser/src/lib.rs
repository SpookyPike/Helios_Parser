//! Fast HELIOS LOG field-table block parser.
//!
//! HELIOS writes one field table per snapshot as fixed rows of ASCII numbers.
//! `helios_parser.parser._parse_field_table_rows` used to hand the whole block
//! to `np.fromstring`, which runs at roughly 146 MB/s on this text, and filtered
//! the lines in Python first. This crate does both in one pass and fills a
//! row-major `f64` matrix the caller already owns.
//!
//! Contract (mirrored by `helios_parser/native_table.py`):
//!
//! * returns the number of rows written to `out`
//! * `-1` row width mismatch, `-2` unparseable token, `-3` bad arguments
//! * any negative return means "use the Python path"
//! * a line whose first token is not an integer is skipped, matching the Python
//!   filter; zone index 0 is skipped; rows outside `1..=n_zones` are not written
//! * float parsing is correctly rounded: Eisel-Lemire through `fast-float`, a
//!   Clinger exact fast path for the common shape, and `str::parse` for wide
//!   exponents. The Fortran forms `1.234-123` and `1.0D+00` parse natively.
//!
//! The function is pure: no globals, no allocation, no I/O. It is safe to call
//! from any thread.

#![allow(clippy::missing_safety_doc)]

/// Powers of ten that are exactly representable in `f64`, for the Clinger fast
/// path. `m * 10^e` with both factors exact rounds once, so the result equals
/// the correctly rounded decimal value.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

#[inline(always)]
fn is_space(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\r'
}

/// Parse the Fortran / HELIOS numeric forms `fast-float` rejects: `1.234-123`
/// (exponent without `E`) and `1.0D+00`. Wide exponents fall through to the
/// correctly rounded reference parser.
fn parse_f64_fallback(tok: &[u8]) -> Option<f64> {
    let mut mant: u64 = 0;
    let mut digits = 0u32;
    let mut seen_dot = false;
    let mut exp10: i32 = 0;
    let mut p = 0usize;
    let mut neg = false;
    if p < tok.len() && (tok[p] == b'+' || tok[p] == b'-') {
        neg = tok[p] == b'-';
        p += 1;
    }
    while p < tok.len() {
        let b = tok[p];
        if b.is_ascii_digit() {
            if digits < 19 {
                mant = mant * 10 + (b - b'0') as u64;
                if seen_dot {
                    exp10 -= 1;
                }
            } else if !seen_dot {
                exp10 += 1;
            }
            digits += 1;
            p += 1;
        } else if b == b'.' && !seen_dot {
            seen_dot = true;
            p += 1;
        } else {
            break;
        }
    }
    if digits == 0 {
        return None;
    }
    let mant_end = p;
    let mut exp_sign_at: Option<usize> = None;
    if p < tok.len() {
        let b = tok[p];
        if b == b'e' || b == b'E' || b == b'd' || b == b'D' {
            p += 1;
            let mut eneg = false;
            if p < tok.len() && (tok[p] == b'+' || tok[p] == b'-') {
                eneg = tok[p] == b'-';
                exp_sign_at = Some(p);
                p += 1;
            }
            let mut e: i32 = 0;
            let mut ed = 0;
            while p < tok.len() && tok[p].is_ascii_digit() {
                e = e.saturating_mul(10).saturating_add((tok[p] - b'0') as i32);
                ed += 1;
                p += 1;
            }
            if ed == 0 {
                return None;
            }
            exp10 += if eneg { -e } else { e };
        } else if seen_dot && (b == b'+' || b == b'-') {
            exp_sign_at = Some(p);
            let eneg = b == b'-';
            let mut q = p + 1;
            let mut e: i32 = 0;
            let mut ed = 0;
            while q < tok.len() && tok[q].is_ascii_digit() {
                e = e.saturating_mul(10).saturating_add((tok[q] - b'0') as i32);
                ed += 1;
                q += 1;
            }
            if !(2..=3).contains(&ed) {
                return None;
            }
            exp10 += if eneg { -e } else { e };
            p = q;
        }
    }
    if p != tok.len() {
        return None;
    }
    if mant <= (1u64 << 53) && exp10.abs() <= 22 {
        let m = mant as f64;
        let v = if exp10 >= 0 {
            m * POW10[exp10 as usize]
        } else {
            m / POW10[(-exp10) as usize]
        };
        return Some(if neg { -v } else { v });
    }
    // Wide exponent: normalise into `mantissa e sign digits` and use the
    // correctly rounded reference parser.
    let mut buf = [0u8; 64];
    let mut len = mant_end;
    if len >= buf.len() - 5 {
        return None;
    }
    buf[..mant_end].copy_from_slice(&tok[..mant_end]);
    buf[len] = b'e';
    len += 1;
    match exp_sign_at {
        Some(at) => {
            for &b in &tok[at..] {
                if len >= buf.len() {
                    return None;
                }
                buf[len] = b;
                len += 1;
            }
        }
        None => {
            buf[len] = b'+';
            len += 1;
        }
    }
    std::str::from_utf8(&buf[..len]).ok()?.parse::<f64>().ok()
}

#[inline(always)]
fn parse_value(tok: &[u8]) -> Option<f64> {
    match fast_float::parse::<f64, &[u8]>(tok) {
        Ok(v) => Some(v),
        Err(_) => parse_f64_fallback(tok),
    }
}

/// Parse a block of table lines into a preallocated row-major matrix.
///
/// `out` must hold `n_zones * n_cols` `f64` values; the caller pre-fills it
/// with NaN so missing rows stay visible. See the module documentation for the
/// return codes and the skipped-line rules.
#[no_mangle]
pub unsafe extern "C" fn parse_table_block(
    text: *const u8,
    len: usize,
    n_zones: u32,
    n_cols: u32,
    out: *mut f64,
) -> i64 {
    if text.is_null() || out.is_null() || n_cols == 0 {
        return -3;
    }
    let s = std::slice::from_raw_parts(text, len);
    let rows = n_zones as usize;
    let cols = n_cols as usize;
    let out = std::slice::from_raw_parts_mut(out, rows * cols);
    let mut parsed: i64 = 0;
    let mut line_start = 0usize;
    while line_start <= s.len() {
        let line_end = match memchr::memchr(b'\n', &s[line_start..]) {
            Some(k) => line_start + k,
            None => s.len(),
        };
        let mut i = line_start;
        while i < line_end && is_space(s[i]) {
            i += 1;
        }
        if i < line_end {
            let mut neg = false;
            if s[i] == b'+' || s[i] == b'-' {
                neg = s[i] == b'-';
                i += 1;
            }
            let digit_start = i;
            let mut zone: i64 = 0;
            while i < line_end && s[i].is_ascii_digit() {
                zone = zone * 10 + (s[i] - b'0') as i64;
                i += 1;
            }
            // Not an integer zone index: the Python filter drops the line.
            if i == digit_start || (i < line_end && !is_space(s[i])) {
                if line_end >= s.len() {
                    break;
                }
                line_start = line_end + 1;
                continue;
            }
            if neg {
                zone = -zone;
            }
            if zone != 0 {
                let in_range = zone >= 1 && zone <= rows as i64;
                let row_base = if in_range {
                    (zone as usize - 1) * cols
                } else {
                    0
                };
                let mut c = 0usize;
                while c < cols {
                    while i < line_end && is_space(s[i]) {
                        i += 1;
                    }
                    if i >= line_end {
                        return -1;
                    }
                    let tok_start = i;
                    let mut vneg = false;
                    if s[i] == b'+' || s[i] == b'-' {
                        vneg = s[i] == b'-';
                        i += 1;
                    }
                    let mut mant: u64 = 0;
                    let mut ndig = 0u32;
                    let mut dot = false;
                    let mut exp10: i32 = 0;
                    while i < line_end {
                        let b = s[i];
                        if b.is_ascii_digit() {
                            if ndig < 19 {
                                mant = mant * 10 + (b - b'0') as u64;
                                if dot {
                                    exp10 -= 1;
                                }
                            } else if !dot {
                                exp10 += 1;
                            }
                            ndig += 1;
                            i += 1;
                        } else if b == b'.' && !dot {
                            dot = true;
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    if ndig == 0 {
                        return -2;
                    }
                    let mut simple = true;
                    if i < line_end {
                        let b = s[i];
                        if b == b'e' || b == b'E' || b == b'd' || b == b'D' {
                            i += 1;
                            let mut eneg = false;
                            if i < line_end && (s[i] == b'+' || s[i] == b'-') {
                                eneg = s[i] == b'-';
                                i += 1;
                            }
                            let mut e: i32 = 0;
                            let mut ed = 0;
                            while i < line_end && s[i].is_ascii_digit() {
                                e = e.saturating_mul(10).saturating_add((s[i] - b'0') as i32);
                                ed += 1;
                                i += 1;
                            }
                            if ed == 0 {
                                return -2;
                            }
                            exp10 += if eneg { -e } else { e };
                        } else if dot && (b == b'+' || b == b'-') {
                            let eneg = b == b'-';
                            let mut q = i + 1;
                            let mut e: i32 = 0;
                            let mut ed = 0;
                            while q < line_end && s[q].is_ascii_digit() {
                                e = e.saturating_mul(10).saturating_add((s[q] - b'0') as i32);
                                ed += 1;
                                q += 1;
                            }
                            if (2..=3).contains(&ed) {
                                exp10 += if eneg { -e } else { e };
                                i = q;
                            } else {
                                simple = false;
                            }
                        }
                        if simple && i < line_end && !is_space(s[i]) {
                            simple = false;
                        }
                    }
                    let value = if simple && mant <= (1u64 << 53) && exp10.abs() <= 22 {
                        let m = mant as f64;
                        let v = if exp10 >= 0 {
                            m * POW10[exp10 as usize]
                        } else {
                            m / POW10[(-exp10) as usize]
                        };
                        if vneg {
                            -v
                        } else {
                            v
                        }
                    } else {
                        // Rare: re-scan the token and use the reference parser.
                        let mut q = tok_start;
                        while q < line_end && !is_space(s[q]) {
                            q += 1;
                        }
                        match parse_value(&s[tok_start..q]) {
                            Some(v) => v,
                            None => return -2,
                        }
                    };
                    if in_range {
                        out[row_base + c] = value;
                    }
                    c += 1;
                }
                while i < line_end && is_space(s[i]) {
                    i += 1;
                }
                if i < line_end {
                    return -1;
                }
                if in_range {
                    parsed += 1;
                }
            }
        }
        if line_end >= s.len() {
            break;
        }
        line_start = line_end + 1;
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, zones: u32, cols: u32) -> (i64, Vec<f64>) {
        let mut out = vec![f64::NAN; (zones * cols) as usize];
        let rc =
            unsafe { parse_table_block(text.as_ptr(), text.len(), zones, cols, out.as_mut_ptr()) };
        (rc, out)
    }

    #[test]
    fn parses_rows_and_skips_zone_zero() {
        let text = "     0  9.9E+99\n     1  1.000E+00  2.000E+00\n     2  3.000E+00  4.000E+00\n";
        let (rc, out) = run(text, 2, 2);
        assert_eq!(rc, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn skips_non_integer_lines() {
        let text = "header text\n   # comment\n     1  1.5  2.5\n";
        let (rc, out) = run(text, 1, 2);
        assert_eq!(rc, 1);
        assert_eq!(out, vec![1.5, 2.5]);
    }

    #[test]
    fn rejects_width_mismatch() {
        let text = "     1  1.5\n     2  1.5  2.5\n";
        assert_eq!(run(text, 2, 2).0, -1);
    }

    #[test]
    fn parses_fortran_forms() {
        let text = "     1  1.234-123  1.0D+00  1.950E-26\n";
        let (rc, out) = run(text, 1, 3);
        assert_eq!(rc, 1);
        assert_eq!(out[0], "1.234E-123".parse::<f64>().unwrap());
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], "1.950E-26".parse::<f64>().unwrap());
    }

    #[test]
    fn drops_out_of_range_zones() {
        let text = "     1  1.0\n     7  2.0\n";
        let (rc, out) = run(text, 2, 1);
        assert_eq!(rc, 1);
        assert_eq!(out[0], 1.0);
        assert!(out[1].is_nan());
    }

    #[test]
    fn parses_crlf_rows() {
        let text = "     1  1.5  2.5\r\n     2  3.5  4.5\r\n";
        let (rc, out) = run(text, 2, 2);
        assert_eq!(rc, 2);
        assert_eq!(out, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn rejects_bad_arguments() {
        let mut out = [0.0f64; 4];
        let text = "1 2";
        let rc = unsafe { parse_table_block(text.as_ptr(), text.len(), 2, 0, out.as_mut_ptr()) };
        assert_eq!(rc, -3);
    }
}
