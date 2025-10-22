use std::io::{self, Write};

use crate::thrift::{USERCOLLECTION, User};

// Compact protocol type ids (low nibble)
const TTYPE_I32: u8 = 0x05;
const TTYPE_BINARY: u8 = 0x08;
const TTYPE_LIST: u8 = 0x09;
const TTYPE_MAP: u8 = 0x0B;
const TTYPE_I64: u8 = 0x06;
const TTYPE_STRUCT: u8 = 0x0C;
const TTYPE_STOP: u8 = 0x00;

#[inline]
fn zigzag_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

#[inline]
fn zigzag_i16(n: i16) -> u16 {
    ((n << 1) ^ (n >> 15)) as u16
}

pub struct CompactWriter<W: Write> {
    out: W,
    prev_field_id: i16,
}

impl<W: Write> CompactWriter<W> {
    pub fn new(out: W) -> Self {
        Self {
            out,
            prev_field_id: 0,
        }
    }

    pub fn write_stop(&mut self) -> io::Result<()> {
        self.out.write_all(&[TTYPE_STOP])
    }

    pub fn write_field_header(&mut self, field_id: i16, ttype: u8) -> io::Result<()> {
        let delta = field_id - self.prev_field_id;
        if (1..=15).contains(&delta) {
            let b = ((delta as u8) << 4) | (ttype & 0x0F);
            self.out.write_all(&[b])?;
        } else {
            self.out.write_all(&[ttype & 0x0F])?;
            self.write_varint_u32(zigzag_i16(field_id) as u32)?;
        }
        self.prev_field_id = field_id;
        Ok(())
    }

    pub fn write_i32(&mut self, v: i32) -> io::Result<()> {
        self.write_varint_u32(zigzag_i32(v))
    }

    pub fn write_string(&mut self, s: &str) -> io::Result<()> {
        let bytes = s.as_bytes();
        self.write_varint_u32(bytes.len() as u32)?;
        self.out.write_all(bytes)
    }

    pub fn write_i64(&mut self, v: i64) -> io::Result<()> {
        // zigzag then varint
        let zz = ((v << 1) ^ (v >> 63)) as u64;
        self.write_varint_u64(zz)
    }

    pub fn write_varint_u32(&mut self, mut x: u32) -> io::Result<()> {
        let mut buf = [0u8; 5];
        let mut i = 0;
        loop {
            let mut b = (x & 0x7F) as u8;
            x >>= 7;
            if x != 0 {
                b |= 0x80;
            }
            buf[i] = b;
            i += 1;
            if x == 0 {
                break;
            }
        }
        self.out.write_all(&buf[..i])
    }

    pub fn write_varint_u64(&mut self, mut x: u64) -> io::Result<()> {
        let mut buf = [0u8; 10];
        let mut i = 0;
        loop {
            let mut b = (x & 0x7F) as u8;
            x >>= 7;
            if x != 0 {
                b |= 0x80;
            }
            buf[i] = b;
            i += 1;
            if x == 0 {
                break;
            }
        }
        self.out.write_all(&buf[..i])
    }

    pub fn write_list_header(&mut self, elem_ttype: u8, len: usize) -> io::Result<()> {
        if len < 15 {
            let b = ((len as u8) << 4) | (elem_ttype & 0x0F);
            self.out.write_all(&[b])
        } else {
            let b = (0xF0) | (elem_ttype & 0x0F);
            self.out.write_all(&[b])?;
            self.write_varint_u32(len as u32)
        }
    }
}

pub fn write_user_compact<W: Write>(cw: &mut CompactWriter<W>, user: &User) -> io::Result<()> {
    // Each struct resets field id delta base
    let saved = cw.prev_field_id;
    cw.prev_field_id = 0;

    cw.write_field_header(1, TTYPE_I32)?;
    cw.write_i32(user.id)?;
    cw.write_field_header(2, TTYPE_BINARY)?;
    cw.write_string(&user.name)?;
    if let Some(email) = &user.email {
        cw.write_field_header(3, TTYPE_BINARY)?;
        cw.write_string(email)?;
    }
    if let Some(tags) = &user.tags {
        cw.write_field_header(4, TTYPE_LIST)?;
        cw.write_list_header(TTYPE_BINARY, tags.len())?;
        for t in tags {
            cw.write_string(t)?;
        }
    }
    if let Some(attrs) = &user.attributes {
        cw.write_field_header(5, TTYPE_MAP)?;
        // map header: size varint, then a byte: (key_type << 4) | value_type
        cw.write_varint_u32(attrs.len() as u32)?;
        let types = ((TTYPE_BINARY & 0x0F) << 4) | (TTYPE_I64 & 0x0F);
        cw.out.write_all(&[types])?;
        for (k, v) in attrs {
            cw.write_string(k)?;
            cw.write_i64(*v)?;
        }
    }
    let r = cw.write_stop();
    cw.prev_field_id = saved;
    r
}

pub fn write_collection_compact<W: Write>(
    cw: &mut CompactWriter<W>,
    coll: &USERCOLLECTION,
) -> io::Result<()> {
    // Outer struct reset
    let saved = cw.prev_field_id;
    cw.prev_field_id = 0;

    cw.write_field_header(1, TTYPE_LIST)?;
    let users = &coll.users;
    cw.write_list_header(TTYPE_STRUCT, users.len())?;
    for u in users {
        write_user_compact(cw, u)?;
    }
    let r = cw.write_stop();
    cw.prev_field_id = saved;
    r
}
