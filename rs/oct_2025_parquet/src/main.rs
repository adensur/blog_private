use std::{
    fs::File,
    io::{BufWriter, Write},
};

pub fn encode_varint_u64(mut x: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let mut byte = (x & 0x7F) as u8;
        x >>= 7;
        if x != 0 {
            byte |= 0x80;
        } // set continuation bit
        out.push(byte);
        if x == 0 {
            break;
        }
    }
    out
}

fn main() {
    let file = File::create("myfile").unwrap();
    let mut writer = BufWriter::new(file);
    let x = 234;
    let b = encode_varint_u64(x);
    x.to_le_be
    writer.write_all(&b).unwrap();
    writer.flush().unwrap();
    println!("Wrote to myfile");
}
