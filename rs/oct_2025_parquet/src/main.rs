// Wrap generated code in a module so inner attributes are valid
mod user {
    include!(concat!(env!("OUT_DIR"), "/user.rs"));
}
use user::User;

use std::{
    fs::File,
    io::{BufWriter, Write},
};

use thrift::protocol::TBinaryOutputProtocol;
use thrift::protocol::TSerializable;
use thrift::transport::TBufferChannel;

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

fn zigzag_decode_i64(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn main() {
    // Initialize a Thrift `User` struct and serialize it into binary using TBinaryOutputProtocol
    let user = User {
        id: 1,
        name: "Alice".to_string(),
    };

    let mut channel = TBufferChannel::with_capacity(0, 128);
    {
        let mut proto = TBinaryOutputProtocol::new(&mut channel, true);
        user.write_to_out_protocol(&mut proto)
            .expect("serialize user");
    }
    let serialized = channel.write_bytes();

    let file = File::create("myfile").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(&serialized).unwrap();
    writer.flush().unwrap();
    println!(
        "Wrote {} bytes of Thrift-binary User to myfile",
        serialized.len()
    );
}
