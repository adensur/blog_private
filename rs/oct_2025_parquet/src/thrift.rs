// Generated thrift code as submodule; suppress warnings from generated code
#[allow(
    dead_code,
    unused_imports,
    unused_extern_crates,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::vec_box,
    clippy::wrong_self_convention
)]
pub mod user_gen {
    include!(concat!(env!("OUT_DIR"), "/user.rs"));
}

pub use user_gen::User;

use thrift::protocol::{TCompactOutputProtocol, TSerializable};
use thrift::transport::TBufferChannel;

pub fn serialize_user_compact(user: &User) -> Vec<u8> {
    let mut channel = TBufferChannel::with_capacity(0, 128);
    {
        let mut proto = TCompactOutputProtocol::new(&mut channel);
        user.write_to_out_protocol(&mut proto)
            .expect("serialize user");
    }
    channel.write_bytes()
}
