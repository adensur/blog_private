use std::{
    fs::File,
    io::{BufWriter, Write},
};

use parquet::{
    my_compact::{CompactWriter, write_user_compact},
    thrift::{User, serialize_user_compact},
};

fn main() {
    // Initialize a Thrift `User` struct and serialize it into binary using TBinaryOutputProtocol
    let user = User {
        id: 123,
        // name: "Alice".to_string(),
    };

    // Saving thrift compact version (for reference)
    let compact = serialize_user_compact(&user);
    let file = File::create("myfile2").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(&compact).unwrap();
    writer.flush().unwrap();
    println!(
        "Wrote {} bytes of Thrift-compact User to myfile2",
        compact.len()
    );

    // saving custom compact version
    let mut file = File::create("myfile").unwrap();
    let mut cw = CompactWriter::new(&mut file);
    write_user_compact(&mut cw, &user).unwrap();
    println!("Wrote bytes of Compact-binary User to myfile");

    // done
}
