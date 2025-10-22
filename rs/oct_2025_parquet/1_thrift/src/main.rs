use std::{
    fs::File,
    io::{BufWriter, Write},
};

use parquet::{
    my_compact::{CompactWriter, write_collection_compact},
    thrift::{USERCOLLECTION, User, serialize_collection_compact},
};

fn main() {
    // Initialize a Thrift `User` struct
    let user1 = User::new(
        123,
        "Alice".to_string(),
        Some("alice@example.com".to_string()),
        Some(vec!["rust".to_string(), "parquet".to_string()]),
        Some(
            vec![("score".to_string(), 100), ("visits".to_string(), 42)]
                .into_iter()
                .collect(),
        ),
    );
    let user2 = User::new(
        456,
        "Bob".to_string(),
        None,
        Some(vec!["thrift".to_string()]),
        None,
    );

    // serialize USER_COLLECTION with two elements
    let collection = USERCOLLECTION::new(vec![user1, user2]);
    // thrift compact -> myfile2
    let coll_bytes = serialize_collection_compact(&collection);
    let file = File::create("myfile2").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(&coll_bytes).unwrap();
    writer.flush().unwrap();
    println!(
        "Wrote {} bytes of Thrift-compact USER_COLLECTION to myfile2",
        coll_bytes.len()
    );

    // our compact -> myfile
    let mut file = File::create("myfile").unwrap();
    let mut cw = CompactWriter::new(&mut file);
    write_collection_compact(&mut cw, &collection).unwrap();
    println!("Wrote bytes of Compact-binary USER_COLLECTION to myfile");
}
