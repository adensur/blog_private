### Simple example
```python
import polars as pl

data = {
    "id": [1, 2, 3, 4, 5],
}

df = pl.DataFrame(data)
df.write_parquet("data.parquet")
```
Execute this code, then run `xxd -b data.parquet` to see binary content:
```
00000000: 01010000 01000001 01010010 00110001 00010101 00000100  PAR1..
00000006: 00010101 01010000 00010101 01000000 01001100 00010101  .P.@L.
...
000001fe: 00000000 00110111 00000001 00000000 00000000 01010000  .7...P
00000204: 01000001 01010010 00110001                             AR1
```
We'll notice that:
- the file is quite big, 512 bytes. Much bigger than one would expect for 5 numbers.
- There are maigic `PAR1` bytes at the beginning and at the end - all parquet files are written like this.  

Parquet file looks like this:  
```
PAR1 # header
<ROW GROUP 1>
    <COLUMN CHUNK 1>
        <Data Page 1>
        <Data Page 2>
        ...
    <COLUMN CHUNK 2>
    ...
<ROW GROUP 2>
...
<FILE_METADATA>
<FOOTER_LENGTH>
PAR1
```
Each parquet file starts with a marker "PAR1". Schema of the file is located in the footer, so to read it, one will have to seek to the end of the file. End of file will contain same magic marker "PAR1", then a single u32 number containing footer length, then - FILE_METADATA, a struct serialized with Thrift that contains schema of the data, layout of the file, byte offsets etc.   
Let's start with schema. To understand the data, one needs a schema. Sometimes the schema can be inferred from the data itself:  
```
{
    "user_email": "user@example.com",
    "age_years": 34
}
```
Sometimes schema is not stored in the data itself, and it's up to the reader to now correct schema. This is the case for thrift/protobuf: to read a thrift binary file, we need to know schema of what we are reading.  
In case of parquet, schema is saved into the parquet file itself. You can think of a parquet file as something like a dataframe:
```
user_email  age_years
user@example.com    34
...
```
But every column might have complex elements as well: arrays, structs, arrays of structs etc.   
Schema will look like a "tree". In simple case (flat columns), it will have root and a bunch of nodes, one node per column. For nested structs, it can be a more complex tree.  
Each schema element will contain name, type, and some other stuff, as we'll see below.  
Main difference to Thrift itself is file layout organization. Dataframe is split into:
- row groups - a horizontal slice of the entire data. Imagine a dataframe with 20 columns and 10m rows. It can be saved into 10 row groups, 20 columns and 1m rows each. So, row group is like a "subslice" of the entire dataframe
- column chunks: parquet uses columnar storage, i.e., each column is stored separately, as a continuous stream of bytes. This makes it easy to select only specified columns of a dataframe (think of a dataframe with a huge "__debug__" column with some extra info. With column-wise selection, we can just skip it, effectively reading dataframe as quickly as we could if that column wasn't there), and also helps with compression (imagine a "timestamp" column: 1762950215, 1762950213, 1762950415, ...; on its own, each number is quite big, barely fits into u32; but we can save a lot of space if we store as first number + deltas: 1762950215, -2, +200, ...)
- Data pages. Each column is further split into pages. Column chunk header will have a pointer (e.g., byte offset within the file) to the first data page. Each data page header will contain that page's byte size, some metainfo, then the actual data. Metainfo will have, for example, data statistics for min/max value of the numbers, and number of values stored. This allows reading more effectively with filtering/indexing:
  - We always read column header, then 1st page header, then 2nd page header etc
  - We skip actual data page content if predicate is not met (i.e., reading all values with value > N), or if we need a specific index within this row group
We still need to scan forward all the data page headers, but not the actual data.  
### Reading parquet schema
Next, we're going to write some rust code that reads an example parquet file. Let's start with just reading schema of the file itself and every row group.  
```rust
const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

let file_path = "data.parquet";
let mut file = File::open(file_path)?;

println!("Reading Parquet file: {}", file_path);

// Step 1: Read and verify magic header
let mut header_magic = [0u8; 4];
file.read_exact(&mut header_magic)?;

if &header_magic != PARQUET_MAGIC {
    return Err(format!(
        "Invalid Parquet file: expected magic header {:?}, got {:?}",
        PARQUET_MAGIC, header_magic
    ).into());
}
println!("✓ Valid magic header: {}", String::from_utf8_lossy(&header_magic));
```
We start with creating a `File` object. Throughout the code, we are going to use a lot of direct file operations (`read_exact`, `seek` etc) just for demonstrative purposes. It's not really recommended because each such call triggers a syscall; something like `BufReader` would be preferable.  
We read and validate that the file starts with magic `PAR1` header.   
```rust
// Step 2: Read and verify magic footer
file.seek(SeekFrom::End(-4))?;
let mut footer_magic = [0u8; 4];
file.read_exact(&mut footer_magic)?;

if &footer_magic != PARQUET_MAGIC {
    return Err(format!(
        "Invalid Parquet file: expected magic footer {:?}, got {:?}",
        PARQUET_MAGIC, footer_magic
    ).into());
}
println!("✓ Valid magic footer: {}", String::from_utf8_lossy(&footer_magic));

// Step 3: Read footer length (4 bytes before the magic footer)
file.seek(SeekFrom::End(-8))?;
let footer_length = file.read_i32::<LittleEndian>()?;
println!("✓ Footer length: {} bytes", footer_length);

if footer_length <= 0 {
    return Err(format!("Invalid footer length: {}", footer_length).into());
}
```
`file.seek` jumps to a specified location. In our case, 4 bytes before the end of the file. We verify that end of file also contains `PAR1` magic header, and then read footer length - length of binary file metadata.
```rust
use thrift::protocol::TCompactInputProtocol;
use parquet_format::FileMetaData;

// Step 4: Read FileMetaData
file.seek(SeekFrom::End(-8 - footer_length as i64))?;
let mut metadata_bytes = vec![0u8; footer_length as usize];
file.read_exact(&mut metadata_bytes)?;

// Step 5: Deserialize FileMetaData using Thrift Compact Protocol
let mut cursor = std::io::Cursor::new(metadata_bytes);
let mut protocol: TCompactInputProtocol<&mut std::io::Cursor<Vec<u8>>> =
    TCompactInputProtocol::new(&mut cursor);
let file_metadata = FileMetaData::read_from_in_protocol(&mut protocol)?;
```
Next, some interesting bits. We read bytes containing FileMetaData - we know precise length by now.  
We import a struct `parquet_format::FileMetaData` that looks something like this:  
```rust
/// Description for file metadata
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FileMetaData {
  /// Version of this file *
  pub version: i32,
  /// Parquet schema for this file.  This schema contains metadata for all the columns.
  /// The schema is represented as a tree with a single root.  The nodes of the tree
  /// are flattened to a list by doing a depth-first traversal.
  /// The column metadata contains the path in the schema for that column which can be
  /// used to map columns to nodes in the schema.
  /// The first element is the root *
  pub schema: Vec<SchemaElement>,
  ...
```
This is autogenerated from official thrift schema for parquet FileMetadata. There is also a generated method `read_from_in_protocol` that decodes this struct from bytes. To view its content, I wrote a simple struct-to-json func:  
```rust
/// Convert FileMetaData to human-readable JSON with enum names
pub fn file_metadata_to_json(metadata: &FileMetaData) -> Value {
    json!({
        "version": metadata.version,
        "num_rows": metadata.num_rows,
        "created_by": metadata.created_by,
        "num_row_groups": metadata.row_groups.len(),
        "schema": metadata.schema.iter().map(|se| {
            json!({
                "type": format!("{:?}", se.type_),
                "type_length": se.type_length,
                "repetition_type": format!("{:?}", se.repetition_type),
                "name": se.name,
                "num_children": se.num_children,
                "converted_type": format!("{:?}", se.converted_type),
                "scale": se.scale,
                "precision": se.precision,
                "field_id": se.field_id,
                "logical_type": format!("{:?}", se.logical_type),
            })
        }).collect::<Vec<_>>(),
        "key_value_metadata": metadata.key_value_metadata.as_ref().map(|kv_list| {
            kv_list.iter().map(|kv| {
                json!({
                    "key": kv.key,
                    "value": kv.value,
                })
            }).collect::<Vec<_>>()
        }),
        "column_orders": metadata.column_orders.as_ref().map(|orders| {
            orders.iter().map(|order| format!("{:?}", order)).collect::<Vec<_>>()
        }),
        "encryption_algorithm": format!("{:?}", metadata.encryption_algorithm),
        "footer_signing_key_metadata": metadata.footer_signing_key_metadata.as_ref().map(hex::encode),
    })
}
```
And this is how it looks like for our toy example:  
```json
{
  "column_orders": [
    "TYPEORDER(TypeDefinedOrder)"
  ],
  "created_by": "Polars",
  "encryption_algorithm": "None",
  "footer_signing_key_metadata": null,
  "key_value_metadata": [
    {
      "key": "ARROW:schema",
      "value": "/////3cAAAAEAAAA8v///xQAAAAEAAEAAAAKAAsACAAKAAQA+P///wwAAAAIAAgAAAAEAAEAAAAEAAAA7P///zgAAAAgAAAAGAAAAAECAAAQABIABAAQABEACAAAAAwAAAAAAPT///9AAAAAAQAAAAgACQAEAAgAAgAAAGlkAA=="
    }
  ],
  "num_row_groups": 1,
  "num_rows": 5,
  "schema": [
    {
      "converted_type": "None",
      "field_id": null,
      "logical_type": "None",
      "name": "root",
      "num_children": 1,
      "precision": null,
      "repetition_type": "None",
      "scale": null,
      "type": "None",
      "type_length": null
    },
    {
      "converted_type": "None",
      "field_id": null,
      "logical_type": "None",
      "name": "id",
      "num_children": null,
      "precision": null,
      "repetition_type": "Some(Optional)",
      "scale": null,
      "type": "Some(Int64)",
      "type_length": null
    }
  ],
  "version": 1
}
```
We can see:
- total number of rows and row groups
- Schema. It will always have a root node. In our case, root has 1 child - meaning, we have a dataframe with 1 column
    - type is Int64
    - repetition_type is "Optional" (can be required, optional, repeated)
We will talk about definition and repetition levels a bit later, but in essence, it's a thing that allows flattening arbitrary nested lists, like this one: 
```
nums = [
    [1, 2, None],   # defined, three elements
    None,           # nums = null
    [],             # nums defined but empty
    [3]
]
```
by encoding repetition/definition levels first, and then a single continuous array of actual elements.  

`converted_type`/`logical_type`: legacy version and new version. These can be used to distinguish different "logical types" that look the same on disk. For example, both `[u8]` (aka bytes) and `String` (aka utf8) will have a type named "BYTE_ARRAY". But the latter will also have "STRING" logical type; bytes will have "none".  

There is also a b64-encoded ARROW_SCHEMA in key_value_metadata. In our case, it decodes to:  
```
{
  "fields": [
    {
      "data_type": "Int64",
      "metadata": {},
      "name": "id",
      "nullable": true
    }
  ],
  "metadata": {}
}
```
`key_value_metadata` means that it's not actually used by parquet reader, and we can put arbitrary stuff here. A lot of parquet readers, for both Rust and Python, rely on Arrow in-memory representation, which has its own, separate Schema concept. They save arrow schema in this metadata to help convert parquet data to arrow during reading. I hope to write a separate post about Arrow and explain all of this a bit better.   

RowGroup metadata is also part of the footer schema, but I decided to print them separately:
```
{
  "columns": [
    {
      "file_offset": 110,
      "file_path": null,
      "metadata": {
        "codec": "Zstd",
        "data_page_offset": 4,
        "dictionary_page_offset": null,
        "encodings": [
          "Plain",
          "Rle",
          "RleDictionary"
        ],
        "index_page_offset": null,
        "num_values": 5,
        "path_in_schema": [
          "id"
        ],
        "statistics": {
          "distinct_count": null,
          "max": "0400000000000000",
          "min": "0000000000000000",
          "null_count": 0
        },
        "total_compressed_size": 106,
        "total_uncompressed_size": 105,
        "type": "Int64"
      }
    }
  ],
  "num_columns": 1,
  "num_rows": 5,
  "row_group_index": 0,
  "total_byte_size": 105
}
```
`file_path`: parquet might store data accross many files in some rare cases. Wild stuff.  
`data_page_offset`, `file_offset`: byte index of start of first page, and of end of the entire chunk. Chunk end is kind of redundant in case there's no multi-file data.  
`codec` - compression used for the actual data pages.
`Plain`, `Rle`, `RleDictionary` encodings: these are just listed out, not necesserily useful. Useful one would be located in the actual data page header. 
    `Rle` stands for run-length encoding. This is useful when we have lists with a lot of repeated values. For example: `1 1 1 1 1 0 1 1 1 1` can be more compactly encoded as `"1" 4 times, "0" one time, "1" 4 times`. We'll discuss specific later
    `RleDictionary`: same as Rle, but with encoded values corresponding to dictionary index, not the final value. To get the final value - look it up in the dictionary.
`statistics` - this is another important parquet block. Statistics allow predicate pushing, i.e., quickly filtering "data with value >= N" without reading ALL the data.  

We are done with the "file schema" now, i.e., we've read a small portion of data, and understand the layout of file a bit. Next, we need to jump to the actual row groups / column chunks / data pages, and read them. 
```rust
for (rg_idx, row_group) in file_metadata.row_groups.iter().enumerate() {
        println!("\n--- Row Group {} Data Pages ---", rg_idx);

        for (col_idx, column_chunk) in row_group.columns.iter().enumerate() {
            let col_meta = column_chunk.meta_data.as_ref().unwrap();
            let path_str = col_meta.path_in_schema.join(".");

            println!("\n  Column: {} ({})", path_str, col_idx);

            // Determine starting offset (dictionary page or data page)
            let start_offset = if let Some(dict_offset) = col_meta.dictionary_page_offset {
                dict_offset
            } else {
                col_meta.data_page_offset
            };

            // Seek to the first page
            file.seek(SeekFrom::Start(start_offset as u64))?;
```
We cannot just scan a file sequentially, like we would in case of formats like .csv, because we need to be very careful about specific byte offsets. So instead, we first look into our metadata and find out specific byte offsets we need to seek to.
```rust
use parquet_format::PageHeader;
// ...
            let mut values_read = 0;
            let total_values = col_meta.num_values;
            let mut page_num = 0;
            let mut dictionary: Option<Vec<i64>> = None; // Store dictionary for this column

            // Read pages until we've consumed all values
            while values_read < total_values {
                // Read PageHeader
                let mut protocol: TCompactInputProtocol<&mut File> =
                    TCompactInputProtocol::new(&mut file);
                let page_header = PageHeader::read_from_in_protocol(&mut protocol)?;
```
We are inside our 3d nested for loop now. We iterate over row groups, column chunks, and data pages. We use thrift to read some bites from our file and convert them to PageHeader struct - another predefined struct from `parquet_format`.    

Note that when we were reading footer, we first read exact number of bytes, then decoded with Thrift. This time, we don't know exact number of bytes ahead of time; but we can trust Thrift reader to read exactly as many bytes as it needs to read a correct PageHeader structure.   
In my case, this is what I got for our toy example:  
```
Page 0:
    {
      "compressed_page_size": 32,
      "crc": null,
      "dictionary_page_header": {
        "encoding": "Plain",
        "is_sorted": null,
        "num_values": 5
      },
      "page_number": 0,
      "type": "DictionaryPage",
      "uncompressed_page_size": 40
    }
Page 1:
    {
      "compressed_page_size": 20,
      "crc": null,
      "data_page_header": {
        "definition_level_encoding": "Rle",
        "encoding": "RleDictionary",
        "num_values": 5,
        "repetition_level_encoding": "Rle",
        "statistics": {
          "distinct_count": null,
          "max": "0400000000000000",
          "min": "0000000000000000",
          "null_count": 0
        }
      },
      "page_number": 1,
      "type": "DataPage",
      "uncompressed_page_size": 11
    }
```
For some reason, python polars encoder decided to create a dictionary for 5 values. Perhaps it has a simple condition on the number of unique values.  
So we get two pages. First - dictionary page, its size; then - actual DataPage, in RleDictionary encoding, with statistics and repetition level annotations. 
```rust
// Read compressed page data
                let mut compressed_data = vec![0u8; page_header.compressed_page_size as usize];
                file.read_exact(&mut compressed_data)?;

                // Decompress the page data
                let decompressed = decompress_page(
                    &compressed_data,
                    col_meta.codec,
                    page_header.uncompressed_page_size,
                )?;
```
We know specific byte size we need to read from data page header. From row group metadata, we also know that it was `zst`-compressed. So we read exact number of bytes and decompress.
```rust
if let Some(ref dict_header) = page_header.dictionary_page_header {
    // Dictionary page - store dictionary values
    println!("    Dictionary Page Values:");
    match decode_page_data(
        &decompressed,
        col_meta.type_,
        dict_header.encoding,
        dict_header.num_values,
    ) {
        Ok(PageValues::Int64(values)) => {
            dictionary = Some(values.clone());
            // Print up to 5 values
            for (i, val) in values.iter().take(5).enumerate() {
                println!("      [{}]: {}", i, val);
            }
            if values.len() > 5 {
                println!("      ... and {} more", values.len() - 5);
            }
        }
        Err(e) => {
            println!("      Failed to decode dictionary: {}", e);
        }
        _ => {
            println!("      Unexpected page value type");
        }
    }
}
```
Remember, our `dictionary` object was defined like this: `let mut dictionary: Option<Vec<i64>> = None`. It will obviously work for i64 values only (so our code is not really production-grade, just a demo). If we know from page header that this page is a dictionary page, we read dictionary values and store them in this object - to be used when reading actual data.  
Dictionary is just a vector, so we always map from index to the actual value. Because of this, the way we store the dictionary for a column of values of type T is exactly the same as we would store values themselves, and decoding function is the same as well.  

Decoding happens in `decode_page_data` call (defined later), which takes in data bytes, type T, encoding (plain/rle/rle dictionary) and the number of values we need.  
```rust
/// Decode page data based on type and encoding
pub fn decode_page_data(
    page_data: &[u8],
    page_type: Type,
    encoding: Encoding,
    num_values: i32,
) -> Result<PageValues, Box<dyn std::error::Error>> {
    match (page_type, encoding) {
        (Type::Int64, Encoding::Plain) => {
            let values = decode_plain_int64(page_data, num_values as usize)?;
            Ok(PageValues::Int64(values))
        }
        (Type::Int64, Encoding::RleDictionary) => {
            // For dictionary encoding, we need the bit width
            let bit_width = page_data[0];
            let indices = decode_rle_dictionary(page_data, num_values as usize, bit_width)?;
            Ok(PageValues::DictionaryIndices(indices))
        }
        _ => Err(format!(
            "Decoding not implemented for type {:?} with encoding {:?}",
            page_type, encoding
        )
        .into()),
    }
}

/// Decode Plain-encoded Int64 values
pub fn decode_plain_int64(data: &[u8], count: usize) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(data);
    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        let value = cursor.read_i64::<LittleEndian>()?;
        values.push(value);
    }

    Ok(values)
}
```
This is the definition for `decode_page_data`. Depending on the type and encoding type, it forwards decoding to a more specific function. `decode_plain_int64` is, indeed, plain as hell: we just read i64 values in little endian one by one, without varint/zigzag or other such mambo jambo.   
Here are the values stored in our toy dictionary page:
```
    Dictionary Page Values:
      [0]: 0
      [1]: 1
      [2]: 2
      [3]: 3
      [4]: 4
```
