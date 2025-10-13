### Intro
In the past few weeks, I struggled to vibe code a proper parquet lib in rust that would stream results from s3, select only necessary stuff, do parallel decoding etc. So I decided it's time to dig into that stuff myself - and turns out there's plenty of interesting details! So I decided to make a blog post out of it. Here is a rough outline of things I want to touch:
- Binary serialization, little/big endian, varint, zigzag
- thrift binary serialization format
- parquet format: schema, row groups, column chunks, pages
- compression
- s3 API, byte range requests, and how to read parquet effectively without downloading the entire file
- Rust async iterators
- Decoding and convenient interfacec
### Reading Files in Rust
Let's start with primitives to work with files in Rust. Here is a minimal program to write something to a file:
```rust
use std::{fs::File, io::Write};

fn main() {
    println!("Hello world!");
    let mut file = File::create("myfile").unwrap();
    let s = "Hello, world!";
    let b = s.as_bytes();
    let bytes_written = file.write(b).unwrap();
    println!("Written {bytes_written} bytes!");
}
```
`File::create` creates a new file (or opens an existing one and truncates it). `file.write`, a method on File object that requires a separate `io::Write` trait - is basically a thin wrapper around syscall. It returns `Result<usize>`: it might error out, for example, if file is closed, disk out of space etc. Or it might succeed and return number of bytes that were written - which might be less then requested, which is typical for syscalls, making them require extra care.  
There is a better alternative - `file.write_all`:  
```rust
file.write_all(b).unwrap();
```
It just returns `Result<()>` - its a convenience wrapper that will call `.write()` many times until all buffer's content is written, or an error occurs.  
There is also a BufWriter: 
```rust
    let mut writer = BufWriter::new(file);
    writer.write_all(b).unwrap();
```
It is usually convenient to just call `.write_all` whenever we want to write something. It is NOT recommended to do the actual syscall every time, e.g., in a for-loop iteration. BufWriter takes care of that - buffers data on its side, does large infrequent writes to underlying Writer.  
It also implements Writer trait, i.e., you can plug it into any function that accepts Writer. Convenient!  
If we read the doc for it, it also recommends to call flush manually:  
```rust
writer.flush().unwrap();
```
This is NOT typically a required pattern in RAII languages (rust/c++) because destructor of the object should take care of it. And it does. However, any errors that occur during flush will kill the program if that happens inside destructor. If we flush manually, we can control that more gracefully.  
Finally, there are higher-level functions and macros that allow writing text:  
```rust
writeln!(&mut writer, "Hello, world!").unwrap();
```
To sum up:  
- `Writer` is a trait for writing something to a "stream": files, memory buffers, anything else.
- `File` represents files in an operating system: actual files on disk, network sockets, stdin/stdout. `File` also implements `Writer`
- We can write bytes (aka binary, aka &[u8]) or text (aka utf8, aka &str), though interfaces for these differ
### Binary vs text
Typically, `text` refers to something a human can read, while `binary` represents something only readable by a program. In rust, `text` definition is even narrower: it has to be correct utf8. We use &str to refer to "text" data, and &[u8] to refer to bytes, i.e., data that might not necesserily be text; as we've seen from the code above, any "text" can be converted to "bytes", but not any "bytes" can be converted to "text".   
Examples of text files: .txt, .json, .svg.  
Examples of binary: executable files, archives, parquet, protobuf, thrift.  
You can read a text file, split it into lines (by looking at \n separator), which is what editors do by default; search some text within the file.  
If you try reading binary file (i.e., bytes that are not utf8), you might see something like: `ELF>L@hrI@8@+*@@@`. In some cases, this will actually break your terminal window, causing current cmd line to be all weird. Same problem will happen if you put binary data inside some parts of text, for example, serialize binary as a json field. Json parsers assume that any string content is correct utf8, and will break if it isn't. To avoid this, ppl sometimes encode binary into base64, which causes it to eat ~33% more space.  

When we saved string content to file, we could either use text wrappers (writeln!) that take text directly, or convert text (&str) to &[u8] by calling `.as_bytes()`. This method doesn't really do anything, it just allows you to look at the actual buffer content of a string, which is *guaranteed* to be correct utf8 in rust. Because of this, the two writing interfaces (bytes and text) actually yield the same result.   
Now let's try to save an `i32` value. For this, we have two options: save it as text or save it as bytes.
1. As text: 
```rust
let i = 234;
writeln!(&mut writer, "{}", i).unwrap();
```
`writeln!`, similar to other format-style macros (`format!`, `println!`, ...) converts an object to text. For integer numbers, that means encoding individual digits of the decimal representation of the number, i.e., '2' + '3' + '4', or simply, "234". This is what we'll see in the file:  
```bash
cat myfile
# 234
```
2. As bytes
```rust
let i: u32 = 234;
let b = i.to_be_bytes();
```
This is a bit more intereseting. `.to_le_bytes()` stands for "to little endian bytes". There is also `.to_be_bytes()` (big endian), and `.to_ne_bytes()` - native, which can be big endian or little endian, depending on the platform.  
This encoding corresponds to how an object is stored in memory. That, in turn, works like we've been taught at school, with a little twist; so, number `0` corresponds to binary `0`, or `00000000 00000000 00000000 00000000` (32 zero bits, because we save u32 number), then: 
```
# big endian
1 -> 1, or:   00000000 00000000 00000000 00000001
2 -> 10, or:  00000000 00000000 00000000 00000010
3 -> 11, or:  00000000 00000000 00000000 00000011
4 -> 100, or: 00000000 00000000 00000000 00000100
234 -> 11101010, or 00000000 00000000 00000000 11101010
```
And so on.   
The twist is that sometimes ppl encode stuff in little endian: indivudual bytes are encoded the same; byte order is right to left instead of left to right:   
```
234 -> 11101010 00000000 00000000 00000000
```
The name can be explained as "least significant byte" first for little endian, or "most significant byte first" for big endian.  
To view contents of a binary file, we can use `hexdump` or `xxd`:
```bash
% hexdump myfile
0000000 00ea 0000
0000004
```
```bash
xxd -b myfile
00000000: 11101010 00000000 00000000 00000000                    ....
# lower-endian "234"
```
`cat` will try to display binary as text. In some cases, some info can be seen like that (for example, in protobuf with a lot of texts - those texts will be visible as-is). Somethimes it's just useless and might screw up terminal.  
`hexdump` displays content in hexadecimal (16-based system). It is more convenient than decimal, because 1 hex digit always corresponds to 4 bits.   
`xxd` displays binary.  

Now that we know how to display bytes content, let's look at examples for other types:
```rust
let i: i16 = 1;
let b = i.to_le_bytes();
// 00000001 00000000
let i: i16 = -1;
let b = i.to_le_bytes();
// 11111111 11111111
```
Positive signed integers look exactly like unsigned integers. Negatives have their most significant bit as 1; content is computed as 2-s complement - invert(number) + 1. So, for `-1`: invert(1) -> `11111110 1111111`, add 1 -> `11111111 11111111`  
```rust
let f: f32 = 0.5;
let b = f.to_be_bytes();
// 00111111 00000000 00000000 00000000
let f: f32 = 1.0;
let b = f.to_be_bytes();
// 00111111 10000000 00000000 00000000
let f: f32 = 101414.39;
let b = f.to_be_bytes();
// 01000111 11000110 00010011 00110010
```
I've switched to big endian here, because it's easier to read.  
f32 are stored as follows: leftmost bit - sign (0 for positive), then 8 bits for exponent, then fraction. 

Before we proceed with some more binary tricks, let's discuss advantages of using binary over text: decoding speed and size.  
*Decoding speed*. If endianness of a stored value is the same as our platform, we don't need to decode it at all - it is ready to use. If endianness is opposite, we just invert byte order, which is a rather quick operation.  
    For text based formats, we read them character by character and decode from decimal. So, for `234`, we read `2` -> `3` -> `4` and compute the resulting number. It is slower than binary, but is rarely noticable on modern systems. 
*Size*. Strings are exactly the same in bytes as in text. Integers can be bigger or smaller, depending on the range. For example, `u32` is always 4 bytes in binary. In text it depends on the size of the number; `234` is 3 bytes; `65536` - 5 bytes. So, storing a lot of small numbers might actually be more compact in text. Same goes for floats. Very simple numbers (like 0.5) are smaller in text. Random float like 3.1415927 will take about 8 bytes in text, and will often lose precision as compared to in-memory representation (so in case of "pi", whose in-memory representation is not ideal as well, text-serialized version might actually end up being closer to the truth).  
### Varint, zig-zag encoding
As we've just seen, encoding integers might actually be more compact in text than in binary for small numbers. For example, if we have data about number of likes on reddit post, it might look something like this: `0, 0, 1, 0, 3, 0, 4, 0, 1, 10394285, 0, 1`. So, mostly small number or zero, but occasionally jumps to large numbers. If we store in binary, we cannot use anything smaller than u32 because of these occasional large numbers (and we still won't be sure that everything fits!). The sequence will be 19 bytes in text (not counting separators; 30 bytes if we separate by commas), and 48 in binary u32.  
Can we do better? Supposedly so, considering that text only uses a fraction of available bit trange to store the actual value. In text, we use bytes (numbers 0-256) to encode digits (0-10), which uses only ~50% of available space, expressed in bits.  
`varint` encoding allows us to effectively encode such numbers in binary. It works like this: 
- Split bits of the number into groups of 7
- Encode first group + continuation bit into a byte. Continuation bit: 1 if there are more non-zero bits to follow; 0 otherwise.   
Let's walk through example of `234`:  
- `234` expressed as binar is `11101010`
- Split into groups of 7: `1` + `1101010`
- For first group, continuation bit is 1; so we write down `11101010` (leftmost bit is continuation bit, than the first group)  
- write down second group: `00000001`. Continuation bit is zero.  
Expressed like this, `234` is just 2 bytes, instead of 3 for text or 4 for binary u32. And we don't have to worry about range at all - we can encode arbitrary large numbers like this.  
Here is rust code for such encoding:  
```rust
pub fn encode_varint_u64(mut x: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10); //u64 will take up to 10 bytes, not more
    loop {
        let mut byte = (x & 0x7F) as u8; // 0x7F = 01111111
        x >>= 7;
        if x != 0 {
            byte |= 0x80; // 0x80 = 10000000
        } // set continuation bit
        out.push(byte);
        if x == 0 {
            break;
        }
    }
    out
}

let x = 234;
let b = encode_varint_u64(x);
```
Let's walk through the code:  
- We take a single number (u64 in this example) and return a vector of u8. Recall that previously, when we used `.to_be_bytes()` for example, we returned `[u8; _]` - fixed-size array, i.e., size known at compile time. Now we don't want that, because we want our result to take less space for smaller numbers!
- In a loop, we start by performing bitwise and operation (`&`) with number 0x7F, or 0111111 in binary. This will zero out all bits of our number except 7 least significant bits
- We convert (x & 0x7F) to `u8`. This will drop everything except 8 least significant bits - our group to write! Currently has 0 as "continution bit".
- We byte shift `x` 7 bits to the right. This erases 7 least significant bits, and move everything else to the right. 
- If current `x` is not zero (still have groups to encode!), we set continuation bit. We compute bitwise OR (`|=`, for "OR assignment") with 0x80, or 1000000 in binary - this will set continuation bit to 1, not changing anything else.
- Continue until all groups are written.

We still have a problem with negatives though. Recall that -1 i32 is `11111111 11111111 11111111 11111111`, which will be 5 bytes in varint encoding. Because of this, 