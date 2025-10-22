namespace rs user

struct User {
  1: required i32 id,
  2: required string name,
  3: optional string email,
  4: optional list<string> tags,
  5: optional map<string, i64> attributes,
}

struct USER_COLLECTION {
  1: required list<User> users,
}


