> Current writer/reader format version: **3**. Version 3 adds a one-byte flags
> field to every decision record so XGBoost's learned missing direction and
> categorical set membership are stored. It also uses 32-bit varints for
> repeated indices and counts, fixed u32 header counts, and q16 classification
> probabilities. Version 2 added a
> distinct strict numeric comparison opcode for XGBoost and float64 numeric
> pools. Native thresholds and regression means retain Python float precision;
> XGBoost inputs and thresholds are normalized to float32.
> Nested native CART nodes use `"<="`; strict XGBoost nodes use `"<"`.
> Readers reject other versions rather than guessing their semantics. In
> particular, format 2 must be re-exported from the training model or retrained.

# .cart Binary Format Specification

Version 3 — Little-endian throughout

---

## Overview

The `.cart` format is a compact binary representation of decision trees, random forests, and XGBoost models optimized for:

- **Size**: Varint encoding, packed bytes, no padding
- **Speed**: Compact fixed-layout pools, no parsing of text
- **Portability**: Simple format readable in any language

---

## File Structure

```
┌──────────────────────────────────────┐
│ Header (52 bytes)                    │
├──────────────────────────────────────┤
│ String Table                         │
├──────────────────────────────────────┤
│ Feature Table                        │
├──────────────────────────────────────┤
│ Class Table                          │
├──────────────────────────────────────┤
│ Float Pool                           │
├──────────────────────────────────────┤
│ Categorical Value Pool               │
├──────────────────────────────────────┤
│ Tree Offsets (varint[])              │
├──────────────────────────────────────┤
│ Decision Nodes (variable size)       │
├──────────────────────────────────────┤
│ Leaf Nodes (variable size)           │
├──────────────────────────────────────┤
│ Distributions (optional)             │
├──────────────────────────────────────┤
│ Case Tables (optional)               │
├──────────────────────────────────────┤
│ Category-Set Tables (optional)        │
├──────────────────────────────────────┤
│ Metadata JSON (optional)             │
└──────────────────────────────────────┘
```

---

## Header (52 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `CART` (0x43 0x41 0x52 0x54) |
| 4 | 2 | version | Format version (currently 3) |
| 6 | 2 | flags | Bitfield (see below) |
| 8 | 4 | n_features | Number of features |
| 12 | 4 | n_classes | Number of class labels |
| 16 | 4 | n_trees | Number of trees |
| 20 | 4 | n_decisions | Number of decision nodes |
| 24 | 4 | n_leaves | Number of leaf nodes |
| 28 | 4 | n_floats | Size of float pool |
| 32 | 4 | n_cat_vals | Size of categorical value pool |
| 36 | 4 | n_dists | Number of distributions |
| 40 | 4 | n_case_tables | Number of case tables |
| 44 | 4 | n_category_sets | Number of category-set tables |
| 48 | 4 | metadata_len | Length of metadata JSON |

### Flags (16-bit bitfield)

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `FLAG_IS_FOREST` | Model is a forest (multiple trees) |
| 1 | `FLAG_IS_REGRESSION` | Regression task (vs classification) |
| 2 | `FLAG_HAS_DISTRIBUTIONS` | Leaf distributions stored |
| 3 | `FLAG_IS_XGBOOST` | XGBoost model (additive prediction) |

---

## String Table

```
n_strings: varint
strings: (byte_length: varint, utf8: bytes[byte_length])[n_strings]
```

All strings (feature names, class labels, categorical values) are stored once and referenced by index.

---

## Feature Table

For each feature (n_features entries):

```
name_idx: varint            # Index into string table
type_flags: u8              # Bits 0-1: type, Bits 2-4: dtype
n_cat: varint               # Number of known categorical values
cat_indices: varint[n_cat]  # String indices for each value
```

### Type Flags

| Bits | Field | Values |
|------|-------|--------|
| 0-1 | type | 0=categorical, 1=numerical |
| 2-4 | dtype | 0=str, 1=int, 2=float, 3=bool |

---

## Class Table

For classification models:

```
class_indices: varint[n_classes]  # String indices for class labels
```

---

## Float Pool

```
floats: f64[n_floats]           # All threshold values and regression outputs
```

Referenced by index from decision nodes (thresholds) and leaf nodes (regression values).

---

## Categorical Value Pool

```
cat_vals: varint[n_cat_vals]    # String indices for categorical comparisons
```

When a decision node does `feature == "value"`, it stores an index into this pool, which contains the string table index.

---

## Tree Offsets

```
offsets: varint[n_trees]        # Index into decision array for each tree's root
```

For forests, each tree's root node index. For single trees, just one entry.

---

## Decision Nodes

Variable-size encoding for each node:

```
feature: varint                 # Feature index
op: u8                          # Operation type
flags: u8                       # Missing direction and set-membership marker
val: varint                     # Index into an operation-specific value table
left: varint                    # Left child index (OP_LE/OP_LT/OP_EQ only)
right: varint                   # Right child index (OP_LE/OP_LT/OP_EQ only)
```

### Operations

| Value | Constant | Meaning | Children |
|-------|----------|---------|----------|
| 0 | `OP_LE` | Numeric: `feature <= floats[val]` | left, right |
| 3 | `OP_LT` | Numeric: `feature < floats[val]` | left, right |
| 1 | `OP_EQ` | Categorical equality, or set membership when `CATEGORY_SET` is set | left, right |
| 2 | `OP_SWITCH` | Case table lookup | In case_tables[val] |

### Decision flags

Every decision record has one flags byte, including native CART and switch
nodes. Bits 0-1 encode the missing direction. Bit 2 marks category-set
membership and is valid only with `OP_EQ`; bits 3-7 are reserved and zero.

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `MISSING_NONE` | No learned route; apply the runner's missing policy |
| 1 | `MISSING_LEFT` | A missing value follows the left/yes branch |
| 2 | `MISSING_RIGHT` | A missing value follows the right/no branch |
| 4 | `CATEGORY_SET` | `val` indexes `category_sets` instead of `cat_vals` |

`CATEGORY_SET` is combined with one missing-direction value, so valid set
flags are 4, 5, or 6. For a present value, the left branch is taken exactly
when its canonical string is in the referenced set. Bool-dtype values are
normalized to `"0"` or `"1"` before lookup, as for `OP_EQ`.

For `OP_SWITCH`, a learned left direction follows the first stored case's child
and a learned right direction follows the default child. Cartlet's own
exporters, including XGBoost export, do not produce switch nodes with a learned
direction; a hand-authored one must list the intended missing-value case first. Every decision record stores feature and value varints around
the two fixed bytes; comparison records then carry two child varints, while
switch records end after the value varint.

### Child Index Encoding

Child indices use a flag bit to distinguish decisions from leaves:

- Bit 31 clear: Index into decision nodes
- Bit 31 set: Index into leaf nodes (mask with `0x7FFFFFFF`)

---

## Leaf Nodes

```
leaf_type: u8                   # Type of leaf value
val: varint                     # Index into appropriate pool
```

### Leaf Types

| Value | Constant | val meaning |
|-------|----------|-------------|
| 0 | `LEAF_CLASS` | String index (class label) |
| 1 | `LEAF_FLOAT` | Float index (regression value) |
| 2 | `LEAF_CLASS_DIST` | Distribution index |

### Stable node IDs

Decision-array and leaf-array indexes are internal references in the `.cart`
format. The format and writer numbering are unchanged, but `predict_path` and
leaf inspection do not expose these indexes. They report root-relative path
IDs instead: binary branches contribute `L` or `R`, switch defaults contribute
`D`, and matched switch cases contribute a length-prefixed canonical key such
as `C3:red`. See [Decision paths](runners.md#decision-paths).

The writer still numbers trees in tree order. Within each nested tree it
reserves each decision in preorder, visits the left subtree before the right
subtree, and, for a switch, visits the default subtree before case subtrees in
stored order. Every leaf node appends one leaf-array entry in that traversal;
leaf nodes are never deduplicated, although their string, float, and
distribution payloads may share pool entries.

---

## Distributions (if FLAG_HAS_DISTRIBUTIONS)

For each distribution (n_dists entries):

```
n_entries: varint
entries: (class_idx: varint, probability_q16: u16)[n_entries]
```

The writer sorts each distribution by the original probability, descending,
then stores `probability_q16 = round(probability * 65535)`. The sort is stable,
so a quantization tie retains the same best class as the in-process argmax.
Distribution deduplication uses the ordered `(class_idx, probability_q16)`
values. Both runners decode each entry to `probability_q16 / 65535`, then divide
the decoded entries by their sum; the rebuild path uses those normalized values.

Before renormalization, rounding gives an absolute error of at most
`1 / 131070`, approximately `7.6e-6`, per probability. For a distribution of
`k` entries whose original probabilities sum to one, let `Q` be the sum of its
stored q16 integers. After renormalization, each absolute error is at most
`(k - 1) / (2Q)`. Since `Q >= 65535 - k/2`, this is at most
`(k - 1) / (131070 - k)` when `k < 131070`. The bound is zero for `k = 1`.
These bounds do not apply to regression or XGBoost float leaves or to numeric
thresholds, which remain float64.

---

## Case Tables (if n_case_tables > 0)

For `OP_SWITCH` nodes (n-ary categorical splits):

```
n_cases: varint
default_child: varint           # Child index for unmatched values
cases: (cat_val_idx: varint, child: varint)[n_cases]
```

Case keys are stored as strings. Bool-dtype case keys are normalized to `"0"`
or `"1"` first. The writer rejects a switch when two authored keys have the
same stored string (for example, `"yes"` and `True` on a bool feature), rather
than emitting an ambiguous table. Hand-authored switch nodes are supported by
the `.cart` writer, runners, and binary rebuild path; the JSON/pickle nested
model validator does not admit switch nodes.

---

## Category-Set Tables (if n_category_sets > 0)

For `OP_EQ` decisions whose `CATEGORY_SET` flag is set:

```
n_values: varint
cat_val_indices: varint[n_values]  # Indices into the categorical value pool
```

Sets occur after all case tables and before metadata. The writer canonicalizes
each category to its stored string, removes duplicates, and sorts the strings
before interning the table. Readers resolve each table once to a set of strings
so membership is O(1) per evaluated decision. The decision retains ordinary
left and right child varints and ordinary learned-missing semantics.

---

## Varint Encoding

Protobuf-style variable-length integers:

- 7 bits per byte, MSB = continuation flag
- Little-endian order
- 1-5 bytes for 32-bit values
- Values above `2^32 - 1`, a continuation chain longer than five bytes, and a
  truncated continuation chain are invalid

```
0xxxxxxx                        # 1 byte: 0-127
1xxxxxxx 0xxxxxxx               # 2 bytes: 128-16383
1xxxxxxx 1xxxxxxx 0xxxxxxx      # 3 bytes: etc.
```

---

## Metadata (optional)

If `metadata_len > 0`, the final bytes contain UTF-8 JSON with arbitrary metadata (training timestamp, feature specs, etc.).

---

## Constants Reference

For implementers, here are the key constants:

```python
# Magic
MAGIC = b"CART"
VERSION = 3
HEADER_SIZE = 52
OFF_VERSION = 4
OFF_FLAGS = 6
OFF_FEATURES = 8
OFF_CLASSES = 12
OFF_TREES = 16
OFF_DECISIONS = 20
OFF_LEAVES = 24
OFF_FLOATS = 28
OFF_CAT_VALS = 32
OFF_DISTS = 36
OFF_CASE_TABLES = 40
OFF_CATEGORY_SETS = 44
OFF_META_LEN = 48

SIZE_F64 = 8
SIZE_Q16 = 2

# Flags
FLAG_IS_FOREST = 0x01
FLAG_IS_REGRESSION = 0x02
FLAG_HAS_DISTRIBUTIONS = 0x04
FLAG_IS_XGBOOST = 0x08

# Decision flags
MISSING_NONE = 0
MISSING_LEFT = 1
MISSING_RIGHT = 2
MISSING_MASK = 0x03
CATEGORY_SET = 0x04
DECISION_FLAGS_MASK = 0x07

# Operations
OP_LE = 0
OP_LT = 3
OP_EQ = 1
OP_SWITCH = 2

# Leaf types
LEAF_CLASS = 0
LEAF_FLOAT = 1
LEAF_CLASS_DIST = 2

# Child index flag
LEAF_FLAG = 0x80000000
INDEX_MASK = 0x7FFFFFFF
```

## Capacity Limits

Format 3 keeps the 52-byte header's model counts and metadata length as fixed
u32 fields. String counts and byte lengths, feature names and vocabulary
counts/indices, class indices, categorical-value indices, decision
feature/value indices, leaf values, distribution counts/class indices, case
counts/indices, category-set counts/indices, child IDs, and tree offsets are
32-bit varints. Both encodings have a format limit of `2^32 - 1`.

Decision and leaf arrays are each limited to `2^31 - 1` entries because bit 31
distinguishes leaf child IDs. This preserves the documented node-ID capacity;
those flagged child IDs still fit in a five-byte 32-bit varint.

The u16 fields that remain are the version, header flags, and q16 probability;
the u8 fields that remain are feature type/dtype flags, the decision operation
and flags, and the leaf type. These encode fixed enums or the requested q16
value rather than model cardinalities. Float-pool values remain f64.

Both runners also apply defensive load-time caps of 10,000 features, 100,000
classes, 100,000 trees, and 10,000,000 decision or leaf nodes. These are
resource-safety checks, not narrower on-disk fields; string and categorical
vocabulary cardinalities and string byte lengths use their full u32 range.

---

## Embedded Models

When bundling a model with the Python runner, the raw `.cart` bytes are
base64-encoded and inserted as a module-level constant
(`_EMBEDDED_MODEL_B64`). At load time the runner decodes that constant when
no explicit model path is supplied.

Feature-table dtype bits are retained by both loaders. Known categorical values
are restored as the declared bool/int/float/str type for vocabulary/OOV checks;
comparison pools remain strings for traversal.
