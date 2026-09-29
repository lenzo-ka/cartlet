> Current writer/reader format version: **3**. Version 3 adds a one-byte flags
> field to every decision record so XGBoost's learned missing direction and
> categorical set membership are stored. Version 2 added a distinct strict
> numeric comparison opcode for XGBoost and float64 numeric/probability pools.
> Native thresholds, regression means and class probabilities retain Python
> float precision; XGBoost inputs and thresholds are normalized to float32.
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
│ Header (36 bytes)                    │
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
│ Leaf Nodes (3 bytes each)            │
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

## Header (36 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `CART` (0x43 0x41 0x52 0x54) |
| 4 | 2 | version | Format version (currently 3) |
| 6 | 2 | flags | Bitfield (see below) |
| 8 | 2 | n_features | Number of features |
| 10 | 2 | n_classes | Number of class labels |
| 12 | 2 | n_trees | Number of trees |
| 14 | 4 | n_decisions | Number of decision nodes |
| 18 | 4 | n_leaves | Number of leaf nodes |
| 22 | 4 | n_floats | Size of float pool |
| 26 | 2 | n_cat_vals | Size of categorical value pool |
| 28 | 2 | n_dists | Number of distributions |
| 30 | 2 | n_case_tables | Number of case tables |
| 32 | 2 | n_category_sets | Number of category-set tables |
| 34 | 2 | metadata_len | Length of metadata JSON |

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
n_strings: u16
offsets: u16[n_strings]     # Offsets into string data
data: bytes[]               # Null-terminated UTF-8 strings
```

All strings (feature names, class labels, categorical values) are stored once and referenced by index.

---

## Feature Table

For each feature (n_features entries):

```
name_idx: u16               # Index into string table
type_flags: u8              # Bits 0-1: type, Bits 2-4: dtype
n_cat: u8                   # Number of known categorical values
cat_indices: u16[n_cat]     # String indices for each value
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
class_indices: u16[n_classes]   # String indices for class labels
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
cat_vals: u16[n_cat_vals]       # String indices for categorical comparisons
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
feat_op: u8                     # Packed feature index + operation
flags: u8                       # Missing direction and set-membership marker
val: u16                        # Index into an operation-specific value table
left: varint                    # Left child index (OP_LE/OP_LT/OP_EQ only)
right: varint                   # Right child index (OP_LE/OP_LT/OP_EQ only)
```

### feat_op Encoding

| Bits | Field | Description |
|------|-------|-------------|
| 0-5 | feature | Feature index (0-63); this is a **hard limit** — models that split on a feature with index > 63 cannot be written to `.cart` and export raises `ValueError`. Use a full-fidelity format (`.json`/`.pkl`) for wider models. |
| 6-7 | op | Operation type |

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

For `OP_SWITCH`, left means the XGBoost yes/category child and right means the
default/no child. The flags field costs exactly one byte per decision node; the
header, pools, leaf records, and child varints are unchanged.

### Child Index Encoding

Child indices use a flag bit to distinguish decisions from leaves:

- Bit 31 clear: Index into decision nodes
- Bit 31 set: Index into leaf nodes (mask with `0x7FFFFFFF`)

---

## Leaf Nodes (3 bytes each)

```
leaf_type: u8                   # Type of leaf value
val: u16                        # Index into appropriate pool
```

### Leaf Types

| Value | Constant | val meaning |
|-------|----------|-------------|
| 0 | `LEAF_CLASS` | String index (class label) |
| 1 | `LEAF_FLOAT` | Float index (regression value) |
| 2 | `LEAF_CLASS_DIST` | Distribution index |

### Stable node IDs

The decision-array index and leaf-array index are the public IDs returned by
`predict_path`. They are model-global and stable across supported save/load
formats. The writer numbers trees in tree order. Within each nested tree it
reserves each decision in preorder, visits the left subtree before the right
subtree, and, for a switch, visits the default subtree before case subtrees in
stored order. Every leaf node appends one leaf-array entry in that traversal;
leaf nodes are never deduplicated, although their string, float, and
distribution payloads may share pool entries.

---

## Distributions (if FLAG_HAS_DISTRIBUTIONS)

For each distribution (n_dists entries):

```
n_entries: u16
entries: (class_idx: u16, prob: f64)[n_entries]
```

Sorted by probability descending. Used for `predict_nbest()` support.

---

## Case Tables (if n_case_tables > 0)

For `OP_SWITCH` nodes (n-ary categorical splits):

```
n_cases: u16
default_child: varint           # Child index for unmatched values
cases: (cat_val_idx: u16, child: varint)[n_cases]
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
n_values: u16
cat_val_indices: u16[n_values]  # Indices into the categorical value pool
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
HEADER_SIZE = 36
OFF_CATEGORY_SETS = 32
OFF_META_LEN = 34

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

# feat_op encoding
OP_SHIFT = 6
OP_MASK = 0xC0
FEAT_MASK = 0x3F

# Leaf types
LEAF_CLASS = 0
LEAF_FLOAT = 1
LEAF_CLASS_DIST = 2

# Child index flag
LEAF_FLAG = 0x80000000
INDEX_MASK = 0x7FFFFFFF
```

---

## Embedded Models

When bundling a model with the Python runner, the raw `.cart` bytes are
base64-encoded and inserted as a module-level constant
(`_EMBEDDED_MODEL_B64`). At load time the runner decodes that constant when
no explicit model path is supplied.

Feature-table dtype bits are retained by both loaders. Known categorical values
are restored as the declared bool/int/float/str type for vocabulary/OOV checks;
comparison pools remain strings for traversal.
