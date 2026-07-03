# .cart Binary Format Specification

Version 1 — Little-endian throughout

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
│ Header (34 bytes)                    │
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
│ Metadata JSON (optional)             │
└──────────────────────────────────────┘
```

---

## Header (34 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `CART` (0x43 0x41 0x52 0x54) |
| 4 | 2 | version | Format version (currently 1) |
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
| 32 | 2 | metadata_len | Length of metadata JSON |

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
floats: f32[n_floats]           # All threshold values and regression outputs
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
val: u16                        # Index into floats, cat_vals, or case_tables
left: varint                    # Left child index (OP_LT/OP_EQ only)
right: varint                   # Right child index (OP_LT/OP_EQ only)
```

### feat_op Encoding

| Bits | Field | Description |
|------|-------|-------------|
| 0-5 | feature | Feature index (0-63); this is a **hard limit** — models that split on a feature with index > 63 cannot be written to `.cart` and export raises `ValueError`. Use a full-fidelity format (`.json`/`.pkl`) for wider models. |
| 6-7 | op | Operation type |

### Operations

| Value | Constant | Meaning | Children |
|-------|----------|---------|----------|
| 0 | `OP_LT` | Numeric: `feature <= floats[val]` | left, right |
| 1 | `OP_EQ` | Categorical: `feature == strings[cat_vals[val]]` | left, right |
| 2 | `OP_SWITCH` | Case table lookup | In case_tables[val] |

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

---

## Distributions (if FLAG_HAS_DISTRIBUTIONS)

For each distribution (n_dists entries):

```
n_entries: u16
entries: (class_idx: u16, prob: f32)[n_entries]
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
VERSION = 1

# Flags
FLAG_IS_FOREST = 0x01
FLAG_IS_REGRESSION = 0x02
FLAG_HAS_DISTRIBUTIONS = 0x04
FLAG_IS_XGBOOST = 0x08

# Operations
OP_LT = 0
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
