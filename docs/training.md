# Training semantics

Cartlet's native and sklearn decision-tree trainers use the same meaning for
minimum-sample limits: min_samples_split and min_samples_leaf count training
rows with positive weight. Instance weights still control class probabilities,
regression statistics and impurity gains. Frequency weights do not turn one row
into several rows for these limits. Zero-weight rows do not contribute to fitting.

Native regression split search uses mergeable centered weighted moments for both
numerical and fast categorical splits. This avoids subtracting large squared
means when targets have a large constant offset. Numerical midpoint thresholds
remain finite for finite feature bounds. The fast categorical strategy may still
choose a different split from the exact strategy on floating-point ties.

Sklearn training one-hot encodes categorical features into a sparse matrix;
numerical-only inputs remain ordinary numerical rows. Conversion keeps an
explicit encoded-column mapping, preserving feature names and category values.
This avoids dense quadratic storage for high-cardinality categorical columns;
it does not promise that high-cardinality trees will fit quickly.

Isolation trees select only columns that vary in the current partition. A
partition becomes an unsplittable leaf when all columns are constant, rather than
when one randomly selected column happens to be constant.

XGBoost export retains strict numerical split comparisons, unlike the native
CART trainer's inclusive threshold convention. Current XGBoost releases can
provide a separate base margin for each class; these intercepts must be retained
for multiclass export. Prediction parity should be checked at split boundaries
as well as ordinary examples. Models whose missing-value direction cannot be
represented still warn explicitly; this limitation is separate from ordinary
numeric boundary semantics.

## Native XGBoost artifacts

XGBoostTree.export writes native Booster JSON for .json, or its native binary
format for .ubj/.xgb. This differs from the generic Cartlet JSON envelope used
by DecisionTree and RandomForest: load it with XGBoostTree.load/load_model, not
the generic convert workflow. A versioned Booster attribute stores Cartlet's
feature specifications, resolved task and class labels; no sidecar is required.
Reloaded models retain predictions and can re-export .cart for either runtime.
External native boosters lacking this metadata are rejected clearly; use the
upstream XGBoost API for those artifacts instead of inventing labels.
