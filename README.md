# fit-framework

Framework for fitting least squares problems. This framework handles situations
where certain fitted parameters are shared between samples depending on another
constant property of the samples. For example, if two samples have the same
"batch" parameter, then they should have the same "signal" parameter.

## Usage

```python
from fit_framework import fit

result_df = fit(...)
```
