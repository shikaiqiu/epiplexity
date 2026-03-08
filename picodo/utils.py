from collections.abc import Mapping


def flatten_dict(d, prefix=None):
  if isinstance(d, Mapping):
    out = {}
    for k, v in d.items():
      nested_prefix = k if prefix is None else f'{prefix}/{k}'
      out |= flatten_dict(v, nested_prefix)
    return out
  else:
    return {prefix: d}
