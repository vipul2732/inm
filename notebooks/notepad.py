"""
Write information on a NotePad

Usage

```
import notepad
pad = notepad.NotePad()
pad.write("Title")
pad.write(dict(a = 1, b = 2))
pad

data = pad.data

# Format output in a new way
pad.fmt_func = my_new_fmt_func # Define this
```
"""

def default_fmt_func(i, tup):
    key, val = tup
    assert "\n" not in key
    if val == None:
        return key + "\n"
    else:
        return f"  {key}  :  {val}\n"

class NotePad:
    def __init__(self, fmt_func = default_fmt_func): 
        self.data = []
        self.fmt_func = fmt_func 
    def write(self, other):
        if isinstance(other, tuple):
            assert len(other) == 2
            assert isinstance(other[0], str)
            self.data.append(other)
        elif isinstance(other, str):
            self.data.append((other, None))
        elif isinstance(other, dict):
            for key, value in other.items():
                assert isinstance(key, str)
                self.data.append((key, value))
        elif isinstance(other, list):
            for tup in other:
                assert isinstance(tup, tuple)
                assert len(tup) == 2
                assert isinstance(tup[0], str)
            self.data = self.data + other    
        else:
            msg = f"type {type(other)} not supported"
            raise TypeError(msg)
    def clear(self):
        self.data = []
    def __repr__(self):
        if len(self.data) == 0:
            return "Empty NotePad at " + super().__repr__() 
        else:
            s = ""
            for i, tup in enumerate(self.data):
                s += self.fmt_func(i, tup)
            return s.removesuffix("\n")

