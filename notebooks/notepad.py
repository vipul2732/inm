"""
Write information on a NotePad
Prepare the viewing of the NotePad
View the NotePad

To write information on the NotePad simply write on it 
To prepare the notepad for viewing, take action(s) 
To view the notepad simply look at the image 

Usage
```
import notepad

pad = notepad.NotePad()

pad.write("DataFrame Name")
pad.write("shape", df.shape)
pad.write(dict(nrows = len(df), ncolumns = len(df.columns)))
pad.write([("key1", 10), ("key2", 20)])

pad.prep(0, human_readable)  # prepares the first line as human readable
pad.prep(0, 10, indent(2))   # indents the first 10 lines by two spaces
pad.prep.clear()                      # clears the notepad_view 
pad.prep.clear(20, 30)                # clears formats on lines [20, 30)

pad.view()
```
Implementation

A NotePad is keeps track of its data and its image 
Actions are higher functions
Action_templates are higher order functions/callables
Actions are applied to certain lines in the notepad 

The possible call signature of prep is *args where the last arguemnt is
action

start, stop action
index, action
slice, action
tuple, action
action

# Apply formatting options to lines
# The word format is both a noun and a transitive verb
# to format vs. a format. The word format is ambigious
# Unambigous words
# arrange, organize, align, tidy, apply_format, edit, adjust, tweak,
# prettyfy, lay_out, styleize, prepare, prep
# 

Actions:

Some example actions
Indent some lines to the left
Make numbers human readable 
Round numbers off

To write to the notebook, record the data in the data list and apply a default action to the image data
  To record data in the data list ...
  To apply a default action to the image data, edit the image data at the corresponding index


Formatting Text For View
- data : list([key, value])
- format_specification : DefaultDict(line_nume : spec)

specs
 - indent
 - round
 - human readable numbers
 - seperator
 - gap1_width 
 - gap2_width
"""

class Spec(NamedTuple):
    indent: int = 0
    float_round: bool = False 




import notepad_line_data_actions as line_data_actions
import notepad_pair_actions as pair_actions

LineData = namedtuple("LineData", "key value")
Pair = namedtuple("Pair", "line data")

# Generic functions
def h(x):
    return "{:,}".format(x)
# line_data actions

def key_human_readable(line_data):
    return LineData(h(line_data.key), line_data.value)
def value_human_readable(line_data):
    return LineData(line_data.key, h(line_data.value))
def human_readable(line_data):
    return LineData(h(line_data.key), h(line_data.key)) 


# pair actions :: (line, line_data) -> (line, line_data)
def key_human_readable(pair):
    return Pair(pair.line, key_human_readable(pair.line_data)) 
def value_human_readable(pair):
    return Pair(pair.line, value_human_readable(pair.line_data)) 
def human_readable(pair):
    return Pair(pair.line, human_readable(pair.line_data)) 

# line_data_action_template :: (Any -> (line, line_data))

def indent(i: int, char=" "):
    def action(line, line_data):
        line = char * i + line
        return line, line_data
    return action
def seperator(sep=":", gap1_width=8, gap2_width=8, gap1_char=" ", gap2_char=" "):
    def action(line, line_data):
        line = line_data.key + gap1_char * gap1_width + sep + gap2_char * gap2_width + line_data.value 
        return line, line_data
    return action




def basic_add_line(self):
    ...

def apply_one(image, index, action):
    #

class NotePad:
    """
    Users interface to the notepad
    """
    def __init__(self, data = None, default_action=None):
        self.data = NotePadKeyValListData(data)
        self.image = Image() 
        self.default_action = default_action if defatult_action else basic_add_line
    def write(self, other):
        self.data = _notepad_write(self, other)
        self.prep(-1, self.default_action)
    def prep(self, *args): 
        """
        action = notepad.human_readable
        pad.prep(action)  # Applies the action to the current image region
        #
        pad.prep(0, action) # updates the image region and applies the action
        pad.prep(-1, action)
        #
        pad.prep(1, 10, action)
        pad.prep(0, 10, 2, action)
        """
        assert callable(args[-1]), "must supply an action"
        if len(args) == 1:
           self.image = apply_all(self.image, args)
        elif len(args) == 2:
           self.image = apply_one(self.image, *args)
        elif len(args) == 3: 
            self.image = apply_range(self.image, *args)
        elif len(args) == 4:
            self.image = apply_step_range(self.image, *args)
        else:
            raise ValueError
    def view(self):
        return self.image
    def __repr__(self):
        return self.image.__repr__()

class Image:
    def __init__(self, image_str  = None):
        self.image_str = image_str
    def __len__(self):
        return len(self.image_str.split("\n")) if self.image_str else 0
    def __repr__(self):
        return self.image_str if self.image_str else super().__repr__()

class NotePadKeyValListData:
    """
    Internal datastructure of the notepad
    """
    def __init__(self, keyval_list = None):
        self.keyval_list = keyval_list

def _notepad_write(self, other):
    if isinstance(other, str):
        return _write_str(self, other)
    elif isinstance(self, other, tuple):
        return _write_tuple(self, other)
    elif isinstance(self, other, dict):
        return _write_dict(self, other)
    elif isinstance(other, list):
        return _write_key_val_list(self, other)
    else:
        raise ValueError

def _write_str(self, other: str):
    return self.data + [(other, None)]

def _write_tuple(other: tuple):
    assert len(other) == 2
    return self.data + [other]

def _write_dict(other: dict):
    return self.data + list(other.items())

def _write_key_val_list(other: list):
    assert len(other) > 0, "Cannot write an empty list"
    assert isinstance(other[0], tuple), "Must be a list of tuples"
    assert len(other[0]) == 2, "tuples must be key value pairs"
    return self.data + other
    

def _notepad_prep(self, other, action):
    if isinstance(other, int):
        return _prep_line(self, other, action)
    elif isinstance(other, slice):
        return _prep_slice(self, other, action)
