# Description

The folder `number_tokenizer/` contains codes that facilitate the process of **num**ber **tok**enizing (NumTok). 

`numtok.py` defines the NumTok class, where a series of class methods are implemented to facilitate the NumTok functionalities. 

`number_process.py` stores different preprocessing methods to the number string recognized by NumTok.


# Usage 

A set of testing examples are included in numtok.py

```python 
python -m number_tokenizer.numtok

# Output:
# 30 June 2018 is parsed into  [('30', 0, 2), ('2018', 8, 12)]
# 16,284 is parsed into  [('16,284', 0, 6)]
# (1,746) is parsed into  [('1,746', 1, 6)]
# ... more output suppressed here ...
```

