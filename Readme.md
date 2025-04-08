## Online Signature Verification

The `sample/` directory contains 40 signature data of a person. 
The first 20 are genuine and rest 20 are forged ones.

`Signature Files → Data Reading → Feature Extraction → Template Creation → Verification`

### Run the code
`cd src/`

`python3 main.py`

### Matching Score
In the verification phase, If the matching score is less than 100,
the signature is genuine otherwise fake.
