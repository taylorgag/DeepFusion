### "DeepFusion: Smart contract vulnerability detection via deep learning and data fusion".

```
DeepFusion is an Ethereum smart contract vulnerability detection method, it can detect 5 types of Solidity smart contract vulnerabilities.
```

## Task Definition

```
Detect re-entrancy, timestamp dependence, integer overflow and underflow, Use tx.origin for authentication and Unprotected Selfdestruct Instruction vulnerabilities in smart contract.
```

## Structure in this project

```
${DeepFusion}
├── dataset
│   ├── IntergerOverOrUnderFlow
│   │   └── ast.txt
│   │   └── code_slicing.txt
│   ├── reentrancy
│   │   └── ast.txt
│   │   └── code_slicing.txt
│   └── Selfdestruct
│   |   └── ast.txt
│   |   └── code_slicing.txt
│   └── timestamp
│   |   └── ast.txt
│   |   └── code_slicing.txt
│   └── txOrigin
│   |   └── ast.txt
│   |   └── code_slicing.txt
├── models
    ├── representation_fu_BLSTM_Attention.py
├── ProgmaSlicing
    ├── ree
    │   └── sourceCode
    │   └── main.py
    │   └── progmaSlicing_code.py
```

- `dataset`: This is results of two kinds of data processing.
- `models/representation_fu_BLSTM_Attention.py`: This is the training and testing model of fused data.
- `ProgmaSlicing/ree/progmaSlicing_code.py`: This is to extract code slicing information of re-entrancy vulnerability.

