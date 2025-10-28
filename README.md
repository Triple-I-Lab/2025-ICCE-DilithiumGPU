# 2025-ICCE-DilithiumGPU

# GPU-Accelerated High-Performance Design for CRYSTALS-Dilithium Digital Signature

## Overview
This repository accompanies the paper:

**A GPU-Accelerated High-Performance Design for CRYSTALS-Dilithium Digital Signature**  
*Hien Nguyen, Bertrand Cambou, and Tuy Tan Nguyen*  
School of Informatics, Computing, and Cyber Systems, Northern Arizona University, USA  
Published in **IEEE ICCE 2025**

---

## Abstract
We present a **GPU-accelerated implementation** of the post-quantum digital signature scheme **CRYSTALS-Dilithium**.  
Using modern GPUs, this design optimizes:
- Number Theoretic Transform (NTT)
- Polynomial arithmetic
- Random sampling and batch signing

Our implementation achieves:
- **53–60% faster** signature generation  
- **20–64% faster** verification  

across NIST security levels 2, 3, and 5 — enabling high-throughput, quantum-resistant cryptography for blockchain and cloud systems.

---

## Key Contributions
- GPU-centric parallel architecture for Dilithium
- Optimized NTT kernel with butterfly operations
- Twiddle-factor caching in constant memory
- Batch signing and verification
- Implemented in **Python + CUDA** (tested on RTX 3090 Ti)

---

## Implementation Details
- **Language:** Python (CUDA bindings)
- **Platform:** NVIDIA GPU (RTX 3090 Ti)
- **Parallelization:** Thread-block hybrid design
- **Memory Optimization:** Shared + constant memory for NTT coefficients

---

## Performance Summary

| Security Level | Signature Speed-up | Verification Speed-up |
|----------------|--------------------|------------------------|
| Level 2        | +59.9 %            | +20.0 %                |
| Level 3        | +59.8 %            | +47.8 %                |
| Level 5        | +53.7 %            | +63.8 %                |

---

## Citation
If you reference this work, please cite:

```bibtex
@INPROCEEDINGS{10929968,
  author={Nguyen, Hien and Cambou, Bertrand and Nguyen, Tuy Tan},
  booktitle={2025 IEEE International Conference on Consumer Electronics (ICCE)}, 
  title={A GPU-Accelerated High-Performance Design for CRYSTALS-Dilithium Digital Signature}, 
  year={2025},
  month={11-14 Jan},
  address={Las Vegas, NV, USA},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/ICCE63647.2025.10929968}}

