# RF-Only 5.9 GHz V2X-Inspired Roadside ISAC Using USRPs

> **M1 research repository — Team `g14_research_isacv2x_ind`**  
> **Target paper submission:** 18 November 2026

[![Status](https://img.shields.io/badge/status-M1%20proposal-blue)](#project-status)
[![Scope](https://img.shields.io/badge/scope-RF--only%20ISAC-0b7a75)](#scope-and-claim-boundaries)
[![Hardware](https://img.shields.io/badge/hardware-USRP%20B210-ef8b2c)](#hardware-roles)

## Project video

> **Loom progress video:** _Add the Loom embed URL or public video link below._

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/ca03f628b9004216a871881f3888bf60" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

## Overview

This project investigates a focused question in roadside **Integrated Sensing and Communication (ISAC)**:

> Can a low-power roadside SDR, designed as a prototype for a future ETSI-oriented RSU, use a 5.9 GHz OFDM signal to detect a moving object despite interference from its own transmitter and reflections from stationary surroundings? At the same time, how reliably can an independent receiver decode the data carried by that signal?

The first experiment is intentionally **RF-only**. It does not use camera input, lidar input, automotive radar point clouds, or a trained camera–radar fusion model. We transmit a custom OFDM waveform, record the reflected radio signal, suppress direct-path leakage and static clutter, and estimate moving-target evidence from delay–Doppler processing.

The work is framed as **6G-oriented V2X ISAC research**. It does **not** claim that the initial waveform is a compliant 3GPP PC5 implementation, a certified ETSI ITS-G5 stack, or a commercial RSU product.

---

## Research boundary

### What this work studies

- A low-power, stationary, 5.9 GHz roadside sensing site.
- A custom **V2X/ITS-G5-parameter-inspired OFDM waveform** with communication payload and known pilot/training symbols.
- A moving non-cooperative target, initially a corner reflector, vehicle, bicycle, or controlled moving reflector.
- Direct TX-to-RX leakage, static clutter, weak target reflections, and receiver saturation risk.
- Moving-target detection, Doppler sign, radial-velocity estimation, and coarse range **only if experimentally resolvable**.
- Communication packet reception at an independent SDR receiver.
- A measured sensing-versus-communication trade-off only when a real waveform/resource parameter is varied.

### What this first paper does not claim

- Full ETSI ITS-G5, LTE-V2X, or NR-V2X PC5 compliance.
- High-resolution automotive radar, imaging radar, dense depth estimation, or 3D scene reconstruction.
- Camera–radar fusion, object classification, or semantic perception.
- Precise angle-of-arrival estimation.
- Multi-RSU distributed sensing results.
- A maximum sensing range before it is experimentally measured.
- Permission to transmit at 5.9 GHz without applicable institutional and regulatory authorization.

---

## Why this problem matters

ISAC aims to reuse wireless spectrum, waveform resources, RF hardware, and baseband processing for both communication and environmental sensing. Existing work has studied this concept analytically and in large cellular-network demonstrations. However, the experimentally achievable sensing capability of a **single, low-power, roadside SDR** using a V2X-like waveform remains a practical question.

This work does not attempt to reproduce Ericsson's distributed, multi-site cellular ISAC system. Instead, it studies a smaller and reproducible operating point: **one roadside sensing RSU site with constrained power, limited bandwidth, direct-path leakage, and a real independent communication receiver**.

---

## Experimental model

![Single-RSU V2X-inspired ISAC experiment](assets/V2X_ISAC_experiment_model_diagram.svg)

### Signal paths

```text
                     Same 5.9 GHz OFDM waveform

      ┌────────────────────────────────────────────────────────────┐
      │                                                            ▼
┌───────────────┐                                            ┌───────────────┐
│ B210 #1       │                                            │ B210 #2       │
│ Roadside RSU  │──────── communication payload ───────────► │ Independent   │
│ TX + Echo RX  │                                            │ packet RX     │
└───────┬───────┘                                            └───────────────┘
        │
        │ transmitted waveform
        ▼
┌─────────────────────┐
│ Moving target       │
│ vehicle / reflector │
└──────────┬──────────┘
           │ weak reflected echo: delay τ and Doppler f_D
           ▼
┌────────────────────────────────────────────────────────────────┐
│ B210 #1 sensing receive path                                    │
│ direct leakage + static clutter + target echo + noise            │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
        IQ synchronization → channel estimation → background removal
                             ▼
                delay–Doppler map → threshold / CFAR detector
                             ▼
        target presence + Doppler sign + radial velocity (+ range if valid)
```

### Hardware roles

| Device | Planned role in the first paper | Notes |
|---|---|---|
| **USRP B210 #1** | Single stationary roadside sensing RSU | One TX path emits OFDM packets; one RX path records echoes. Separate TX/RX antennas are used to improve isolation. |
| **USRP B210 #2** | Independent communication receiver | Decodes the OFDM payload and logs packet reception, PER, goodput, and received-SNR proxies. |
| **USRP B210 #3** | Optional reference / cross-check receiver | Used only if a direct-path reference measurement is needed. It is not treated as a second sensing RSU in Phase 1. |
| **USRP B210 #4–#6** | Future extension / backup hardware | Reserved for later synchronized multi-RSU or distributed-sensing experiments. |

---

## Signal model

Let `X[m,n]` be the known transmitted OFDM symbol at OFDM time index `m` and subcarrier index `n`. A first-order received frequency-domain model is:

\[
Y[m,n] = X[m,n] \Big(H_{\mathrm{leak}}[m,n] + H_{\mathrm{static}}[m,n]
+ \alpha e^{-j2\pi n\Delta f\tau} e^{j2\pi mT_{\mathrm{sym}}f_D}\Big) + W[m,n].
\]

where:

| Term | Meaning |
|---|---|
| `H_leak[m,n]` | Direct transmit-to-receive leakage / self-interference. |
| `H_static[m,n]` | Repeatable stationary clutter from roadside objects and environment. |
| `α` | Complex target reflection coefficient. |
| `τ` | Round-trip propagation delay. |
| `f_D` | Doppler frequency. |
| `W[m,n]` | Residual noise, interference, phase noise, and modelling error. |

The channel estimate is formed from known transmitted symbols:

\[
\widehat{H}[m,n] = \frac{Y[m,n]}{X[m,n]}.
\]

After reference/background subtraction, a delay–Doppler map is generated to detect moving-target components.

For monostatic sensing:

\[
\widehat{R} = \frac{c\widehat{\tau}}{2},
\qquad
\widehat{v}_r = \frac{\lambda\widehat{f}_D}{2},
\qquad
\lambda = \frac{c}{f_c},
\qquad
f_c \approx 5.9\ \text{GHz}.
\]

The nominal resolution limits are:

\[
\Delta R \approx \frac{c}{2B_{\mathrm{occ}}},
\qquad
\Delta v_r \approx \frac{\lambda}{2T_{\mathrm{coh}}}.
\]

For example, an occupied bandwidth of 10 MHz has nominal range resolution near 15 m. Therefore, the initial emphasis is on **moving-target presence and radial velocity**, not high-resolution ranging or imaging.

---

## Testable hypothesis

> After avoiding receiver saturation and suppressing repeatable direct-path leakage and static clutter, a moving high-reflectivity target will produce a detectable Doppler component at a controlled false-alarm rate, while B210 #2 independently decodes the OFDM payload.

If the experiment does not support this hypothesis, the project will report the measured operating limit and failure mode rather than claim target detection without evidence.

---

## Measurement methodology

1. **Cabled and attenuated validation**
   - Validate OFDM generation, IQ capture, timing, packet decoding, and receiver headroom.
   - Confirm that transmit leakage does not saturate the sensing receiver.

2. **Empty-scene measurement**
   - Record direct-path leakage and static clutter with no moving target.
   - Construct a reference/background estimate.

3. **Controlled moving-target trials**
   - Collect repeated target-present and target-absent captures.
   - Begin with a corner reflector or controlled moving reflector.
   - Progress to a vehicle/bicycle only after the basic measurement chain is validated.

4. **RF sensing processing**
   - Estimate the OFDM channel using known training/pilot symbols.
   - Compare raw processing, empty-scene subtraction, and threshold/CFAR-style detection.
   - Estimate target Doppler sign and radial velocity.

5. **Independent communication measurement**
   - Use B210 #2 to measure packet reception using the same transmitted waveform.
   - Log packet error rate, successfully received payload bits, and receiver SNR/RSSI proxies.

6. **Ground-truth validation**
   - Use marked distances, a known target path, speed reference, or video only for offline validation.
   - Do not use camera output as an input to the sensing detector.

---

## Evaluation metrics

| Category | Metric | Definition / interpretation |
|---|---|---|
| Target detection | Detection probability | \(\widehat{P}_D = N_{\text{detected} \mid \text{target}} / N_{\text{target-present}}\) |
| False alarms | False-alarm probability | \(\widehat{P}_{FA} = N_{\text{detected} \mid \text{empty}} / N_{\text{target-absent}}\) |
| Doppler / motion | Radial-velocity RMSE | \(\mathrm{RMSE}_v = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(\widehat v_{r,k}-v_{r,k})^2}\) |
| Range | Range error | Reported only when a target delay peak is demonstrably separable from leakage/clutter. |
| Communication | Packet error rate | \(\mathrm{PER}=N_{\text{failed packets}}/N_{\text{transmitted packets}}\) |
| Communication | Goodput | Correctly received payload bits divided by observation time. |
| RF health | Leakage, clipping, noise floor | Used to determine whether the sensing RX is physically usable. |
| ISAC trade-off | Sensing versus payload/pilot configuration | Measured only when an actual waveform or resource parameter is varied. |

---

## Project status

| Item | Status |
|---|---|
| Research question and scope | Defined |
| Related-work review and SOTA position | Initial review completed; ongoing refinement required |
| Single-site experimental model | Designed |
| Network / experiment diagram | Created |
| OFDM waveform implementation | Planned |
| Cabled timing and RX-headroom validation | Planned |
| Leakage/clutter measurement | Planned |
| Controlled moving-target dataset | Planned |
| Target-detection results | Not yet claimed |
| Multi-RSU distributed sensing | Future extension |

---

## Timeline

| Period | Work | Expected evidence |
|---|---|---|
| **25 Sep – 2 Oct 2026** | Finalize waveform terminology, literature matrix, mathematical model, target protocol, permissions, and M1 GitHub package. | M1 source files, literature review matrix, network diagram, scope and safety checklist. |
| **3 Oct – 16 Oct 2026** | Implement known-symbol OFDM TX, B210 #1 sensing IQ capture, B210 #2 packet receiver, and cabled timing tests. | Reproducible waveform scripts, independent receiver baseline, RX clipping check. |
| **17 Oct – 30 Oct 2026** | Measure empty-scene leakage, optimize antenna isolation, and collect authorized target/no-target trials. | Timestamped IQ captures, experiment configuration logs, target-path ground truth. |
| **31 Oct – 7 Nov 2026** | Process delay–Doppler maps; compare raw and cancelled detectors; calculate detection, false alarm, Doppler error, and PER. | Figures with trial counts, uncertainty intervals, and failure cases. |
| **8 Nov – 17 Nov 2026** | Audit novelty claims; write, revise, and package code/configuration/results. | Complete manuscript and reproducibility package. |
| **18 Nov 2026** | Submit paper. | Submission confirmation. |

---

## Repository layout

```text
.
├── README.md
├── assets/
│   └── V2X_ISAC_experiment_model_diagram.svg
├── m1/
│   ├── M1_AWC.pdf
│   ├── proposal.tex
│   ├── literature-matrix.csv
│   ├── experiment-plan.md
│   ├── regulatory-and-risks.md
│   └── README.md
├── waveform/
│   ├── tx_ofdm.py
│   ├── rx_packet_decoder.py
│   └── parameters.yaml
├── sensing/
│   ├── channel_estimation.py
│   ├── clutter_cancellation.py
│   ├── delay_doppler.py
│   └── detector.py
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── ground_truth/
└── results/
    ├── figures/
    └── tables/
```

> File names may change as implementation proceeds. Raw IQ data should not be committed if its size exceeds repository limits; store a manifest, acquisition metadata, and instructions for retrieving approved datasets instead.

---

## Reproducibility notes

The planned software stack is expected to include Python, GNU Radio and/or UHD, NumPy, SciPy, Matplotlib, and project-specific SDR control scripts. Exact versions, sample rates, gain values, OFDM parameters, antenna details, and experiment configuration files will be committed before reporting results.

Example environment placeholder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pyyaml
# Install UHD / GNU Radio using the operating-system-specific procedure.
```

> Do not treat the command above as a complete USRP installation procedure. UHD/GNU Radio installation differs by operating system, driver version, and lab hardware setup.

---

## Safety and spectrum note

This repository does not grant permission to transmit. Before over-the-air operation, verify applicable institutional safety requirements and current DoT/WPC rules for the selected center frequency, bandwidth, EIRP, duty cycle, antenna configuration, and roadside/RSU deployment. Until authorization is confirmed, use attenuated cabled tests, dummy loads, shielded setups, or other approved methods.

---

## Related work

1. Ericsson, [Drone detection with ISAC for defense](https://www.ericsson.com/en/industries/defense/drone-detection-isac).
2. N. Decarli, S. Bartoletti, A. Bazzi, R. A. Stirling-Gallacher, and B. M. Masini, “Performance Characterization of Joint Communication and Sensing With Beyond 5G NR-V2X Sidelink,” *IEEE Transactions on Vehicular Technology*, 2024. DOI: [10.1109/TVT.2024.3365770](https://doi.org/10.1109/TVT.2024.3365770).
3. Z. Li, P. Wang, Y. Shen, and S. Li, “Reinforcement Learning-Based Resource Allocation Scheme of NR-V2X Sidelink for Joint Communication and Sensing,” *Sensors*, 2025. DOI: [10.3390/s25020302](https://doi.org/10.3390/s25020302).
4. ETSI EN 302 663 V1.3.1, [ITS-G5 Access Layer Specification](https://www.etsi.org/deliver/etsi_en/302600_302699/302663/01.03.01_60/en_302663v010301p.pdf), 2020.
5. A. D. Singh et al., “Depth Estimation from Camera Image and mmWave Radar Point Cloud,” *CVPR*, 2023. Used here to distinguish dedicated radar-plus-camera depth fusion from this RF-only experiment.
6. Y. Wang et al., “TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-stage Fusion,” *CVPR*, 2025. Used here to distinguish dedicated radar-plus-camera depth fusion from this RF-only experiment.

---

## Team

- **Repository / team name:** `g14_research_isacv2x_ind`
- **Project area:** V2X, ITS, SDR, OFDM, ISAC, roadside sensing
- **Paper target:** 18 November 2026

---

## License

Add a project license before public release. Until then, all rights are reserved by the project team and affiliated institution.
