# RF-Only 5.9 GHz V2X-Inspired Roadside ISAC Using USRPs

> **M1 research repository — Team `g14_research_isacv2x_ind`**  
> **Target paper submission:** 18 November 2026

## Project video

▶️ **[Watch our two-minute progress video on Loom](https://www.loom.com/embed/ca03f628b9004216a871881f3888bf60)**

GitHub README pages do not display Loom iframe players. The link above is the GitHub-friendly way to watch the video. For a webpage that supports iframe embeds, the original embed code is preserved below:

<details>
<summary>Show original Loom iframe code</summary>

```html
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/ca03f628b9004216a871881f3888bf60" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
```

</details>

---

## Overview

We investigate a specific question in roadside **integrated sensing and communication (ISAC)**:

> Can a low-power roadside SDR, designed as a prototype for a future ETSI-oriented RSU, use a 5.9 GHz OFDM signal to detect a moving object despite interference from its own transmitter and reflections from stationary surroundings? At the same time, how reliably can a separate receiver decode the data carried by that signal?

Our initial system is **RF-only**. It uses a USRP B210 as the roadside sensing node and a second USRP B210 as an independent communication receiver. We have six B210s available, but multi-RSU sensing is reserved for future work. We use a custom **V2X/ITS-G5-parameter-inspired OFDM waveform**, not a claimed fully compliant ETSI ITS-G5 or 3GPP PC5 protocol implementation.

At M1, the research question, system model, related-work position, and experiment plan are defined. **Experimental target-detection performance has not yet been established.**

## Research boundary

### What we are studying

- A stationary, low-power, 5.9 GHz roadside sensing site.
- OFDM packets carrying data and known reference/training symbols.
- Direct TX-to-RX leakage, static roadside clutter, and weak moving-target echoes.
- Target presence, approach/recede decision, and radial velocity.
- Coarse range only if a reflected delay component can be reliably separated and calibrated.
- Independent packet reception and communication metrics from the same transmitted waveform.

### What we do not claim in the first paper

- Certified ETSI ITS-G5 or 3GPP PC5 sidelink implementation.
- Automotive-radar-grade ranging, imaging, or 3D scene reconstruction.
- Camera–radar fusion or object-class recognition.
- Precise angle estimation or multi-RSU localization.
- Any target-detection distance or detection probability before measurements exist.

A camera may be used **only to produce offline ground truth**; camera data are not an input to the RF detector.

## Experimental model


![Proposed single-RSU V2X-inspired ISAC experiment](assets/V2X_ISAC_experiment_model_diagram.svg)

```text
                          ONE TRANSMITTED OFDM WAVEFORM
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
        Moving roadside target                   USRP B210 #2
          reflected echo                    independent packet RX
                  │                                       │
                  ▼                                       ▼
USRP B210 #1: transmit + sensing RX                  PER / goodput
         │
         ├── direct TX→RX leakage
         ├── stationary roadside clutter
         ├── weak moving-target echo
         └── receiver noise
                  │
                  ▼
  IQ sync → channel estimation → clutter/leakage reduction
                  │
                  ▼
     delay–Doppler map → detector → target and motion estimates
```

### Hardware roles

| SDR | Role | First-paper status |
|---|---|---|
| B210 #1 | Single roadside sensing RSU; transmits OFDM data and receives echoes with separate TX/RX antennas. | Core setup |
| B210 #2 | Independent communication receiver for decoding the same transmitted packets. | Core setup |
| B210 #3 | Optional direct-path/reference receiver if required for diagnostics. | Optional |
| B210 #4–#6 | Spares or later synchronized multi-RSU work. | Outside the first paper |

"Single RSU" means **one sensing site**, not that the lab has only one SDR.

## Mathematical signal model

## Mathematical signal model

Let $X[m,n]$ be the known OFDM symbol at time index $m$ and subcarrier index $n$. The received signal contains direct transmitter leakage, static reflections, a moving-target echo, and noise:

```math
Y[m,n] =
X[m,n]\left(
H_{\mathrm{leak}}[m,n]
+ H_{\mathrm{static}}[m,n]
+ \alpha e^{-j2\pi n\Delta f\tau}
         e^{j2\pi mT_{\mathrm{sym}}f_D}
\right)
+ W[m,n].
```

Here, $\tau$ is the target echo delay and $f_D$ is its Doppler shift.

Here $H_{\mathrm{leak}}$ is direct transmitter-to-receiver leakage; $H_{\mathrm{static}}$ represents static clutter; $\alpha$ is complex target reflectivity; $\tau$ is round-trip propagation delay; $f_D$ is Doppler frequency; and $W$ includes noise and residual interference. This is an initial model: the experiment must determine whether synchronization error, receiver clipping, phase noise, and multipath limit its usefulness.

We estimate the channel on known transmitted symbols:

$$
\widehat{H}[m,n] = \frac{Y[m,n]}{X[m,n]},
\qquad X[m,n] \neq 0.
$$

After suppressing repeatable leakage and static components, we form a delay–Doppler map. For a monostatic sensing geometry:

$$
\widehat{R} = \frac{c\widehat{\tau}}{2},
\qquad
\widehat{v}_{r} = \frac{\lambda\widehat{f}_{D}}{2},
\qquad
\lambda = \frac{c}{f_c},
\qquad f_c \approx 5.9\,\mathrm{GHz}.
$$

The approximate nominal resolution limits are:

$$
\Delta R \approx \frac{c}{2B_{\mathrm{occ}}},
\qquad
\Delta v_{r} \approx \frac{\lambda}{2T_{\mathrm{coh}}}.
$$

Here $B_{\mathrm{occ}}$ is **occupied sensing bandwidth**, not simply the nominal channel width, and $T_{\mathrm{coh}}$ is the effective coherent observation time. A waveform with 10 MHz **occupied** bandwidth has a nominal two-target range-resolution scale of about 15 m. That does not automatically mean every isolated-target range estimate has 15 m error, but it does rule out an unsupported high-resolution imaging claim.

The approximate monostatic echo-power trend is:

$$
P_{\mathrm{echo}} \approx
\frac{P_tG_tG_r\lambda^2\sigma}{(4\pi)^3R^4L}.
$$

This model motivates controlled short-range trials, careful antenna isolation, and explicit measurement of receiver headroom rather than an assumed sensing distance.

## Testable hypothesis

> After avoiding receiver saturation and suppressing repeatable TX leakage and static clutter, a moving high-reflectivity target will produce a detectable Doppler component at a controlled false-alarm rate, while B210 #2 independently decodes the OFDM payload.

If the measurements do not support that statement, we will report the observed operating limit and failure mode instead of claiming a successful detector.

## Measurement methodology

1. **Cabled/attenuated baseline:** Verify the OFDM waveform, sample timing, packet decoder, and safe receive power before OTA trials.
2. **Empty-scene capture:** Measure direct-path leakage and stationary background with no moving target present.
3. **Controlled target trials:** Repeat target-present and target-absent experiments on a marked path; start with a high-reflectivity controlled target.
4. **Signal processing:** Compare raw processing, empty-scene subtraction, and subtraction plus a fixed threshold or CFAR-style detector.
5. **Communication validation:** Use B210 #2 to count successfully received packets and payload bits from the same waveform.
6. **External ground truth:** Use marked distances, a speed reference, or video solely to validate the RF estimates.

One statistical trial will be a predefined observation window and a predefined detector decision; adjacent FFT frames from one pass are not counted as independent trials without justification.

## Evaluation metrics

For $N_1$ target-present trials, $N_0$ target-absent trials, and $K$ velocity-labeled detections:

$$
\widehat{P}_{D} =
\frac{N_{\mathrm{detections\ in\ target\ trials}}}{N_1},
\qquad
\widehat{P}_{FA} =
\frac{N_{\mathrm{detections\ in\ empty\ trials}}}{N_0}.
$$

$$
\mathrm{RMSE}_{v} =
\sqrt{\frac{1}{K}\sum_{k=1}^{K}
\left(\widehat{v}_{r,k}-v_{r,k}\right)^2}.
$$

The independent communication receiver enables:

$$
\mathrm{PER} =
\frac{N_{\mathrm{failed\ packets}}}{N_{\mathrm{transmitted\ packets}}},
\qquad
G =
\frac{N_{\mathrm{correctly\ received\ payload\ bits}}}{T}.
$$

| Output | Planned evidence |
|---|---|
| Moving-target detection | Detection and false-alarm estimates at stated threshold/false-alarm setting; trial counts and uncertainty intervals. |
| Doppler / radial velocity | Approach/recede sign, estimated velocity, and error against ground truth. |
| Range | Delay/range error **only if** echo separation from leakage is demonstrably valid. |
| Communication | Packet error rate and goodput at B210 #2. |
| RF integrity | Receiver clipping check, leakage level, background behavior, and acquisition parameters. |
| ISAC trade-off | A plot of sensing and communication metrics against an **actually varied** packet, pilot, or resource parameter, if implemented. |

## Related work and our position

| Reference | What it studies | Why our experiment is different |
|---|---|---|
| Decarli *et al.* (2024) | Analytical performance of sensing through beyond-5G NR-V2X sidelink. | We are testing a single low-power roadside SDR under measured leakage/clutter, using a clearly identified custom waveform. |
| Li *et al.* (2025) | Resource allocation for NR-V2X joint communication and sensing in simulation. | We focus first on repeatable RF measurements and an independently measured communication link. |
| Ericsson ISAC demonstration | Distributed cellular sensing of non-cooperative drones. | Our prototype uses one roadside sensing site, a V2X-like channel, and a constrained SDR setup. |
| Singh *et al.* (2023); TacoDepth (2025) | Depth estimation with camera images and dedicated automotive radar point clouds. | Our model has no camera input and starts from SDR IQ and channel estimates, not radar point clouds. |

This SOTA position is preliminary. We will re-check the literature for comparable 5.9 GHz single-site SDR demonstrations before claiming a unique contribution.

## Project status

| Component | M1 status |
|---|---|
| Problem statement and scope | Defined |
| Related-work review | Initial review completed; SOTA position remains open to refinement |
| Signal model and planned evaluation | Defined |
| Network/experiment diagram | Created |
| OFDM waveform implementation | Planned |
| Cabled RX safety and packet test | Planned |
| Leakage/clutter measurement | Planned |
| Moving-target IQ dataset | Planned |
| Experimental detection and communication results | Not yet claimed |
| Multi-RSU sensing | Future extension |

## Timeline

| Period | Work | Expected evidence |
|---|---|---|
| **25 Sep – 2 Oct 2026** | Finalize waveform terminology, literature, mathematical model, permissions, and M1 GitHub package. | Proposal PDF/source, literature matrix, network diagram, experiment checklist. |
| **3 Oct – 16 Oct 2026** | Implement known-symbol OFDM TX, B210 #1 IQ capture, B210 #2 packet receiver, and cabled tests. | Waveform scripts, independent receiver test, RX-headroom check. |
| **17 Oct – 30 Oct 2026** | Characterize empty-scene leakage and collect authorized target/no-target trials. | Timestamped IQ, configuration logs, ground-truth trajectory. |
| **31 Oct – 7 Nov 2026** | Compare detectors and calculate detection, false alarm, Doppler error, PER, and goodput. | Plots with trial counts, uncertainty intervals, and limitations. |
| **8 Nov – 17 Nov 2026** | Validate novelty, write paper, and package reproducible evidence. | Manuscript and documented experiment configuration. |
| **18 Nov 2026** | Submit paper. | Submission confirmation. |

> The **M1 GitHub deadline** is separate from the 18 November paper deadline. Commit all required M1 components before the deadline specified by the course/team; no M1 date is assumed here.

## Proposed repository layout

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
│   └── regulatory-and-risks.md
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
│   └── logs/
└── results/
    ├── figures/
    └── tables/
```

This layout is a **plan**, not a claim that all scripts or datasets already exist. Store large IQ captures outside GitHub if needed and provide acquisition metadata and access instructions.

## Reproducibility and safe operation

Before reporting any experimental results, record software versions, occupied bandwidth, subcarrier map, sample rate, antenna separation, transmit/receive gains, target path, timestamping, and detector thresholds.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pyyaml
# UHD/GNU Radio must be installed and tested for the lab operating system.
```

The above command is a Python environment placeholder, **not** a complete USRP installation procedure.

This repository does not authorize over-the-air transmission. Before using 5.9 GHz in a roadside or campus setting, verify applicable institutional and Indian DoT/WPC requirements for the selected band, emissions, EIRP, duty cycle, and deployment. Start with cabled/attenuated or otherwise approved shielded testing.

## References

1. Ericsson, [Drone detection with ISAC for defense](https://www.ericsson.com/en/industries/defense/drone-detection-isac).
2. N. Decarli, S. Bartoletti, A. Bazzi, R. A. Stirling-Gallacher, and B. M. Masini, “[Performance Characterization of Joint Communication and Sensing With Beyond 5G NR-V2X Sidelink](https://doi.org/10.1109/TVT.2024.3365770),” *IEEE Transactions on Vehicular Technology*, 2024.
3. Z. Li, P. Wang, Y. Shen, and S. Li, “[Reinforcement Learning-Based Resource Allocation Scheme of NR-V2X Sidelink for Joint Communication and Sensing](https://doi.org/10.3390/s25020302),” *Sensors*, 2025.
4. ETSI, [ITS-G5 Access Layer Specification — EN 302 663 V1.3.1](https://www.etsi.org/deliver/etsi_en/302600_302699/302663/01.03.01_60/en_302663v010301p.pdf), 2020.
5. A. D. Singh *et al.*, “Depth Estimation from Camera Image and mmWave Radar Point Cloud,” *CVPR*, 2023. Included only for context on dedicated radar and camera fusion.
6. Y. Wang *et al.*, “TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-stage Fusion,” *CVPR*, 2025. Included only for context on dedicated radar and camera fusion.

## Team

- **Team / repository name:** `g14_research_isacv2x_ind`
- **Research focus:** V2X, OFDM, SDR, RF-only ISAC, roadside sensing
- **Paper target:** 18 November 2026

## License

Add an explicit repository license before public release. Until then, permissions for reuse should not be assumed.
