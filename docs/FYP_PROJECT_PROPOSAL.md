JOMO KENYATTA UNIVERSITY OF AGRICULTURE  
AND TECHNOLOGY  

COLLEGE OF ENGINEERING AND TECHNOLOGY  
SCHOOL OF MECHANICAL, MANUFACTURING AND MATERIALS  
ENGINEERING  
DEPARTMENT OF MECHATRONIC ENGINEERING  

**Implementation of In-Situ Monitoring and Dynamic Control  
of Robotic MIG-Based Wire Arc Additive Manufacturing**

Project Proposal  

**Project Team:**  
David Mwanga Bukosia  ENM221-0225/2022  
Gareth Githiri Ruhiu  ENM221-0049/2022  

**Supervisors:**  
Dr. Shohei Aoki  
Dr. James Wamai Mwangi  

November 2026  

---

## Contents

| | |
|---|---|
| Abstract | 3 |
| 1 Introduction | 4 |
| 1.1 Background | 4 |
| 1.2 Problem Statement | 5 |
| 1.3 Objectives | 5 |
| 1.3.1 Main Objective | 5 |
| 1.3.2 Specific Objectives | 5 |
| 2 Literature Review | 6 |
| 2.1 Overview of Wire Arc Additive Manufacturing | 6 |
| 2.2 Process Instability and Geometric Defects | 6 |
| 2.3 Past Solutions | 7 |
| 2.3.1 Open-Loop Parameter Selection and Optimisation | 7 |
| 2.3.2 In-Situ Monitoring | 7 |
| 2.3.3 Closed-Loop and Supervisory Control | 8 |
| 2.3.4 Process Modelling, Soft Sensing and Digital Twins | 9 |
| 2.4 Research Gaps | 10 |
| 3 Methodology | 11 |
| 3.1 System Overview | 11 |
| 3.2 Job Preparation and Nominal Parameter Derivation | 12 |
| 3.3 Command of the Welding Power Source | 13 |
| 3.4 In-Situ Sensing and Data Acquisition | 13 |
| 3.5 Offline Melt-Pool Modelling and Soft-Sensor Development | 14 |
| 3.6 Supervisory Closed-Loop Correction | 15 |
| 3.7 Testing and Validation | 16 |
| 4 Expected Outcomes | 16 |
| References | 17 |
| Appendices | 20 |

---

## Abstract

Wire arc additive manufacturing (WAAM) builds metallic components by melting filler wire with an electric arc and depositing material layer by layer. The process offers high deposition rates and comparatively low feedstock cost, and is therefore attractive for large structural parts and for repair. The same arc that makes the process productive also makes bead geometry difficult to hold: voltage, current, wire-feed speed, travel speed and accumulated heat jointly determine bead width and height. When these quantities are left on a fixed schedule while heat builds through successive layers, the deposit may overflow, hump, spat, become porous, or lose the arc.

Much of the published work on WAAM emphasises sensing and offline characterisation. Demonstrated closed-loop schemes exist, yet many rely on industrial welding power sources that already expose a fieldbus for online parameter change. The robotic WAAM cell at Jomo Kenyatta University of Agriculture and Technology (JKUAT) can already follow a planned path and record electrical, acoustic and limited thermal measurements. What it cannot yet do is change the setpoints of its conventional Rustler EM250i metal inert gas (MIG) inverter during a bead so that the deposited track remains close to the geometry intended in the part file.

This proposal describes a programme of work that closes that gap. Intended bead width and layer height will be extracted from the existing G-code workflow and converted into a nominal wire-feed schedule using the cross-sectional area of the filler wire. A first-principles melt-pool model will be developed and checked offline against independent numerical results and sparse experimental beads. Because that model is too slow to run inside the welding loop, a compact soft sensor will be trained from it so that live current, voltage, wire-feed speed and travel can be mapped to estimated bead geometry at control rates. Bounded supervisory corrections of voltage, inductance and wire-feed speed will then be issued to the Rustler EM250i. Performance will be judged by depositing the same qualified part once with fixed parameters and once under supervision, and by determining which deposit remains closer to the intended bead.

At the conclusion of the project, the expected outcomes are a working computer-to-welder communication link, a structured geometric description of each job taken from G-code, a validated offline process model and soft sensor, and experimental evidence that supervised deposition improves tracking of the intended bead on a conventional MIG power source.

---

## 1 Introduction

### 1.1 Background

Additive manufacturing of metals has moved from laboratory demonstration into industrial use for components that are large, customised, or difficult to obtain by subtractive routes. Among directed-energy processes, wire arc additive manufacturing uses an electric arc as the heat source and a continuous wire as the feedstock. Reviews of the field consistently identify high deposition rate, comparatively inexpensive wire, and suitability for large envelopes as the principal advantages of the process [1]–[3]. The same literature records that bead geometry and metallurgical quality remain sensitive to welding voltage, current, wire-feed speed, travel speed, and the thermal history of previous layers [4], [5].

In a typical robotic WAAM cell, a computer-aided design model is sliced into a toolpath, the robot follows that path, and a MIG or related power source melts the wire into a melt pool that solidifies as a bead. If the welding schedule is frozen at the start of the build, progressive heat accumulation changes the pool size and the resulting bead even though the commanded path has not changed. Dinovitzer et al. showed systematically that process parameters alter both bead geometry and microstructure [4]. Subsequent reviews of steel WAAM reiterate overflow, humping, porosity and loss of arc as recurring failure modes when control is inadequate [5], [6].

Closed-loop and supervisory schemes have been reported. Inter-pass temperature gating with forced cooling has been used to stabilise small near-net-shape prints [7]. Travel-speed adjustment at constant wire feed, multi-robot scan-and-print strategies, and multivariable layer-geometry control have likewise been demonstrated [8], [9]. Many of these cells, however, employ industrial welders that already expose Ethernet or similar interfaces, so that a supervisory computer can rewrite setpoints without reverse-engineering a consumer MIG panel. Digital-twin architectures for wire-arc directed energy deposition further assume bidirectional data flow between the physical cell and a virtual counterpart [10], [11].

The JKUAT robotic WAAM cell already follows a planned path derived from G-code and can observe melt-pool-related and acoustic signals. The power source selected for this project is the Rustler EM250i. That inverter does not presently offer a documented industrial fieldbus for online parameter change. Closing the loop on this cell therefore requires both a communication path into the welder and a means of estimating bead geometry from measurements that the cell can acquire while the arc is active.

### 1.2 Problem Statement

A robotic WAAM system intended for local production and repair should deposit beads whose width and height remain close to the values specified in the manufacturing file, even as heat accumulates through successive layers. Welding parameters should be adjustable during deposition on the basis of live process information, without requiring a specialist industrial welder or a separate materials-science campaign.

In the present JKUAT cell, the robot path and several monitoring channels are available, yet welding voltage, inductance and wire-feed speed on the Rustler EM250i cannot yet be changed under computer supervision while a bead is being laid. Fixed schedules therefore remain the default. Under those schedules, bead geometry drifts as the thermal state of the part evolves, and the deposit may depart from the intended cross-section or suffer arc instability.

Without a means to command the inverter online and to relate live measurements to bead geometry, the cell cannot progress from demonstration prints to a process that holds the intended bead. This project addresses that gap by establishing computer-to-welder communication, deriving geometric intent from the existing G-code workflow, validating an offline melt-pool model, distilling a fast soft sensor for geometry estimation, and applying bounded supervisory corrections during deposition.

### 1.3 Objectives

#### 1.3.1 Main Objective

To implement in-situ process observation and closed-loop control of robotic MIG-based wire arc additive manufacturing on the Rustler EM250i, so that welding parameters can be changed during deposition using live measurements and a process model that has been validated offline.

#### 1.3.2 Specific Objectives

i. To implement control of welding voltage, inductance and wire-feed speed on the Rustler EM250i, and to verify each commanded value against measured arc voltage and current.  

ii. To extract intended bead width and layer height from the existing Prusa-style G-code and process profile, and to derive a nominal wire-feed speed from those geometric targets and the cross-sectional area of the filler wire.  

iii. To develop and validate an offline first-principles melt-pool model that predicts bead width, height and penetration depth; to distill from that model a fast soft sensor that estimates the same quantities from live process measurements; and to use the soft-sensor estimates as supervisory feedback for corrective signals during deposition.

---

## 2 Literature Review

### 2.1 Overview of Wire Arc Additive Manufacturing

Wire arc additive manufacturing is a directed-energy deposition process in which an electric arc melts a continuously fed metal wire and a part is built layer by layer [1], [2]. Variants based on gas metal arc welding, gas tungsten arc welding and plasma transferred arc welding have been reported; the present cell uses a MIG configuration. Costello et al., in a state-of-the-art survey of wire-arc directed energy deposition for large metallic components, emphasise deposition rate, buy-to-fly ratio and suitability for structural envelopes as distinguishing industrial motivations [3]. Liu et al. similarly frame arc additive manufacture as a route to near-net-shape metallic parts with reduced material waste relative to machining from solid stock [12].

Hardware typically comprises a motion system (robot or gantry), a welding power source, a wire feeder, shielding gas delivery, and increasingly a suite of sensors for quality assurance [1], [13]. Sagili Arthur et al. have argued that flexible, open architectures reduce proprietary barriers and the cost of adding monitoring and control [13]. That observation is relevant to laboratories that must work with conventional MIG inverters rather than fieldbus-equipped industrial sources.

### 2.2 Process Instability and Geometric Defects

Bead width, bead height and penetration depth are the geometric quantities that determine whether successive layers stack into a usable wall or part. Dinovitzer et al. demonstrated that travel speed, wire-feed rate and heat-related settings alter both geometry and microstructure in WAAM [4]. Dekis et al., reviewing steel WAAM, catalogue overflow, humping, spatter, porosity and loss of arc among the defects that appear when heat input and material delivery are poorly matched to the thermal state of the build [5]. Shah et al. likewise identify geometric control and thermal management as persistent challenges [2].

Because each layer remelts and reheats previous metal, a schedule that is acceptable on the first pass may be excessive on the tenth. Heat-source management studies show that toolpath, dwell and power sequencing strongly influence distortion and residual stress even when the nominal bead recipe is unchanged [14]. The practical consequence for a production cell is that open-loop constancy of setpoints does not imply constancy of bead cross-section.

### 2.3 Past Solutions

#### 2.3.1 Open-Loop Parameter Selection and Optimisation

A first class of solutions selects a fixed welding schedule by experiment or by learning from prior trials. Taguchi and related designs have been used to optimise voltage, wire-feed rate and travel for joint quality in gas metal arc welding [15]. Learning-based methods have been proposed to predict bead geometry or to recommend process parameters for robotic welding, including ensemble and deep models trained on experimental matrices [16], [17]. So et al. report deep neural networks that forecast bead width and height for cold-metal-transfer WAAM from travel speed, wire-feed rate and previous-layer dimensions [18]. Kesse et al. similarly combine fuzzy logic and deep networks to predict TIG bead width from current, arc length and speed [17].

These approaches improve the initial recipe. They do not, by themselves, correct the recipe when heat accumulates mid-build. Offline optimisation therefore remains necessary but insufficient for multi-layer consistency on a cell that cannot yet adjust setpoints online.

#### 2.3.2 In-Situ Monitoring

A second class of solutions observes the process while metal is being deposited. Arjomandi et al. review in-situ monitoring technologies for WAAM and document electrical, optical, thermal and acoustic modalities, together with the practical difficulty of turning raw signals into actionable geometry estimates [6]. Electrical signatures (current and voltage) remain the most immediate indicators of arc state. Acoustic sensing has received substantial attention as a low-cost complement: Surovi and Soh show that acoustic features can identify geometric defect segments in WAAM [19]; Rajesh et al. combine voltage, current and audio with machine learning for anomaly detection across nickel alloys [20]; further studies examine Mel-spectrogram deep learning and audible-arc descriptors for porosity and transfer instability [21], [22].

Thermal sensing with pyrometers and thermocouples is widely used for inter-pass temperature control. Its usefulness for melt-pool peak temperature is limited wherever instrument range or viewing geometry cannot capture the hottest region of the pool. Optical and vision systems can measure bead profile directly, but they add cost, require clean optical paths, and are not always available on laboratory cells.

Monitoring alone diagnoses; it does not change the weld unless a control path into the power source exists.

#### 2.3.3 Closed-Loop and Supervisory Control

Closed-loop WAAM has been demonstrated in several forms. Kozamernik et al. combine inter-pass temperature control with forced cooling for near-net-shape printing of small metal components [7]. Lu et al. describe multi-robot scan-and-print strategies in which sensing and deposition are coordinated [8]. Bendia et al. present multivariable closed-loop control for layer geometry [9]. Coutinho et al. describe iWAAM, an automated monitoring and control framework based on the Robot Operating System, with power-source configuration and deposition sequencing embedded in the same software stack [23].

These results establish that supervisory adjustment of travel, wire feed or related setpoints can improve geometric consistency. They also reveal a recurring assumption: the welding equipment must accept online commands. Where that assumption fails, as on many conventional MIG inverters, the sensing literature cannot be translated into closed-loop practice without an additional actuation interface.

#### 2.3.4 Process Modelling, Soft Sensing and Digital Twins

Physics-based models of welding heat sources and melt pools provide a complementary route to geometry prediction. The double-ellipsoidal heat source of Goldak and related semi-analytical calibrations remain standard tools for linking current, voltage and travel to fusion-zone estimates [24], [25]. Numerical studies of WAAM heat transfer and bead formation support process planning when experiment alone would be prohibitive [14]. High-fidelity fluid models, however, rarely run at the rates required for segment-by-segment feedback on a shop floor.

Soft sensors and data-driven estimators address that speed gap by mapping measurable process quantities to bead width and height without resolving the full melt-pool field in real time [17], [18]. Digital-twin architectures formalise the coupling of such estimators with the physical cell. Kim et al. develop a digital-twin architecture for wire-arc directed energy deposition under the ISO 23247 framing, with bidirectional data flow and anomaly detection [10]. Skryhunets et al. investigate multi-layer geometry simulation intended for digital-twin integration, noting the need for simulation times comparable to process duration if planning loops are to remain practical [11]. Knapp et al. earlier articulated the building blocks of an additive-manufacturing digital twin, including synchronisation between physical and virtual states [26].

The emerging pattern is therefore clear: detailed models inform and train; compact estimators and supervisory controllers act online; digital-twin frameworks organise the data exchange. What remains thin in the literature is the explicit grounding of a sim-trained soft sensor for geometry feedback on a conventional MIG power source that must first be opened to computer command.

### 2.4 Research Gaps

Despite extensive reviews of WAAM hardware, defects, monitoring and modelling [1]–[6], [12], several gaps remain relevant to the JKUAT cell.

First, closed-loop demonstrations frequently presume industrial welders with native fieldbus access. Methods for commanding a conventional MIG inverter during deposition, and for verifying commanded setpoints against measured arc quantities, are less often treated as first-class research objectives.

Second, monitoring studies outnumber closed-loop studies [6]. Acoustic and electrical sensing can flag instability, yet few published pipelines convert those streams, together with a geometry intent taken from the part file, into bounded setpoint updates on a non-fieldbus power source.

Third, high-fidelity melt-pool simulation and real-time control are often discussed as if they were the same layer of the stack. In practice, detailed models are too slow for control-rate inference. Soft sensors trained on simulation exist [18], but proposals that force validation of the teacher model before distillation, and that retire out-of-range melt-pool pyrometry as the primary geometry gate, remain uncommon in student-scale cell implementations.

Fourth, geometric intent is not always made explicit. Many cells treat bead width and layer height as operator knowledge rather than as quantities extracted from G-code and linked by volume balance to wire-feed speed through the filler-wire cross-section. Without that link, supervisory error has no unambiguous reference.

This project addresses those gaps on the existing JKUAT robotic WAAM cell by combining G-code geometric intent, computer command of the Rustler EM250i, an offline validated melt-pool model, a distilled soft sensor, and a supervised versus open-loop experimental comparison.

---

## 3 Methodology

This section describes the methods that will be used to accomplish the stated objectives. The study will be conducted on the existing robotic WAAM cell at Jomo Kenyatta University of Agriculture and Technology. The robot path planner, operator interface and monitoring channels already available on the cell will be retained. The welding power source will be the Rustler EM250i. Software previously written for a different inverter will not be reused.

### 3.1 System Overview

The proposed system takes a part from a computer-aided design model to a supervised deposit through five coordinated functions. First, the model is sliced to G-code with a qualified process profile, and intended bead width and layer height are read from that file. Second, a nominal welding schedule is derived so that the volume of wire delivered per unit length is consistent with the intended bead cross-section. Third, the cell computer issues voltage, inductance and wire-feed speed to the Rustler EM250i over a dedicated communication link and records electrical, motion and acoustic measurements on a common time base. Fourth, an offline first-principles melt-pool model is validated and used to train a soft sensor that estimates bead geometry from live measurements. Fifth, bounded corrections are applied to subsequent weld segments whenever the estimated geometry departs from the G-code intent.

The system does not attempt to execute a full melt-pool fluid simulation inside the welding loop. Detailed simulation is reserved for offline generation of labelled process-to-geometry data. Online estimation is performed by the soft sensor. Path planning remains with the existing robot controller; only welding setpoints are supervised.

Success is defined as a supervised multi-layer deposit that remains closer to the intended bead width and layer height than an otherwise identical open-loop deposit, without a worse incidence of arc loss or gross surface defects.

### 3.2 Job Preparation and Nominal Parameter Derivation

A computer-aided design model of each trial part will be sliced to G-code using a process profile already qualified on the cell. The existing path transpiler will continue to extract torch path, programmed travel speed and arc on–off state from that file. In addition, intended bead width \(W^*\) and intended layer height \(H^*\) will be read from the G-code and the associated process profile. These two quantities constitute the geometric reference that the supervisor will track.

Nominal wire-feed speed will then be obtained from volume balance. The cross-sectional area of the filler wire is

\[
A_w = \frac{\pi d_w^2}{4},
\]

where \(d_w\) is the wire diameter stated on the process sheet. If the intended bead cross-section is taken as \(A_b \approx W^* H^*\) (or that product multiplied by a shape factor when a non-rectangular profile is assumed), conservation of deposited volume per unit time requires

\[
A_w \, v_{\mathrm{wire}} = A_b \, v_{\mathrm{travel}},
\]

and therefore

\[
v_{\mathrm{wire}} = \frac{W^* \, H^* \, v_{\mathrm{travel}}}{A_w}.
\]

Nominal voltage and inductance will be taken from the qualified schedule corresponding to that wire-feed and travel combination, or from characterisation already performed on the Rustler EM250i. The resulting schedule is the open-loop baseline and the centre of the parameter region later explored when generating soft-sensor training data. At this stage no closed-loop correction is applied.

### 3.3 Command of the Welding Power Source

A communication link will be established between the cell computer and the Rustler EM250i, beginning from the serial connection between the inverter and its control panel as characterised for this machine. Through that link the computer will issue welding voltage, inductance and wire-feed speed for each deposition segment.

Commanded values will be verified against measured arc voltage and current so that actuation is confirmed rather than assumed. Failure of the link, disagreement beyond an agreed tolerance between command and measurement, or loss of arc will disable automatic updates until the process is restored to a safe state. The robot motion programme will not be rewritten by the supervisor; only welding setpoints will change.

### 3.4 In-Situ Sensing and Data Acquisition

During every trial the following quantities will be recorded with a shared time stamp: commanded and measured electrical signals, wire-feed related measurements, robot pose and travel speed, and acoustic features used as a process-stability indicator. Temperature channels will be logged where the available instruments can read reliably, for example in cooler regions of the build or in the heat-affected zone.

Peak melt-pool temperature on this process commonly exceeds the usable range of the cell’s existing pyrometer. Agreement between a modelled melt-pool temperature and a pyrometer reading will therefore not be used as the primary condition that unlocks geometry feedback. Acoustic and electrical instability flags will instead be used to pause or inhibit corrective updates when the arc is not in a trustworthy state.

### 3.5 Offline Melt-Pool Modelling and Soft-Sensor Development

A first-principles melt-pool model will relate process inputs—current, voltage or heat input, wire-feed speed, travel speed and torch path—to predicted bead width, bead height and penetration depth. The model will be executed offline. Its role is to generate labelled examples and to support design of the soft sensor, not to complete a full simulation of each segment before the robot reaches the next waypoint.

Before any model output is used as training truth, three checks will be applied. Internal consistency will be required: within the explored process window, predicted trends must be physically sensible. Selected cases will be compared with an independent established numerical solver under the same process intent, and large unexplained disagreement will halt progress until the cause is resolved. Where sectioned or scanned beads from the cell are available, predicted geometry will be compared with measured geometry so that residual error is known and stated. Absolute reproduction of every published coupon from another laboratory is not required; knowledge of residual error on this cell is required.

Once those checks are satisfied, a soft sensor

\[
g:\ (I,\ V,\ v_{\mathrm{wire}},\ v_{\mathrm{travel}},\ \ldots)\ \longrightarrow\ (\hat{W},\ \hat{H},\ \hat{D})
\]

will be trained on the labelled set. Hold-out cases never used in training will be reserved for error reporting on width and height. Where a small number of real beads can be measured, a residual correction will be fitted so that estimates move toward the physical cell. If hold-out error remains large, the estimator will be simplified or the training set enlarged before automatic correction is enabled. The soft sensor will be evaluated online at a modest rate sufficient for segment-level supervision.

### 3.6 Supervisory Closed-Loop Correction

For each weld segment, or for each low-rate window along a longer bead, the geometry error

\[
e = y^* - \hat{y}
\]

will be formed, where \(y^* = (W^*, H^*)\) is the G-code intent and \(\hat{y} = (\hat{W}, \hat{H})\) is the soft-sensor estimate. A bounded update of the welding setpoints will then be applied,

\[
u \leftarrow \mathrm{sat}\bigl(u + K\, e\bigr),
\]

where \(u\) may include wire-feed speed, voltage and inductance, \(\mathrm{sat}(\cdot)\) enforces machine limits and rate limits, and \(K\) is a gain chosen from process characterisation. As a first-order guide, wire-feed speed will be preferred for height tracking and heat-related setpoints or travel for width tracking, while the inverter’s inner arc regulation remains intact. Corrections will act on the following admissible segment, not as high-bandwidth replanning of the robot path.

Automatic correction will be enabled only when the soft sensor has passed its hold-out checks, the computer–welder link is healthy, the arc is stable, and commanded setpoints remain inside the qualified process envelope. Otherwise the cell will revert to the last safe nominal schedule or pause deposition in accordance with existing safety practice.

### 3.7 Testing and Validation

One qualified multi-layer or multi-segment part will be deposited twice under otherwise identical conditions. The first trial will use the fixed nominal schedule of Section 3.2. The second will use the same starting schedule with the supervisory corrections of Section 3.6 enabled.

The two deposits will be compared using bead width and layer height measured after welding by section, scan or other metrology available on site; electrical traces of current, voltage and wire feed; recorded arc-loss events; and qualitative notes on spatter, humping and overflow. The supervised deposit will be judged successful if it remains closer to the intended bead from the G-code than the open-loop deposit, without a worse rate of process faults.

Prior to that comparison, the computer-to-welder link will be verified on dry and live trials, the volume-balance schedule will be checked against measured wire consumption where practical, and the soft sensor will be required to meet agreed hold-out error limits. Stages that fail their checks will be corrected before closed-loop claims are made.

---

## 4 Expected Outcomes

At the conclusion of this research, the following outcomes are anticipated:

1. **Computer-to-welder actuation.** A working communication link between the cell computer and the Rustler EM250i that can change welding voltage, inductance and wire-feed speed during a print, with commanded values verified against measured arc quantities.

2. **Structured geometric job description.** A repeatable pipeline that extracts torch path, travel speed, arc state, intended bead width and intended layer height from G-code, and that derives a nominal wire-feed speed from the filler-wire cross-section and those geometric targets.

3. **Validated offline process model and soft sensor.** A melt-pool model checked against independent numerical results and sparse experimental beads, together with a soft sensor capable of estimating bead width, height and penetration from live process measurements at a rate suitable for supervision.

4. **Demonstrated supervisory improvement.** Experimental evidence, from open-loop versus supervised deposition of the same qualified part, that closed-loop correction keeps the deposit closer to the intended bead without degrading arc stability.

---

## References

[1] Y. Li, C. Su, and J. Zhu, “Comprehensive review of wire arc additive manufacturing: Hardware system, physical process, monitoring, property characterization, application and future prospects,” *Results in Engineering*, vol. 13, p. 100330, 2022.

[2] A. Shah, R. Aliyev, H. Zeidler, and S. Krinke, “A review of the recent developments and challenges in wire arc additive manufacturing (WAAM) process,” *Journal of Manufacturing and Materials Processing*, vol. 7, no. 3, p. 97, 2023.

[3] S. C. A. Costello, C. R. Cunningham, F. Xu, A. Shokrani, V. Dhokia, and S. T. Newman, “The state-of-the-art of wire arc directed energy deposition (WA-DED) as an additive manufacturing process for large metallic component manufacture,” *International Journal of Computer Integrated Manufacturing*, vol. 36, no. 3, pp. 469–510, 2023.

[4] M. Dinovitzer, X. Chen, J. Laliberte, X. Huang, and H. Frei, “Effect of wire and arc additive manufacturing (WAAM) process parameters on bead geometry and microstructure,” *Additive Manufacturing*, vol. 26, pp. 138–146, 2019.

[5] M. Dekis, M. Tawfik, M. Egiza, and M. Dewidar, “Challenges and developments in wire arc additive manufacturing of steel: A review,” *Results in Engineering*, vol. 26, p. 104657, 2025.

[6] M. Arjomandi, J. Motley, Q. Ngo, Y. Anees, M. A. Afzal, and T. Mukherjee, “A review on in-situ monitoring in wire arc additive manufacturing: Technologies, applications, challenges, and needs,” *Machines*, vol. 14, no. 1, 2026.

[7] N. Kozamernik, D. Bračun, and D. Klobčar, “WAAM system with interpass temperature control and forced cooling for near-net-shape printing of small metal components,” *The International Journal of Advanced Manufacturing Technology*, vol. 110, no. 7–8, pp. 1955–1968, 2020.

[8] C.-L. Lu et al., “Multi-robot scan-n-print for wire arc additive manufacturing,” *ASME Letters in Translational Robotics*, 2024.

[9] R. Bendia, F. Lizarralde, and F. Coutinho, “Multivariable closed-loop control for layer geometry in wire-arc additive manufacturing,” in *XV Simpósio Brasileiro de Automação Inteligente*, vol. 1, 2021.

[10] C. J. Kim et al., “Architecture development of digital twin-based wire arc directed energy deposition,” *International Journal of Precision Engineering and Manufacturing-Green Technology*, vol. 12, pp. 885–904, 2025.

[11] B. Denkena, V. Böß, K. M. Heide, A. Skryhunets, and T. Malek, “Modeling and simulation of multi-layer WAAM structures for digital twin integration,” *Journal of Manufacturing and Materials Processing*, vol. 10, no. 106, 2026.

[12] D. Liu, B. Lee, A. Babkin, and Y. Chang, “Research progress of arc additive manufacture technology,” *Materials*, vol. 14, no. 1415, 2021.

[13] N. J. R. Sagili Arthur, L. Liu, and J. P. Oliveira, “Design and development of open-source, multi-purpose wire arc additive manufacturing machine,” *The International Journal of Advanced Manufacturing Technology*, vol. 138, pp. 4337–4356, 2025.

[14] K. Oyama, S. Diplas, M. M’hamdi, A. E. Gunnæs, and A. S. Azar, “Heat source management in wire-arc additive manufacturing process for Al-Mg and Al-Si alloys,” *Additive Manufacturing*, vol. 26, pp. 392–402, 2019.

[15] F. K. Tesfaye, “Parameter optimization of gas metal arc welding process for welding dissimilar steels,” *American Journal of Mechanical and Industrial Engineering*, vol. 8, no. 1, pp. 1–6, 2023.

[16] Y. Zhang, J. Xiao, Z. Zhang, and H. Dong, “Intelligent design of robotic welding process parameters using learning-based methods,” *IEEE Access*, vol. 10, pp. 13432–13441, 2022.

[17] M. A. Kesse, E. Buah, H. Handroos, and G. K. Ayetor, “Development of an artificial intelligence powered TIG welding algorithm for the prediction of bead geometry for TIG welding processes using hybrid deep learning,” *Metals*, vol. 10, no. 451, 2020.

[18] M. S. So, M. M. Mahdi, D. B. Kim, and J.-H. Shin, “Prediction of metal additively manufactured bead geometry using deep neural network,” *Sensors*, vol. 24, no. 6250, 2024.

[19] N. A. Surovi and G. S. Soh, “Acoustic feature based geometric defect identification in wire arc additive manufacturing,” *Virtual and Physical Prototyping*, vol. 18, no. 1, e2210553, 2023.

[20] A. Rajesh, W. Ya, and M. Hermans, “Anomaly detection in WAAM deposition of nickel alloys—single-material and cross-material analysis,” *Metals*, vol. 13, no. 1820, 2023.

[21] F. Gao et al., “Mel spectrogram–driven deep learning framework for acoustic emission-based manufacturing monitoring in wire arc additive manufacturing,” *Journal of Reliability Science and Engineering*, vol. 2, p. 025301, 2026.

[22] M. Cudina, J. Prezelj, and I. Polajnar, “Use of audible sound for on-line monitoring of gas metal arc welding process,” *Metalurgija*, vol. 47, no. 2, pp. 81–85, 2008.

[23] F. Coutinho et al., “iWAAM: An automated system for monitoring and control of wire-arc additive manufacturing,” *IFAC-PapersOnLine*, vol. 56, no. 2, pp. 6576–6581, 2023.

[24] J. Goldak, A. Chakravarti, and M. Bibby, “A new finite element model for welding heat sources,” *Metallurgical Transactions B*, vol. 15, pp. 299–305, 1984.

[25] S. Trupiano et al., “A semi-analytical method for the calculation of double-ellipsoidal heat source parameters in welding simulation,” *IOP Conference Series: Materials Science and Engineering*, vol. 1214, p. 012023, 2022.

[26] G. L. Knapp et al., “Building blocks for a digital twin of additive manufacturing,” *Acta Materialia*, vol. 135, pp. 390–399, 2017.


---

## Appendices

### Appendix A: Volume Balance for Nominal Wire-Feed Speed

The filler wire is treated as a circular cylinder of diameter \(d_w\). Its cross-sectional area is

\[
A_w = \frac{\pi d_w^2}{4}. \tag{A.1}
\]

Over a short interval the volume of wire advanced into the arc is \(A_w v_{\mathrm{wire}}\,\Delta t\). Over the same interval the torch advances a distance \(v_{\mathrm{travel}}\,\Delta t\), and the intended solid bead occupies a cross-sectional area \(A_b\). Equating delivered and deposited volume gives

\[
A_w v_{\mathrm{wire}} = A_b v_{\mathrm{travel}}. \tag{A.2}
\]

Taking \(A_b = W^* H^*\) for a rectangular idealisation of the intended bead yields

\[
v_{\mathrm{wire}} = \frac{W^* H^* v_{\mathrm{travel}}}{A_w}. \tag{A.3}
\]

If a shape factor \(\kappa\) is introduced to represent a non-rectangular bead profile, \(A_b = \kappa W^* H^*\) and the same relation holds with that substitution. Mass-rate form \(\dot{m} = \rho A_w v_{\mathrm{wire}}\) is equivalent when density is constant and spatter losses are neglected; those losses, where measured, may be absorbed into an efficiency factor on the right-hand side of (A.2).

### Appendix B: Equipment and Facilities

| Item | Availability | Role |
|------|--------------|------|
| JKUAT robotic WAAM cell (KUKA motion) | Available | Torch path execution |
| Rustler EM250i MIG power source | Available | Arc power and wire feed |
| Existing G-code / path transpiler | Available | Job preparation |
| Electrical and acoustic sensing channels | Available / to be completed | In-situ observation |
| Computing workstation for modelling and soft-sensor training | Available | Offline simulation and estimator development |
| Bead metrology (section / scan as available) | Departmental / project arrangement | Open-loop versus supervised comparison |

### Appendix C: Work Packages (Indicative)

| Work package | Focus |
|--------------|--------|
| 1 | Literature consolidation and final problem framing |
| 2 | Computer-to-EM250i communication and electrical verification |
| 3 | G-code geometric intent and nominal volume-balance schedule |
| 4 | Offline melt-pool model validation |
| 5 | Soft-sensor training, hold-out testing and residual fit |
| 6 | Supervisory correction and gated enable logic |
| 7 | Open-loop versus supervised experimental comparison |
| 8 | Documentation and thesis writing |
