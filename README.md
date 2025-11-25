
# **AMI & GIS Data Processing Toolkit**

A Python module for **processing, cleaning, integrating, and analyzing Advanced Metering Infrastructure (AMI) datasets** and **GIS-based electrical network data**.
It provides unified structures for working with:

* **Daily or sub-daily consumption data**
* **Voltage profiles**
* **Active/reactive power profiles**
* **Geospatial meter locations and circuit layers**
* **Energy deltas, max demand, and phase-level measurements**

The toolkit is designed for **utility data analytics**, **distribution network studies**, and **research applications** involving smart meters.

Author: Mario R. Peralta A. <br>
Contact: mario.peralta@ieee.org


## Core Features

1. Geospatial Integration: Load, process, and visualize GIS layers with automatic CRS handling and interactive mapping via Folium
1. Multi-source Data Handling: Dedicated classes for consumption, voltage, and power data from AMI meters
1. Phase Analysis: Extract and organize three-phase electrical measurements (voltage, current, power factor)
1. Energy Profiling: Daily and time-series energy increment analysis with delivered/received decomposition
1. Scenario Generation: Create calendar-aware and seasonal load profiles for microgrid analysis
1. Data Standardization: Automatic column renaming, type conversion, and temporal alignment

## Key Applications

### 1. Load Profiling & Energy Analytics

Use `ConsumptionData` to compute:

* Daily **ΔkWh consumption** (“Delivered” & “Received”)
* Daily **max demand** (MDE)
* Long-term customer load patterns
* Customer segmentation based on consumption behavior
* Weekday vs weekend load shape analysis


### 2. Voltage Quality & Power Quality Monitoring

Using `VoltageData`, perform:

* Daily voltage profile reconstruction
* Phase-specific voltage analytics
* Voltage deviation detection
* Compliance monitoring (e.g., per EN50160 / IEEE Std. 1159)
* Aggregation of voltage data across customers or feeders


### 3. Feeder-Level Power Flow Behavior

`PowerData` allows:

* Reconstruction of **15-min (or 5/10-min) active/reactive profiles**
* Identification of reverse power flow (net generation periods)
* Feeder demand forecasting inputs
* DER hosting capacity analysis (using aggregated kWh/kVArh intervals)


### 4. GIS-Integrated AMI Analysis

**GISCircuit** + any AMI class enables:

* Mapping each AMI device to its **actual network location** (NISE → geometry)
* Overlaying customers, transformers, and conductors
* Visualizing feeders, transformers, phases, and loads
* Conducting geospatial queries:

  * Neighboring customer analysis
  * Phase balancing
  * Spatial clustering


### 5. Utility Engineering Studies

This module is suited for:

* **EV integration studies** (spatial + temporal loads)
* **PV hosting analysis** (AMI reverse flow detection)
* **Feeder reconstruction** with AMI + GIS
* **Transformer loading estimates** using AMI kW data
* **Loss analysis** with reactive energy datasets

---


## Architecture Overview

### **1. Abstract Interfaces**

* `GIS` — contract for loading spatial layers
* `AMI` — contract for loading and normalizing AMI datasets

These ensure every dataset uses the same data preparation workflow.


### **2. GISCircuit**

Reads all GIS layers (shapefiles) under a folder:

* Auto-imports GIS layers
* Applies CRS conversions (EPSG:5367 → EPSG:4326 when needed)
* Adds consistent coloring
* Provides an interactive **Folium map** of the circuit

Useful for spatial analysis and visualization.


### **3. ConsumptionData**

Handles **daily energy**, **max demand**, **phase-level quantities**, and **geographic mapping**.

Processes:

* kWh delivered/received
* Δ daily energy
* Instantaneous kW, kVAr, kVA
* Voltages, currents, power factor
* Phase count per customer

Outputs:

* `energy_df`: Daily ΔkWh table
* `power_data`, `voltage_data`, `pf_data`, `current_data`
* `ami_gdf`: Geospatial AMI location points


### **4. VoltageData**

Processes high-frequency voltage snapshots:

* Per-phase RMS voltage
* Time-ordered voltage dataframe
* Ready for voltage regulation studies


### **5. PowerData**

Handles active and reactive energy (kWh/kVArh):

* Separates **Delivered** / **Received**
* Builds interval-based active and reactive datasets
* Suitable for forecasting, feeder modeling, and DER analysis

---

## Example Workflow

### **1. Load GIS layers**

```python
gis = GISCircuit(gis_path="./GIS/*.shp")
m = gis.explore_network()
m
```

### **2. Load consumption data**

```python
cons = ConsumptionData(data_path="./Data/Consumption", gis=gis)
cons.energy_df.head()
```

### **3. Map AMI devices to geographic points**

```python
ami_points = cons.put_geometry()
ami_points.plot()
```

### **4. Load voltage profiles**

```python
volt = VoltageData(data_path="./Data/Voltages", gis=gis)
volt.voltage_df.head()
```

### **5. Load interval power data**

```python
pwr = PowerData(data_path="./Data/Power", gis=gis)
pwr.active_df.head()
```

---

## Highlights & Strengths

* Automates cleaning of dirty AMI datasets
* Standardizes timestamps, IDs, and formats
* Unifies multiple AMI types under the same interface
* Interoperable with GIS for spatial analysis
* Fast (uses Parquet + vectorized Pandas operations)
* Ready for downstream ML, clustering, or time-series analysis

---

## Typical Use Cases

| Application                 | Classes Used                   |
| --------------------------- | ------------------------------ |
| Build daily load profiles   | `ConsumptionData`              |
| Voltage compliance analysis | `VoltageData`                  |
| DER impact studies          | `ConsumptionData`, `PowerData` |
| GIS + AMI visualization     | `GISCircuit` + any AMI class   |
| Clustering of load shapes   | `energy_df`, `active_df`       |
| Transformer loading         | `power_data`, `kWh_df`         |
| Load forecasting            | All power/energy datasets      |

---

## Project Structure

```
/GIS               Shapefiles for network
/Data
    /Consumption   Daily energy + profile data
    /Voltages      Sub-daily voltage readings
    /Power         Interval power/energy data
ami.py             (GISCircuit, ConsumptionData, VoltageData, PowerData, ScenariosManager)
README.md
```

---
