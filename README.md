# 🎓 Student Performance Prediction System

A professional, high-performance web dashboard built with **Streamlit** designed to analyze, visualize, and predict student performance matrices. The system integrates advanced runtime configurations, interactive prediction forms, dynamic neural network simulation maps, and instant reporting downloads.

---

## 🚀 Features & Layout Architecture

The application is structured into a clean, borderless dark-themed multi-panel system containing the following modules:

* **👤 Input Section (Form-Locked):** An interactive input controller featuring robust sliders and inputs mapping key student vectors. It is securely locked inside a `st.form` module—preventing live dynamic reloads until the user explicitly hits the prediction trigger.
* **☊ Model Architecture:** Visual interactive breakdown of a Deep Neural Network mapping input layers through multi-layered dense stacks down to regression and classification targets.
* **🎯 Output Panel:** Live responsive panel generating the **Predicted Final Score**, **Risk Level Status** (with real-time conditional color mapping), and **Model Confidence Rating**.
* **📊 Dataset Overview:** Monitors structural counts dynamically reflecting the loaded dataset metadata shape (Rows, Features, Targets).
* **📈 Training Monitor:** Displays simulated loss decay curves and accuracy tracking visualizations mirroring structural deep learning iterations.
* **🍩 Risk Distribution Map:** Generates real-time custom donut charts visualizing categorical target splits matching user criteria.

---

## 🛠️ System Parameters & 9-Feature Configuration

The core prediction algorithm evaluates student status metrics based on the tracking architecture layout across key computational vectors, including:
1.  **Student ID** (`ST001`) - Relational Key Input
2.  **Attendance Percentage (%)** - Scale: 0 to 100
3.  **Assignment Average** - Scale: 0 to 100
4.  **Quiz Average** - Scale: 0 to 100
5.  **Study Hours Per Week** - Scale: 0 to 50
6.  **Previous GPA** - Scale: 0.0 to 10.0
7.  **Participation Score** - Scale: 1 to 10
8.  *Target 1 (Regression Component):* **Final Score Outcomes**
9.  *Target 2 (Classification Component):* **Risk Level States**

---

## 📂 Project Directory Structure

```text
student-performance-system/
│
├── data/
│   └── student_performance.csv   # Target dataset repository (e.g., 20 Rows / 9 Features)
│
├── app.py                        # Core Streamlit Application and Layout UI Code
├── README.md                     # Project Documentation File
└── requirements.txt              # Project Library Dependencies
```

💻 Technical Stack & Installation

Prerequisites
Make sure you have Python 3.9, 3.10, or 3.11 (64-bit) installed on your system for environment compatibility.

1. Clone the Project Repository
Open your terminal inside your project folder:
cd student-performance-system

 2. Install Project Dependencies
Install the required packages using the following command:
pip install streamlit pandas numpy matplotlib

 3. Run the Streamlit Web Application
Fire up the local runtime server using:
streamlit run app.py

 ⚙️ Advanced Customizations Embedded

Anti-Border UI Engine: Overrides standard Streamlit styling wrappers (stVerticalBlock) to enforce clean, nested-box-free flat layouts aligned with enterprise design wireframes.

State Retention: Leverages st.session_state to decouple input alterations from immediate calculations, recalculating values purely on specific user command events.

Export Engine: Includes data formatting using StringIO streams to package calculated predictive indexes into immediate .csv file downloads.

Dashboard Overview
<img width="1285" height="629" alt="Image" src="https://github.com/user-attachments/assets/648762ff-cf6a-4367-aff0-cef49191c687" />

<img width="1285" height="615" alt="Image" src="https://github.com/user-attachments/assets/0d88dec6-e8f9-4c41-ae25-9ae3f6b6b21d" />

