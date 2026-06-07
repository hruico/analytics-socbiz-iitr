## **Society of Business** 

**Open Project 2026** Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks Using Large-Scale Charging Session Data. 

## **Problem Statement** 

The rapid electrification of mobility has exposed a critical gap in EV charging infrastructure static, fixed-rate tariff models that remain blind to real-world operational dynamics. As EV adoption accelerates across India and globally, charging stations operating on flat ₹/kWh pricing face mounting inefficiencies: peak-hour congestion, charger underutilization during off-peak windows, rising electricity procurement costs, and deteriorating user experience. 

To solve this, this project utilizes real-world session data from largescale adaptive networks to build an Agentic AI framework. This selfimproving pricing engine autonomously predicts demand, recommends dynamic tariffs in real time, and continuously learns from outcomes to maximize revenue, balance grid demand, and optimize EV infrastructure efficiency. 

## **Objective** 

Given real-time and historical charging session data, predict and optimize: 

- **Demand Forecast:** How will charging demand and station utilization vary across time of day, day of week, and location? **Dynamic Tariff:** What is the optimal per-kWh tariff to maximize revenue while minimizing congestion and wait times? 

- **Charger Utilization:** Which stations are underutilized or overloaded, and when? 

- **Congestion & Wait Time Reduction:** Smoother demand distribution across time slots, directly reducing peak-hour queues and user wait times 

- **Autonomous Pricing Intelligence:** A self-improving system that continuously refines tariff decisions through the Monitoring & Learning Agent's feedback loop 

**2** 

## **Datasets** 

You will work with two official datasets: 

These datasets are given in the OpenProject Drive. But you are free to explore these sites in depth. 

## 1. **ACN-Data (Adaptive Charging Network)** 

- a.URL: https://ev.caltech.edu/dataset.html 

- b.Coverage: 30,000+ EV charging sessions from Caltech and JPL sites 

- c.Format: JSON - convert it into CSV for analysis 

- d.Use: Core charging session data; timestamps, energy delivered, session duration, station IDs, user behavior 

- e.Location: Caltech/JPL/US workplace sites 

## 2. **UrbanEV Dataset (ST-EVCDP)** 

## - a.URL: https://github.com/IntelligentSystemsLab/ST EVCDP 

- b.Coverage: 24,798 charging piles, 5-minute interval data, large-scale urban charging 

- c.Format: CSV 

- d.Use: Temporal demand variation, spatial charging patterns, peak-hour analysis 

- e.Location: Shenzhen, China 

**3** 

## **Overview** 

## **Data Preprocessing** 

- Align all datasets by timestamp, station ID, and session granularity to create a unified analytical base 

- Engineer economically meaningful features including Charger Utilization Rate (Charging Time / Total Available Time), Revenue per Session, Energy Cost per kWh, Queue Length Proxy, and Occupancy Density 

- Apply transparent missing value handling strategies with documented assumptions at each stage 

## **Exploratory Data Analysis (EDA)** 

- Examine long-run demand trends and short-run utilization fluctuations across station types and geographies 

- Profile temporal charging behavior intraday cycles, weekday vs. weekend patterns, and fleet vs. public usage signatures 

- Quantify volatility and stability differences across peak, shoulder, and off-peak periods 

- All visualizations are insight-driven, well-labeled, and tied directly to pricing implications 

## **Agents/ML Modelling** 

- **Demand Prediction Agent:** Model future charging demand and station utilization using ML models on historical session features; outputs include predicted utilization rate, congestion probability, and expected charging load 

- **Tariff Pricing Agent:** Translate demand forecasts into optimal dynamic tariffs, with surge pricing recommendations when utilization exceeds 80% and discount signals when it falls below 30% 

- **Monitoring & Learning Agent:** Systematically evaluate each pricing decision against live operational outcomes like revenue generated, charger utilization achieved, customer wait times, and pricing efficiency. 

**4** 

## **Evaluation Metrics** 

## **Demand Prediction Agent** 

- **RMSE:** Penalizes large errors in predicted station utilization or charging load 

- **MAE:** Average absolute error in predicted demand across time slots 

- **R² Score:** How well the model explains variance in actual charging demand. 

## **Tariff Pricing Agent** 

- **Revenue Gain %:** ((New Revenue − Old Revenue) / Old Revenue) × 100, compared against the ₹15/kWh fixed baseline 

- **Charger Utilization Rate:** Charging Time / Total Available Time, measured before and after dynamic pricing 

- **Off-Peak Uplift:** Increase in sessions during low-demand periods (utilization < 30%) after discount pricing is applied 

## **Monitoring & Learning Agent** 

- **Average Waiting Time Reduction:** Reduction in queue length across peak periods, tracked by the monitoring agent over evaluation episodes 

- **Customer Response Rate:** Shift in session volume in response to tariff changes (demand elasticity proxy) 

- **Pricing Efficiency Score:** Revenue per kWh delivered, tracked over time to measure if the feedback loop is improving decisions 

**5** 

## **Deliverables** 

## **Code / Notebooks** 

Clean, well-structured, and reproducible code. 

- Any outputs, such as scores, values, or rankings, can be placed in separate CSV files within the submission folder. 

## **Presentation Deck (5-7 slides) ( excluding cover page, executive summary and appendix )** 

- Data landscape and preprocessing decisions 

- Key EDA findings and demand behavior insights 

- Demand prediction modeling and results 

- Dynamic tariff optimization logic and pricing outcomes Monitoring agent evaluation and feedback loop performance Business, operational, and policy implications 

- Supporting visualizations for all major conclusions 

- Additional analysis and robustness checks in appendix 

## **Important Notes** 

- Causal claims should be avoided unless clearly justified Transparency in assumptions and limitations is expected 

## **Timeline** 

## **25th May:** Case Release 

**5th June:** Google Form for submission will be released 

(Submissions once made cannot be altered) 

**7th June:** Deadline for Submission 

## **10th June:** Result 

**6** 

