

# Q8. Large-Scale Weather and Environmental Data Analytics

## Input

First line:

```
N K S
```

Next  $N$  lines:

```
timestamp station_id temperature humidity pressure rainfall wind_speed
```

## Required Computations

- Total measurements.
- Average, minimum, and maximum temperature, humidity, and pressure.
- Total and maximum rainfall; average and maximum wind speed.
- Top- $K$  stations by measurement count, with average temperature and total rainfall.
- Hottest and coldest measurements.
- Busiest 60-second interval.
- Number of measurements with temperature  $\geq 40.0$  or  $\leq 0.0$ .

Top- $K$  stations are sorted by decreasing count, then increasing station ID. For hottest/coldest measurements, ties use smaller timestamp, then station ID.

## Output

```
TOTAL_MEASUREMENTS <value>
AVERAGE_TEMPERATURE <value>
MIN_TEMPERATURE <value>
MAX_TEMPERATURE <value>
AVERAGE_HUMIDITY <value>
MIN_HUMIDITY <value>
MAX_HUMIDITY <value>
AVERAGE_PRESSURE <value>
MIN_PRESSURE <value>
MAX_PRESSURE <value>
TOTAL_RAINFALL <value>
MAX_RAINFALL <value>
AVERAGE_WIND_SPEED <value>
MAX_WIND_SPEED <value>
EXTREME_TEMPERATURE_EVENTS <value>
HOTTEST_MEASUREMENT <temperature> <station_id> <timestamp>
COLDEST_MEASUREMENT <temperature> <station_id> <timestamp>
BUSIEST_INTERVAL <interval_id> <count>
TOP_STATIONS
<station_id> <measurement_count> <average_temperature> <total_rainfall>
...
```

## Dataset Generation and Benchmarking

The supplied examples are for correctness testing. Generate larger reproducible datasets using a fixed seed and document the generation method, parameters, sizes, and assumptions.

Benchmark both implementations using multiple input sizes and MPI process counts. Report execution time, speedup/efficiency, and a short analysis of computation, communication, data distribution, and scalability.

## **Deliverables for this section**

- Sequential implementation, MPI implementation, and dataset generator.
- README with compilation, execution, generation, and benchmark instructions.
- Correctness verification using the provided samples.
- Reproducible dataset-generation code/procedure.
- Benchmark results for multiple input sizes and process counts.
- Speedup/efficiency plots and a short performance analysis.