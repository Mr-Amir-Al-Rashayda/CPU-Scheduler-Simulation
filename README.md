# CPU Scheduler Simulation

## Overview

This project implements a **CPU Scheduler Simulation** using Python. It simulates the behavior of an operating system's CPU scheduler, handling multiple processes with varying arrival times, priorities, and burst types (CPU and I/O). The system incorporates a priority-based scheduling algorithm with Round Robin within each priority level, and basic resource management (requesting and releasing resources).

## Description

The simulation reads process definitions from an input file (`Processes.txt`), each describing a process's ID, arrival time, priority, and a sequence of bursts. Bursts can be CPU execution, I/O operations, resource requests, or resource releases. The scheduler manages a ready queue organized by priority, an I/O queue, and tracks resource allocation. It steps through time units, executing processes based on the scheduling algorithm and handling resource contention.

## Files

*   `OS_Amir_Nour.py`: The main Python script containing the `Process`, `Resource`, and `Scheduler` classes, as well as the simulation logic, input parsing, and output generation.
*   `Processes.txt`: The input file containing the definitions of processes to be simulated.
*   `TestCases.txt`: Contains example process definitions and scenarios, illustrating the input file format and various test cases.

## Features

*   Reads process data from a text file (`Processes.txt`).
*   Implements a CPU scheduler combining **Priority Scheduling** and **Round Robin** (within each priority level).
*   Handles **CPU bursts** (execution time).
*   Handles **I/O bursts** (waiting time for I/O completion).
*   Manages **Resources**, supporting **Request (R)** and **Release (F)** operations.
*   Maintains a **Priority Ready Queue** and an **I/O Queue**.
*   Simulates execution time step by time step.
*   Calculates and displays key performance metrics: **Average Waiting Time** and **Average Turnaround Time**.
*   Generates a **visual Gantt Chart** using `matplotlib` to show CPU utilization, I/O activity, and ready queue states over time.
*   Generates detailed **trace logs** showing system state changes at each time step, written to `scheduler.log` and `trace_logs.txt`.
*   Includes robust **input validation** for process definitions.

## Input File Format (`Processes.txt`)

Each line in `Processes.txt` defines a single process with the following format:

`PID ARRIVAL_TIME PRIORITY BURST1 BURST2 ...`

*   `PID`: A unique non-negative integer (Process ID).
*   `ARRIVAL_TIME`: A non-negative integer indicating when the process enters the system.
*   `PRIORITY`: A non-negative integer representing the process priority (lower number usually indicates higher priority, as per typical OS conventions).
*   `BURSTs`: A sequence of burst definitions, separated by spaces. Each burst is defined as `TYPE{operations}`.
    *   `TYPE`: Can be `CPU` or `IO`.
    *   `operations`: A comma-separated list of operations within the burst.
        *   For `CPU` bursts:
            *   Numeric value (e.g., `50`): CPU execution time in time units.
            *   `R{ResourceID}`: Request for the resource with `ResourceID`.
            *   `F{ResourceID}`: Release of the resource with `ResourceID`.
        *   For `IO` bursts: Usually a single numeric value representing the I/O duration (e.g., `IO{30}`).

Example (`Processes.txt` based on `TestCases.txt` and code parsing):
0 0 1 CPU{R{1}, 50, F{1}}
1 5 1 CPU{20} IO{30} CPU{20, R{2}, 30, F{2}, 10}


## Requirements

*   Python 3.x
*   Standard Python libraries (`re`, `collections`, `heapq`, `dataclasses`, `typing`, `logging`, `time`, `contextlib`)
*   `matplotlib` library for Gantt chart visualization (`pip install matplotlib`).
*   The input file `Processes.txt` must exist in the same directory as the script.

## How to Build and Run

1.  Save the provided Python code content as `OS_Amir.py`.
2.  Create the input file `Processes.txt` in the same directory, following the format described above. You can use examples from `TestCases.txt`.
3.  Ensure `matplotlib` is installed (`pip install matplotlib`).
4.  Open a terminal or command prompt.
5.  Navigate to the directory where you saved the files.
6.  Run the script using the Python interpreter:
    ```bash
    python OS_Amir.py
    ```
7.  Observe the simulation progress printed to the console and logged to `scheduler.log`. A `matplotlib` window will pop up showing the Gantt chart after the simulation completes. A `trace_logs.txt` file will also be created with detailed step-by-step logs.

## Implementation Details

*   **Process Class:** Represents a process with attributes like PID, arrival time, priority, state, remaining burst time, and burst operations.
*   **Resource Class:** Represents a resource with an ID, tracking which process holds it and which processes are waiting for it.
*   **Scheduler Class:** Manages the system state. It uses a `defaultdict(deque)` for the priority ready queues and a `heapq` (min-heap) to quickly access the highest priority level with ready processes. Round Robin is implemented using a `time_quantum` and rotating processes back to their queue ends. Resource requests and releases are handled, moving processes to/from a waiting state.
*   **Simulation Loop:** Increments time step by time step, managing the CPU, I/O, and ready queues, processing bursts, and handling state transitions.
*   **Gantt Chart:** The simulation records process activity over time, which is then visualized using `matplotlib` to show CPU usage, I/O blocks, and time spent in the ready queue.

## Output

The script produces:
1.  Console output and log file (`scheduler.log`, `trace_logs.txt`) detailing process state changes, scheduling decisions, burst executions, and resource events at each time unit.
2.  A final summary printed to the console including Average Waiting Time and Average Turnaround Time.
3.  A graphical Gantt chart displaying the timeline of process execution, I/O, and ready queue states.

## Author(s)

*   Name: Amir Al-Rashayda (ID: 1222596)
