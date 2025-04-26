
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
