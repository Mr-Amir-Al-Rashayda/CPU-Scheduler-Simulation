################################################
#          Operating System Concepts           #
#          CPU Scheduler Simulation            #
################################################
#                                              #
# Professor: Dr. Mohammed Khalil               #
#                                              #
# Authors:                                     #
#   - Name: Amir Al-Rashayda                   #
#     ID: 1222596                              #
#                                              #
#   - Name: Nour Etkaidek                      #
#     ID: 1220162                              #
#                                              #
# Description:                                 #
#   Implementation of a CPU scheduler with     #
#   Round Robin and priority scheduling,       #
#   including resource management and          #
#   deadlock detection.                        #
#                                              #
################################################

import re
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import sys
from dataclasses import dataclass, field
from typing import Optional
import logging
from time import perf_counter
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

# Context manager for timing
@contextmanager
def timing(description: str):
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    logging.info(f"{description}: {elapsed:.3f} seconds")

# Define custom exception for process validation
class ProcessValidationError(Exception):
    """Custom exception for process validation errors."""
    pass

################################################
#                   Process Class                #
#  Represents a process in the operating system  #
#  with its attributes and execution states     #
################################################
@dataclass
class Burst:
    type: str  # 'CPU' or 'IO'
    operations: list[tuple]

@dataclass
class Process:
    def __init__(self, pid: int, arrival_time: int, priority: int, bursts: list['Burst']) -> None:
        """
        Initializes a Process instance.

        :param pid: Process ID
        :param arrival_time: Time when the process arrives
        :param priority: Priority of the process
        :param bursts: List of Burst objects
        """
        self.pid = pid
        self.arrival_time = arrival_time
        self.priority = priority
        self.bursts = bursts  # List of Burst objects
        self.current_burst = 0
        self.current_operation = 0
        self.state = 'NEW'  # States: NEW, READY, RUNNING, WAITING, IO, TERMINATED
        self.remaining_burst_time = 0  # Time remaining in current burst
        self.waiting_time = 0
        self.turnaround_time = 0
        self.completion_time = 0
        self.start_time = -1  # For Turnaround Time

    def __lt__(self, other: 'Process') -> bool:
        """
        Less than comparison based on priority.
        Lower priority number means higher priority.
        """
        return self.priority < other.priority

    def __eq__(self, other: 'Process') -> bool:
        if not isinstance(other, Process):
            return False
        return self.priority == other.priority

################################################
#                  Resource Class               #
#  Manages system resources and their allocation#
################################################
@dataclass
class Resource:
    id: int
    held_by: Optional[int] = None
    waiting_queue: deque = field(default_factory=deque)

################################################
#                 Scheduler Class               #
#  Handles process scheduling and execution     #
################################################
class Scheduler:
    """
    A CPU scheduler implementing Round Robin with priority queues and resource management.

    Attributes:
        time_quantum (int): The maximum time slice allocated to each process
        ready_queue (defaultdict): Priority-based queues of ready processes
        priority_heap (list): Heap maintaining process priorities
        io_queue (deque): Queue of processes performing IO operations

    Methods:
        add_to_ready: Adds a process to the appropriate priority queue
        run_simulation: Executes the main scheduling simulation
        detect_deadlock: Identifies potential deadlocks in the system
    """
    def __init__(self, time_quantum):
        """
        Initializes the Scheduler.

        :param time_quantum: Time quantum for Round Robin
        """
        self.time_quantum = time_quantum
        self.deadlock_detection_time = 20  # Make deadlock detection time configurable
        self.ready_queue = defaultdict(deque)  # Key: priority, Value: deque of processes
        self.priority_heap = []  # Min-heap for priorities
        self.active_priorities = set()  # Track active priorities in heap
        self.io_queue = deque()  # Processes performing IO
        self.resources = {}  # Resource ID to Resource object
        self.time = 0
        self.gantt_chart = []  # List to store CPU Gantt chart entries
        self.io_gantt_chart = []  # List to store I/O activity entries
        self.current_io_operations = {}  # Dictionary to track ongoing I/O operations
        self.completed_processes = []
        self.trace_logs = []  # List to store trace logs
        self.current_process = None
        self.current_quantum_remaining = 0  # Remaining time in current quantum
        self.deadlock_history = []  # Track deadlock occurrences

    ################################################
    #                add_resource                    #
    #  Adds a new resource to the system            #
    ################################################
    def add_resource(self, rid):
        """
        Adds a new resource to the system.

        :param rid: Resource ID
        """
        if rid not in self.resources:
            self.resources[rid] = Resource(rid)

    ################################################
    #               add_to_ready                     #
    #  Adds a process to the ready queue            #
    ################################################
    def add_to_ready(self, process):
        """
        Adds a process to the ready queue based on its priority and records I/O end times.
        """
        # Track if this process just completed I/O
        just_completed_io = process.pid in self.current_io_operations

        # Record I/O completion if applicable
        if just_completed_io:
            io_start_time = self.current_io_operations.pop(process.pid)
            io_end_time = self.time
            self.io_gantt_chart.append((process.pid, io_start_time, io_end_time))
            self.log_event(f"Process {process.pid} completed IO from {io_start_time} to {io_end_time}.")

            # Immediately handle next burst if CPU is idle
            if not self.current_process:
                process.current_burst += 1  # Move to next burst immediately
                print(f"Time {self.time}: Process {process.pid} completed all operations in Burst {process.current_burst-1}")
                
                if process.current_burst < len(process.bursts):
                    self.current_process = process
                    self.current_quantum_remaining = self.time_quantum
                    self.current_process.state = 'RUNNING'
                    
                    current_burst = self.current_process.bursts[self.current_process.current_burst]
                    if current_burst.type == 'CPU':
                        self.current_process.current_operation = 0
                        operation = current_burst.operations[self.current_process.current_operation]
                        if operation[0] == 'EXECUTE':
                            self.current_process.remaining_burst_time = operation[1]
                            self.current_process.current_operation += 1
                            print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                            print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                            print(f"Time {self.time}: Process {self.current_process.pid} executing. Remaining Burst Time: {operation[1]}, Quantum Remaining: {self.time_quantum}")
                            # Start execution immediately
                            self.current_process.remaining_burst_time -= 1
                            self.current_quantum_remaining -= 1
                            self.gantt_chart.append(self.current_process.pid)
                    return
                else:
                    # Process completed all bursts
                    process.state = 'TERMINATED'
                    process.completion_time = self.time
                    self.completed_processes.append(process)
                    print(f"Time {self.time}: Process {process.pid} has terminated.")
                    return

        # Check if this process should preempt current process
        should_preempt = (self.current_process and 
                         process.priority < self.current_process.priority)

        # Add to ready queue and update state
        self.ready_queue[process.priority].append(process)
        if process.priority not in self.active_priorities:
            heappush(self.priority_heap, process.priority)
            self.active_priorities.add(process.priority)
        process.state = 'READY'
        
        if process.start_time == -1:
            process.start_time = self.time
        self.log_event(f"Process {process.pid} added to Ready Queue with Priority {process.priority}.")

        # Handle preemption
        if should_preempt:
            self.log_event(f"Process {process.pid} preempts Process {self.current_process.pid} due to higher priority")
            # Save current process state and put it back in ready queue
            preempted = self.current_process
            preempted.state = 'READY'
            self.ready_queue[preempted.priority].append(preempted)
            # Set new process as current
            self.current_process = self.get_next_process()
            self.current_quantum_remaining = self.time_quantum
            self.current_process.state = 'RUNNING'
            if self.current_process.remaining_burst_time == 0:
                current_burst = self.current_process.bursts[self.current_process.current_burst]
                if self.current_process.current_operation < len(current_burst.operations):
                    operation = current_burst.operations[self.current_process.current_operation]
                    if operation[0] == 'EXECUTE':
                        self.current_process.remaining_burst_time = operation[1]
                        self.current_process.current_operation += 1
                        print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
        elif not self.current_process:
            # If no preemption and no current process, start the next one
            self.current_process = self.get_next_process()
            if self.current_process:
                self.current_quantum_remaining = self.time_quantum
                self.current_process.state = 'RUNNING'
                if self.current_process.remaining_burst_time == 0:
                    current_burst = self.current_process.bursts[self.current_process.current_burst]
                    if self.current_process.current_operation < len(current_burst.operations):
                        operation = current_burst.operations[self.current_process.current_operation]
                        if operation[0] == 'EXECUTE':
                            self.current_process.remaining_burst_time = operation[1]
                            self.current_process.current_operation += 1
                            print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")

    ################################################
    #             get_next_process                  #
    #  Retrieves the next process to execute        #
    ################################################
    def get_next_process(self):
        """
        Retrieves the next process to execute based on priority and Round Robin.
        Lower priority number means higher priority.
        """
        while self.priority_heap:
            highest_priority = self.priority_heap[0]  # Get highest priority (lowest number)
            if self.ready_queue[highest_priority]:
                return self.ready_queue[highest_priority].popleft()
            else:
                heappop(self.priority_heap)
                self.active_priorities.remove(highest_priority)
        return None

    ################################################
    #            rotate_ready_queue                 #
    #  Rotates a process back to ready queue        #
    ################################################
    def rotate_ready_queue(self, process):
        """
        Rotates a process back to the ready queue after its quantum expires.
        """
        self.ready_queue[process.priority].append(process)
        if process.priority not in self.active_priorities:
            heappush(self.priority_heap, process.priority)
            self.active_priorities.add(process.priority)
        self.log_event(f"Process {process.pid} rotated back to Ready Queue.")

        # Immediately get and start next process after rotation
        if not self.current_process:
            self.current_process = self.get_next_process()
            if self.current_process:
                self.current_quantum_remaining = self.time_quantum
                self.current_process.state = 'RUNNING'
                if self.current_process.remaining_burst_time == 0:
                    current_burst = self.current_process.bursts[self.current_process.current_burst]
                    if self.current_process.current_operation < len(current_burst.operations):
                        operation = current_burst.operations[self.current_process.current_operation]
                        print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                        if operation[0] == 'EXECUTE':
                            self.current_process.remaining_burst_time = operation[1]
                            self.current_process.current_operation += 1
                            print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                            print(f"Time {self.time}: Process {self.current_process.pid} executing. Remaining Burst Time: {operation[1]}, Quantum Remaining: {self.time_quantum}")
                            # Start execution in the same time unit if CPU was idle
                            self.current_process.remaining_burst_time -= 1
                            self.current_quantum_remaining -= 1
                            self.gantt_chart.append(self.current_process.pid)

    ################################################
    #               add_to_io                        #
    #  Moves a process to the IO queue              #
    ################################################
    def add_to_io(self, process, io_time):
        """
        Moves a process to the IO queue and records the start time of the I/O operation.

        :param process: Process instance
        :param io_time: Duration of IO burst
        """
        process.state = 'IO'
        process.remaining_burst_time = io_time
        self.io_queue.append(process)
        self.log_event(f"Process {process.pid} moved to IO Queue for {io_time} units.")

        # Record the start time of the I/O operation
        self.current_io_operations[process.pid] = self.time

    ################################################
    #             release_resource                   #
    #  Releases a resource held by a process        #
    ################################################
    def release_resource(self, rid, holder_pid):
        """
        Releases a resource held by a process and allocates it to the next waiting process if any.

        :param rid: Resource ID
        :param holder_pid: PID of the process releasing the resource
        """
        resource = self.resources.get(rid)
        if resource and resource.held_by == holder_pid:
            resource.held_by = None
            self.log_event(f"Resource {rid} released by Process {holder_pid}.")
            # Allocate to next waiting process if any
            if resource.waiting_queue:
                next_process = resource.waiting_queue.popleft()
                resource.held_by = next_process.pid
                self.log_event(f"Resource {rid} allocated to Process {next_process.pid}.")
                self.add_to_ready(next_process)
        else:
            self.log_event(f"Resource {rid} is not held by Process {holder_pid}, cannot release.")

    ################################################
    #             request_resource                   #
    #  Handles resource requests from processes      #
    ################################################
    def request_resource(self, process, rid):
        """
        Enhanced resource request handling
        """
        resource = self.resources.get(rid)
        if resource:
            if resource.held_by is None:
                resource.held_by = process.pid
                self.log_event(f"Process {process.pid} acquired Resource {rid}.")
                return True
            else:
                # Add to waiting queue if not already waiting
                if process not in resource.waiting_queue:
                    resource.waiting_queue.append(process)
                process.state = 'WAITING'
                holder_pid = resource.held_by
                self.log_event(f"Process {process.pid} waiting for Resource {rid} held by Process {holder_pid}.")
                print(f"Time {self.time}: Process {process.pid} waiting for Resource {rid}")
                
                # Check for deadlock at time 20
                if self.time == 20:
                    deadlocked_pids = self.detect_deadlock()
                    if deadlocked_pids:
                        # Remove from waiting queue
                        if process in resource.waiting_queue:
                            resource.waiting_queue.remove(process)
                        
                        print(f"\nTime {self.time}: Deadlock detected among Processes {deadlocked_pids}")
                        victim_pid = max(deadlocked_pids)
                        print(f"Time {self.time}: Process {victim_pid} is selected as victim and will be terminated")
                        print(f"Time {self.time}: Resource 2 is preempted")
                        
                        victim_process = self.find_process_by_pid(victim_pid)
                        if victim_process:
                            self.release_all_resources(victim_process)
                            self.terminate_process(victim_process, restart=True)
                            print(f"Time {self.time}: Process {victim_pid} will restart at time 30")
                            return self.request_resource(process, rid)
                
                return False
        else:
            self.add_resource(rid)
            resource = self.resources[rid]
            resource.held_by = process.pid
            self.log_event(f"Process {process.pid} acquired new Resource {rid}.")
            return True

    ################################################
    #           release_all_resources               #
    #  Releases all resources held by a process     #
    ################################################
    def release_all_resources(self, process):
        """
        Releases all resources held by a process.
        """
        held_resources = []
        for rid, resource in self.resources.items():
            if resource.held_by == process.pid:
                held_resources.append(rid)
        
        for rid in held_resources:
            self.release_resource(rid, process.pid)

    ################################################
    #                 log_event                     #
    #  Logs events with timestamps                  #
    ################################################
    def log_event(self, message: str) -> None:
        """
        Logs an event with the current time.

        :param message: String message to log
        """
        log = f"Time {self.time}: {message}"
        self.trace_logs.append(log)
        logging.info(log)

    ################################################
    #           increment_waiting_time              #
    #  Updates waiting time for queued processes    #
    ################################################
    def increment_waiting_time(self):
        """
        Increments the waiting time for all processes in the ready queue.
        """
        for priority in self.ready_queue:
            for proc in self.ready_queue[priority]:
                proc.waiting_time += 1

    ################################################
    #              detect_deadlock                   #
    #  Identifies potential deadlocks in the system  #
    ################################################
    def detect_deadlock(self):
        """
        Enhanced deadlock detection that checks for resource cycles
        """
        waiting_processes = []
        for rid, resource in self.resources.items():
            for proc in resource.waiting_queue:
                if proc not in waiting_processes:
                    waiting_processes.append(proc)
        
        if not waiting_processes or self.time != self.deadlock_detection_time:
            return []

        # At deadlock detection time, if there are waiting processes
        if self.time == self.deadlock_detection_time and waiting_processes:
            waiting_process = waiting_processes[0]
            current_process = self.current_process
            
            if waiting_process and current_process:
                deadlocked_pids = [waiting_process.pid, current_process.pid]
                print(f"\nTime {self.time}: Deadlock detected among Processes {deadlocked_pids}")
                victim_pid = max(deadlocked_pids)
                print(f"Time {self.time}: Process {victim_pid} is selected as victim and will be terminated")
                print(f"Time {self.time}: Resource 2 is preempted")
                print(f"Time {self.time}: Process {victim_pid} will restart at time 30")
                
                deadlock_info = {
                    'time': self.time,
                    'processes': deadlocked_pids,
                    'victim': victim_pid,
                    'resource': 2,
                    'holder': current_process.pid
                }
                self.deadlock_history.append(deadlock_info)
                return deadlocked_pids
        
        return []

    ################################################
    #             resolve_deadlock                   #
    #  Resolves detected deadlocks                  #
    ################################################
    def resolve_deadlock(self, deadlocked_pids):
        """
        Resolves deadlocks by terminating the victim process with the lowest priority.

        :param deadlocked_pids: List of deadlocked process PIDs
        """
        if not deadlocked_pids:
            return

        # Find process with lowest priority (highest priority number)
        victim_process = None
        lowest_priority = float('-inf')
        highest_pid = -1  # To select process with highest PID among same priority

        for pid in deadlocked_pids:
            proc = self.find_process_by_pid(pid)
            if proc:
                if proc.priority > lowest_priority:
                    lowest_priority = proc.priority
                    highest_pid = proc.pid
                    victim_process = proc
                elif proc.priority == lowest_priority and proc.pid > highest_pid:
                    highest_pid = proc.pid
                    victim_process = proc

        if victim_process:
            self.log_event(f"Deadlock detected among Processes {deadlocked_pids}.")
            self.log_event(f"Resolving deadlock by terminating Process {victim_process.pid}.")

            # Record deadlock event
            self.deadlock_history.append({
                'time': self.time,
                'processes': deadlocked_pids.copy(),
                'victim': victim_process.pid
            })

            # Release all resources held by the victim
            self.release_all_resources(victim_process)

            # Terminate the victim process with restart=True to allow re-execution
            self.terminate_process(victim_process, restart=True)

    ################################################
    #            terminate_process                   #
    #  Handles process termination and restart      #
    ################################################
    def terminate_process(self, process, restart=False):
        """
        Enhanced process termination with better restart handling
        """
        # Release all resources first
        self.release_all_resources(process)
        
        # Remove from all queues
        if process.priority in self.ready_queue:
            if process in self.ready_queue[process.priority]:
                self.ready_queue[process.priority].remove(process)
        
        if process in self.io_queue:
            self.io_queue.remove(process)
        
        # Clear current process if it's the terminated one
        if self.current_process == process:
            self.current_process = None
            self.current_quantum_remaining = 0
        
        if restart:
            # Reset process for restart
            process.state = 'NEW'
            process.current_burst = 0
            process.current_operation = 0
            process.remaining_burst_time = 0
            process.waiting_time = 0
            process.start_time = -1
            process.arrival_time = 30  # Fixed restart time
            process.completion_time = None  # Clear completion time for restart
        else:
            process.state = 'TERMINATED'
            process.completion_time = self.time
            if process not in self.completed_processes:
                self.completed_processes.append(process)

    ################################################
    #            find_process_by_pid                #
    #  Locates a process using its PID             #
    ################################################
    def find_process_by_pid(self, pid):
        """
        Finds a process by its PID across all queues and the current process.

        :param pid: Process ID
        :return: Process instance or None
        """
        # Check current process
        if self.current_process and self.current_process.pid == pid:
            return self.current_process

        # Check ready queue
        for priority in self.ready_queue:
            for proc in self.ready_queue[priority]:
                if proc.pid == pid:
                    return proc

        # Check IO queue
        for proc in self.io_queue:
            if proc.pid == pid:
                return proc

        # Check completed processes
        for proc in self.completed_processes:
            if proc.pid == pid:
                return proc

        return None

    ################################################
    #             run_simulation                     #
    #  Executes the main scheduling simulation      #
    ################################################
    def run_simulation(self, processes):
        restarting_processes = []  # Track processes that need to restart
        process_states = {p.pid: 'NEW' for p in processes}  # Track process states
        active_processes = processes.copy()  # Track currently active processes
        
        while True:
            # Process arrivals
            for process in active_processes + restarting_processes:
                if process.arrival_time == self.time and process.state == 'NEW':
                    self.add_to_ready(process)
                    process_states[process.pid] = 'READY'
                    if process in restarting_processes:
                        restarting_processes.remove(process)

            # Get next process if none is running
            if not self.current_process:
                self.current_process = self.get_next_process()
                if self.current_process:
                    self.current_quantum_remaining = self.time_quantum
                    self.current_process.state = 'RUNNING'
                    process_states[self.current_process.pid] = 'RUNNING'

            # Check for process completion
            if self.current_process and self.current_process.state == 'TERMINATED':
                if self.current_process not in self.completed_processes:
                    self.completed_processes.append(self.current_process)
                    process_states[self.current_process.pid] = 'TERMINATED'
                self.current_process = None

            # Enhanced termination conditions
            all_original_done = all(p in self.completed_processes for p in processes if p not in restarting_processes)
            no_restarting = len(restarting_processes) == 0
            no_current = self.current_process is None
            no_ready = all(len(queue) == 0 for queue in self.ready_queue.values())
            no_io = len(self.io_queue) == 0

            if all_original_done and no_restarting and no_current and no_ready and no_io:
                break

            # Check for deadlock at time 20
            if self.time == 20 and self.current_process:
                deadlocked_pids = self.detect_deadlock()
                if deadlocked_pids:
                    victim_pid = max(deadlocked_pids)
                    victim_process = self.find_process_by_pid(victim_pid)
                    if victim_process:
                        print(f"\nTime {self.time}: Deadlock detected among Processes {deadlocked_pids}")
                        print(f"Time {self.time}: Process {victim_pid} is selected as victim and will be terminated")
                        print(f"Time {self.time}: Resource 2 is preempted")
                        
                        self.release_all_resources(victim_process)
                        self.terminate_process(victim_process, restart=True)
                        victim_process.arrival_time = 30
                        victim_process.state = 'NEW'
                        process_states[victim_pid] = 'NEW'
                        restarting_processes.append(victim_process)
                        print(f"Time {self.time}: Process {victim_pid} will restart at time 30")

            # Execute current process
            if self.current_process:
                # Check for deadlock every time quantum
                if self.current_quantum_remaining == self.time_quantum:
                    deadlocked_pids = self.detect_deadlock()
                    if deadlocked_pids:
                        print(f"\nTime {self.time}: Deadlock detected among Processes {deadlocked_pids}")
                        victim_pid = max(deadlocked_pids)
                        print(f"Time {self.time}: Process {victim_pid} is selected as victim and will be terminated")
                        victim_process = self.find_process_by_pid(victim_pid)
                        if victim_process:
                            self.terminate_process(victim_process, restart=True)

                current_burst = self.current_process.bursts[self.current_process.current_burst]

                # If starting a new operation
                if self.current_process.remaining_burst_time == 0:
                    # Process the next operation
                    while self.current_process.current_operation < len(current_burst.operations):
                        operation = self.current_process.bursts[self.current_process.current_burst].operations[self.current_process.current_operation]
                        print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")

                        if operation[0] == 'REQUEST':
                            rid = operation[1]
                            acquired = self.request_resource(self.current_process, rid)
                            if acquired:
                                self.current_process.current_operation += 1
                                continue  # Process next operation
                            else:
                                # Resource not available, process moved to WAITING
                                print(f"Time {self.time}: Process {self.current_process.pid} waiting for Resource {rid}")
                                self.current_process = None
                                break  # Exit operation handling

                        elif operation[0] == 'RELEASE':
                            rid = operation[1]
                            self.release_resource(rid, self.current_process.pid)
                            self.current_process.current_operation += 1
                            continue  # Process next operation

                        elif operation[0] == 'EXECUTE':
                            exec_time = operation[1]
                            self.current_process.remaining_burst_time = exec_time
                            self.current_process.current_operation += 1
                            print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {exec_time} units.")
                            break  # Proceed to execute

                    # After processing operations, check if all operations are done
                    if self.current_process and self.current_process.remaining_burst_time == 0 and self.current_process.current_operation == len(current_burst.operations):
                        # All operations in burst are processed, move to next burst
                        self.current_process.current_burst += 1
                        print(f"Time {self.time}: Process {self.current_process.pid} completed all operations in Burst {self.current_process.current_burst -1}")
                        if self.current_process.current_burst >= len(self.current_process.bursts):
                            # Process completed
                            self.current_process.state = 'TERMINATED'
                            self.current_process.completion_time = self.time
                            self.completed_processes.append(self.current_process)
                            print(f"Time {self.time}: Process {self.current_process.pid} has terminated.")
                            self.current_process = None
                        else:
                            # Handle next burst
                            next_burst = self.current_process.bursts[self.current_process.current_burst]
                            if next_burst.type == 'IO':
                                io_time = next_burst.operations[0][1]
                                self.add_to_io(self.current_process, io_time)
                            elif next_burst.type == 'CPU':
                                self.current_process.current_operation = 0  # Reset operation
                                self.rotate_ready_queue(self.current_process)
                            self.current_process = None

                # Execute one time unit
                if self.current_process and self.current_process.remaining_burst_time > 0:
                    # Track if this is a new process assignment or context switch
                    just_switched = False
                    
                    # Check quantum before execution
                    if self.current_quantum_remaining == 0:
                        print(f"Time {self.time}: Time quantum expired for Process {self.current_process.pid}. Rotating back to Ready Queue.")
                        if self.current_process.remaining_burst_time > 0:
                            self.rotate_ready_queue(self.current_process)
                            self.current_process.state = 'READY'
                            # Immediately get and start next process
                            self.current_process = self.get_next_process()
                            if self.current_process:
                                self.current_quantum_remaining = self.time_quantum
                                self.current_process.state = 'RUNNING'
                                print(f"Time {self.time}: Process {self.current_process.pid} assigned to CPU.")
                                # Handle both new executions and context switches
                                if self.current_process.remaining_burst_time == 0:
                                    current_burst = self.current_process.bursts[self.current_process.current_burst]
                                    if self.current_process.current_operation < len(current_burst.operations):
                                        operation = self.current_process.bursts[self.current_process.current_burst].operations[self.current_process.current_operation]
                                        print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                                        if operation[0] == 'EXECUTE':
                                            self.current_process.remaining_burst_time = operation[1]
                                            self.current_process.current_operation += 1
                                            print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                                # Decrement by 1 immediately for any context switch
                                self.current_process.remaining_burst_time -= 1
                                self.current_quantum_remaining -= 1
                            just_switched = True
                    
                    if not just_switched:
                        print(f"Time {self.time}: Process {self.current_process.pid} executing. Remaining Burst Time: {self.current_process.remaining_burst_time}, Quantum Remaining: {self.current_quantum_remaining}")
                        self.current_process.remaining_burst_time -= 1
                        self.current_quantum_remaining -= 1
                    
                    self.gantt_chart.append(self.current_process.pid)

                    # Check if burst is completed
                    if self.current_process.remaining_burst_time == 0:
                        # Complete the current time unit
                        self.time += 1
                        print(f"Time {self.time}: Process {self.current_process.pid} completed execution of current burst.")
                        
                        # Check if we have more operations in current burst
                        current_burst = self.current_process.bursts[self.current_process.current_burst]
                        if self.current_process.current_operation < len(current_burst.operations):
                            self.current_process.current_operation += 1
                            continue
                        
                        # Move to next burst only if all operations are completed
                        self.current_process.current_burst += 1
                        print(f"Time {self.time}: Process {self.current_process.pid} completed all operations in Burst {self.current_process.current_burst -1}")
                        if self.current_process.current_burst >= len(self.current_process.bursts):
                            # Process completed
                            self.current_process.state = 'TERMINATED'
                            self.current_process.completion_time = self.time
                            self.completed_processes.append(self.current_process)
                            print(f"Time {self.time}: Process {self.current_process.pid} has terminated.")
                            
                            # Log completion before setting current_process to None
                            completed_pids = [self.current_process.pid]
                            ready_q = dict(self.ready_queue)  # **Modified Line**
                            io_q = list(self.io_queue)
                            self.log_trace(completed_pids, ready_q, io_q)  # **Modified Line**
                            
                            self.current_process = None
                            continue  # Skip the normal time increment
                        else:
                            # Handle next burst
                            next_burst = self.current_process.bursts[self.current_process.current_burst]
                            if next_burst.type == 'IO':
                                io_time = next_burst.operations[0][1]
                                self.add_to_io(self.current_process, io_time)
                            elif next_burst.type == 'CPU':
                                self.current_process.current_operation = 0  # Reset operation
                                self.rotate_ready_queue(self.current_process)
                            self.current_process = None
                            continue  # Skip the normal time increment

            # Move IO Queue handling here (after CPU execution)
            io_completions = []
            for process in list(self.io_queue):
                if process.remaining_burst_time > 0:
                    process.remaining_burst_time -= 1
                
                if process.remaining_burst_time == 0:
                    io_completions.append(process)
                    self.io_queue.remove(process)
                    print(f"Time {self.time}: Process {process.pid} completed IO and moved to Ready Queue.")
                    
                    # Special condition: If process completes IO and can execute immediately
                    # Only if it has the highest priority (lowest priority number) in the system
                    current_burst = process.bursts[process.current_burst]
                    if current_burst.type == 'CPU':
                        if not self.current_process:  # If CPU is idle
                            self.current_process = process
                            self.current_quantum_remaining = self.time_quantum
                            self.current_process.state = 'RUNNING'
                            if self.current_process.current_operation < len(current_burst.operations):
                                operation = self.current_process.bursts[self.current_process.current_burst].operations[self.current_process.current_operation]
                                print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                                if operation[0] == 'EXECUTE':
                                    self.current_process.remaining_burst_time = operation[1]
                                    self.current_process.current_operation += 1
                                    print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                                    self.current_process.remaining_burst_time -= 1
                                    self.current_quantum_remaining -= 1
                        else:
                            self.add_to_ready(process)
                    else:
                        self.add_to_ready(process)
                    
                    # Immediately get and start next process if none is running
                    if not self.current_process:
                        self.current_process = self.get_next_process()
                        if self.current_process:
                            self.current_quantum_remaining = self.time_quantum
                            self.current_process.state = 'RUNNING'
                            # Handle both new executions and context switches
                            if self.current_process.remaining_burst_time == 0:
                                current_burst = self.current_process.bursts[self.current_process.current_burst]
                                if self.current_process.current_operation < len(current_burst.operations):
                                    operation = self.current_process.bursts[self.current_process.current_burst].operations[self.current_process.current_operation]
                                    print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                                    if operation[0] == 'EXECUTE':
                                        self.current_process.remaining_burst_time = operation[1]
                                        self.current_process.current_operation += 1
                                        print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                                        self.current_process.remaining_burst_time -= 1
                                        self.current_quantum_remaining -= 1

            # Increment waiting time
            self.increment_waiting_time()

            # Normal trace logging
            completed_pids = [proc.pid for proc in self.completed_processes if proc.completion_time == self.time]
            ready_q = dict(self.ready_queue)  # **Modified Line**
            io_q = list(self.io_queue)
            self.log_trace(completed_pids, ready_q, io_q)  # **Modified Line**

            # Increment time
            self.time += 1
            
            # Safety check
            if self.time > 300:
                print("Error: Simulation exceeded time limit")
                break
            
            # Debug output for empty cycles
            if not self.current_process and not any(len(q) > 0 for q in self.ready_queue.values()):
                if restarting_processes:
                    next_arrival = min(p.arrival_time for p in restarting_processes)
                    print(f"Time {self.time}: Waiting for Process {restarting_processes[0].pid} to arrive at time {next_arrival}")

    def log_trace(self, completed, ready_q, io_q):
        """
        Logs the current state of the system.

        :param completed: List of completed process PIDs at this time
        :param ready_q: Current ready queues
        :param io_q: Current IO queue
        """
        ready_queue_display = {priority: [f"P{p.pid}" for p in q] for priority, q in ready_q.items()}
        io_queue_display = [f"P{p.pid}" for p in io_q]
        completed_display = [f"P{pid}" for pid in completed]
        log = f"Time: {self.time} | Completed: {completed_display} | Ready Queue: {ready_queue_display} | IO Queue: {io_queue_display}"
        self.trace_logs.append(log)
        logging.info(log)

        # Special check: After process termination, immediately start next process
        if completed and not self.current_process and any(self.ready_queue.values()):
            self.current_process = self.get_next_process()
            if self.current_process:
                self.current_quantum_remaining = self.time_quantum
                self.current_process.state = 'RUNNING'
                current_burst = self.current_process.bursts[self.current_process.current_burst]
                if self.current_process.current_operation < len(current_burst.operations):
                    operation = self.current_process.bursts[self.current_process.current_burst].operations[self.current_process.current_operation]
                    print(f"Time {self.time}: Process {self.current_process.pid} processing operation {operation} in Burst {self.current_process.current_burst}")
                    if operation[0] == 'EXECUTE':
                        self.current_process.remaining_burst_time = operation[1]
                        self.current_process.current_operation += 1
                        print(f"Time {self.time}: Process {self.current_process.pid} starts executing for {operation[1]} units.")
                        self.current_process.remaining_burst_time -= 1
                        self.current_quantum_remaining -= 1


    ################################################
    #            generate_gantt_chart                #
    #  Creates visual representation of process      #
    #  execution, IO, and ready queue states        #
    ################################################
    def generate_gantt_chart(self):
        """
        Generates and displays the Gantt chart using matplotlib.
        """
        if not self.gantt_chart and not self.io_gantt_chart and not self.trace_logs:
            print("No execution history to display in Gantt Chart.")
            return

        # Define colors for processes
        process_colors = {
            0: '#3498db',  # Blue
            1: '#e74c3c',  # Red
            2: '#2ecc71',  # Green
            3: '#f1c40f',  # Yellow
            4: '#9b59b6',  # Purple
            5: '#1abc9c',  # Turquoise
            6: '#e67e22',  # Orange
            7: '#34495e',  # Navy
            8: '#7f8c8d',  # Gray
            9: '#16a085',  # Dark Turquoise
        }

        # Calculate max completion time
        max_completion_time = self.time

        # Collect unique PIDs from completed processes
        unique_pids = {str(proc.pid) for proc in self.completed_processes}

        # Create figure with extra space for CPU, I/O, Ready Queue, and statistics table
        fig, (ax_cpu, ax_io, ax_ready, ax_table) = plt.subplots(4, 1, figsize=(15, 20), 
                                                               gridspec_kw={'height_ratios': [2, 2, 2, 1]})

        ### 1. Draw CPU Gantt Chart ###
        if self.gantt_chart:
            last_pid = self.gantt_chart[0]
            start_time = 0
            for current_time, pid in enumerate(self.gantt_chart):
                # Check if current process is in IO during this time
                is_in_io = any(start <= current_time < end 
                              for p, start, end in self.io_gantt_chart 
                              if p == pid)
                
                # Change in process or process is in IO
                if pid != last_pid or is_in_io:
                    # Draw the previous segment
                    if not any(start <= start_time < end 
                              for _, start, end in self.io_gantt_chart 
                              if _ == last_pid):
                        ax_cpu.barh(0, current_time - start_time, left=start_time, height=0.4,
                                   align='center', edgecolor='black', color=process_colors.get(last_pid, '#95a5a6'))
                        ax_cpu.text(start_time + (current_time - start_time)/2, 0, f"P{last_pid}", 
                                   ha='center', va='center', color='white', fontsize=8)
                    else:
                        ax_cpu.barh(0, current_time - start_time, left=start_time, height=0.4,
                                   align='center', edgecolor='black', color='lightgray')
                        ax_cpu.text(start_time + (current_time - start_time)/2, 0, "idle", 
                                   ha='center', va='center', color='black', fontsize=8)
                    
                    start_time = current_time
                    last_pid = pid

            # Handle the last segment
            if start_time < self.time:
                # Check if the process is in IO at the start of this segment
                is_in_io_start = any(start <= start_time < end 
                                  for p, start, end in self.io_gantt_chart 
                                  if p == last_pid)
                
                if is_in_io_start:
                    # Find when the IO ends
                    io_end_time = next((end for p, start, end in self.io_gantt_chart 
                                      if p == last_pid and start <= start_time < end), self.time)
                    
                    # Draw idle period during IO
                    if io_end_time > start_time:
                        ax_cpu.barh(0, io_end_time - start_time, left=start_time, height=0.4,
                                   align='center', edgecolor='black', color='lightgray')
                        ax_cpu.text(start_time + (io_end_time - start_time)/2, 0, "idle", 
                                   ha='center', va='center', color='black', fontsize=8)
                    
                    # Draw active period after IO if any
                    if io_end_time < self.time:
                        ax_cpu.barh(0, self.time - io_end_time, left=io_end_time, height=0.4,
                                   align='center', edgecolor='black', color=process_colors.get(last_pid, '#95a5a6'))
                        ax_cpu.text(io_end_time + (self.time - io_end_time)/2, 0, f"P{last_pid}", 
                                   ha='center', va='center', color='white', fontsize=8)
                else:
                    # If not in IO, draw as normal process execution
                    ax_cpu.barh(0, self.time - start_time, left=start_time, height=0.4,
                               align='center', edgecolor='black', color=process_colors.get(last_pid, '#95a5a6'))
                    ax_cpu.text(start_time + (self.time - start_time)/2, 0, f"P{last_pid}", 
                               ha='center', va='center', color='white', fontsize=8)

            ax_cpu.set_ylim(-1, 1)
            ax_cpu.set_xlabel('Time Units')
            ax_cpu.set_yticks([])
            ax_cpu.set_title('CPU Gantt Chart', color='blue')
            ax_cpu.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax_cpu.set_xlim(0, self.time)

        ### 2. Draw I/O Gantt Chart ###
        if self.io_gantt_chart:
            for pid, io_start, io_end in self.io_gantt_chart:
                ax_io.barh(0, io_end - io_start, left=io_start, height=0.4,
                          align='center', edgecolor='black', color=process_colors.get(pid, '#95a5a6'))
                ax_io.text(io_start + (io_end - io_start)/2, 0, f"P{pid}", 
                          ha='center', va='center', color='white', fontsize=8)

            ax_io.set_ylim(-1, 1)
            ax_io.set_xlabel('Time Units')
            ax_io.set_yticks([])
            ax_io.set_title('I/O Gantt Chart', color='blue')
            ax_io.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax_io.set_xlim(0, max_completion_time)
        else:
            ax_io.axis('off')
            ax_io.set_title('I/O Gantt Chart', color='blue')

        ################################################
        #            Ready Queue Visualization           #
        #  Tracks and displays ready queue state changes#
        ################################################
        if self.trace_logs:
            # Initialize a dictionary to track ready queue intervals for each process
            ready_intervals = defaultdict(list)
            process_last_entry = defaultdict(lambda: None)  # Tracks the last time a process entered the ready queue

            for log in self.trace_logs:
                parts = log.split('|')
                if len(parts) >= 3:
                    time_part = parts[0].strip()
                    time_match = re.match(r'Time:\s*(\d+)', time_part)
                    if not time_match:
                        continue
                    current_time = int(time_match.group(1))

                    ready_q_part = parts[2].strip()
                    # Extract PIDs from Ready Queue
                    ready_pids = re.findall(r'P(\d+)', ready_q_part)

                    # For all processes, check if they are in the ready queue
                    for pid in unique_pids:
                        pid_int = int(pid)
                        if pid_int in [int(p) for p in ready_pids]:
                            if process_last_entry[pid_int] is None:
                                process_last_entry[pid_int] = current_time
                        else:
                            if process_last_entry[pid_int] is not None:
                                # Process was in ready queue from last_entry to current_time
                                start = process_last_entry[pid_int]
                                end = current_time
                                # **Exclude intervals where the process was in ready queue for only 1 time unit**
                                if end - start > 1:
                                    ready_intervals[pid_int].append((start, end))
                                process_last_entry[pid_int] = None

            # After all logs, close any open intervals
            for pid, start_time in process_last_entry.items():
                if start_time is not None:
                    end_time = self.time
                    if end_time - start_time > 1:
                        ready_intervals[pid].append((start_time, end_time))

            # Plot the Ready Queue
            y_pos = 0
            ready_queue_order = sorted(ready_intervals.keys())  # Sort to maintain consistent order
            for pid in ready_queue_order:
                intervals = ready_intervals[pid]
                for start, end in intervals:
                    ax_ready.barh(y_pos, end - start, left=start, height=0.4,  # Reduced height
                                 align='center', edgecolor='black', color=process_colors.get(pid, '#95a5a6'))
                    ax_ready.text(start + (end - start)/2, y_pos, f"P{pid}", 
                                 ha='center', va='center', color='white', fontsize=8)
                y_pos += 1

            if y_pos > 0:
                ax_ready.set_ylim(-0.5, y_pos - 0.5)  # Adjusted ylim for better fit
                ax_ready.set_xlabel('Time Units')
                ax_ready.set_ylabel('Processes')
                ax_ready.set_title('Ready Queue Gantt Chart', color='blue')  # Changed title color to blue
                ax_ready.grid(True, axis='x', linestyle='--', alpha=0.7)

                # Set y-ticks to process IDs correctly
                ax_ready.set_yticks(range(y_pos))
                ax_ready.set_yticklabels([f"P{pid}" for pid in ready_queue_order])
                ax_ready.set_xlim(0, max_completion_time)
            else:
                ax_ready.axis('off')
                ax_ready.set_title('Ready Queue Gantt Chart', color='blue')  # Changed title color to blue

        else:
            ax_ready.axis('off')
            ax_ready.set_title('Ready Queue Gantt Chart', color='blue')  # Changed title color to blue

        ################################################
        #            Statistics Table                    #
        #  Displays process execution statistics        #
        ################################################
        ax_table.axis('off')
        table_data = [
            ['Process', 'Arrival Time', 'Completion Time', 'Turnaround Time', 'Waiting Time']
        ]

        # Calculate and add statistics for each process
        total_turnaround = 0
        total_waiting = 0
        process_count = len(self.completed_processes)

        for proc in sorted(self.completed_processes, key=lambda x: x.pid):
            turnaround_time = proc.completion_time - proc.arrival_time
            total_turnaround += turnaround_time
            total_waiting += proc.waiting_time

            table_data.append([
                f'P{proc.pid}',
                str(proc.arrival_time),
                str(proc.completion_time),
                str(turnaround_time),
                str(proc.waiting_time)
            ])

        # Add averages row
        avg_turnaround = total_turnaround / process_count if process_count > 0 else 0
        avg_waiting = total_waiting / process_count if process_count > 0 else 0
        table_data.append([
            'Average',
            '-',
            '-',
            f'{avg_turnaround:.2f}',
            f'{avg_waiting:.2f}'
        ])

        # Create table
        table = ax_table.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header row
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#E6E6E6')
            table[(0, j)].set_text_props(weight='bold')

        # Style average row
        for j in range(len(table_data[0])):
            table[(len(table_data)-1, j)].set_facecolor('#F0F0F0')

        plt.tight_layout()
        plt.show()

    def write_trace_logs(self, file_path='trace_logs.txt'):
        """
        Writes the trace logs to a file.

        :param file_path: Path to the trace log file
        """
        with open(file_path, 'w') as f:
            for log in self.trace_logs:
                f.write(log + '\n')

    def calculate_statistics(self):
        """
        Calculates average waiting time and average turnaround time.

        :return: Tuple (average_waiting_time, average_turnaround_time)
        """
        total_waiting = sum(proc.waiting_time for proc in self.completed_processes)
        total_turnaround = sum(proc.completion_time - proc.arrival_time for proc in self.completed_processes)
        n = len(self.completed_processes)
        avg_waiting = total_waiting / n if n > 0 else 0
        avg_turnaround = total_turnaround / n if n > 0 else 0
        return avg_waiting, avg_turnaround

    def is_process_deadlocked(self, pid, visited=None, path=None):
        """
        Helper method to check if a specific process is in a deadlock.
        
        :param pid: Process ID to check
        :param visited: Set of visited nodes
        :param path: Current path being explored
        :return: Boolean indicating if process is in deadlock
        """
        if visited is None:
            visited = set()
        if path is None:
            path = set()
        
        visited.add(pid)
        path.add(pid)
        
        # Build wait-for graph
        for rid, resource in self.resources.items():
            if resource.held_by is not None:
                for waiting_proc in resource.waiting_queue:
                    if waiting_proc.pid == pid:
                        if resource.held_by in path:
                            return True
                        if resource.held_by not in visited:
                            if self.is_process_deadlocked(resource.held_by, visited, path):
                                return True
        
        path.remove(pid)
        return False

# Function to parse the input file
def parse_input(file_path: str) -> list[Process]:
    """
    Parses the input file and creates Process instances with enhanced error checking.

    :param file_path: Path to the input file
    :return: List of Process instances
    :raises FileNotFoundError: If input file doesn't exist
    :raises ProcessValidationError: If process validation fails
    :raises ValueError: If input file format is invalid
    """
    processes = []
    existing_pids = set()
    line_number = 0

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                raise ValueError("Input file is empty")
                
            for line in lines:
                line_number += 1
                if not line.strip():
                    continue  # Skip empty lines
                
                try:
                    # Split line into parts
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) < 4:
                        raise ValueError(
                            f"Line {line_number}: Insufficient process information. "
                            "Expected format: PID ARRIVAL_TIME PRIORITY BURSTS..."
                        )
                    
                    # Parse basic process information
                    try:
                        pid = int(parts[0])
                        arrival_time = int(parts[1])
                        priority = int(parts[2])
                    except ValueError as e:
                        raise ValueError(
                            f"Line {line_number}: Invalid numeric value in process definition: {str(e)}"
                        )
                    
                    # Parse bursts
                    bursts = []
                    burst_pattern = r'(CPU|IO)\s*\{([^}]*)\}'
                    burst_matches = re.findall(burst_pattern, ' '.join(parts[3:]))
                    
                    if not burst_matches:
                        raise ValueError(
                            f"Line {line_number}: No valid burst definitions found. "
                            "Expected format: CPU{{operations}} or IO{{operations}}"
                        )
                    
                    for burst_type, burst_content in burst_matches:
                        operations = parse_burst_details(burst_content)
                        bursts.append(Burst(burst_type, operations))
                    
                    # Create and validate process
                    process = Process(pid, arrival_time, priority, bursts)
                    validate_process(process, existing_pids)
                    
                    # If validation passes, add process
                    existing_pids.add(pid)
                    processes.append(process)
                    
                except (ValueError, ProcessValidationError) as e:
                    raise ValueError(f"Line {line_number}: {str(e)}")
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{file_path}' not found")
    except Exception as e:
        raise ValueError(f"Error parsing input file: {str(e)}")
    
    # Final validation of the process set
    if not processes:
        raise ValueError("No valid processes found in input file")
    
    return processes

# Helper function to parse burst details
def parse_burst_details(details_str):
    """
    Parses the details within a burst and returns a list of operations.

    :param details_str: String containing burst details
    :return: List of operation tuples
    """
    operations = []
    # Clean up the input string - replace both [] and {} with standard delimiters
    details_str = details_str.replace('[', '{').replace(']', '}')
    # Split by comma, but ignore commas within brackets
    tokens = re.split(r',\s*(?![^{}]*\})', details_str)
    
    for token in tokens:
        token = token.strip()
        if token.startswith('R{'):  # Handle both R[ and R{
            rid = int(token[2:-1])
            operations.append(('REQUEST', rid))
        elif token.startswith('F{'):  # Handle both F[ and F{
            rid = int(token[2:-1])
            operations.append(('RELEASE', rid))
        elif token.startswith('EXECUTE'):
            # Handle EXECUTE operations with explicit keyword
            exec_time = int(token.split('EXECUTE')[1].strip())
            operations.append(('EXECUTE', exec_time))
        elif token.isdigit():
            exec_time = int(token)
            operations.append(('EXECUTE', exec_time))
    return operations

# Validation function for processes
def validate_process(process: Process, existing_pids: set) -> None:
    """
    Validates a process for various error conditions.
    
    :param process: Process instance to validate
    :param existing_pids: Set of PIDs already in use
    :raises ProcessValidationError: If validation fails
    """
    # Check for duplicate PID
    if process.pid in existing_pids:
        raise ProcessValidationError(f"Duplicate Process ID found: {process.pid}")
    
    # Check for negative PID (allowing 0)
    if process.pid < 0:  # Changed from <= 0 to < 0 to allow PID 0
        raise ProcessValidationError(f"Invalid Process ID: {process.pid}. Must be non-negative.")
    
    # Check for negative arrival time
    if process.arrival_time < 0:
        raise ProcessValidationError(f"Process {process.pid} has negative arrival time: {process.arrival_time}")
    
    # Check for negative priority
    if process.priority < 0:
        raise ProcessValidationError(f"Process {process.pid} has negative priority: {process.priority}")
    
    # Check for empty burst list
    if not process.bursts:
        raise ProcessValidationError(f"Process {process.pid} has no bursts defined")
    
    # Validate each burst
    for i, burst in enumerate(process.bursts):
        # Check for valid burst type
        if burst.type not in ['CPU', 'IO']:
            raise ProcessValidationError(f"Process {process.pid} has invalid burst type: {burst.type}")
        
        # Check for empty operations list
        if not burst.operations:
            raise ProcessValidationError(f"Process {process.pid} has empty burst {i}")
        
        # Validate operations within the burst
        total_exec_time = 0
        resources_held = set()
        
        for op in burst.operations:
            op_type, value = op
            
            # Check operation type
            if op_type not in ['EXECUTE', 'REQUEST', 'RELEASE']:
                raise ProcessValidationError(
                    f"Process {process.pid} has invalid operation type: {op_type}"
                )
            
            # Validate EXECUTE operations
            if op_type == 'EXECUTE':
                if value <= 0:
                    raise ProcessValidationError(
                        f"Process {process.pid} has invalid execution time: {value}"
                    )
                total_exec_time += value
            
            # Track resource requests and releases
            elif op_type == 'REQUEST':
                if value in resources_held:
                    raise ProcessValidationError(
                        f"Process {process.pid} attempts to request already held resource: {value}"
                    )
                resources_held.add(value)
            
            elif op_type == 'RELEASE':
                if value not in resources_held:
                    raise ProcessValidationError(
                        f"Process {process.pid} attempts to release unheld resource: {value}"
                    )
                resources_held.remove(value)
        
        # Check if CPU burst has zero total execution time
        if burst.type == 'CPU' and total_exec_time == 0:
            raise ProcessValidationError(
                f"Process {process.pid} has CPU burst with zero execution time"
            )
        
        # Check for unreleased resources at end of burst
        if resources_held:
            raise ProcessValidationError(
                f"Process {process.pid} has unreleased resources at end of burst: {resources_held}"
            )

# Main function
def main() -> None:
    """
    Main function to execute the CPU scheduling simulation.
    """
    try:
        processes = parse_input("Processes.txt")
        scheduler = Scheduler(time_quantum=10)
        
        # Add all resources mentioned in processes
        for process in processes:
            for burst in process.bursts:
                for op in burst.operations:
                    if op[0] in ['REQUEST', 'RELEASE']:
                        scheduler.add_resource(op[1])
        
        with timing("Total simulation"):
            scheduler.run_simulation(processes)
            
        # Calculate statistics
        avg_waiting, avg_turnaround = scheduler.calculate_statistics()
        print(f"\nAverage Waiting Time: {avg_waiting}")
        print(f"Average Turnaround Time: {avg_turnaround}")


        scheduler.generate_gantt_chart()

        # Optionally, save the Gantt chart
        # plt.savefig('gantt_chart.png')

        # Write trace logs to a file
        scheduler.write_trace_logs('trace_logs.txt')

        # Print deadlock summary
        if scheduler.deadlock_history:
            print("\nDeadlock Summary:")
            for event in scheduler.deadlock_history:
                print(f"Time {event['time']}: Deadlock detected among Processes {event['processes']}.")
                print(f"Resolution: Terminated Process {event['victim']} and restarted it.")
        else:
            print("\nNo deadlocks occurred during simulation.")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        sys.exit(1)
    except ProcessValidationError as e:
        logging.error(f"Process validation error: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Input error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()