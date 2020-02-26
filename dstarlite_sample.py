# This is the D* Lite algorithm (original version - not optimized for the queue U)
import time
import threading
import cv2
import numpy as np
import random
import heapq

continue_display = True

map_img = cv2.resize(cv2.imread('testlab.png', cv2.IMREAD_UNCHANGED), (200, 150))
map_img_temp = cv2.resize(cv2.imread('testlab.png', cv2.IMREAD_UNCHANGED), (200, 150))

mutex_mat = threading.Lock()
buffer_mat = np.zeros(map_img.shape, np.uint8)

def display_thread():
    while continue_display:
        mutex_mat.acquire()
        cv2.imshow('Mat image',buffer_mat)
        cv2.waitKey(1)
        mutex_mat.release()
        time.sleep(1.0/60.0)
    cv2.destroyAllWindows()

### DSTAR LITE (BASE VERSION)
class NodeState:
    def __init__(self, state_id):
        self.g = np.Inf
        self.rhs = np.Inf
        self.successors = {}
        self.predecessors = {}
        self.id = state_id

    def __repr__(self):
        return "( " + self.id + ": g = "+ str(self.g) + ", rhs = "+ str(self.rhs) + ", succ = " + str(len(self.successors)) + ", pred = "+ str(len(self.predecessors))+" )"


class StateGraph:
    def __init__(self, obstacles, start, goal):  # The Initialize procedure
        self.S = {}  # All the states (the graph)
        self.obstacles = obstacles
        self.ack_obstacles = np.zeros(self.obstacles.shape, dtype=np.int16)
        self.size = [self.obstacles.shape[1], self.obstacles.shape[0]]
        self.goal = goal
        self.start = start
        self.goal_state_id = 'x'+str(self.goal[0])+'y'+str(self.goal[1])
        self.start_state_id = 'x'+str(self.start[0])+'y'+str(self.start[1])
        self.k_m = np.Inf
        self.U = None  # The queue
        self.past_node = None
        self.past_past_node = None

    def SetupGraph(self, init_map, with_obstacles):
        # create all the states and initialize them
        for idx in range(0, self.size[0]):
            for jdx in range(0, self.size[1]):
                state_id = 'x'+str(idx)+'y'+str(jdx)
                self.S[state_id] = NodeState(state_id)  # rhs = g = Inf
                for sidx in range(-1, 2):
                    for sjdx in range(-1, 2):
                        if not (sidx == 0 and sjdx == 0):
                            nidx = idx + sidx
                            njdx = jdx + sjdx
                            if nidx >= 0 and nidx < self.size[0] and njdx >= 0 and njdx < self.size[1]:
                                dist_x = nidx - idx
                                dist_y = njdx - jdx
                                cost = np.sqrt(dist_x*dist_x + dist_y*dist_y)
                                #print(cost)
                                if(with_obstacles):
                                    obstacle_curr = self.get_obstacle_status([idx, jdx])
                                    obstacle_next = self.get_obstacle_status([nidx, njdx])
                                    if obstacle_curr == 0 and obstacle_next == 0:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                    elif obstacle_curr == 0 and obstacle_next == 1:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                    elif obstacle_curr == 1 and obstacle_next == 0:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                    elif obstacle_curr == 1 and obstacle_next == 1:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                    else:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                elif init_map:
                                    self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = cost  # initialize successor transition cost
                                    self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = cost  # initialize predecessor transition cost
                                else:
                                    obstacle_curr = self.get_ack_obstacle_status([idx, jdx])
                                    obstacle_next = self.get_ack_obstacle_status([nidx, njdx])
                                    if obstacle_curr == 0 and obstacle_next == 0:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                    elif obstacle_curr == 0 and obstacle_next == 1:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                    elif obstacle_curr == 1 and obstacle_next == 0:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = cost
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                    elif obstacle_curr == 1 and obstacle_next == 1:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                    else:
                                        self.S[state_id].successors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                        self.S[state_id].predecessors['x'+str(nidx)+'y'+ str(njdx)] = np.Inf
                                
    def U_Insert(self, s, key): # TODO: REVISE
        new_key = key + (s,)  # e.g. (130, 0, 'x140y10')
        heapq.heappush(self.U, new_key) 

    def Initialize(self, s_start, initialize_map, initialize_obstacles):
        self.past_node = s_start
        self.past_past_node = None
        self.U = []
        self.k_m = 0
        self.SetupGraph(initialize_map, initialize_obstacles) # dont initialize cost for obstacles
        self.set_rhs(self.goal_state_id, 0)
        self.U_Insert(self.goal_state_id,
                      self.CalculateKey(s_start, self.goal_state_id))

    def StateIDToCoods(self, state_id):
        x_coord = int(state_id.split('y', -1)[0].split('x', -1)[1])
        y_coord = int(state_id.split('y', -1)[1])
        return [x_coord, y_coord]

    def h(self, s_start, s):
        dest_coords = self.StateIDToCoods(s)
        start_coords = self.StateIDToCoods(s_start)
        x_distance = abs(dest_coords[0] - start_coords[0])
        y_distance = abs(dest_coords[1] - start_coords[1])
        return max(x_distance, y_distance)

    def g(self, s):
        return self.S[s].g

    def rhs(self, s):
        return self.S[s].rhs

    def set_g(self, s, value):
        self.S[s].g = value

    def set_rhs(self, s, value):
        self.S[s].rhs = value

    def CalculateKey(self, s_start, s):
        return (min(self.g(s), self.rhs(s)) +
                self.h(s_start, s) + self.k_m,
                min(self.g(s), self.rhs(s)))

    def GetTopKey(self):
        self.U.sort()
        if len(self.U) > 0:
            return self.U[0][:2]
        else:
            return (np.Inf, np.Inf)

    def Is_Inside_U(self, u): # TODO: NEED TO OPTIMIZE
        #id_in_queue = [item for item in self.U if u in item]
        found = False
        for item in self.U:
            if item[2] == u:
                found = True
                return found

        #if id_in_queue != []:
        #    return True
        #else:
        return found

    def U_Update(self, u, key): # TODO: REVISE
        n = len(self.U)
        for idx in range(0, n):
            if self.U[idx][2] == u:
                new_key = key + (u,)
                self.U[idx] = new_key
                break

    def U_Remove(self, u):  # TODO: NEED TO OPTIMIZE
        n = len(self.U)
        for idx in range(0, n):
            if self.U[idx][2] == u:
                del self.U[idx]
                break

    def UpdateVertex(self, s_start, u):
        if u != self.goal_state_id:
            min_rhs = np.Inf
            for s_p in self.S[u].successors:
                min_rhs = min(min_rhs, self.g(s_p) + self.S[u].successors[s_p])
            self.set_rhs(u, min_rhs)
        
        u_contained_in_U = self.Is_Inside_U(u)
        if u_contained_in_U:
            self.U_Remove(u)

        if self.g(u) != self.rhs(u):
            self.U_Insert(u, self.CalculateKey(s_start, u))

    def U_TopKey(self):
        if len(self.U) > 0:
            #self.U.sort()  # TODO: REVISE
            self.U = sorted(sorted(sorted(self.U, key = lambda x : x[2]), key = lambda x : x[1]), key = lambda x : x[0])
            return self.U[0][:2]
        else:
            return (np.Inf, np.Inf)

    def U_Top(self):
        if len(self.U) > 0:
            #self.U.sort()  # TODO: REVISE
            #self.U = sorted(sorted(sorted(a, key = lambda x : x[2]), key = lambda x : x[1]), key = lambda x : x[0])
            return self.U[0][2]
        else:
            return None

    def U_Pop(self):
        return heapq.heappop(self.U)[2]

    def ComputeShortestPath(self, s_start):
        k_old = self.U_TopKey()
        new_key = self.CalculateKey(s_start, s_start)
        while((k_old[0] < new_key[0]) or (k_old[0] == new_key[0] and k_old[1] < new_key[1]) or self.rhs(s_start) != self.g(s_start)):
            u = self.U_Pop()
            #print(u + " : " + str(k_old) + " / " + str(self.g(u)) + ", " + str(self.rhs(u)))
            #k_old = self.U_TopKey()
            k_new = self.CalculateKey(s_start, u)
            if (k_old[0] < k_new[0]) or (k_old[0] == k_new[0] and k_old[1] < k_new[1]):
                self.U_Insert(u, k_new)
            elif self.g(u) > self.rhs(u):
                self.set_g(u, self.rhs(u))
                for s in self.S[u].predecessors:
                    self.UpdateVertex(s_start, s)
            else:
                self.set_g(u, np.Inf)
                for s in self.S[u].predecessors:
                    self.UpdateVertex(s_start, s)
                self.UpdateVertex(s_start, u)

            k_old = self.U_TopKey()
            new_key = self.CalculateKey(s_start, s_start)

    def RunProcedure(self, s_start, s_last, scan_range):
        if self.past_past_node == s_start: # Terminate the loop if cannot decide where to go
            print("TIED - NEED TO DECIDE WHERE TO GO")
            return s_start, None

        self.past_past_node = self.past_node
        self.past_node = s_start
        if self.g(s_start) == np.Inf:
            print("NO KNOWN PATH")
            return s_start, None

        if self.rhs(s_start) == np.Inf: # Terminate the loop immediatly after being trapped inside an obstacle
            print("GOT STUCK")
            return s_start, None

        s_lowest_cost = s_start
        lowest_cost = np.Inf
        for s_p in self.S[s_start].successors:
            temp_cost = self.S[s_start].successors[s_p] + self.g(s_p)
            if temp_cost < lowest_cost:
                lowest_cost = temp_cost
                s_lowest_cost = s_p

        s_start = s_lowest_cost
        
        start_coords = self.StateIDToCoods(s_start) # Handle suddenly appearing obstacles on next selected step
        if self.get_ack_obstacle_status(start_coords) != self.get_obstacle_status(start_coords):
            s_start = self.past_node
            s_last = s_start
  
        old_s_last = s_last
        s_last = self.ScanForNewObstacles(s_start, scan_range, old_s_last)

        if s_last != old_s_last:
            self.ComputeShortestPath(s_start)

        return s_start, s_last

    def ScanForNewObstacles(self, s_start, scan_range, s_last):
        states_to_update = {}
        start_coords = self.StateIDToCoods(s_start) # Handle suddenly appearing obstacles on current step
        if self.get_ack_obstacle_status(start_coords) != self.get_obstacle_status(start_coords):
            states_to_update[s_start] = self.get_obstacle_status(start_coords)

        range_checked = 0
        if scan_range >= 1: # Discover immediate obstacles (one step aside of the agent)
            for neighbor in self.S[s_start].successors:
                neighbor_coords = self.StateIDToCoods(neighbor)
                if self.get_ack_obstacle_status(neighbor_coords) != self.get_obstacle_status(neighbor_coords):
                    states_to_update[neighbor] = self.get_obstacle_status(neighbor_coords)
            range_checked = 1

        while range_checked < scan_range:# Check obstacles connected to the originally discovered immediate one (ignoring not associated obstacles)
            new_set = {}
            for state in states_to_update:
                new_set[state] = states_to_update[state]
                for neighbor in self.S[state].successors:
                    if neighbor not in new_set:
                        neighbor_coords = self.StateIDToCoods(neighbor)
                        if self.get_ack_obstacle_status(neighbor_coords) != self.get_obstacle_status(neighbor_coords):
                            new_set[neighbor] = self.get_obstacle_status(neighbor_coords)
                # Not necessary to check predecessors as they are the same as the successors
            range_checked += 1
            states_to_update = new_set

        new_updates = False
        for state in states_to_update:
            if states_to_update[state] == 1:  # Obstacle appeared
                for neighbor in self.S[state].successors:
                    if(self.S[state].successors[neighbor] != np.Inf):
                        state_coords = self.StateIDToCoods(state)
                        self.set_ack_obstacle_status(state_coords, self.get_obstacle_status(state_coords))
                        self.S[state].successors[neighbor] = np.Inf  # Force an rhs = Inf when updating the vertex (and effectively banning this node)
                        self.UpdateVertex(s_start, state)
                        new_updates = True
            else:  # Obstacle dissapeared
                for neighbor in self.S[state].successors:
                    if(self.S[state].successors[neighbor] == np.Inf or self.rhs(state) == np.Inf):
                        state_coords = self.StateIDToCoods(state)
                        self.set_ack_obstacle_status(state_coords, self.get_obstacle_status(state_coords))
                        curr_coords = self.StateIDToCoods(state)
                        dist_x = curr_coords[0] - state_coords[0]
                        dist_y = curr_coords[1] - state_coords[1]
                        cost = np.sqrt(dist_x*dist_x + dist_y*dist_y)
                        self.S[state].successors[neighbor] = cost # Force recalculating the rhs to the minimum possible value
                        self.UpdateVertex(s_start, state)
                        new_updates = True
            
        if (new_updates):
            self.k_m += self.h(s_last, s_start)
            return s_start
        else:
            return s_last

    def get_obstacle_status(self, coord):
        if self.obstacles[coord[1]][coord[0]] < 255:
            return 1
        else:
            return 0

    def set_obstacle_status(self, coord, new_status):
        if new_status == 0:
            self.obstacles[coord[1], coord[0]] = 255
        else:
            self.obstacles[coord[1], coord[0]] = 0

    def get_ack_obstacle_status(self, coord):
        return self.ack_obstacles[coord[1]][coord[0]]

    def set_ack_obstacle_status(self, coord, new_status):
        self.ack_obstacles[coord[1], coord[0]] = new_status

### DSTAR LITE (BASE VERSION) -END-

threading.Thread(target=display_thread).start()

obstacle_map = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

start = [10, 140]
finish = [140,140]

states = StateGraph(obstacle_map, start, finish)
states.set_obstacle_status([40, 138],1)
states.set_obstacle_status([40, 139],1)
states.set_obstacle_status([40, 140],1)
states.set_obstacle_status([40, 141],1)
states.set_obstacle_status([40, 142],1)
states.set_obstacle_status([40, 143],1)
states.set_obstacle_status([40, 144],1)
states.set_obstacle_status([40, 145],1)
states.set_obstacle_status([40, 146],1)
states.set_obstacle_status([40, 147],1)
states.set_obstacle_status([40, 148],1)
states.set_obstacle_status([40, 149],1)
states.set_obstacle_status([41, 140],1)
states.set_obstacle_status([41, 141],1)
states.set_obstacle_status([41, 142],1)
states.set_obstacle_status([41, 143],1)
states.set_obstacle_status([41, 144],1)
states.set_obstacle_status([41, 145],1)
states.set_obstacle_status([41, 146],1)
states.set_obstacle_status([41, 147],1)
states.set_obstacle_status([41, 148],1)
states.set_obstacle_status([41, 149],1)

scan_range = 20
path_to_start = []
s_start = states.start_state_id
first_run = True
try:
    while(True): # AVOID "TIED" BUG WITH ADDING first_run FLAG
        s_last = s_start
        states.Initialize(s_start, first_run, False)
        first_run = False # Handles keeping the discovered obstacles so far
        states.ComputeShortestPath(s_start)
        counter = 0
        paint = 0
        obs_counter = 29

        while(s_start != states.goal_state_id):
            obs_counter -= 1

            if obs_counter < 0:
                states.set_obstacle_status([40, 138],0)
                states.set_obstacle_status([40, 139],0)
                states.set_obstacle_status([40, 140],0)
                states.set_obstacle_status([40, 141],0)
                states.set_obstacle_status([40, 142],0)
                states.set_obstacle_status([40, 143],0)
                states.set_obstacle_status([40, 144],0)
                states.set_obstacle_status([40, 145],0)
                states.set_obstacle_status([40, 146],0)
                states.set_obstacle_status([40, 147],0)
                states.set_obstacle_status([40, 148],0)
                states.set_obstacle_status([40, 149],1)
                states.set_obstacle_status([41, 140],0)
                states.set_obstacle_status([41, 141],0)
                states.set_obstacle_status([41, 142],0)
                states.set_obstacle_status([41, 143],0)
                states.set_obstacle_status([41, 144],0)
                states.set_obstacle_status([41, 145],0)
                states.set_obstacle_status([41, 146],0)
                states.set_obstacle_status([41, 147],0)
                states.set_obstacle_status([41, 148],0)
                states.set_obstacle_status([41, 149],1)
            if obs_counter < -1:
                states.set_obstacle_status([40, 138],1)
                states.set_obstacle_status([40, 139],1)
                states.set_obstacle_status([40, 140],1)
                states.set_obstacle_status([40, 141],1)
                states.set_obstacle_status([40, 142],1)
                states.set_obstacle_status([40, 143],1)
                states.set_obstacle_status([40, 144],1)
                states.set_obstacle_status([40, 145],1)
                states.set_obstacle_status([40, 146],1)
                states.set_obstacle_status([40, 147],1)
                states.set_obstacle_status([40, 148],1)
                states.set_obstacle_status([40, 149],1)
                states.set_obstacle_status([41, 140],1)
                states.set_obstacle_status([41, 141],1)
                states.set_obstacle_status([41, 142],1)
                states.set_obstacle_status([41, 143],1)
                states.set_obstacle_status([41, 144],1)
                states.set_obstacle_status([41, 145],1)
                states.set_obstacle_status([41, 146],1)
                states.set_obstacle_status([41, 147],1)
                states.set_obstacle_status([41, 148],1)
                states.set_obstacle_status([41, 149],1)

            s_start, s_last = states.RunProcedure(s_start, s_last, scan_range)

            counter += 1
            if s_start is None or s_last is None:
                break
            pos_coords = states.StateIDToCoods(s_start)
            pos_last = states.StateIDToCoods(s_last)
            path_to_start.append([pos_coords[0], pos_coords[1]])

            if paint > scan_range/2:
                map_img = map_img_temp.copy()
                    
                for idx in range(0, states.size[0]):
                    for jdx in range(0, states.size[1]):
                        node_name = 'x' + str(idx) + 'y' + str(jdx)

                        if(states.S[node_name].g != np.Inf):
                            coords = states.StateIDToCoods(node_name)
                            map_img[jdx, idx] = (int(255 - states.S[node_name].g//1.5), 0, int(states.S[node_name].g//1.5))
                        
                        if(states.S[node_name].g == np.Inf):
                            map_img[jdx, idx] = (0, 0, 128)

                        if states.get_obstacle_status([idx, jdx]) == 1:
                            map_img[jdx, idx] = (128, 128, 128)

                        if states.get_ack_obstacle_status([idx, jdx]) == 1:
                            map_img[jdx, idx] = (0, 0, 0)

                cv2.rectangle(map_img, (pos_last[0] - scan_range, pos_last[1] - scan_range), (pos_last[0] + scan_range, pos_last[1] + scan_range), (0,255,255), 1)

                map_img[start[1], start[0]] = [255, 0, 0]
                map_img[finish[1], finish[0]] = [0, 128, 200]
                map_img[pos_coords[1], pos_coords[0]] = [0, 255, 255]
                        
                for item in path_to_start:
                    map_img[item[1],item[0]] = [0,255,0]
                map_img[path_to_start[len(path_to_start)-1][1],path_to_start[len(path_to_start)-1][0]] = [128,128,0]

                mutex_mat.acquire()
                buffer_mat = cv2.resize(map_img.copy(),(map_img.shape[1]*1,map_img.shape[0]*1), interpolation=cv2.INTER_NEAREST)#.data = map_img.data.copy()
                mutex_mat.release()
                paint = 0
            paint += 1

        if s_start == states.goal_state_id:
            print("GOAL REACHED!")
            map_img = map_img_temp.copy()
                
            for idx in range(0, states.size[0]):
                for jdx in range(0, states.size[1]):
                    node_name = 'x' + str(idx) + 'y' + str(jdx)

                    if(states.S[node_name].g != np.Inf):
                        coords = states.StateIDToCoods(node_name)
                        map_img[jdx, idx] = (int(255 - states.S[node_name].g), 0, int(states.S[node_name].g))
                    
                    if(states.S[node_name].g == np.Inf):
                        map_img[jdx, idx] = (0, 0, 128)

                    if states.get_obstacle_status([idx, jdx]) == 1:
                        map_img[jdx, idx] = (128, 128, 128)

                    if states.get_ack_obstacle_status([idx, jdx]) == 1:
                        map_img[jdx, idx] = (0, 0, 0)

            map_img[start[1], start[0]] = [255, 0, 0]
            map_img[finish[1], finish[0]] = [0, 255, 0]
            map_img[pos_coords[1], pos_coords[0]] = [0, 255, 255]
                    
            for item in path_to_start:
                map_img[item[1],item[0]] = [0,255,0]

            mutex_mat.acquire()
            buffer_mat = cv2.resize(map_img.copy(),(map_img.shape[1]*1,map_img.shape[0]*1), interpolation=cv2.INTER_NEAREST)#.data = map_img.data.copy()
            mutex_mat.release()
        else:
            print("REINITIALIZING")
except KeyboardInterrupt:
    print("FINISH PROGRAM")
    continue_display = False
