import numpy as np
from copy import deepcopy
from collections import Counter

def most_frequent(arr):
    return Counter(arr).most_common(1)[0][0]

def sorted_random_numbers(count, N):
    res = []
    for _ in range(count):
        r = np.random.randint(0, N)
        while r in res:
            r = np.random.randint(0, N)
        res.append(r)
    res.sort()
    return res

class Crossover:
    
    @staticmethod
    def PMX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)
        point_1 = np.random.randint(0, N // 2)
        point_2 = np.random.randint(N // 2, N)

        not_seen_1 = []
        not_seen_2 = []

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        for i in range(N):
            if point_1 <= i < point_2:
                child_1[i] = parent_1[i]
                child_2[i] = parent_2[i]
            else:
                not_seen_1.append(parent_1[i])
                not_seen_2.append(parent_2[i])
                
        for i in range(N):
            if child_1[i] != -1 and parent_2[i] in not_seen_1:
                ans = parent_2[i]
                j = i
                while child_1[j] != -1:
                    j = np.where(parent_2 == child_1[j])[0][0]
                child_1[j] = ans

        for i in range(N):
            if parent_2[i] in not_seen_1 and child_1[i] == -1:
                child_1[i] = parent_2[i]

        for i in range(N):
            if child_2[i] != -1 and parent_1[i] in not_seen_2:
                ans = parent_1[i]
                j = i
                while child_2[j] != -1:
                    j = np.where(parent_1 == child_2[j])[0][0]
                child_2[j] = ans

        for i in range(N):
            if parent_1[i] in not_seen_2 and child_2[i] == -1:
                child_2[i] = parent_1[i]

        return child_1, child_2

    @staticmethod
    def CX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        tmp_1 = deepcopy(parent_1)
        tmp_2 = deepcopy(parent_2)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        swap = True
        count, pos = 0, 0

        while True:
            if count > N:
                break
            for i in range(N):
                if child_1[i] == -1:
                    pos = i
                    break

            if swap:
                while True:
                    child_1[pos] = parent_1[pos]
                    count += 1
                    pos = np.where(parent_2 == parent_1[pos])[0][0]
                    if tmp_1[pos] == -1:
                        swap = False
                        break
                    tmp_1[pos] = -1
            else:
                while True:
                    child_1[pos] = parent_2[pos]
                    count += 1
                    pos = np.where(parent_1 == parent_2[pos])[0][0]
                    if tmp_2[pos] == -1:
                        swap = True
                        break
                    tmp_2[pos] = -1
        
        for i in range(N):
            if child_1[i] == parent_1[i]:
                child_2[i] = parent_2[i]
            else:
                child_2[i] = parent_1[i]

        for i in range(N):
            if child_1[i] == -1:
                if tmp_1[i] == -1:
                    child_1[i] = parent_2[i]
                else:
                    child_1[i] = parent_1[i]
                    
        return child_1, child_2

    @staticmethod
    def OX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)
        point_1 = np.random.randint(0, N // 2)
        point_2 = np.random.randint(N // 2, N)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        trans_1 = []
        trans_2 = []

        list_1 = []
        list_2 = []

        for i in range(point_1, point_2):
            child_1[i] = parent_1[i]
            child_2[i] = parent_2[i]
            trans_1.append(child_1[i])
            trans_2.append(child_2[i])

        for i in range(point_2, N):
            list_1.append(parent_2[i])
            list_2.append(parent_1[i])

        for i in range(0, point_2):
            list_1.append(parent_2[i])
            list_2.append(parent_1[i])

        delete_1 = []
        delete_2 = []

        for i in range(N):
            if list_1[i] in trans_1:
                delete_1.append(i)
            if list_2[i] in trans_2:
                delete_2.append(i)

        count = 0
        for i in range(len(delete_1)):
            list_1.pop(delete_1[i] - count)
            list_2.pop(delete_2[i] - count)
            count += 1

        count = 0
        for i in range(point_2, N):
            child_1[i] = list_1[count]
            child_2[i] = list_2[count]
            count += 1

        for i in range(point_1):
            child_1[i] = list_1[count]
            child_2[i] = list_2[count]
            count += 1

        return child_1, child_2

    @staticmethod
    def MOC(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        hold_1 = []
        hold_2 = []

        pos = np.random.randint(2, N)
        for i in range(pos):
            hold_1.append(parent_2[i])
            hold_2.append(parent_1[i])

        for i in range(N):
            if parent_1[i] in hold_1:
                child_1[i] = parent_1[i]
            if parent_2[i] in hold_2:
                child_2[i] = parent_2[i]

        hold_1 = []
        hold_2 = []

        for i in range(pos, N):
            hold_1.append(parent_2[i])
            hold_2.append(parent_1[i])

        k, m = 0, 0
        for i in range(N):
            if child_1[i] == -1:
                child_1[i] = hold_1[k]
                k += 1
            if child_2[i] == -1:
                child_2[i] = hold_2[m]
                m += 1

        return child_1, child_2

    @staticmethod
    def MPMX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        point_1, point_2 = sorted_random_numbers(2, N)

        seen_1 = []
        seen_2 = []
        not_seen_1 = []
        not_seen_2 = []

        for i in range(point_1, point_2):
            child_1[i] = parent_1[i]
            seen_1.append(parent_1[i])
            child_2[i] = parent_2[i]
            seen_2.append(parent_2[i])

        for i in range(N):
            if i < point_1 or i >= point_2:
                if parent_2[i] not in seen_1:
                    child_1[i] = parent_2[i]
                    seen_1.append(parent_2[i])
                if parent_1[i] not in seen_2:
                    child_2[i] = parent_1[i]
                    seen_2.append(parent_1[i])

        for i in range(N):
            if parent_1[i] not in seen_1:
                not_seen_1.append(parent_1[i])
            if parent_2[i] not in seen_2:
                not_seen_2.append(parent_2[i])

        np.random.shuffle(not_seen_1)
        np.random.shuffle(not_seen_2)

        k, m = 0, 0
        for i in range(N):
            if child_1[i] == -1:
                child_1[i] = not_seen_1[k]
                k += 1
            if child_2[i] == -1:
                child_2[i] = not_seen_2[m]
                m += 1

        return child_1, child_2
        
    @staticmethod
    def OBX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        not_seen_1 = []
        not_seen_2 = []

        points = sorted_random_numbers(3, N)

        for i in range(len(points)):
            not_seen_1.append(parent_1[points[i]])
            not_seen_2.append(parent_2[points[i]])

        for i in range(N):
            if i not in points:
                child_1[i] = parent_1[i]
                child_2[i] = parent_2[i]

        k, m = 0, 0
        for i in range(N):
            if parent_2[i] in not_seen_1:
                child_1[points[k]] = parent_2[i]
                k += 1
            if parent_1[i] in not_seen_2:
                child_2[points[m]] = parent_1[i]
                m += 1

        return child_1, child_2

    @staticmethod
    def POS(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        seen_1 = []
        seen_2 = []

        points = sorted_random_numbers(3, N)

        for i in points:
            seen_1.append(parent_2[i])
            seen_2.append(parent_1[i])
            
        for i in points:
            child_1[i] = parent_2[i]
            child_2[i] = parent_1[i]

        points = np.where(child_1 == -1)[0]
        k = 0
        for i in range(N):
            if parent_1[i] not in seen_1:
                child_1[points[k]] = parent_1[i]
                k += 1
                
        k = 0
        for i in range(N):
            if parent_2[i] not in seen_2:
                child_2[points[k]] = parent_2[i]
                k += 1

        return child_1, child_2

    @staticmethod
    def VR(*parents):
        N = len(parents[0])

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)
        threshold = 3

        seen = []
        not_seen = []

        for i in range(N):
            column = []
            for j in range(len(parents)):
                column.append(parents[j][i])

            vote = most_frequent(column)
            votes_num = column.count(vote)
            if votes_num >= threshold:
                child_1[i] = vote
                child_2[i] = vote
                seen.append(vote)

        for i in range(N):
            if parents[0][i] not in seen:
                not_seen.append(parents[0][i])

        np.random.shuffle(not_seen)
        k = 0
        for i in range(N):
            if child_1[i] == -1:
                child_1[i] = not_seen[k]
                k += 1

        np.random.shuffle(not_seen)
        k = 0
        for i in range(N):
            if child_2[i] == -1:
                child_2[i] = not_seen[k]
                k += 1

        return child_1, child_2

    @staticmethod
    def MPX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        seen_1 = []
        seen_2 = []
        
        point_1, point_2 = sorted_random_numbers(2, N)

        for i in range(point_1, point_2):
            child_1[i] = parent_1[i]
            seen_1.append(parent_1[i])

        points = np.where(child_1 == -1)[0]

        k = 0
        for i in range(N):
            if parent_2[i] not in seen_1:
                child_1[points[k]] = parent_2[i]
                k += 1

        point_1, point_2 = sorted_random_numbers(2, N)
        for i in range(point_1, point_2):
            child_2[i] = parent_2[i]
            seen_2.append(parent_2[i])

        points = np.where(child_2 == -1)[0]

        k = 0
        for i in range(N):
            if parent_1[i] not in seen_2:
                child_2[points[k]] = parent_1[i]
                k += 1

        return child_1, child_2
        
    @staticmethod     
    def PX(parent_1: np.array, parent_2: np.array):
        N = len(parent_1)

        child_1 = np.full(N, -1)
        child_2 = np.full(N, -1)

        seen_1 = []
        seen_2 = []

        points_num = np.random.randint(2, N - 2)
        points = sorted_random_numbers(points_num, N)

        for i in points:
            child_1[i] = parent_1[i]
            seen_1.append(parent_1[i])

        points = np.where(child_1 == -1)[0]
        k = 0
        for i in range(N):
            if parent_2[i] not in seen_1:
                child_1[points[k]] = parent_2[i]
                k += 1

        points_num = np.random.randint(2, N - 2)
        points = sorted_random_numbers(points_num, N)

        for i in points:
            child_2[i] = parent_2[i]
            seen_2.append(parent_2[i])

        points = np.where(child_2 == -1)[0]
        k = 0
        for i in range(N):
            if parent_1[i] not in seen_2:
                child_2[points[k]] = parent_1[i]
                k += 1

        return child_1, child_2
                
    @staticmethod
    def ER(parent_1, parent_2):
        def create_neighbor_list(parent):
            N = len(parent)
            neighbor_list = {i: [] for i in parent}
            for i, node in enumerate(parent):
                left_neighbor = parent[i - 1]
                right_neighbor = parent[(i + 1) % N]
                neighbor_list[node].extend([left_neighbor, right_neighbor])
            return neighbor_list
        
        def merge_neighbor_lists(parent_1, parent_2):
            neighbor_list_1 = create_neighbor_list(parent_1)
            neighbor_list_2 = create_neighbor_list(parent_2)
            
            merged_list = deepcopy(neighbor_list_1)
            for node, neighbors in neighbor_list_2.items():
                merged_list[node].extend(neighbors)
                merged_list[node] = list(set(merged_list[node]))
            return merged_list
        
        def generate_child(neighbor_list, start_node):
            N = len(neighbor_list)
            child, current_node = [start_node], start_node
            
            while len(child) < N:
                neighbors = [n for n in neighbor_list[current_node] if n not in child]
                if not neighbors:
                    break 
                next_node = min(neighbors, key=lambda x: len([n for n in neighbor_list[x] if n not in child]))
                child.append(next_node)
                current_node = next_node
                
            if len(child) < N:
                missing = np.setdiff1d(list(neighbor_list.keys()), child, assume_unique=True)
                np.random.shuffle(missing)
                child.extend(missing)
                
            return np.array(child)
        merged_neighbor_list = merge_neighbor_lists(parent_1, parent_2)
        
        child_1 = generate_child(merged_neighbor_list, parent_1[0])
        child_2 = generate_child(merged_neighbor_list, parent_2[0])
        
        return child_1, child_2
