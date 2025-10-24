from enum import Enum


# simplified representation
class TLS(Enum):
    ABSENT = -1
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


# direction
class Direction(Enum):
    L = 0
    S = 1
    R = 2


class Pt:
    def __init__(self, x, y, z=None) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # 路径压缩
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

    def form_groups(self) -> list[list[int]]:
        """
        Given a UnionFind object uf, which represents the edge grouping result, returns a list of groups of edges that
        are meant to be joined into single junctions. The criteria is _is_connection_group().

        For example,
        [ [1,2,3],
          [4],
          [5,6]]
        """

        root_to_elements: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in root_to_elements:
                root_to_elements[root] = []
            root_to_elements[root].append(i)

        groups = [elements for elements in root_to_elements.values()]
        return groups
