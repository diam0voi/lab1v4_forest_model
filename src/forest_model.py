import numpy as np
from enum import Enum

class CellState(Enum):
    EMPTY = 0
    TREE = 1
    FIRING = 2
    

class NeighborhoodType(Enum):
    # окрестность фон Неймана
    NEUMAN = "+"
    # окрестность Мура
    MOORE = "x"
    

def _has_burning_neighbor(grid, r, c, neighborhood_type):
    """
    Проверяет, есть ли у клетки (r, c) хотя бы
    один горящий сосед с учётом сетки
    """
    h, w = grid.shape
      
    # cмещения для разных окрестностей
    if neighborhood_type == NeighborhoodType.NEUMAN:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif neighborhood_type == NeighborhoodType.MOORE:
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        return False
    
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        # cосед в пределах сетки
        if 0 <= nr < h and 0 <= nc < w:
            if grid[nr, nc] == CellState.FIRING.value:
                return True
    
    return False


def initialize_grid(w, h, eta, f, seed):
    np.random.seed(seed)
    grid = np.full((h, w), CellState.EMPTY.value)
    n = int(eta * w * h)
    
    empty_cells_indices = np.where(grid == CellState.EMPTY.value)
    if len(empty_cells_indices[0]) < n:
        raise ValueError("мало места для деревьев!")
    
    tree_indices = np.random.choice(len(empty_cells_indices[0]), n, replace=False)
    grid[empty_cells_indices[0][tree_indices], empty_cells_indices[1][tree_indices]] = CellState.TREE.value
    
    tree_cells_indices = np.where(grid == CellState.TREE.value)
    if len(tree_cells_indices[0]) < f:
        raise ValueError(f"мало деревьев! Найдено {len(tree_cells_indices[0])}, нужно {f}...")
    
    fire_indices = np.random.choice(len(tree_cells_indices[0]), f, replace=False)
    grid[tree_cells_indices[0][fire_indices], tree_cells_indices[1][fire_indices]] = CellState.FIRING.value
    
    return grid

def apply_rules(grid, Pf, Pg, neighborhood_type):
    h, w = grid.shape
    new_grid = grid.copy()
    
    for r in range(h):
        for c in range(w):
            state = grid[r, c]
            
            if state == CellState.FIRING.value:
                new_grid[r, c] = CellState.EMPTY.value
            elif state == CellState.EMPTY.value:
                if np.random.random() < Pg:
                    new_grid[r, c] = CellState.TREE.value
            elif state == CellState.TREE.value:
                if _has_burning_neighbor(grid, r, c, neighborhood_type):
                    new_grid[r, c] = CellState.FIRING.value
                elif np.random.random() < Pf:
                    new_grid[r, c] = CellState.FIRING.value
    return new_grid

def run_simulation(w, h, eta, Pf, Pg, f, num_iterations, neighborhood_type, seed):
    grid = initialize_grid(w, h, eta, f, seed)
    history = [grid]
    
    stats = {
        'fire': [np.sum(grid == CellState.FIRING.value)],
        'tree': [np.sum(grid == CellState.TREE.value)],
        'empty': [np.sum(grid == CellState.EMPTY.value)]
    }

    for i in range(num_iterations):
        grid = apply_rules(grid, Pf, Pg, neighborhood_type)
        history.append(grid)
        
        stats['fire'].append(np.sum(grid == CellState.FIRING.value))
        stats['tree'].append(np.sum(grid == CellState.TREE.value))
        stats['empty'].append(np.sum(grid == CellState.EMPTY.value))
        
    return {'history': history, 'stats': stats}
