class GreedySetCover:
    """Greedy O(log n) approximation algorithm for Set Cover."""
    
    def solve(self, universe, sets_dict):
        """Solve set cover greedily. Returns list of selected set names."""
        if not universe:
            return []
        
        if not sets_dict:
            return []
        
        # Convert to sets for efficient operations
        universe = set(universe)
        sets_dict = {name: set(elements) for name, elements in sets_dict.items()}
        
        uncovered = universe.copy()
        solution = []
        available_sets = sets_dict.copy()
        
        while uncovered and available_sets:
            # Find the set that covers the most uncovered elements
            best_set_name = self._find_best_set(uncovered, available_sets)
            
            if best_set_name is None:
                # No set covers any uncovered element - shouldn't happen 
                # if input is valid, but handle gracefully
                break
            
            # Add to solution
            solution.append(best_set_name)
            
            # Update uncovered elements
            uncovered -= available_sets[best_set_name]
            
            # Remove selected set to avoid reselection
            del available_sets[best_set_name]
        
        return solution
    
    def _find_best_set(self, uncovered, available_sets):
        """Find set covering most uncovered elements."""
        best_set_name = None
        max_coverage = 0
        
        for set_name, set_elements in available_sets.items():
            # Count how many uncovered elements this set covers
            coverage = len(set_elements & uncovered) # TODO: Might remove empty sets
            
            if coverage > max_coverage:
                max_coverage = coverage
                best_set_name = set_name
        
        return best_set_name
    
    def validate_solution(self, solution, universe, sets_dict):
        """Check if solution covers all elements in universe."""
        universe = set(universe)
        covered = set()
        
        for set_name in solution:
            if set_name in sets_dict:
                covered.update(sets_dict[set_name])
        
        return universe.issubset(covered)
    
    def compute_solution_stats(self, solution, universe, sets_dict):
        """Compute solution statistics (size, coverage, validity)."""
        universe = set(universe)
        covered = set()
        
        for set_name in solution:
            if set_name in sets_dict:
                covered.update(sets_dict[set_name])
        
        return {
            'solution_size': len(solution),
            'universe_size': len(universe),
            'elements_covered': len(covered),
            'coverage_ratio': len(covered) / len(universe) if universe else 1.0,
            'is_valid': universe.issubset(covered),
            'selected_sets': solution.copy()
        }
