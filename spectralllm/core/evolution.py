"""
Evolutionary Optimization for Basis Functions
=============================================

Complete implementation of the HRFEvo (Harmonic Response Function Evolution) system
for adaptive basis function optimization in SpectralLLM.
"""

import torch
import torch.nn as nn
import random
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class BasisFunction:
    """
    Represents a basis function that can be evolved for optimal performance.
    
    This class encapsulates different types of basis functions (wavelets, Fourier, etc.)
    and provides genetic operators for evolutionary optimization.
    """
    
    WAVELET_FAMILIES = [
        'db4', 'db8', 'db16', 'sym4', 'sym8', 'sym16',
        'coif2', 'coif4', 'coif6', 'bior2.2', 'bior4.4', 'dmey'
    ]
    
    def __init__(self, type_name: str = 'db4', params: Optional[Dict] = None):
        """
        Initialize a basis function.
        
        Args:
            type_name: Type of basis function (wavelet family, etc.)
            params: Additional parameters for the basis function
        """
        self.type_name = type_name
        self.params = params or {}
        self.fitness = 0.0
        self.age = 0
        
        # Default parameters for different basis types
        self._set_default_params()
    
    def _set_default_params(self):
        """Set default parameters based on basis function type."""
        if self.type_name in self.WAVELET_FAMILIES:
            self.params.setdefault('levels', 3)
            self.params.setdefault('mode', 'reflect')
            self.params.setdefault('frequency_weight', 1.0)
            self.params.setdefault('compression_ratio', 0.1)
        elif self.type_name == 'fourier':
            self.params.setdefault('num_frequencies', 64)
            self.params.setdefault('frequency_range', [0.1, 3.14])
            self.params.setdefault('phase_shift', 0.0)
        else:
            # Custom basis function
            self.params.setdefault('custom_weight', 1.0)
    
    def __repr__(self) -> str:
        return f"BasisFunction(type={self.type_name}, fitness={self.fitness:.4f}, params={self.params})"
    
    def mutate(self, mutation_rate: float = 0.1) -> 'BasisFunction':
        """
        Create a mutated copy of this basis function.
        
        Args:
            mutation_rate: Probability of mutation for each parameter
            
        Returns:
            New mutated BasisFunction
        """
        new_basis = self.copy()
        
        # Mutate type with small probability
        if random.random() < mutation_rate * 0.1:
            if self.type_name in self.WAVELET_FAMILIES:
                new_basis.type_name = random.choice(self.WAVELET_FAMILIES)
            else:
                new_basis.type_name = random.choice(['fourier'] + self.WAVELET_FAMILIES)
            new_basis._set_default_params()
        
        # Mutate parameters
        for key, value in new_basis.params.items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    if key == 'levels':
                        new_basis.params[key] = max(1, min(5, value + random.randint(-1, 1)))
                    elif key == 'num_frequencies':
                        new_basis.params[key] = max(8, min(128, value + random.randint(-8, 8)))
                    elif key in ['frequency_weight', 'compression_ratio', 'custom_weight']:
                        new_basis.params[key] = max(0.1, min(2.0, value * random.uniform(0.8, 1.2)))
                    elif key == 'phase_shift':
                        new_basis.params[key] = (value + random.uniform(-0.5, 0.5)) % (2 * 3.14159)
                elif key == 'mode' and value in ['reflect', 'symmetric', 'periodization']:
                    new_basis.params[key] = random.choice(['reflect', 'symmetric', 'periodization'])
                elif key == 'frequency_range' and isinstance(value, list):
                    new_basis.params[key] = [
                        max(0.05, value[0] * random.uniform(0.9, 1.1)),
                        min(3.14, value[1] * random.uniform(0.9, 1.1))
                    ]
        
        new_basis.age = 0  # Reset age for new mutant
        return new_basis
    
    def crossover(self, other: 'BasisFunction') -> 'BasisFunction':
        """
        Create offspring through crossover with another basis function.
        
        Args:
            other: Another BasisFunction to crossover with
            
        Returns:
            New BasisFunction combining features of both parents
        """
        # Choose dominant parent
        if random.random() < 0.5:
            new_basis = self.copy()
            other_params = other.params
        else:
            new_basis = other.copy()
            other_params = self.params
        
        # Mix parameters
        for key in new_basis.params:
            if key in other_params and random.random() < 0.5:
                if isinstance(new_basis.params[key], (int, float)):
                    # Blend numerical parameters
                    alpha = random.uniform(0.3, 0.7)
                    new_basis.params[key] = (
                        alpha * new_basis.params[key] + 
                        (1 - alpha) * other_params[key]
                    )
                elif isinstance(new_basis.params[key], list):
                    # Mix list parameters
                    for i in range(len(new_basis.params[key])):
                        if i < len(other_params[key]):
                            alpha = random.uniform(0.3, 0.7)
                            new_basis.params[key][i] = (
                                alpha * new_basis.params[key][i] + 
                                (1 - alpha) * other_params[key][i]
                            )
                else:
                    # Copy discrete parameters
                    new_basis.params[key] = other_params[key]
        
        new_basis.age = 0
        new_basis.fitness = 0.0
        return new_basis
    
    def copy(self) -> 'BasisFunction':
        """Create a deep copy of this basis function."""
        new_params = {}
        for key, value in self.params.items():
            if isinstance(value, list):
                new_params[key] = value.copy()
            else:
                new_params[key] = value
        
        new_basis = BasisFunction(self.type_name, new_params)
        new_basis.fitness = self.fitness
        new_basis.age = self.age
        return new_basis
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'type_name': self.type_name,
            'params': self.params,
            'fitness': self.fitness,
            'age': self.age
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BasisFunction':
        """Create BasisFunction from dictionary."""
        basis = cls(data['type_name'], data['params'])
        basis.fitness = data.get('fitness', 0.0)
        basis.age = data.get('age', 0)
        return basis


class HRFEvoController:
    """
    Harmonic Response Function Evolution Controller.
    
    Manages the evolutionary optimization of basis functions for optimal
    spectral representation in SpectralLLM.
    """
    
    def __init__(self, config):
        """
        Initialize the HRFEvo controller.
        
        Args:
            config: Configuration object with evolution parameters
        """
        self.config = config
        self.population_size = getattr(config, 'evolution_population', 20)
        self.num_generations = getattr(config, 'evolution_generations', 10)
        self.mutation_rate = getattr(config, 'mutation_rate', 0.15)
        self.crossover_rate = getattr(config, 'crossover_rate', 0.7)
        self.elite_size = max(1, self.population_size // 10)
        
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize the population with diverse basis functions."""
        self.population = []
        
        # Add some known good basis functions
        good_bases = ['db4', 'db8', 'sym4', 'sym8', 'coif2', 'bior2.2']
        for i, base_type in enumerate(good_bases):
            if i < self.population_size:
                basis = BasisFunction(base_type)
                self.population.append(basis)
        
        # Fill rest with random variations
        while len(self.population) < self.population_size:
            base_type = random.choice(BasisFunction.WAVELET_FAMILIES + ['fourier'])
            basis = BasisFunction(base_type)
            
            # Add some random variation
            basis = basis.mutate(mutation_rate=0.3)
            self.population.append(basis)
    
    def evaluate_population(self, evaluation_func: callable) -> None:
        """
        Evaluate fitness of all individuals in the population.
        
        Args:
            evaluation_func: Function that takes a BasisFunction and returns fitness dict
        """
        for individual in self.population:
            if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                try:
                    fitness_dict = evaluation_func(individual)
                    
                    # Combine multiple fitness metrics
                    if isinstance(fitness_dict, dict):
                        # Weighted combination of metrics
                        perplexity = fitness_dict.get('perplexity', 100.0)
                        efficiency = fitness_dict.get('computation_efficiency', 1.0)
                        
                        # Lower perplexity is better, higher efficiency is better
                        individual.fitness = efficiency / max(perplexity, 1.0)
                    else:
                        individual.fitness = float(fitness_dict)
                        
                except Exception as e:
                    print(f"Error evaluating basis {individual}: {e}")
                    individual.fitness = 0.001  # Very low fitness for failed evaluation
            
            individual.age += 1
    
    def select_parents(self) -> List[BasisFunction]:
        """
        Select parents for reproduction using tournament selection.
        
        Returns:
            List of selected parent BasisFunctions
        """
        parents = []
        tournament_size = max(2, self.population_size // 10)
        
        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def create_offspring(self, parents: List[BasisFunction]) -> List[BasisFunction]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            parents: List of parent BasisFunctions
            
        Returns:
            List of offspring BasisFunctions
        """
        offspring = []
        
        while len(offspring) < len(parents):
            parent1 = random.choice(parents)
            
            if random.random() < self.crossover_rate and len(parents) > 1:
                # Crossover
                parent2 = random.choice(parents)
                child = parent1.crossover(parent2)
            else:
                # Just copy
                child = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = child.mutate(self.mutation_rate)
            
            offspring.append(child)
        
        return offspring
    
    def evolve_generation(self, evaluation_func: callable) -> Dict:
        """
        Run one generation of evolution.
        
        Args:
            evaluation_func: Function to evaluate fitness of basis functions
            
        Returns:
            Dictionary with generation statistics
        """
        # Evaluate current population
        self.evaluate_population(evaluation_func)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track statistics
        best_fitness = self.population[0].fitness
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        diversity = self._calculate_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)
        
        # Elite selection (keep best individuals)
        elite = self.population[:self.elite_size]
        
        # Select parents and create offspring
        parents = self.select_parents()
        offspring = self.create_offspring(parents)
        
        # New population: elite + offspring
        self.population = elite + offspring[:self.population_size - self.elite_size]
        
        self.generation += 1
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'best_basis': self.population[0].to_dict()
        }
    
    def run_evolution(self, evaluation_func: callable) -> BasisFunction:
        """
        Run complete evolution for specified number of generations.
        
        Args:
            evaluation_func: Function to evaluate fitness of basis functions
            
        Returns:
            Best BasisFunction found
        """
        print(f"Starting HRFEvo with population size {self.population_size} for {self.num_generations} generations")
        
        for gen in range(self.num_generations):
            stats = self.evolve_generation(evaluation_func)
            
            if gen % max(1, self.num_generations // 10) == 0:
                print(f"Generation {gen}: Best fitness = {stats['best_fitness']:.4f}, "
                      f"Diversity = {stats['diversity']:.4f}")
        
        # Return best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_basis = self.population[0]
        
        print(f"Evolution complete. Best basis: {best_basis}")
        return best_basis
    
    def get_best_basis(self) -> BasisFunction:
        """Get the current best basis function."""
        if not self.population:
            return BasisFunction()
        
        return max(self.population, key=lambda x: x.fitness)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on basis function types."""
        type_counts = {}
        for individual in self.population:
            type_counts[individual.type_name] = type_counts.get(individual.type_name, 0) + 1
        
        # Shannon diversity index
        total = len(self.population)
        diversity = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                diversity -= p * np.log(p)
        
        return diversity
    
    def save_state(self) -> Dict:
        """Save the current evolution state."""
        return {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history,
            'config': {
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }
    
    def load_state(self, state: Dict) -> None:
        """Load evolution state from dictionary."""
        self.generation = state.get('generation', 0)
        self.population = [BasisFunction.from_dict(ind_data) 
                          for ind_data in state.get('population', [])]
        self.best_fitness_history = state.get('best_fitness_history', [])
        self.diversity_history = state.get('diversity_history', [])
        
        # Load config if provided
        config = state.get('config', {})
        self.population_size = config.get('population_size', self.population_size)
        self.mutation_rate = config.get('mutation_rate', self.mutation_rate)
        self.crossover_rate = config.get('crossover_rate', self.crossover_rate)


class SpectralGapAnalyzer:
    """
    Implements spectral gap analysis for wavelet coefficients as recommended
    by Dr. Tao to measure effectiveness in separating linguistic structures.
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
    
    def compute_laplacian(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the graph Laplacian for embeddings.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_length, embed_dim]
            
        Returns:
            Graph Laplacian matrix
        """
        # Compute pairwise similarities (using cosine similarity)
        # Reshape to [batch_size * seq_length, embed_dim]
        batch_size, seq_length, embed_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        
        # Normalize embeddings for cosine similarity
        norms = torch.norm(flat_embeddings, dim=1, keepdim=True)
        normalized_embeddings = flat_embeddings / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Create adjacency matrix (threshold similarities for sparse graph)
        adjacency = torch.zeros_like(similarity)
        top_k = min(10, seq_length)  # Connect each node to top-k neighbors
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
        
        # Populate adjacency matrix with top-k connections
        for i in range(similarity.size(0)):
            adjacency[i, top_k_indices[i]] = top_k_values[i]
        
        # Make symmetric
        adjacency = 0.5 * (adjacency + adjacency.t())
        
        # Compute degree matrix
        degree = torch.sum(adjacency, dim=1)
        degree_matrix = torch.diag(degree)
        
        # Compute Laplacian: L = D - A
        laplacian = degree_matrix - adjacency
        
        return laplacian
    
    def compute_spectral_gap(self, laplacian: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Compute the spectral gap of the Laplacian matrix.
        
        Args:
            laplacian: Graph Laplacian matrix
            
        Returns:
            Spectral gap and eigenvalues
        """
        # Compute eigenvalues (use only a subset for efficiency)
        try:
            eigenvalues, _ = torch.linalg.eigh(laplacian)
        except:
            # Fallback to CPU if eigendecomposition fails on GPU
            eigenvalues, _ = torch.linalg.eigh(laplacian.cpu())
            eigenvalues = eigenvalues.to(self.device)
        
        # Sort eigenvalues (they should already be sorted, but just to be sure)
        eigenvalues, _ = torch.sort(eigenvalues)
        
        # Compute spectral gap (difference between first non-zero and zero eigenvalue)
        # First eigenvalue should be close to zero
        spectral_gap = eigenvalues[1] - eigenvalues[0]
        
        return spectral_gap.item(), eigenvalues
    
    def analyze_wavelet_representation(self, 
                                      coeffs: Tuple[torch.Tensor, List[torch.Tensor]]) -> Dict:
        """
        Analyze the spectral properties of wavelet coefficients.
        
        Args:
            coeffs: Tuple of (approximation, details) from wavelet transform
            
        Returns:
            Dictionary with spectral analysis metrics
        """
        approx, details = coeffs
        
        # Compute Laplacian for approximation coefficients
        approx_laplacian = self.compute_laplacian(approx)
        approx_gap, approx_eigenvalues = self.compute_spectral_gap(approx_laplacian)
        
        # Compute Laplacians for detail coefficients at each level
        detail_gaps = []
        detail_eigenvalues = []
        
        for detail in details:
            detail_laplacian = self.compute_laplacian(detail)
            gap, eigenvalues = self.compute_spectral_gap(detail_laplacian)
            detail_gaps.append(gap)
            detail_eigenvalues.append(eigenvalues[:5])  # Keep only first few eigenvalues
        
        # Compute overall spectral gap as weighted combination
        # Weight approximation gap higher than detail gaps
        weights = [0.6] + [0.4 / len(detail_gaps)] * len(detail_gaps)
        overall_gap = approx_gap * weights[0] + sum(g * w for g, w in zip(detail_gaps, weights[1:]))
        
        return {
            'approx_gap': approx_gap,
            'detail_gaps': detail_gaps,
            'overall_gap': overall_gap,
            'approx_eigenvalues': approx_eigenvalues[:5],  # Keep only first few eigenvalues
            'detail_eigenvalues': detail_eigenvalues
        } 