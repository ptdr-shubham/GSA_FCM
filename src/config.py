from dataclass import dataclass , field
from typing import List

@dataclass
class GSAparameters:
    """Hyperparameters for the Gravitational Search Algorithm."""
    G0: float = 100.0           # Initial gravitational constant
    alpha: float = 20.0         # Gravitational constant decay rate
    w1: float = 0.9             # Weight for inter-cluster distance (maximize)
    w2: float = 100.0           # Weight for intra-cluster variance (minimize)
    max_iter: int = 100         # Maximum number of iterations


@dataclass
class FCMparameters:
    """Hyperparameters for the Fuzzy C-Means algorithm."""
    m: float = 2.0              # Fuzziness parameter
    epsilon: float = 1e-7       # Convergence threshold
    max_iter: int = 100         # Maximum number of iterations

@dataclass
class ProjectConfig:
    """Centralized configuration for the GSA-FCM clustering project."""
    random_seed: int = 42

    supported_datasets: List[str] = field(default_factory=lambda: ["iris", "wine", "cancer"])

    # Nested configurations
    gsa: GSAParameters = field(default_factory=GSAParameters)
    fcm: FCMParameters = field(default_factory=FCMParameters)

# Instantiate a global config object to be imported by other modules
config = ProjectConfig()