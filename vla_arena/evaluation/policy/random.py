import random
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry 
@PolicyRegistry.register("random")
class RandomPolicy(Policy):
    
    def predict(self, obs, **kwargs):
        
        return [random.uniform(-0.1, 0.1) for _ in range(7)]
    
    @property
    def name(self):
        return "random"