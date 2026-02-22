import math

class WarmupCosineScheduler:
      '''
            Warmup + Cosine Annealing 学习率调度器
            - Linear warmup for 10k steps
            - Cosine decay
      '''
      def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.base_lr = base_lr
            self.min_lr = min_lr
            self.current_step = 0

      def step(self):
            self.current_step += 1
            if self.current_step < self.warmup_steps:
                  lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                  progress = (self.current_step-self.warmup_steps)/(self.total_steps-self.warmup_steps)
                  lr = self.min_lr+(self.base_lr-self.min_lr)*0.5*(1+math.cos(math.pi*progress))
            for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr
            return lr