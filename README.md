# CS 4391/5391 — Reinforcement Learning HW4
## Team 15: Abhijeet Pandey, Ganesh Adhikari, Vishal Thapa

---

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## File Structure

| File | Description |
|------|-------------|
| `buffer.py` | Task 1 — Buffer, environment loop, action sampling |
| `vpg.py` | Task 2 — Vanilla policy gradient (REINFORCE) |
| `gae.py` | Task 3 — Critic network and GAE |
| `ppo.py` | Tasks 4 & 5 — PPO surrogate loss and full PPO |
| `Modules.py` | Shared — NormalModule (policy output head) |
| `plotting.py` | Shared — learning curve and loss curve plotters |
| `video.py` | Shared — video recording utilities |
| `pg.py` | Entry point — runs all tasks end to end |

---

## Running Each Task

### Task 2 — Vanilla Policy Gradient
```bash
python vpg.py
```
Runs VPG at two learning rates (`1e-4` and `3e-4`) and plots learning curves.

### Task 3 — GAE
```bash
python gae.py
```
Compares rewards-to-go vs GAE learning curves.

### Tasks 4 & 5 — PPO
```bash
python ppo.py
```
Runs full PPO for 500 iterations (Task 5). The Task 4 clipped vs unclipped block is included in `ppo.py` and can be uncommented.

### All Tasks
```bash
python pg.py
```
Runs everything end to end; Task 5 uses a learning-rate sweep (1e-4 vs 3e-4). Comment/uncomment blocks for individual tasks.

---

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `gamma` | 0.99 | `train_ppo` |
| `lam` | 0.95 | `compute_gae` |
| `eps_clip` | 0.2 | `ppo_surrogate_loss` |
| `learning_rate` | 3e-4 (Task 5 sweep also uses 1e-4) | `train_ppo` |
| `steps_per_iter` | 2048 | `train_ppo` |
| `sgd_epochs` | 10 | `train_ppo` |
| `hidden_size` | 64 | `train_ppo` |

---

## Notes
- All code runs on CPU — no GPU needed for base assignment
- Training Task 2/3 takes ~10-15 min on CPU for 200 epochs
- Training Task 5 takes ~20-30 min on CPU for 500 iterations
- Videos saved to `videos/` directory if `record_video` is uncommented
