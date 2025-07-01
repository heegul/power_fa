# plan.md

## Issue: Generalization Gap in DNN vs. Brute-Force (FS) for Power/FA Allocation

### Observation
- **Single-sample mode:**  
  DNN/FS sum-rate ratio ≈ 0.98 (DNN nearly matches brute-force on the *same* sample it was trained on).
- **Multi-sample mode:**  
  DNN/FS sum-rate ratio drops by ~10%p (e.g., from 0.98 to ~0.88) when training and validating on multiple, diverse samples.

---

### Explanation

1. **Overfitting in Single-Sample Mode**
   - DNN can memorize the optimal mapping for a single scenario.
   - Nearly matches brute-force, but does not generalize.

2. **Generalization in Multi-Sample Mode**
   - DNN must learn a function that works across many channel realizations.
   - Cannot memorize; must generalize, so performance drops.

3. **Why the Drop?**
   - **Diversity:** More diverse training set = harder to fit all cases.
   - **Model Capacity:** DNN may not be large enough for all scenarios.
   - **Optimization:** More complex loss landscape.
   - **Normalization:** Global normalization helps, but not a cure-all.

---

### What Can We Do?

#### A. Improve Generalization
- Increase model capacity (deeper/wider networks).
- Data augmentation (more/different training samples).
- Regularization (dropout, weight decay, etc.).
- Ensemble methods (combine multiple DNNs).
- Curriculum learning (start easy, increase diversity).

#### B. Improve Training
- Hyperparameter tuning (learning rate, batch size, etc.).
- Longer training (more epochs, early stopping).
- Advanced architectures (attention, residuals, etc.).

#### C. Accept the Trade-off
- Some gap is inevitable due to the complexity of the problem.
- The goal is to minimize the gap while maintaining generalization.

---

### Next Steps / To-Do
- [ ] Visualize DNN/FS ratio for single vs. multi-sample training.
- [ ] Experiment with larger models and regularization.
- [ ] Try data augmentation and curriculum learning.
- [ ] Document findings and update this plan as progress is made.

---

*This plan will be continuously updated as we address the generalization gap and improve DNN performance for wireless resource allocation.*

---

*This plan will be continuously updated as we address the generalization gap and improve DNN performance for wireless resource allocation.*
